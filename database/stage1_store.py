import time
from contextlib import ExitStack, closing
from dataclasses import dataclass
from sqlite3 import Connection
from uuid import uuid4

from .sqlite import get_connection


_ADVANCE_PHASE2_SQL = """
    INSERT INTO memory_jobs (kind, job_key, input_version, updated_at)
    VALUES ('phase2', 'global', 1, ?)
    ON CONFLICT (kind, job_key) DO UPDATE SET
        input_version = memory_jobs.input_version + 1,
        status = CASE
            WHEN memory_jobs.status = 'running' THEN 'running'
            ELSE 'pending'
        END,
        updated_at = excluded.updated_at
"""


@dataclass(frozen=True)
class Stage1Claim:
    """一次 Phase 1 领取的日志路径和执行凭证。"""

    session_id: str
    event_log_path: str
    ownership_token: str


class Stage1Store:
    def claim_stage1(
        self,
        *,
        scan_limit: int,
        max_running: int,
        max_age_days: int,
        min_idle_seconds: int,
        lease_seconds: int,
        max_attempts: int,
        excluded_sessions: tuple[str, ...] = (),
    ) -> Stage1Claim | None:
        """原子领取最近的待处理会话；活跃租约和处于退避的同一版本不重复领取。"""
        placeholders = ",".join("?" for _ in excluded_sessions)
        exclude = f"AND c.session_id NOT IN ({placeholders})" if excluded_sessions else ""
        with closing(get_connection()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            now = int(time.time())
            row = conn.execute(
                f"""
                WITH eligible AS (
                    SELECT s.session_id, s.event_log_path, o.last_completed_seq, o.source_max_seq,
                           COALESCE(s.updated_at, o.last_completed_at, s.created_at) AS activity
                    FROM sessions AS s
                    JOIN stage1_outputs AS o ON o.session_id = s.session_id
                    WHERE s.is_archived = 0 AND s.memory_enabled = 1
                      AND o.last_completed_seq IS NOT NULL
                      AND COALESCE(s.updated_at, o.last_completed_at, s.created_at) BETWEEN ? AND ?
                    ORDER BY activity DESC, s.session_id
                    LIMIT ?
                )
                SELECT c.session_id, c.event_log_path, c.last_completed_seq,
                       CASE WHEN j.job_key IS NULL OR j.status = 'pending'
                                  OR c.last_completed_seq > j.input_version
                            THEN ? ELSE j.retry_remaining END AS attempts
                FROM eligible AS c
                LEFT JOIN memory_jobs AS j ON j.kind = 'stage1' AND j.job_key = c.session_id
                WHERE (c.source_max_seq IS NULL OR c.source_max_seq < c.last_completed_seq)
                  AND (j.last_success_version IS NULL OR j.last_success_version < c.last_completed_seq)
                  AND (j.status IS NULL OR j.status <> 'running' OR j.lease_until <= ?)
                  AND (j.job_key IS NULL OR j.status = 'pending'
                       OR c.last_completed_seq > j.input_version
                       OR (j.retry_remaining > 0 AND COALESCE(j.retry_at, 0) <= ?))
                  AND (SELECT COUNT(*) FROM memory_jobs
                       WHERE kind = 'stage1' AND status = 'running' AND lease_until > ?) < ?
                  {exclude}
                ORDER BY c.activity DESC, c.session_id
                LIMIT 1
                """,
                (now - max_age_days * 86400, now - min_idle_seconds,
                 scan_limit, max_attempts, now, now, now, max_running, *excluded_sessions),
            ).fetchone()
            if row is None:
                return None

            claim = Stage1Claim(
                row["session_id"], row["event_log_path"], uuid4().hex,
            )
            conn.execute(
                """
                INSERT INTO memory_jobs (
                    kind, job_key, status, input_version, claimed_version, ownership_token,
                    lease_until, retry_remaining, started_at, updated_at
                ) VALUES ('stage1', ?, 'running', ?, NULL, ?, ?, ?, ?, ?)
                ON CONFLICT (kind, job_key) DO UPDATE SET
                    status = 'running', input_version = excluded.input_version,
                    claimed_version = excluded.claimed_version,
                    ownership_token = excluded.ownership_token,
                    lease_until = excluded.lease_until, retry_at = NULL,
                    retry_remaining = excluded.retry_remaining, last_error = NULL,
                    started_at = excluded.started_at, finished_at = NULL,
                    updated_at = excluded.updated_at
                """,
                (claim.session_id, row["last_completed_seq"], claim.ownership_token,
                 now + lease_seconds, row["attempts"] - 1, now, now),
            )
            return claim

    def fail_stage1(self, claim: Stage1Claim, error: str, *, retry_delay_seconds: int) -> bool:
        """仅将仍属于该凭证的任务设为失败，保留上次成功结果。"""
        now = int(time.time())
        with closing(get_connection()) as conn, conn:
            cursor = conn.execute(
                """
                UPDATE memory_jobs
                SET status = 'error', claimed_version = NULL, ownership_token = NULL,
                    lease_until = NULL, retry_at = ?, last_error = ?, finished_at = ?, updated_at = ?
                WHERE kind = 'stage1' AND job_key = ? AND status = 'running'
                  AND ownership_token = ?
                """,
                (now + retry_delay_seconds, error, now, now,
                 claim.session_id, claim.ownership_token),
            )
            return cursor.rowcount == 1

    def record_session_activity(self, session_id: str, updated_at: int, completed_seq: int | None) -> None:
        """同步事件日志活动时间；完整 Turn 水位在同一个事务中更新。"""
        with closing(get_connection()) as conn, conn:
            conn.execute(
                """
                UPDATE sessions SET updated_at = MAX(COALESCE(updated_at, created_at), ?)
                WHERE session_id = ?
                """,
                (updated_at, session_id),
            )
            if completed_seq is not None:
                self.record_turn_end(session_id, completed_seq, updated_at, connection=conn)

    def set_memory_enabled(self, session_id: str, enabled: bool) -> bool:
        """设置会话能否被 Phase 1 领取，返回开关是否发生变化。"""
        with closing(get_connection()) as conn, conn:
            cursor = conn.execute(
                "UPDATE sessions SET memory_enabled = ? WHERE session_id = ? AND memory_enabled <> ?",
                (enabled, session_id, enabled),
            )
            if cursor.rowcount:
                conn.execute(_ADVANCE_PHASE2_SQL, (int(time.time()),))
                if not enabled:
                    conn.execute("""UPDATE memory_jobs SET status='pending', ownership_token=NULL,
                                 claimed_version=NULL, lease_until=NULL WHERE kind='stage1'
                                 AND job_key=? AND status='running'""", (session_id,))
            return cursor.rowcount == 1

    def record_turn_end(
        self,
        session_id: str,
        seq: int,
        completed_at: int,
        *,
        connection: Connection | None = None,
    ) -> bool:
        """仅记录更新的完整 Turn，可复用调用方已开启的事务。"""
        with ExitStack() as stack:
            if connection is None:
                connection = stack.enter_context(closing(get_connection()))
                stack.enter_context(connection)
            cursor = connection.execute(
                """
                INSERT INTO stage1_outputs (
                    session_id, last_completed_seq, last_completed_at
                ) VALUES (?, ?, ?)
                ON CONFLICT (session_id) DO UPDATE SET
                    last_completed_seq = excluded.last_completed_seq,
                    last_completed_at = excluded.last_completed_at
                WHERE stage1_outputs.last_completed_seq IS NULL
                   OR excluded.last_completed_seq > stage1_outputs.last_completed_seq
                """,
                (session_id, seq, completed_at),
            )
            return cursor.rowcount == 1

    def commit_stage1_result(
        self,
        session_id: str,
        source_seq: int,
        summary: str,
        slug: str,
        ownership_token: str,
    ) -> bool:
        """原子提交已领取版本的提取结果及任务状态。"""
        with closing(get_connection()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            previous = conn.execute(
                """
                SELECT outputs.last_completed_seq, outputs.rollout_summary,
                       outputs.allow_global
                FROM stage1_outputs AS outputs
                JOIN memory_jobs AS jobs
                  ON jobs.kind = 'stage1' AND jobs.job_key = outputs.session_id
                WHERE outputs.session_id = ?
                  AND jobs.status = 'running'
                  AND jobs.ownership_token = ?
                  AND jobs.ownership_token <> ''
                  AND jobs.input_version <= ?
                  AND EXISTS (SELECT 1 FROM sessions WHERE session_id=outputs.session_id AND memory_enabled=1)
                  AND outputs.last_completed_seq >= ?
                  AND (outputs.source_max_seq IS NULL OR outputs.source_max_seq < ?)
                """,
                (session_id, ownership_token, source_seq, source_seq, source_seq),
            ).fetchone()
            if previous is None:
                return False

            now = int(time.time())
            conn.execute(
                """
                UPDATE stage1_outputs
                SET source_max_seq = ?, rollout_summary = ?, rollout_slug = ?,
                    generated_at = ?
                WHERE session_id = ?
                """,
                (source_seq, summary, slug, now, session_id),
            )
            status = 'pending' if previous["last_completed_seq"] > source_seq else 'done'
            conn.execute(
                """
                UPDATE memory_jobs
                SET last_success_version = ?, input_version = ?, status = ?,
                    claimed_version = NULL, ownership_token = NULL, lease_until = NULL,
                    retry_at = NULL, last_error = NULL, finished_at = ?, updated_at = ?
                WHERE kind = 'stage1' AND job_key = ?
                """,
                (source_seq, source_seq, status, now, now, session_id),
            )
            if previous["allow_global"] and (previous["rollout_summary"] or summary):
                conn.execute(_ADVANCE_PHASE2_SQL, (now,))
            return True

    def set_allow_global(self, session_id: str, allow_global: bool) -> bool:
        """更新全局参与权限，并在实际变化时推进 Phase 2 输入版本。"""
        with closing(get_connection()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """
                INSERT INTO stage1_outputs (session_id, allow_global)
                VALUES (?, ?)
                ON CONFLICT (session_id) DO UPDATE SET
                    allow_global = excluded.allow_global
                WHERE stage1_outputs.allow_global <> excluded.allow_global
                """,
                (session_id, allow_global),
            )
            if cursor.rowcount == 0:
                return False

            conn.execute(_ADVANCE_PHASE2_SQL, (int(time.time()),))
            return True
