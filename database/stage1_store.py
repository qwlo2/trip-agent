import time
from contextlib import ExitStack, closing
from sqlite3 import Connection

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


class Stage1Store:
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
                       outputs.allow_global, jobs.input_version
                FROM stage1_outputs AS outputs
                JOIN memory_jobs AS jobs
                  ON jobs.kind = 'stage1' AND jobs.job_key = outputs.session_id
                WHERE outputs.session_id = ?
                  AND jobs.status = 'running'
                  AND jobs.ownership_token = ?
                  AND jobs.ownership_token <> ''
                  AND jobs.claimed_version = ?
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
            input_version = max(previous["input_version"], previous["last_completed_seq"])
            status = 'pending' if input_version > source_seq else 'done'
            conn.execute(
                """
                UPDATE memory_jobs
                SET last_success_version = ?, input_version = ?, status = ?,
                    claimed_version = NULL, ownership_token = NULL, lease_until = NULL,
                    retry_at = NULL, last_error = NULL, finished_at = ?, updated_at = ?
                WHERE kind = 'stage1' AND job_key = ?
                """,
                (source_seq, input_version, status, now, now, session_id),
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
