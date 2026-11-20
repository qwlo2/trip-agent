"""使用共享 SQLite 领取全局记忆任务及保存最近成功的入选版本。"""

import time
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from uuid import uuid4

from .sqlite import get_connection


@dataclass(frozen=True)
class LocalMemory:
    """此次全局整合使用的局部记忆版本。"""

    session_id: str
    source_seq: int
    summary: str


@dataclass(frozen=True)
class Phase2Claim:
    """全局任务的所有权、输入版本和同一事务内选出的候选快照。"""

    token: str
    input_version: int
    memories: tuple[LocalMemory, ...]


class Phase2Store:
    """管理唯一的 Phase 2 全局任务。"""

    def claim(self, *, limit: int, active_days: int, interval: int, max_retries: int) -> Phase2Claim | None:
        """在同一事务中检查冷却、领取任务并提取本轮入选的局部记忆。"""
        now = int(time.time())
        with closing(get_connection()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            job = conn.execute(
                "SELECT * FROM memory_jobs WHERE kind = 'phase2' AND job_key = 'global'"
            ).fetchone()
            version = job["input_version"] if job else 0
            retries = max_retries
            if job:
                if job["started_at"] is not None and now < job["started_at"] + interval:
                    return None
                if job["status"] == "running" and (job["lease_until"] or 0) > now:
                    return None
                if (job["status"] in ("error", "running") and job["claimed_version"] == version):
                    if job["retry_remaining"] <= 0:
                        return None
                    retries = min(max_retries, job["retry_remaining"]) - 1

            memories = tuple(LocalMemory(*row) for row in conn.execute(
                """
                SELECT o.session_id, o.source_max_seq, o.rollout_summary
                FROM stage1_outputs AS o
                JOIN sessions AS s ON s.session_id = o.session_id
                WHERE o.allow_global = 1 AND s.memory_enabled = 1
                  AND o.source_max_seq IS NOT NULL AND TRIM(o.rollout_summary) <> ''
                  AND COALESCE(o.last_usage, o.generated_at) BETWEEN ? AND ?
                ORDER BY o.usage_count DESC, COALESCE(o.last_usage, o.generated_at) DESC, o.session_id
                LIMIT ?
                """,
                (now - active_days * 86400, now, limit),
            ))
            token = uuid4().hex
            conn.execute(
                """
                INSERT INTO memory_jobs (
                    kind, job_key, status, input_version, claimed_version, ownership_token,
                    lease_until, started_at, retry_remaining, updated_at
                ) VALUES ('phase2', 'global', 'running', ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (kind, job_key) DO UPDATE SET
                    status = 'running', claimed_version = excluded.claimed_version,
                    ownership_token = excluded.ownership_token, lease_until = excluded.lease_until,
                    started_at = excluded.started_at, retry_remaining = excluded.retry_remaining,
                    retry_at = NULL, finished_at = NULL, last_error = NULL, updated_at = excluded.updated_at
                """,
                (version, version, token, now + interval, now, retries, now),
            )
            return Phase2Claim(token, version, memories)

    def finish(self, claim: Phase2Claim, *, error: str | None = None) -> bool:
        """仅结束自己领取的任务；成功时原子更新入选标记和成功版本。"""
        now = int(time.time())
        with closing(get_connection()) as conn, conn:
            cursor = conn.execute(
                """
                UPDATE memory_jobs
                SET status = CASE WHEN ? IS NOT NULL THEN 'error'
                                  WHEN input_version > claimed_version THEN 'pending' ELSE 'done' END,
                    claimed_version = CASE WHEN ? IS NULL THEN NULL ELSE claimed_version END,
                    last_success_version = CASE WHEN ? IS NULL THEN claimed_version ELSE last_success_version END,
                    ownership_token = NULL, lease_until = NULL, last_error = ?, finished_at = ?, updated_at = ?
                WHERE kind = 'phase2' AND job_key = 'global' AND status = 'running'
                  AND ownership_token = ? AND claimed_version = ?
                """,
                (error, error, error, error, now, now, claim.token, claim.input_version),
            )
            if cursor.rowcount and error is None:
                conn.execute(
                    """
                    UPDATE stage1_outputs SET selected_for_phase2 = 0, selected_for_phase2_source_max_seq = NULL
                    WHERE selected_for_phase2 = 1
                    """
                )
                conn.executemany(
                    """
                    UPDATE stage1_outputs SET selected_for_phase2 = 1, selected_for_phase2_source_max_seq = ?
                    WHERE session_id = ?
                    """,
                    ((item.source_seq, item.session_id) for item in claim.memories),
                )
            return cursor.rowcount == 1

    def record_usage(self, session_ids: Iterable[str]) -> None:
        """记录调用方实际使用过的记忆，同批重复引用只计一次。"""
        now = int(time.time())
        with closing(get_connection()) as conn, conn:
            conn.executemany(
                """
                UPDATE stage1_outputs SET usage_count = usage_count + 1, last_usage = ?
                WHERE session_id = ? AND source_max_seq IS NOT NULL AND TRIM(rollout_summary) <> ''
                """,
                ((now, session_id) for session_id in set(session_ids)),
            )
