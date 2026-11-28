"""使用共享 SQLite 领取全局记忆任务。"""

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
                if job["status"] == "running":
                    return None
                if job["status"] == "error" and job["claimed_version"] == version:
                    if job["retry_remaining"] <= 0:
                        return None
                    retries = min(max_retries, job["retry_remaining"]) - 1

            memories = tuple(LocalMemory(*row) for row in conn.execute(
                """
                SELECT o.session_id, o.rollout_summary
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
                    started_at, retry_remaining, updated_at
                ) VALUES ('phase2', 'global', 'running', ?, ?, ?, ?, ?, ?)
                ON CONFLICT (kind, job_key) DO UPDATE SET
                    status = 'running', claimed_version = excluded.claimed_version,
                    ownership_token = excluded.ownership_token,
                    started_at = excluded.started_at, retry_remaining = excluded.retry_remaining,
                    retry_at = NULL, finished_at = NULL, last_error = NULL, updated_at = excluded.updated_at
                """,
                (version, version, token, now, retries, now),
            )
            return Phase2Claim(token, version, memories)

    def can_publish(self, claim: Phase2Claim) -> bool:
        """在一次读取事务内核对所有权及来源权限，允许普通版本推进。"""
        with closing(get_connection()) as conn:
            conn.execute("BEGIN")
            owned = conn.execute("""SELECT 1 FROM memory_jobs WHERE kind='phase2' AND job_key='global'
                AND status='running' AND ownership_token=? AND claimed_version=?""",
                (claim.token, claim.input_version)).fetchone()
            return bool(owned) and all(conn.execute("""SELECT 1 FROM sessions JOIN stage1_outputs USING(session_id)
                WHERE session_id=? AND memory_enabled=1 AND allow_global=1""", (item.session_id,)).fetchone()
                for item in claim.memories)

    def finish(self, claim: Phase2Claim, *, error: str | None = None) -> bool:
        """仅结束自己领取的任务；根据领取期间的版本推进决定是否继续等待整合。"""
        now = int(time.time())
        with closing(get_connection()) as conn, conn:
            cursor = conn.execute(
                """
                UPDATE memory_jobs
                SET status = CASE WHEN ? IS NOT NULL THEN 'error'
                                  WHEN input_version > claimed_version THEN 'pending' ELSE 'done' END,
                    claimed_version = CASE WHEN ? IS NULL THEN NULL ELSE claimed_version END,
                    ownership_token = NULL, last_error = ?, finished_at = ?, updated_at = ?
                WHERE kind = 'phase2' AND job_key = 'global' AND status = 'running'
                  AND ownership_token = ? AND claimed_version = ?
                """,
                (error, error, error, now, now, claim.token, claim.input_version),
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
