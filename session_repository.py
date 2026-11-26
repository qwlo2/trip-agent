"""用 SQLite 管理会话目录，为每个会话加载独立的事件日志和上下文。"""

import sqlite3
import time
from contextlib import closing
from typing import Any
from uuid import uuid4

from client import AsyncTripClient
from compact import Compact
from context_builder import ContextBuilder
from data_paths import get_data_paths
from database.sqlite import get_connection, init_database
from database.stage1_store import Stage1Store, _ADVANCE_PHASE2_SQL
from history import HistoryStore
from session import Session, SessionIntegrityError


SESSION_PAGE_SIZE = 25


class SessionNotFound(KeyError):
    """目录中没有请求的 Session。"""


class LogCleanupError(OSError):
    """目录记录已确认删除，日志文件未能清理。"""


class SessionRepository:
    """创建、打开、列出、归档和删除会话。"""

    def __init__(
        self,
        system_prompt: str,
        compact_settings: dict[str, Any],
    ) -> None:
        """初始化目录和数据库；加载时显式接收本次使用的客户端。"""
        init_database()
        self.paths = get_data_paths()
        self.system_prompt = system_prompt
        self.compact_settings = dict(compact_settings)

    def record_open(self, session_id: str) -> str:
        """只记录用户选择，不重新创建日志写入者。"""
        with closing(get_connection()) as conn, conn:
            if not conn.execute("UPDATE sessions SET usage=usage+1 WHERE session_id=?", (session_id,)).rowcount:
                raise SessionNotFound(f"会话不存在：{session_id}")
        return session_id

    def load(self, session_id: str, client: AsyncTripClient) -> Session:
        """在事务外重放及验证，随后用短事务检查并补录完整 Turn 水位。"""
        with closing(get_connection()) as conn:
            row = conn.execute("SELECT event_log_path FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if row is None:
            raise SessionNotFound(f"会话不存在：{session_id}")
        path = self.paths.resolve_event_log(session_id, row["event_log_path"])
        compact = Compact(client, **self.compact_settings)
        session = Session(HistoryStore(path), ContextBuilder(), self.system_prompt, compact, session_id=session_id)
        activity = int(path.stat().st_mtime)
        with closing(get_connection()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT last_completed_seq FROM stage1_outputs WHERE session_id=?",
                               (session_id,)).fetchone()
            watermark = row[0] if row else None
            latest = session.last_completed_seq
            if watermark is not None:
                if latest is None or watermark > latest or watermark < 0:
                    raise SessionIntegrityError("SQLite 水位超出日志的 completed 事件范围")
                event = session.history.get(watermark)
                if event["type"] != "turn/end" or event.get("reason") != "completed":
                    raise SessionIntegrityError("SQLite 水位未对应日志的 completed 事件")
            conn.execute("""
                UPDATE sessions SET updated_at=MAX(created_at, ?)
                WHERE session_id=? AND (updated_at IS NULL OR updated_at < ?)
            """, (activity, session_id, activity))
            if latest is not None and latest != watermark:
                Stage1Store().record_turn_end(session_id, latest, activity, connection=conn)
        return session

    def list_sessions(self, limit: int = SESSION_PAGE_SIZE, offset: int = 0) -> list[dict[str, Any]]:
        """按使用次数降序返回未归档会话。

        每页默认 25 条，参考 Codex 会话选择器；offset 用于获取后续页。
        使用次数相同时优先显示新建会话，再按 ID 排序以保持顺序稳定。
        """
        if limit <= 0 or offset < 0:
            raise ValueError("limit 必须为正整数，offset 不能小于 0")
        with closing(get_connection()) as conn:
            rows = conn.execute(
                """
                SELECT session_id, event_log_path, name, is_archived, usage, created_at
                FROM sessions
                WHERE is_archived = 0
                ORDER BY usage DESC, created_at DESC, session_id
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [dict(row) for row in rows]

    def archive(self, session_id: str, is_archived: bool = True) -> bool:
        """设置归档标记，传入 False 可恢复展示；返回标记是否实际改变。"""
        with closing(get_connection()) as conn, conn:
            cursor = conn.execute(
                """
                UPDATE sessions SET is_archived = ?
                WHERE session_id = ? AND is_archived <> ?
                """,
                (is_archived, session_id, is_archived),
            )
        return cursor.rowcount == 1

    def delete(self, session_id: str) -> bool:
        """删除日志、会话、局部记忆和对应 Phase 1 任务，返回是否找到会话。

        已进入全局记忆的内容留待下一次 Phase 2 筛选淘汰；相关输入版本会推进。
        目录提交成功后才移除日志，清理失败单独报告。
        """
        commit_error = None
        with closing(get_connection()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT sessions.event_log_path, outputs.allow_global, outputs.rollout_summary
                FROM sessions
                LEFT JOIN stage1_outputs AS outputs USING (session_id)
                WHERE sessions.session_id = ?
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                return False
            path = self.paths.resolve_event_log(session_id, row["event_log_path"])
            conn.execute(
                "DELETE FROM memory_jobs WHERE kind = 'stage1' AND job_key = ?",
                (session_id,),
            )
            conn.execute("DELETE FROM stage1_outputs WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            if row["allow_global"] and row["rollout_summary"]:
                conn.execute(_ADVANCE_PHASE2_SQL, (int(time.time()),))
            try:
                conn.commit()
            except sqlite3.Error as exc:
                commit_error = exc
        if commit_error is not None:
            try:
                with closing(get_connection()) as conn:
                    exists = conn.execute("SELECT 1 FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            except sqlite3.Error as exc:
                raise SessionIntegrityError("无法确认会话删除事务结果") from exc
            if exists:
                raise commit_error
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise LogCleanupError(f"数据库已删除会话，日志清理失败：{path}") from exc
        return True

    def create(self, name: str) -> str:
        """先建立独有的日志，再发布目录记录；提交不确定时保留已有记录的文件。"""
        name = name.strip()
        if not name:
            raise ValueError("会话名称不能为空")
        session_id = str(uuid4())
        path = self.paths.event_log(session_id)
        path.parent.mkdir()
        try:
            HistoryStore(path).load(self.system_prompt)
            with closing(get_connection()) as conn, conn:
                conn.execute(
                    "INSERT INTO sessions(session_id,event_log_path,name,created_at,usage) VALUES (?,?,?,?,1)",
                    (session_id, path.relative_to(self.paths.root).as_posix(), name, int(time.time())),
                )
                conn.execute("INSERT INTO stage1_outputs(session_id) VALUES (?)", (session_id,))
        except Exception:
            with closing(get_connection()) as conn:
                exists = conn.execute("SELECT 1 FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if not exists:
                path.unlink(missing_ok=True)
                path.parent.rmdir()
            raise
        return session_id
