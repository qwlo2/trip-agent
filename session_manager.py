"""用 SQLite 管理会话目录，为每个会话加载独立的事件日志和上下文。"""

import time
from contextlib import closing
from typing import Any
from uuid import uuid4

from client import Trip_Client
from compact import Compact
from context_builder import ContextBuilder
from data_paths import get_data_paths
from database.sqlite import get_connection, init_database
from database.stage1_store import Stage1Store, _ADVANCE_PHASE2_SQL
from history import HistoryStore
from session import Session


SESSION_PAGE_SIZE = 25


class SessionManager:
    """创建、打开、列出、归档和删除会话，缓存最近一次列表的日志路径。"""

    def __init__(
        self,
        client: Trip_Client,
        system_prompt: str,
        compact_settings: dict[str, Any],
    ) -> None:
        """初始化表结构；复用客户端，每次打开会话时新建上下文及 Compact。

        compact_settings 使用入口已校验的压缩配置，例如 load_compact_settings()。
        客户端的创建和关闭由调用方负责。
        """
        init_database()
        self.paths = get_data_paths()
        self.client = client
        self.system_prompt = system_prompt
        self.compact_settings = dict(compact_settings)
        self.cache: dict[str, str] = {}
        self.current_session: Session | None = None

    def open(self, session_id: str) -> Session:
        """重放日志，补齐已完成轮次的记忆记录，再增加 usage。

        ID 不存在时抛出 KeyError；日志丢失时抛出 FileNotFoundError。
        归档会话仍可通过明确的 ID 打开，归档标记保持不变。
        """
        with closing(get_connection()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT event_log_path FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"会话不存在：{session_id}")
            path = self.paths.resolve_event_log(session_id, row["event_log_path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            session = Session(
                HistoryStore(path),
                ContextBuilder(),
                self.system_prompt,
                Compact(self.client, **self.compact_settings),
                session_id=session_id,
            )
            log_updated_at = int(path.stat().st_mtime)
            if session.last_completed_seq is not None:
                # 日志不含结束时间，补录时用文件修改时间估算。
                Stage1Store().record_turn_end(
                    session_id,
                    session.last_completed_seq,
                    log_updated_at,
                    connection=conn,
                )
            conn.execute(
                """
                UPDATE sessions
                SET usage = usage + 1, updated_at = MAX(COALESCE(updated_at, created_at), ?)
                WHERE session_id = ?
                """,
                (log_updated_at, session_id),
            )
        self.current_session = session
        return session

    def get_sessions(self, limit: int = SESSION_PAGE_SIZE, offset: int = 0) -> list[dict[str, Any]]:
        """按使用次数降序返回未归档会话，并将本页路径保存为 ID 到路径的字典。

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
        sessions = [dict(row) for row in rows]
        self.cache = {row["session_id"]: row["event_log_path"] for row in sessions}
        return sessions

    def archived(self, session_id: str, is_archived: bool = True) -> bool:
        """设置归档标记，传入 False 可恢复展示；返回标记是否实际改变。"""
        with closing(get_connection()) as conn, conn:
            cursor = conn.execute(
                """
                UPDATE sessions SET is_archived = ?
                WHERE session_id = ? AND is_archived <> ?
                """,
                (is_archived, session_id, is_archived),
            )
        self.cache.pop(session_id, None)
        return cursor.rowcount == 1

    def delete(self, session_id: str) -> bool:
        """删除日志、会话、局部记忆和对应 Phase 1 任务，返回是否找到会话。

        已进入全局记忆的内容留待下一次 Phase 2 筛选淘汰；相关输入版本会推进。
        日志删除失败时异常直接传播，数据库事务回滚。
        """
        with closing(get_connection()) as conn, conn:
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
            path.unlink(missing_ok=True)
        self.cache.pop(session_id, None)
        if self.current_session is not None and self.current_session.session_id == session_id:
            self.current_session = None
        return True

    def create(self, name: str) -> Session:
        """生成 UUID，在 sessions 和 stage1_outputs 中建档，再打开新会话。

        日志保存在会话目录下的 events.jsonl，首次打开写入系统消息。
        新会话打开成功后的 usage 为 1。
        """
        name = name.strip()
        if not name:
            raise ValueError("会话名称不能为空")
        session_id = str(uuid4())
        path = self.paths.event_log(session_id)
        log_path = path.relative_to(self.paths.root).as_posix()
        with closing(get_connection()) as conn, conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, event_log_path, name, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, log_path, name, int(time.time())),
            )
            conn.execute("INSERT INTO stage1_outputs (session_id) VALUES (?)", (session_id,))
            path.parent.mkdir()
            path.touch(exist_ok=False)
        return self.open(session_id)
