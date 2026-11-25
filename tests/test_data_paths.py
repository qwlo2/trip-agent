"""验证数据目录解析、会话日志安全和 SQLite 兼容性。"""

import os
import shutil
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import data_paths
from data_paths import DataPaths, get_data_paths
from database.sqlite import get_connection
from database.stage1_store import Stage1Store
from history import HistoryStore
from memory.phase1.event_reader import EventReader
from session_manager import SessionManager


class DataPathsTests(unittest.TestCase):
    """所有读写使用临时目录，避免操作项目实有数据。"""

    def setUp(self) -> None:
        """准备独立配置和数据目录，并在每次测试后恢复路径缓存。"""
        workspace = TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.project = Path(workspace.name).resolve()
        project_root = patch.object(data_paths, "PROJECT_ROOT", self.project)
        self.addCleanup(project_root.stop)
        project_root.start()
        environment = patch.dict(os.environ)
        self.addCleanup(environment.stop)
        environment.start()
        os.environ.pop("DSH_DATA_DIR", None)
        get_data_paths.cache_clear()
        self.addCleanup(get_data_paths.cache_clear)

    def test_config_is_project_relative(self) -> None:
        """默认值和相对配置不受工作目录影响，系统环境可指定绝对路径。"""
        self.assertEqual(DataPaths("data").root, self.project / "data")
        self.assertEqual(get_data_paths().root, self.project / "data")
        (self.project / ".env").write_text("DSH_DATA_DIR=./custom/state\n", encoding="utf-8")
        original_cwd = Path.cwd()
        with TemporaryDirectory() as another_cwd:
            try:
                os.chdir(another_cwd)
                get_data_paths.cache_clear()
                self.assertEqual(get_data_paths().root, self.project / "custom/state")
                absolute = self.project / "override"
                os.environ["DSH_DATA_DIR"] = str(absolute)
                get_data_paths.cache_clear()
                self.assertEqual(get_data_paths().root, absolute)
            finally:
                os.chdir(original_cwd)

    def test_directories_and_session_paths(self) -> None:
        """公共目录可重复初始化，保留已有数据且可单独创建会话。"""
        paths = get_data_paths()
        paths.ensure_directories()
        self.assertTrue(paths.sessions.is_dir())
        self.assertTrue(paths.rollout_summaries.is_dir())
        self.assertFalse(paths.database.exists())
        self.assertEqual(list(paths.sessions.iterdir()), [])
        self.assertEqual(paths.session_dir("trip-1"), paths.root / "sessions/trip-1")
        self.assertEqual(paths.event_log("trip-1"), paths.root / "sessions/trip-1/events.jsonl")
        self.assertEqual(paths.phase2_workspace_diff, paths.memories / "phase2_workspace_diff.md")
        paths.memory_summary.write_text("已有记忆", encoding="utf-8")
        paths.ensure_directories()
        self.assertEqual(paths.memory_summary.read_text(encoding="utf-8"), "已有记忆")

    def test_paths_cannot_escape_session(self) -> None:
        """拒绝非法 ID、越界数据库路径以及指向其他文件的符号链接。"""
        paths = get_data_paths()
        paths.ensure_directories()
        for session_id in ("", ".", "..", "../outside", "/outside", "a/b", "a\\b"):
            with self.subTest(session_id=session_id), self.assertRaises(ValueError):
                paths.event_log(session_id)
        for log_path in ("../outside", "/outside", "sessions/s2/events.jsonl", "sessions/s1/../s1/events.jsonl"):
            with self.subTest(log_path=log_path), self.assertRaises(ValueError):
                paths.resolve_event_log("s1", log_path)
        outside = self.project / "outside"
        outside.mkdir()
        paths.sessions.joinpath("redirect").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            paths.event_log("redirect")
        directory = paths.session_dir("s1")
        directory.mkdir()
        directory.joinpath("events.jsonl").symlink_to(outside / "events.jsonl")
        with self.assertRaises(ValueError):
            paths.resolve_event_log("s1", "sessions/s1/events.jsonl")

    def test_legacy_standalone_history(self) -> None:
        """默认问答继续读取旧日志，新目录使用标准事件日志路径。"""
        paths = get_data_paths()
        self.assertEqual(HistoryStore().path, paths.event_log("default"))
        paths.ensure_directories()
        legacy = HistoryStore(paths.root / "history.jsonl")
        legacy.load("旧系统提示")
        self.assertEqual(HistoryStore().load("新系统提示")[0]["message"]["content"], "旧系统提示")

    def test_sessions_and_memory_share_relocatable_state(self) -> None:
        """迁移测试副本后，会话日志和局部记忆仍从同一个状态库读取。"""
        settings = {
            "context_window": 100_000,
            "max_output_tokens": 5_000,
            "threshold_ratio": 0.8,
            "retain_ratio": 0.16,
        }
        manager = SessionManager(None, "system", settings)
        session = manager.create("测试会话")
        session_id = session.session_id
        session.append_history([{
            "type": "message", "turn": 1, "message": {"role": "user", "content": "你好"},
        }])
        # 创建真实的日志完成边界，再验证水位可以随数据目录迁移。
        session.history.append([{"type": "message", "turn": 1, "message": {"role": "assistant", "content": "你好"}},
                                {"type": "turn/end", "turn": 1, "reason": "completed"}])
        self.assertTrue(Stage1Store().record_turn_end(session_id, 3, 100))
        paths = get_data_paths()
        with closing(get_connection()) as conn, conn:
            conn.execute("UPDATE stage1_outputs SET rollout_summary = '旧摘要' WHERE session_id = ?", (session_id,))
        log_path = manager.get_sessions()[0]["event_log_path"]
        self.assertEqual(log_path, f"sessions/{session_id}/events.jsonl")
        self.assertEqual(paths.resolve_event_log(session_id, log_path), session.history.path)
        self.assertEqual(list(paths.root.glob("*.db")), [paths.database])

        moved = self.project / "moved"
        shutil.copytree(paths.root, moved)
        os.environ["DSH_DATA_DIR"] = str(moved)
        get_data_paths.cache_clear()
        manager = SessionManager(None, "system", settings)
        self.assertEqual(manager.open(session_id).history.get(1)["message"]["content"], "你好")
        self.assertEqual(EventReader.for_session(session_id, log_path=log_path).read()[-1].seq, 1)
        with closing(get_connection()) as conn:
            row = conn.execute(
                "SELECT rollout_summary FROM stage1_outputs WHERE session_id = ?", (session_id,),
            ).fetchone()
        self.assertEqual(row["rollout_summary"], "旧摘要")
        session = manager.open(session_id)
        session.history.append([{"type": "message", "turn": 2, "message": {"role": "assistant", "content": "继续"}},
                                {"type": "turn/end", "turn": 2, "reason": "completed"}])
        self.assertTrue(Stage1Store().record_turn_end(session_id, 5, 101))

        paths = get_data_paths()
        legacy = paths.sessions / "legacy.jsonl"
        HistoryStore(legacy).load("旧会话")
        with closing(get_connection()) as conn, conn:
            conn.execute(
                "INSERT INTO sessions (session_id, event_log_path, name, created_at) VALUES (?, ?, ?, ?)",
                ("legacy", str(legacy), "旧会话", 100),
            )
        self.assertEqual(manager.open("legacy").history.path, legacy)
        self.assertEqual(EventReader.for_session("legacy", log_path=legacy).path, legacy)
        with closing(get_connection()) as conn, conn:
            conn.execute("UPDATE sessions SET event_log_path = ? WHERE session_id = 'legacy'", (str(paths.database),))
        with self.assertRaises(ValueError):
            manager.open("legacy")
        with self.assertRaises(ValueError):
            manager.delete("legacy")
        self.assertTrue(paths.database.is_file())
        self.assertTrue(manager.delete(session_id))
        self.assertFalse(paths.event_log(session_id).exists())
