"""Resolve all agent data paths from the project-root configuration."""

import re
from functools import lru_cache
from pathlib import Path

from settings_loader import load_settings


PROJECT_ROOT = Path(__file__).resolve().parent


class DataPaths:
    """提供数据目录和文件路径，不读写日志、数据库或记忆内容。"""

    def __init__(self, data_dir: str | Path) -> None:
        """以项目根目录解析相对路径，并保留绝对路径的含义。"""
        self.root = (PROJECT_ROOT / data_dir).resolve()
        self.database = self.root / "state.db"
        self.sessions = self.root / "sessions"
        self.memories = self.root / "memories"

    @staticmethod
    def _without_redirects(path: Path) -> Path:
        """拒绝把会话路径重定向到其他位置的符号链接。"""
        if path.resolve() != path:
            raise ValueError(f"数据路径不能重定向到其他位置：{path}")
        return path

    def session_dir(self, session_id: str) -> Path:
        """返回会话目录，ID 仅允许字母、数字、下划线和连字符。"""
        if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id):
            raise ValueError("无效的 session_id")
        return self._without_redirects(self.sessions / session_id)

    def event_log(self, session_id: str) -> Path:
        """返回新会话使用的 events.jsonl 路径。"""
        return self._without_redirects(self.session_dir(session_id) / "events.jsonl")

    def resolve_event_log(self, session_id: str, log_path: str | Path) -> Path:
        """以数据根目录解析数据库路径，兼容同一会话的旧版平铺日志。"""
        directory = self.session_dir(session_id)
        stored = Path(log_path)
        path = self.root / stored
        if ".." in stored.parts or "\\" in str(log_path):
            raise ValueError("事件日志路径包含非法路径片段")
        legacy = self.sessions / f"{session_id}.jsonl"
        if path != legacy and (path == directory or not path.is_relative_to(directory)):
            raise ValueError("事件日志路径不属于指定会话")
        return self._without_redirects(path)

    def default_event_log(self) -> Path:
        """保留独立问答入口的默认会话，并兼容原有单文件历史。"""
        path = self.event_log("default")
        if path.exists():
            return path
        legacy = self._without_redirects(self.root / "history.jsonl")
        return legacy if legacy.is_file() else path

    @property
    def rollout_summaries(self) -> Path:
        """返回 Phase 1 局部记忆目录。"""
        return self.memories / "rollout_summaries"

    @property
    def memory_summary(self) -> Path:
        """返回全局记忆摘要路径。"""
        return self.memories / "memory_summary.md"

    @property
    def phase2_workspace_diff(self) -> Path:
        """返回 Phase 2 工作区差异文件路径。"""
        return self.memories / "phase2_workspace_diff.md"

    def ensure_directories(self) -> None:
        """创建数据根目录和公共子目录，保留现有内容。"""
        for directory in (self.root, self.sessions, self.rollout_summaries):
            self._without_redirects(directory).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_data_paths() -> DataPaths:
    """通过公共加载器读取 DSH_DATA_DIR，并在进程内复用同一份路径配置。"""
    values = load_settings(
        path=PROJECT_ROOT / ".env",
        defaults_path=PROJECT_ROOT / ".env.example",
        fields={"DSH_DATA_DIR": str},
        allow_empty=("DSH_DATA_DIR",),
    )
    return DataPaths(values["DSH_DATA_DIR"] or "./data")
