import sqlite3
from contextlib import closing
from pathlib import Path

from data_paths import get_data_paths

SCHEMA_PATH = Path(__file__).resolve().with_name("schema.sql")


def get_connection() -> sqlite3.Connection:
    """获取数据库连接，由调用方负责关闭。"""
    conn = sqlite3.connect(get_data_paths().database)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_database() -> None:
    """初始化数据库，并在已有会话表中补齐 Phase 1 的筛选字段。"""
    get_data_paths().ensure_directories()
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    with closing(get_connection()) as conn:
        conn.executescript(schema)
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)")}
            if "updated_at" not in columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN updated_at INTEGER")
            if "memory_enabled" not in columns:
                conn.execute(
                    "ALTER TABLE sessions ADD COLUMN memory_enabled INTEGER NOT NULL DEFAULT 1 "
                    "CHECK (memory_enabled IN (0, 1))"
                )
