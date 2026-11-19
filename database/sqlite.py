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
    """创建数据库，并按 schema.sql 初始化表和索引，可重复调用。"""
    get_data_paths().ensure_directories()
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    with closing(get_connection()) as conn:
        conn.executescript(schema)
