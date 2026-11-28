import sqlite3
import time
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
    """初始化数据目录及数据库表结构。"""
    get_data_paths().ensure_directories()
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    with closing(get_connection()) as conn:
        conn.executescript(schema)


def recover_interrupted_jobs(retry_delay_seconds: int) -> None:
    """在单执行者开始下一轮前结束遗留的领取，保留其重试额度。"""
    now = int(time.time())
    with closing(get_connection()) as conn, conn:
        conn.execute("""
            UPDATE memory_jobs SET status='error', ownership_token=NULL,
                claimed_version=CASE WHEN kind='phase2' THEN claimed_version END,
                retry_at=CASE WHEN kind='stage1' THEN ? END,
                last_error='Interrupted', finished_at=?, updated_at=?
            WHERE status='running'
        """, (now + retry_delay_seconds, now, now))
