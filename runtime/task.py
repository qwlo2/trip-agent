"""Runtime 内部业务任务记录。"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any


class TaskKind(str, Enum):
    TURN = "turn"
    COMPACT = "compact"


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskRecord:
    task_id: str
    session_id: str
    kind: TaskKind
    question: str | None
    completion: asyncio.Future
    status: TaskStatus = TaskStatus.QUEUED
    task: asyncio.Task | None = None
    answer: str | None = None
    result: Any = None
    error: str | None = None

    def snapshot(self) -> dict:
        """结果只包含字符串和不可变 CompactResult。"""
        return {"task_id": self.task_id, "session_id": self.session_id, "kind": self.kind.value,
                "status": self.status.value, "answer": self.answer, "result": self.result, "error": self.error}
