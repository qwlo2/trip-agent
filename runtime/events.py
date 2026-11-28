"""单个前台任务的内存事件流。所有读写均在 Runtime 的事件循环中执行。"""

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


MAX_TASK_EVENTS = 512
MAX_TASK_EVENT_BYTES = 256 * 1024


@dataclass(frozen=True)
class TaskEvent:
    seq: int
    task_id: str
    type: str
    data: dict[str, Any]


class TaskEventBuffer:
    def __init__(self, task_id: str):
        self._task_id = task_id
        self._events: deque[tuple[TaskEvent, int]] = deque()
        self._bytes = 0
        self._seq = 0
        self._ready = asyncio.Event()

    def publish(self, event_type: str, data: dict[str, Any]) -> None:
        """同步发布；大块正文分片，终态快照与普通事件使用不同的字节预算。"""
        if event_type == "llm/content_delta":
            delta = data["delta"]
            for start in range(0, len(delta), 8192):
                self._append(event_type, {**data, "delta": delta[start:start + 8192]})
        else:
            self._append(event_type, data)

    def _append(self, event_type: str, data: dict[str, Any]) -> None:
        size = 0 if event_type == "task/end" else len(json.dumps(data, ensure_ascii=False).encode("utf-8"))
        self._seq += 1
        event = TaskEvent(self._seq, self._task_id, event_type, deepcopy(data))
        self._events.append((event, size))
        self._bytes += size
        while len(self._events) > MAX_TASK_EVENTS or self._bytes > MAX_TASK_EVENT_BYTES:
            _, removed = self._events.popleft()
            self._bytes -= removed
        self._ready.set()

    async def subscribe(self, after: int) -> AsyncIterator[TaskEvent]:
        """每个订阅者独立推进游标，被淘汰的连续区间通过 gap 告知。"""
        if type(after) is not int or not 0 <= after <= self._seq:
            raise ValueError("after 必须是已发布范围内的非负整数")
        while True:
            first = self._events[0][0].seq if self._events else self._seq + 1
            if after < first - 1:
                missing = {"missing_from": after + 1, "missing_to": first - 1}
                after = first - 1
                yield TaskEvent(after, self._task_id, "stream/gap", missing)
            elif after < self._seq:
                event, _ = self._events[after + 1 - first]
                after = event.seq
                yield deepcopy(event)
            elif self._events and self._events[-1][0].type == "task/end":
                return
            else:
                self._ready.clear()
                await self._ready.wait()
