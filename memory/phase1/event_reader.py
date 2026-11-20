"""读取 Event Log 的尾部快照，将有效消息转换为保留原文的候选项。"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_paths import PROJECT_ROOT, get_data_paths
from memory.settings_loader import load_config


@dataclass
class MemoryCandidate:
    """同一来源事件可拆成多个候选项，共享 raw_event。

    priority 从统一配置读取，数字越小越优先；没有 step 的事件记为 0。
    token_count 在读取时为 None，由格式化函数在脱敏、截断后填入。
    """

    seq: int
    turn: int
    step: int
    priority: int
    kind: str
    raw_event: dict
    token_count: int | None = None
    call_id: str | None = None


class EventReader:
    """从指定 JSONL 文件反向读取，不依赖 Session 的内存历史缓存。"""

    def __init__(self, path: str | Path, config: dict[str, Any] | None = None) -> None:
        """接收日志路径及已加载的配置；默认通过统一加载器读取 memory/.config。"""
        self.path = PROJECT_ROOT / path
        self.config = config if config is not None else load_config()
        self.source_max_seq = -1

    @classmethod
    def for_session(
        cls,
        session_id: str,
        *,
        log_path: str | Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> "EventReader":
        """读取指定会话，可传入数据库记录的路径以兼容旧日志。"""
        paths = get_data_paths()
        path = paths.event_log(session_id) if log_path is None else paths.resolve_event_log(session_id, log_path)
        return cls(path, config)

    def read(self, *, max_seq: int | None = None) -> list[MemoryCandidate]:
        """扫描文件末尾最多 max_read_bytes，返回按来源事件由新到旧的候选项。

        固定打开时的文件大小，忽略未以换行提交的末行和扫描边界上的残行。
        超过 max_record_bytes 的原始行直接跳过，不解析或截断原始事件。
        max_seq 限制纳入的来源版本；source_max_seq 包含该范围内的无关事件。
        文件及完整 JSON 行的读取、解析错误直接向调用方传播。
        """
        self.source_max_seq = -1
        with self.path.open("rb") as source:
            snapshot_end = os.fstat(source.fileno()).st_size
            start = max(0, snapshot_end - self.config["max_read_bytes"])
            # 多读一个边界字节，仅用于判断窗口首行是否完整。
            source.seek(start - 1 if start else 0)
            data = source.read(snapshot_end - source.tell())

        lower = 0
        if start:
            lower = 1 if data[0:1] == b"\n" else data.find(b"\n", 1) + 1
            if lower == 0:
                return []

        candidates = []
        end = data.rfind(b"\n")
        while end >= lower:
            previous = data.rfind(b"\n", lower, end)
            begin = previous + 1 if previous >= 0 else lower
            record_bytes = end - begin + 1
            if record_bytes <= self.config["max_record_bytes"]:
                event = json.loads(data[begin:end])
                if max_seq is None or event["seq"] <= max_seq:
                    self.source_max_seq = max(self.source_max_seq, event["seq"])
                    if event["type"] == "message":
                        candidates.extend(self._candidates(event))
            end = previous
        return candidates

    def _candidates(self, event: dict) -> list[MemoryCandidate]:
        """跳过无关角色，保留原始引用，拆分 Assistant 正文及工具调用。"""
        message = event["message"]
        common = {
            "seq": event["seq"], "turn": event["turn"],
            "step": event.get("step", 0), "raw_event": event,
        }
        role = message["role"]
        if role == "user":
            return [MemoryCandidate(**common, priority=self.config["priority_user"], kind="user_message")]
        if role == "tool":
            return [MemoryCandidate(
                **common, priority=self.config["priority_tool"], kind="tool_result",
                call_id=message["tool_call_id"],
            )]
        if role != "assistant":
            return []

        candidates = []
        if message.get("content") or message.get("reasoning_content"):
            candidates.append(MemoryCandidate(
                **common, priority=self.config["priority_assistant"], kind="assistant_message",
            ))
        for call in message.get("tool_calls") or []:
            candidates.append(MemoryCandidate(
                **common, priority=self.config["priority_tool"], kind="tool_call", call_id=call["id"],
            ))
        return candidates
