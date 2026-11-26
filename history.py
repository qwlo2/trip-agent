"""单会话 JSONL 事件存储，记录消息及问答、步骤的开始和结束。"""

import json
import os
import shutil
import time
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any

from data_paths import PROJECT_ROOT, get_data_paths


class SessionIntegrityError(RuntimeError):
    """日志、目录或投影不一致，禁止复用会话。"""


class TurnEndReason(str, Enum):
    """问答结束原因：正常完成、执行异常或达到步骤上限。"""

    COMPLETED = "completed"
    ERROR = "error"
    MAX_STEPS = "max_steps"


class HistoryStore:
    """追加带全局 seq 的历史事件，seq=0 固定为系统身份提示词。"""

    def __init__(self, path: str | Path | None = None) -> None:
        """接收日志路径，省略时使用数据目录内的默认会话。"""
        self.path = PROJECT_ROOT / path if path is not None else get_data_paths().default_event_log()
        self._next_seq = 0
        self._events: dict[int, dict[str, Any]] = {}
        self._write_failed = False

    def load(self, system_prompt: str, *, existing: bool = False) -> list[dict[str, Any]]:
        """读取事件并接续 seq；空历史先写入 seq=0、turn=0 的系统消息。

        未完成的末行不视为已提交事件，备份后截去；完整行损坏则明确报错。
        """
        try:
            content = self.path.read_bytes()
        except FileNotFoundError:
            content = b""
        if content:
            committed_end = content.rfind(b"\n") + 1
            if committed_end != len(content):
                backup = self.path.with_name(f"{self.path.name}.incomplete.{time.time_ns()}.bak")
                shutil.copy2(self.path, backup)
                target = self.path.open("r+b")
                try:
                    with target:
                        target.truncate(committed_end)
                        target.flush()
                        os.fsync(target.fileno())
                except OSError as exc:
                    self._write_failed = True
                    raise SessionIntegrityError(f"日志尾部修复失败：{exc}") from exc
            try:
                records = [json.loads(line) for line in content[:committed_end].splitlines()]
            except ValueError as exc:
                raise SessionIntegrityError(f"日志 JSON 损坏：{exc}") from exc
        else:
            records = []

        if records:
            if any(not isinstance(record, dict) or type(record.get("seq")) is not int or record["seq"] != index
                   for index, record in enumerate(records)):
                raise SessionIntegrityError("History seq 必须从 0 开始连续递增，不能重复或乱序")
            self._events = {record["seq"]: deepcopy(record) for record in records}
            self._next_seq = records[-1]["seq"] + 1
            self._write_failed = False
            return records

        if existing:
            raise SessionIntegrityError("已登记会话缺少完整的系统消息")
        system_event = {
            "type": "message", "turn": 0,
            "message": {"role": "system", "content": system_prompt},
        }
        self._next_seq = 0
        self._events.clear()
        self._write_failed = False
        return self.append([system_event])

    def append(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """给新增事件分配全局递增的 seq，并追加到 JSONL，返回已编号的事件。

        调用方传入 type、turn、step 等事件字段；编号在写入成功后递增。
        每次写入后 flush 并 fsync，不重写已有事件。
        """
        if self._write_failed:
            raise RuntimeError("历史提交曾失败，请重启加载日志后再写入")
        records = []
        for index, event in enumerate(events):
            record = {**event, "seq": self._next_seq + index}
            records.append(record)
        content = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("a", encoding="utf-8") as target:
                target.write(content)
                target.flush()
                os.fsync(target.fileno())
        except OSError:
            self._write_failed = True
            raise
        self._next_seq += len(records)
        self._events.update({record["seq"]: deepcopy(record) for record in records})
        return records

    def get(self, seq: int) -> dict[str, Any]:
        """按 seq 查询包括被压缩来源在内的原始事件，返回独立副本。"""
        if seq not in self._events:
            raise ValueError(f"History 中不存在来源 seq={seq}")
        return deepcopy(self._events[seq])

    def turn_events(self, turn: int) -> list[dict[str, Any]]:
        """返回最新一轮已经提交的事件，供独占的 Session 安全收尾。"""
        start = self._next_seq
        while start and self._events[start - 1]["turn"] == turn:
            start -= 1
        return [self._events[i] for i in range(start, self._next_seq)]

    def clear(self) -> None:
        """清空事件文件并重置 seq。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self._next_seq = 0
        self._events.clear()
        self._write_failed = False


def tool_call_ids(message: dict[str, Any]) -> list[str]:
    """在持久化前检查整批调用是否具有可逐个配对的 ID。"""
    calls = message.get("tool_calls") or []
    ids = [call.get("id") for call in calls]
    if any(not isinstance(key, str) or not key.strip() for key in ids) or len(set(ids)) != len(ids):
        raise ValueError("工具调用 ID 为空或同批重复")
    return ids


def track_message(message: dict[str, Any], pending: set[str]) -> None:
    """按原始 ID 验证模型可见消息序列。"""
    role = message["role"]
    if role == "tool":
        key = message.get("tool_call_id")
        if key not in pending:
            raise ValueError("工具结果 ID 无对应调用")
        pending.remove(key)
    elif pending:
        raise ValueError("上批工具结果未闭合")
    elif role == "assistant":
        pending.update(tool_call_ids(message))
    elif role not in {"system", "user"}:
        raise ValueError("无效的消息角色")


def validate_history(events: list[dict[str, Any]]) -> None:
    """验证生命周期和原始工具配对，兼容早期仅记录消息的历史前缀。"""
    first = events[0]
    if (first["type"] != "message" or first["turn"] != 0 or first["message"]["role"] != "system"
            or not isinstance(first["message"].get("content"), str)):
        raise ValueError("缺少有效的 seq=0 系统消息")
    turn = step = None
    previous_turn = previous_step = 0
    structured = False
    legacy_turn = None
    pending = set()
    last_role = None
    for event in events[1:]:
        kind, number = event["type"], event["turn"]
        if type(number) is not int or number < 0:
            raise ValueError("无效的 Turn 编号")
        if turn is not None and number != turn:
            raise ValueError("事件不属于当前 Turn")
        if kind == "turn/start":
            if turn is not None or pending or number <= previous_turn:
                raise ValueError("非法 turn/start")
            structured = True
            has_input = False
            answer_step = None
            last_role = None
            previous_turn = turn = number
            previous_step = 0
        elif kind == "step/start":
            if turn is None or step is not None or type(event["step"]) is not int or event["step"] != previous_step + 1:
                raise ValueError("非法 step/start")
            previous_step = step = event["step"]
        elif kind == "step/end":
            if step is None or type(event["step"]) is not int or step != event["step"] or pending:
                raise ValueError("非法 step/end")
            step = None
        elif kind == "turn/end":
            if (turn is None and (structured or legacy_turn != number)) or step is not None or pending:
                raise ValueError("非法 turn/end")
            reason = TurnEndReason(event["reason"])
            if reason == TurnEndReason.COMPLETED:
                if last_role != "assistant" or (structured and (not has_input or answer_step != previous_step)):
                    raise ValueError("completed Turn 缺少用户输入或已结束的回答步骤")
            turn = legacy_turn = None
        elif kind == "message":
            usage = event.get("usage")
            if usage is not None and (not isinstance(usage, dict) or type(usage.get("total_tokens")) is not int):
                raise ValueError("消息 usage 缺少有效的 total_tokens")
            if event["message"]["role"] == "system":
                raise ValueError("重复系统消息")
            if structured and (turn is None or event.get("step") != step):
                raise ValueError("消息不属于当前 Turn/Step")
            if not structured:
                if number < previous_turn or (pending and number != legacy_turn):
                    raise ValueError("旧会话 Turn 顺序错误")
                previous_turn = legacy_turn = number
            last_role = event["message"]["role"]
            if structured:
                if last_role == "user" and step is None and not previous_step:
                    content = event["message"].get("content")
                    has_input = isinstance(content, str) and bool(content.strip())
                elif last_role == "assistant":
                    content = event["message"].get("content")
                    answer_step = step if isinstance(content, str) and content.strip() else None
            track_message(event["message"], pending)
        elif kind == "compaction/summary":
            if number != previous_turn:
                raise ValueError("摘要不属于当前会话轮次")
        elif kind not in {"tool/call", "tool/result"}:
            raise ValueError(f"未知历史事件：{kind}")
    if turn is not None or step is not None or pending:
        raise ValueError("历史存在未闭合 Turn、Step 或工具调用")
