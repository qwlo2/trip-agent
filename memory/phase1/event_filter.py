"""仅在候选项准备进入输入时，脱敏、截断、序列化为 JSON 并统计 Token。"""

import json
import re
from typing import Any

from memory.phase1.event_reader import MemoryCandidate


_BEARER = re.compile(r"(?i)\bBearer\s+[^\s\"',;]+")
_API_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")
_SECRET_FIELD = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|"
    r"password|passwd|secret|authorization)\b[\"']?\s*[:=]\s*)"
    r"(?:\"[^\"\n]*\"|'[^'\n]*'|[^\s,;}]+)"
)


def redact(text: str) -> str:
    """隐藏常见 API Key、Bearer 凭据及密码、Token 等字段值。"""
    text = _BEARER.sub("Bearer [REDACTED]", text)
    text = _API_KEY.sub("[REDACTED]", text)
    return _SECRET_FIELD.sub(lambda match: match[1] + "[REDACTED]", text)


def truncate_utf8(text: str, limit: int) -> str:
    """在指定 UTF-8 字节数内保留文本，截断时追加标记且不切坏字符。"""
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    marker = b"\n[truncated]"
    if limit < len(marker):
        return marker[:limit].decode("utf-8")
    return encoded[:limit - len(marker)].decode("utf-8", errors="ignore") + marker.decode()


def _text(value: object) -> str:
    """保留字符串正文，其余消息内容使用紧凑 JSON 表达。"""
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def render_candidate(candidate: MemoryCandidate, config: dict[str, Any]) -> str:
    """生成 Memory 自定义角色的消息 JSON，更新 token_count，不修改 raw_event。

    Tool Result 的正文先按配置的 bytes_per_token 截断，所有记录再受 max_row_bytes
    限制。每条记录包含来源 seq，文本和各次调用分别输出，不包含 Turn 边界。
    role 使用候选项 kind 的四种分类值，此格式用于 Memory 内部提取输入。
    """
    message = candidate.raw_event["message"]
    record = {"seq": candidate.seq, "role": candidate.kind}
    if candidate.kind == "tool_call":
        call = next(call for call in message["tool_calls"] if call["id"] == candidate.call_id)
        function = call["function"]
        record["content"] = None
        record["tool_calls"] = [{
            "id": candidate.call_id,
            "type": "function",
            "function": {
                "name": function["name"],
                "arguments": redact(_text(function["arguments"])),
            },
        }]
    elif candidate.kind == "tool_result":
        record["tool_call_id"] = candidate.call_id
        record["content"] = truncate_utf8(
            redact(_text(message.get("content"))),
            config["tool_output_tokens"] * config["bytes_per_token"],
        )
    else:
        record["content"] = redact(_text(message.get("content")))
    if candidate.kind == "assistant_message" and message.get("reasoning_content"):
        record["reasoning_content"] = redact(_text(message["reasoning_content"]))

    text = serialize_message(record, config)
    bytes_per_token = config["bytes_per_token"]
    candidate.token_count = (len(text.encode("utf-8")) + bytes_per_token - 1) // bytes_per_token
    return text


def serialize_message(message: dict, config: dict[str, Any]) -> str:
    """序列化已脱敏的消息副本，超限时缩减正文或参数，保持消息结构完整。

    每条候选记录独立执行字节限制，同一来源的记录不合并。
    """
    payload_fields = [
        (message, name) for name in ("content", "reasoning_content")
        if isinstance(message.get(name), str)
    ]
    payload_fields.extend((call["function"], "arguments") for call in message.get("tool_calls", []))
    text = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
    size = len(text.encode("utf-8"))
    while size > config["max_row_bytes"]:
        # JSON 转义也占字节，缩减正文后重新序列化，避免直接切坏 JSON。
        container, field = max(payload_fields, key=lambda item: len(item[0][item[1]].encode("utf-8")))
        if not container[field]:
            raise ValueError("max_row_bytes 不足以容纳消息的角色及工具调用标识")
        limit = max(0, len(container[field].encode("utf-8")) - (size - config["max_row_bytes"]))
        container[field] = truncate_utf8(container[field], limit)
        text = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
        size = len(text.encode("utf-8"))
    return text
