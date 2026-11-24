"""注册局部记忆检索工具。"""

from memory.reader import read_memory, search_memory
from memory.search import MATCH_MODES, MAX_CONTEXT, MAX_QUERIES, MAX_QUERY_CHARS, MAX_SESSIONS, MAX_WINDOW
from tool_executor import ToolExecutor


def register_memory_tools(executor: ToolExecutor) -> None:
    """在现有执行器中注册局部搜索和读取的模型接口。"""
    executor.register_tool("search_memory", '按关键词查找历史摘要或 slug，例如 {"queries":["杭州","预算"]}。', {
        "type": "object", "required": ["queries"], "additionalProperties": False,
        "properties": {
            "queries": {"type": "array", "minItems": 1, "maxItems": MAX_QUERIES,
                        "items": {"type": "string", "minLength": 1, "maxLength": MAX_QUERY_CHARS}},
            "match_mode": {"type": "string", "enum": list(MATCH_MODES), "default": "any"},
            "line_count": {"type": "integer", "minimum": 1, "maximum": MAX_WINDOW, "default": 5,
                           "description": "跨行匹配的最大窗口行数"},
            "context_lines": {"type": "integer", "minimum": 0, "maximum": MAX_CONTEXT, "default": 1,
                              "description": "命中前后各保留的行数"},
            "case_sensitive": {"type": "boolean", "default": False, "description": "区分大小写"},
            "normalized": {"type": "boolean", "default": False, "description": "比较时仅保留字母和数字"},
            "limit": {"type": "integer", "minimum": 1, "maximum": MAX_SESSIONS, "default": 5,
                      "description": "最多返回的会话数"},
        },
    }, search_memory)
    executor.register_tool("read_memory", "按 Session ID 读取历史会话的局部摘要。", {
        "type": "object", "required": ["session_id"], "additionalProperties": False,
        "properties": {"session_id": {"type": "string"}},
    }, read_memory)
