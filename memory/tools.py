"""注册局部记忆检索工具。"""

from memory.reader import read_memory, search_memory
from tool_executor import ToolExecutor


def register_memory_tools(executor: ToolExecutor) -> None:
    """在现有执行器中注册局部搜索和读取的模型接口。"""
    executor.register_tool("search_memory", "按单个关键词查询历史会话的摘要或 slug。", {
        "type": "object", "required": ["query"], "additionalProperties": False,
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 10}},
    }, search_memory)
    executor.register_tool("read_memory", "按 Session ID 读取历史会话的局部摘要。", {
        "type": "object", "required": ["session_id"], "additionalProperties": False,
        "properties": {"session_id": {"type": "string"}},
    }, read_memory)
