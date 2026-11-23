"""注册记忆检索工具及 GuideAgent 的结束协议。"""

from memory.reader import read_memory, search_memory
from tool_executor import ToolExecutor


def finish_answer(answer: str, used_session_ids: list[str]) -> tuple[str, set[str]]:
    """验证结束参数并合并本轮重复的来源 ID。"""
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer 必须是非空字符串")
    if not isinstance(used_session_ids, list) or not all(isinstance(i, str) for i in used_session_ids):
        raise ValueError("used_session_ids 必须是字符串数组")
    return answer, set(used_session_ids)


def register_memory_tools(executor: ToolExecutor) -> None:
    """在现有执行器中注册局部读取和最终提交的模型接口。"""
    executor.register_tool("search_memory", "按单个关键词查询历史会话的摘要或 slug。", {
        "type": "object", "required": ["query"], "additionalProperties": False,
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 10}},
    }, search_memory)
    executor.register_tool("read_memory", "按 Session ID 读取历史会话的局部摘要。", {
        "type": "object", "required": ["session_id"], "additionalProperties": False,
        "properties": {"session_id": {"type": "string"}},
    }, read_memory)
    executor.register_tool("finish_answer", "单独调用以提交最终回答及整轮实际使用的来源。", {
        "type": "object", "required": ["answer", "used_session_ids"], "additionalProperties": False,
        "properties": {
            "answer": {"type": "string"},
            "used_session_ids": {"type": "array", "items": {"type": "string"}},
        },
    }, finish_answer)
