"""协调程序启动时的局部记忆提取和全局记忆整合。"""

from client import Trip_Client
from memory.phase1.phase1 import Phase1Agent
from memory.phase2.phase2 import Phase2Agent


class MemoryManager:
    """持有两个记忆阶段，并按顺序运行。"""

    def __init__(self, client: Trip_Client) -> None:
        """让两个阶段共用调用方管理的模型客户端。"""
        self._phase1 = Phase1Agent(client)
        self._phase2 = Phase2Agent(client)

    def run(self, *, current_session_id: str | None = None) -> str:
        """Phase 1 未领取任何会话时跳过 Phase 2；否则返回 Phase 2 的状态。"""
        outcomes = self._phase1.run(current_session_id=current_session_id)
        return self._phase2.run() if outcomes else "skipped"
