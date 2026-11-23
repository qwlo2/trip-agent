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
        """先处理局部记忆，再由 Phase 2 独立判断是否领取全局任务。"""
        self._phase1.run(current_session_id=current_session_id)
        return self._phase2.run()
