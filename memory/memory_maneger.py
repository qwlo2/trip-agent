"""在共享事件循环上串行执行周期记忆任务。"""

import asyncio
import logging

from memory.phase1.phase1 import Phase1Agent
from memory.phase2.phase2 import Phase2Agent
from memory.reader import turn_memory_prompt
from runtime.blocking import blocking


class MemoryManager:
    def __init__(self, client, lock, config):
        self._phase1 = Phase1Agent(client)
        self._phase2 = Phase2Agent(client, lock, config)
        self._interval = config["scan_interval_seconds"]
        self._stopped = asyncio.Event()

    async def recover(self):
        try:
            await blocking(self._phase2.recover)
        except Exception as exc:
            self._phase2.trusted = False
            logging.getLogger(__name__).warning("全局记忆恢复失败（%s）", type(exc).__name__)

    async def prompt(self):
        async with self._phase2.lock:
            return await blocking(turn_memory_prompt, self._phase2.trusted)

    async def run(self):
        for phase in (self._phase1, self._phase2):
            if self._stopped.is_set():
                break
            try:
                result = await phase.run(self._stopped)
            except Exception as exc:
                logging.getLogger(__name__).warning("记忆阶段 %s 失败（%s）", type(phase).__name__, type(exc).__name__)
            else:
                if phase is self._phase2:
                    return result

    async def run_forever(self):
        while not self._stopped.is_set():
            await self.run()
            try:
                await asyncio.wait_for(self._stopped.wait(), self._interval)
            except asyncio.TimeoutError:
                pass

    def stop(self):
        self._stopped.set()
