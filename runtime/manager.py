"""单 Event Loop 的业务任务和会话所有权管理。"""

import asyncio
from collections import deque
from contextlib import contextmanager
from enum import Enum
from uuid import uuid4

from agent import GuideAgent
from client import AsyncTripClient
from memory.settings_loader import load_config
from memory.memory_maneger import MemoryManager
from runtime.blocking import blocking, wait_finished
from runtime.task import TaskKind, TaskRecord, TaskStatus
from session import SessionIntegrityError
from session_repository import LogCleanupError, SessionRepository
from tool_executor import ToolExecutor


class RuntimeState(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    CLOSING = "closing"
    CLOSED = "closed"


class RuntimeManager:
    def __init__(self, directory: SessionRepository, *, max_concurrent_tasks: int = 4,
                 max_pending_tasks: int = 32, max_completed_tasks: int = 128,
                 client_factory=AsyncTripClient, executor_factory=ToolExecutor):
        """配置会话目录、客户端与工具执行器工厂，以及运行任务、未结束任务和保留记录的容量。"""
        if any(type(n) is not int or n < 1 for n in (max_concurrent_tasks, max_pending_tasks, max_completed_tasks)):
            raise ValueError("Runtime 容量必须为正整数")
        self._directory = directory
        self._client_factory = client_factory
        self._executor_factory = executor_factory
        self._client = None
        self._sessions: dict[str, GuideAgent] = {}
        # 只有不能构造 Session 的加载故障才在这里保存，已发布会话使用 Session.failure_reason。
        self._load_failures: dict[str, str] = {}
        self._owners: dict[str, str] = {}
        self._tasks: dict[str, TaskRecord] = {}
        self._completed = deque()
        self._management = set()
        self._limit = max_concurrent_tasks
        self._max_pending = max_pending_tasks
        self._max_completed = max_completed_tasks
        self._loop = None
        self._closing = None
        self.state = RuntimeState.CREATED

    async def start(self) -> None:
        """绑定当前事件循环，初始化共享资源并恢复本地摘要，然后调度后台记忆任务。"""
        loop = asyncio.get_running_loop()
        if self.state == RuntimeState.RUNNING and loop is self._loop:
            return
        if self._loop is not None or self.state != RuntimeState.CREATED:
            raise RuntimeError("Runtime 已启动、启动中或已关闭")
        self._loop = loop
        try:
            self._semaphore = asyncio.Semaphore(self._limit)
            self._client = self._client_factory()
            self._memory = MemoryManager(self._client, asyncio.Lock(), await blocking(load_config))
            await self._memory.recover()
            self._memory_task = asyncio.create_task(self._memory.run_forever())
            self.state = RuntimeState.RUNNING
        except BaseException:
            if self._client is not None:
                await self._client.close()
            self.state = RuntimeState.CLOSED
            raise

    def _check_loop(self) -> None:
        """确认调用发生在启动 Runtime 的事件循环中。"""
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("Runtime 必须由所属 Event Loop 访问")

    def _accepting(self) -> None:
        """确认 Runtime 处于运行状态，可以接收新的前台任务或管理操作。"""
        self._check_loop()
        if self.state != RuntimeState.RUNNING:
            raise RuntimeError("Runtime 未接受新操作")

    def _check_idle(self, session_id: str) -> None:
        """校验会话 ID，并拒绝操作已有所有者的会话。"""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("Session ID 不能为空")
        if session_id in self._owners:
            raise RuntimeError("Session 已有已接受的任务")

    def submit(self, session_id: str, question: str) -> str:
        """提交一轮导游问答，返回可用于查询或等待结果的任务 ID。"""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("问题不能为空")
        return self._submit(session_id, TaskKind.TURN, question)

    def submit_compact(self, session_id: str) -> str:
        """提交独立的会话压缩任务，返回任务 ID。"""
        return self._submit(session_id, TaskKind.COMPACT, None)

    def _submit(self, session_id, kind, question) -> str:
        """校验会话和容量，登记任务及会话所有权，再调度执行。Task 创建失败时撤销登记。"""
        self._accepting()
        self._check_idle(session_id)
        if session_id in self._load_failures:
            raise SessionIntegrityError(self._load_failures[session_id])
        if session_id in self._sessions:
            self._sessions[session_id].session.require_usable()
        if len(self._tasks) - len(self._completed) >= self._max_pending:
            raise RuntimeError("任务接收容量已满")
        key = uuid4().hex
        record = TaskRecord(key, session_id, kind, question, self._loop.create_future())
        self._owners[session_id] = key
        self._tasks[key] = record
        coroutine = self._execute(record)
        try:
            record.task = asyncio.create_task(coroutine)
        except BaseException:
            coroutine.close()
            del self._tasks[key]
            self._release(session_id, key)
            raise
        record.task.add_done_callback(lambda task: self._done(record, task))
        return key

    async def _get_or_open(self, session_id: str) -> GuideAgent:
        """返回已缓存的 Agent，或加载会话、初始化工具并缓存新 Agent。记录会话完整性故障以拒绝后续提交。"""
        if session_id not in self._sessions:
            try:
                session = await blocking(self._directory.load, session_id, self._client)
            except SessionIntegrityError as exc:
                self._load_failures[session_id] = str(exc)
                raise
            executor = await blocking(self._executor_factory)
            await executor.initialize()
            self._sessions[session_id] = GuideAgent(self._client, executor, session, memory_prompt=self._memory.prompt)
        return self._sessions[session_id]

    async def _execute(self, record: TaskRecord) -> None:
        """取得并发许可后执行问答或独立压缩并保存结果。退出时归还许可，执行异常传播至 Task。"""
        async with self._semaphore:
            record.status = TaskStatus.RUNNING
            agent = await self._get_or_open(record.session_id)
            if record.kind == TaskKind.TURN:
                record.answer = await agent.ask(record.question)
            else:
                record.result = await agent.session.compact_now(standalone=True)

    def _done(self, record: TaskRecord, task: asyncio.Task) -> None:
        """根据 Task 的实际结果发布终态，释放会话所有权，通知等待方并按容量回收最旧的记录。"""
        if record.completion.done():
            return
        if task.cancelled():
            record.status = TaskStatus.CANCELLED
        else:
            error = task.exception()
            if error is None:
                record.status = TaskStatus.COMPLETED
            else:
                record.status = TaskStatus.FAILED
                record.error = f"{type(error).__name__}: {error}"
        self._release(record.session_id, record.task_id)
        record.completion.set_result(record.snapshot())
        self._completed.append(record.task_id)
        if len(self._completed) > self._max_completed:
            del self._tasks[self._completed.popleft()]

    def _release(self, session_id, owner):
        """仅在所有者 ID 匹配时释放会话，避免旧任务回调误释放新所有者。"""
        if self._owners.get(session_id) == owner:
            del self._owners[session_id]

    def cancel(self, task_id: str) -> bool:
        """请求取消已接受的前台任务。首次接受返回 True，正在取消或已结束时返回 False。"""
        self._check_loop()
        record = self._tasks[task_id]
        if record.completion.done() or record.status == TaskStatus.CANCELLING:
            return False
        if not record.task.cancel():
            return False
        record.status = TaskStatus.CANCELLING
        return True

    def get_task(self, task_id: str) -> dict:
        """按 ID 获取任务当前状态与结果的快照。ID 不存在时抛出 KeyError。"""
        self._check_loop()
        return self._tasks[task_id].snapshot()

    def list_tasks(self) -> list[dict]:
        """返回当前仍登记的所有前台任务快照，包含未结束任务和保留的终态任务。"""
        self._check_loop()
        return [record.snapshot() for record in self._tasks.values()]

    async def wait(self, task_id: str) -> dict:
        """等待结束回调发布最终快照。取消等待方时，业务 Task 和结果 Future 仍独立运行。"""
        self._check_loop()
        return dict(await asyncio.shield(self._tasks[task_id].completion))

    @contextmanager
    def _manage(self, exclusive_id=None):
        """登记会话管理操作供关闭流程等待。指定会话时独占它，并在操作退出后释放所有权。"""
        self._accepting()
        finished = self._loop.create_future()
        if exclusive_id is not None:
            self._check_idle(exclusive_id)
            owner = uuid4().hex
            self._owners[exclusive_id] = owner
        self._management.add(finished)
        try:
            yield
        finally:
            if exclusive_id is not None:
                self._release(exclusive_id, owner)
            self._management.remove(finished)
            finished.set_result(None)

    async def create_session(self, name: str) -> str:
        """通过会话目录创建会话和初始日志，返回新会话 ID。"""
        with self._manage():
            return await blocking(self._directory.create, name)

    async def list_sessions(self, **page) -> list[dict]:
        """按传入的 limit 和 offset 查询未归档会话，结果按使用次数排序。"""
        with self._manage():
            return await blocking(self._directory.list_sessions, **page)

    async def select_session(self, session_id: str) -> str:
        """记录指定会话被打开，增加使用次数并返回会话 ID。"""
        with self._manage():
            return await blocking(self._directory.record_open, session_id)

    async def archive_session(self, session_id: str, archived: bool = True) -> bool:
        """独占会话并设置归档或恢复状态，返回归档标记是否实际改变。"""
        with self._manage(session_id):
            return await blocking(self._directory.archive, session_id, archived)

    async def delete_session(self, session_id: str) -> bool:
        """独占会话并提交目录删除，随后清理日志和缓存。返回目录中是否找到该会话。"""
        with self._manage(session_id):
            try:
                deleted = await blocking(self._directory.delete, session_id, finish_on_cancel=True)
            except SessionIntegrityError as exc:
                # 目录提交结果无法确认，旧写入者不能复用。
                agent = self._sessions.get(session_id)
                if agent:
                    agent.session.failure_reason = agent.session.failure_reason or str(exc)
                raise
            except LogCleanupError:
                self._forget_deleted(session_id)
                raise
            self._forget_deleted(session_id)
            return deleted

    def _forget_deleted(self, session_id):
        """移除已删除会话的 Agent 和故障记录，并禁用仍被引用的旧会话对象。"""
        agent = self._sessions.pop(session_id, None)
        if agent:
            agent.session.failure_reason = "会话目录已删除"
        self._load_failures.pop(session_id, None)

    async def shutdown(self) -> None:
        """停止接收新操作和后台领取，等待共享关闭任务排空工作。取消等待方时关闭仍继续。"""
        if self.state == RuntimeState.CREATED:
            if self._loop is not None:
                raise RuntimeError("启动尚未完成")
            self.state = RuntimeState.CLOSED
            return
        if self.state == RuntimeState.CLOSED and self._closing is None:
            return
        self._check_loop()
        if self._closing is None or self._closing.cancelled():
            self.state = RuntimeState.CLOSING
            self._memory.stop()
            self._closing = asyncio.create_task(self._drain())
        await asyncio.shield(self._closing)

    async def _drain(self) -> None:
        """等待前台任务最终化、管理操作及后台记忆任务结束，再清理缓存并关闭共享客户端。"""
        await wait_finished(asyncio.gather(*(r.completion for r in self._tasks.values()), *self._management,
                                           self._memory_task, return_exceptions=True),
                            finish_on_cancel=True)
        self._sessions.clear()
        await wait_finished(asyncio.create_task(self._client.close()), finish_on_cancel=True)
        self.state = RuntimeState.CLOSED
