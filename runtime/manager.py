"""单 Event Loop 的业务任务和会话所有权管理。"""

import asyncio
import logging
from collections import deque
from contextlib import contextmanager
from enum import Enum
from uuid import uuid4

from agent import GuideAgent
from client import AsyncTripClient, Trip_Client
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


def startup_memory(session_id: str) -> None:
    """同步客户端的创建、记忆流水线和关闭都在启动 worker 中结束。"""
    with Trip_Client() as client:
        MemoryManager(client).run(current_session_id=session_id)


class RuntimeManager:
    def __init__(self, directory: SessionRepository, *, max_concurrent_tasks: int = 4,
                 max_pending_tasks: int = 32, max_completed_tasks: int = 128,
                 client_factory=AsyncTripClient, executor_factory=ToolExecutor):
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

    async def start(self, initial_session_id: str | None = None) -> None:
        """绑定当前循环，在接收前台操作前结束可选的启动记忆任务。"""
        loop = asyncio.get_running_loop()
        if self.state == RuntimeState.RUNNING and loop is self._loop:
            return
        if self._loop is not None or self.state != RuntimeState.CREATED:
            raise RuntimeError("Runtime 已启动、启动中或已关闭")
        self._loop = loop
        try:
            if initial_session_id is not None:
                try:
                    await blocking(startup_memory, initial_session_id)
                except Exception as exc:
                    logging.getLogger(__name__).warning("启动记忆处理失败（%s），继续问答", type(exc).__name__)
            self._semaphore = asyncio.Semaphore(self._limit)
            self._client = self._client_factory()
            self.state = RuntimeState.RUNNING
        except BaseException:
            self.state = RuntimeState.CLOSED
            raise

    def _check_loop(self) -> None:
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("Runtime 必须由所属 Event Loop 访问")

    def _accepting(self) -> None:
        self._check_loop()
        if self.state != RuntimeState.RUNNING:
            raise RuntimeError("Runtime 未接受新操作")

    def _check_idle(self, session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("Session ID 不能为空")
        if session_id in self._owners:
            raise RuntimeError("Session 已有已接受的任务")

    def submit(self, session_id: str, question: str) -> str:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("问题不能为空")
        return self._submit(session_id, TaskKind.TURN, question)

    def submit_compact(self, session_id: str) -> str:
        return self._submit(session_id, TaskKind.COMPACT, None)

    def _submit(self, session_id, kind, question) -> str:
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
        if session_id not in self._sessions:
            try:
                session = await blocking(self._directory.load, session_id, self._client)
            except SessionIntegrityError as exc:
                self._load_failures[session_id] = str(exc)
                raise
            executor = await blocking(self._executor_factory)
            await executor.initialize()
            self._sessions[session_id] = GuideAgent(self._client, executor, session)
        return self._sessions[session_id]

    async def _execute(self, record: TaskRecord) -> None:
        async with self._semaphore:
            record.status = TaskStatus.RUNNING
            try:
                agent = await self._get_or_open(record.session_id)
                if record.kind == TaskKind.TURN:
                    record.answer = await agent.ask(record.question)
                else:
                    record.result = await agent.session.compact_now(standalone=True)
            except (Exception, asyncio.CancelledError) as exc:
                self._finalize_once(record, exc)
            else:
                self._finalize_once(record, None)

    def _done(self, record, task) -> None:
        """包括第一次运行前、等待许可时被取消的路径。"""
        error = asyncio.CancelledError() if task.cancelled() else task.exception()
        self._finalize_once(record, error)

    def _release(self, session_id, owner):
        if self._owners.get(session_id) == owner:
            del self._owners[session_id]

    def _finalize_once(self, record, error) -> None:
        if record.completion.done():
            return
        record.status = TaskStatus.COMPLETED if error is None else TaskStatus.FAILED
        if error is not None:
            record.error = f"{type(error).__name__}: {error}"
        self._release(record.session_id, record.task_id)
        record.completion.set_result(record.snapshot())
        self._completed.append(record.task_id)
        if len(self._completed) > self._max_completed:
            del self._tasks[self._completed.popleft()]

    def get_task(self, task_id: str) -> dict:
        self._check_loop()
        return self._tasks[task_id].snapshot()

    def list_tasks(self) -> list[dict]:
        self._check_loop()
        return [record.snapshot() for record in self._tasks.values()]

    async def wait(self, task_id: str) -> dict:
        self._check_loop()
        return dict(await asyncio.shield(self._tasks[task_id].completion))

    @contextmanager
    def _manage(self, exclusive_id=None):
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
        with self._manage():
            return await blocking(self._directory.create, name)

    async def list_sessions(self, **page) -> list[dict]:
        with self._manage():
            return await blocking(self._directory.list_sessions, **page)

    async def select_session(self, session_id: str) -> str:
        with self._manage():
            return await blocking(self._directory.record_open, session_id)

    async def archive_session(self, session_id: str, archived: bool = True) -> bool:
        with self._manage(session_id):
            return await blocking(self._directory.archive, session_id, archived)

    async def delete_session(self, session_id: str) -> bool:
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
        agent = self._sessions.pop(session_id, None)
        if agent:
            agent.session.failure_reason = "会话目录已删除"
        self._load_failures.pop(session_id, None)

    async def shutdown(self) -> None:
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
            self._closing = asyncio.create_task(self._drain())
        await asyncio.shield(self._closing)

    async def _drain(self) -> None:
        await wait_finished(asyncio.gather(*(r.completion for r in self._tasks.values()), *self._management),
                            finish_on_cancel=True)
        self._sessions.clear()
        await wait_finished(asyncio.create_task(self._client.close()), finish_on_cancel=True)
        self.state = RuntimeState.CLOSED
