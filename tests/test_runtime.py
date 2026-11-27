"""真实 Session、SQLite 和 JSONL 上的 Runtime 并发及故障边界。"""

import asyncio
import json
import os
import sqlite3
import threading
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from data_paths import get_data_paths
from database.phase2_store import Phase2Store
from database.sqlite import get_connection
from database.stage1_store import Stage1Store
from history import HistoryStore
from memory.phase1.event_reader import EventReader
from runtime.manager import RuntimeManager, RuntimeState
from session import Session, SessionIntegrityError
from session_repository import SessionRepository
from test_compact import completion, response, tool_call
from tool_executor import ToolExecutionMode, ToolExecutor


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = patch.dict(os.environ, DSH_DATA_DIR=tmp.name)
        env.start()
        self.addCleanup(env.stop)
        get_data_paths.cache_clear()
        self.addCleanup(get_data_paths.cache_clear)
        self.directory = SessionRepository("导游", dict(context_window=32000, max_output_tokens=500,
                                                        threshold_ratio=.8, retain_ratio=.02))
        self.a, self.b = (self.directory.create(key) for key in ("A", "B"))
        self.client = AsyncMock()
        self.client.model = "test"
        self.client.ask.return_value = completion("完成")
        self.runtime = None
        self.release = []

    async def setup_runtime(self, **kwargs):
        self.runtime = RuntimeManager(self.directory, client_factory=lambda: self.client,
                                      executor_factory=lambda: ToolExecutor(include_amap=False), **kwargs)
        await self.runtime.start()
        return self.runtime

    async def asyncTearDown(self):
        for gate in self.release:
            gate.set()
        if self.runtime:
            await asyncio.wait_for(self.runtime.shutdown(), 5)

    def gate(self, threaded=False):
        gate = threading.Event() if threaded else asyncio.Event()
        self.release.append(gate)
        return gate

    async def wait_task(self, key):
        return await asyncio.wait_for(self.runtime.wait(key), 5)

    def events(self, session_id):
        return [json.loads(line) for line in get_data_paths().event_log(session_id).read_text().splitlines()]

    async def test_two_sessions_progress_independently(self):
        rt = await self.setup_runtime(max_concurrent_tasks=2)
        entered = {q: asyncio.Event() for q in ("A", "B")}
        release = {q: self.gate() for q in entered}

        async def ask(messages, **kwargs):
            q = messages[-1]["content"]
            entered[q].set()
            await asyncio.wait_for(release[q].wait(), 5)
            return completion(q)

        self.client.ask.side_effect = ask
        a = rt.submit(self.a, "A")
        await asyncio.wait_for(entered["A"].wait(), 5)
        await rt.select_session(self.b)
        b = rt.submit(self.b, "B")
        await asyncio.wait_for(entered["B"].wait(), 5)
        with self.assertRaises(RuntimeError):
            await rt.delete_session(self.a)
        release["B"].set()
        self.assertEqual((await self.wait_task(b))["answer"], "B")
        self.assertEqual(rt.get_task(a)["status"], "running")
        release["A"].set()
        self.assertEqual((await self.wait_task(a))["answer"], "A")
        for sid, answer in ((self.a, "A"), (self.b, "B")):
            self.assertEqual(json.loads(self.events(sid)[-3]["message"]["content"])["answer"], answer)
            self.assertEqual(self.events(sid)[-1]["reason"], "completed")

    async def test_capacity_queued_owner_and_waiter_cancellation(self):
        rt = await self.setup_runtime(max_concurrent_tasks=1, max_pending_tasks=2, max_completed_tasks=1)
        entered, release = asyncio.Event(), self.gate()

        async def ask(*args, **kwargs):
            entered.set()
            await asyncio.wait_for(release.wait(), 5)
            return completion("A")

        self.client.ask.side_effect = ask
        a = rt.submit(self.a, "A")
        await asyncio.wait_for(entered.wait(), 5)
        b = rt.submit_compact(self.b)
        self.assertEqual(rt.get_task(b)["status"], "queued")
        with self.assertRaises(RuntimeError):
            rt.submit(self.b, "重复")
        with self.assertRaisesRegex(RuntimeError, "容量"):
            rt.submit("other", "满")
        waiter = asyncio.create_task(rt.wait(a))
        await asyncio.sleep(0)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        waiting = asyncio.create_task(rt.wait(a))
        release.set()
        self.assertEqual((await self.wait_task(b))["result"].status, "skipped")
        self.assertEqual((await waiting)["answer"], "A")
        with self.assertRaises(KeyError):
            rt.get_task(a)

    async def test_creation_failure_early_cancel_and_old_callback(self):
        rt = await self.setup_runtime()
        with patch("runtime.manager.asyncio.create_task", side_effect=RuntimeError("task factory")):
            with self.assertRaisesRegex(RuntimeError, "task factory"):
                rt.submit(self.a, "A")
        self.assertEqual(rt.list_tasks(), [])
        key = rt.submit(self.a, "A")
        old = rt._tasks[key]
        old.task.cancel()
        self.assertEqual((await self.wait_task(key))["status"], "failed")
        new = rt.submit(self.a, "B")
        rt._done(old, old.task)
        self.assertEqual(rt._owners[self.a], new)
        self.assertEqual((await self.wait_task(new))["status"], "completed")

    async def test_cancellation_of_queued_and_running_tasks(self):
        rt = await self.setup_runtime(max_concurrent_tasks=1)
        entered, release = asyncio.Event(), self.gate()

        async def ask(*args, **kwargs):
            entered.set()
            await asyncio.wait_for(release.wait(), 5)

        self.client.ask.side_effect = ask
        a = rt.submit(self.a, "A")
        await asyncio.wait_for(entered.wait(), 5)
        b = rt.submit_compact(self.b)
        await asyncio.sleep(0)
        rt._tasks[b].task.cancel()
        self.assertEqual((await self.wait_task(b))["status"], "failed")
        retry = rt.submit_compact(self.b)
        rt._tasks[a].task.cancel()
        self.assertEqual((await self.wait_task(a))["status"], "failed")
        self.assertEqual(self.events(self.a)[-1]["reason"], "error")
        self.assertEqual((await self.wait_task(retry))["status"], "completed")

    async def test_repeated_cancel_during_message_write_keeps_owner_and_permit(self):
        rt = await self.setup_runtime(max_concurrent_tasks=1)
        entered, release = threading.Event(), self.gate(True)
        loop_thread = threading.get_ident()
        append = HistoryStore.append

        def held(history, events):
            if events[0].get("message", {}).get("role") == "user":
                self.assertNotEqual(threading.get_ident(), loop_thread)
                entered.set()
                self.assertTrue(release.wait(5))
            return append(history, events)

        with patch.object(HistoryStore, "append", held):
            a = rt.submit(self.a, "A")
            self.assertTrue(await asyncio.to_thread(entered.wait, 5))
            b = rt.submit_compact(self.b)
            for _ in range(2):
                rt._tasks[a].task.cancel()
                await asyncio.sleep(0)
                self.assertEqual(rt.get_task(a)["status"], "running")
                self.assertEqual(rt.get_task(b)["status"], "queued")
                with self.assertRaises(RuntimeError):
                    rt.submit_compact(self.a)
            release.set()
            self.assertEqual((await self.wait_task(a))["status"], "failed")
            self.assertEqual((await self.wait_task(b))["status"], "completed")
        self.directory.load(self.a, self.client)

    async def test_cancel_during_completed_commit_and_usage(self):
        rt = await self.setup_runtime()
        self.client.ask.return_value = completion("已完成", "source")
        entered, release = threading.Event(), self.gate(True)
        usage_entered, usage_release = threading.Event(), self.gate(True)
        original = Stage1Store.record_session_activity

        def activity(store, sid, when, seq):
            if seq is not None:
                entered.set()
                self.assertTrue(release.wait(5))
            original(store, sid, when, seq)

        def usage(ids):
            usage_entered.set()
            self.assertTrue(usage_release.wait(5))
            raise sqlite3.OperationalError("locked")

        with patch.object(Stage1Store, "record_session_activity", activity), \
                patch.object(Phase2Store, "record_usage", side_effect=usage) as record, self.assertLogs("agent"):
            key = rt.submit(self.a, "A")
            self.assertTrue(await asyncio.to_thread(entered.wait, 5))
            rt._tasks[key].task.cancel()
            release.set()
            self.assertTrue(await asyncio.to_thread(usage_entered.wait, 5))
            rt._tasks[key].task.cancel()
            await asyncio.sleep(0)
            self.assertEqual(rt.get_task(key)["status"], "running")
            usage_release.set()
            result = await self.wait_task(key)
            self.assertEqual((result["status"], result["answer"]), ("completed", "已完成"))
            record.assert_called_once_with({"source"})

    async def test_shutdown_drains_management_and_queued_tasks(self):
        rt = await self.setup_runtime(max_concurrent_tasks=1)
        entered, release = threading.Event(), self.gate(True)
        started, allow = asyncio.Event(), self.gate()
        create = self.directory.create

        def held_create(name):
            entered.set()
            self.assertTrue(release.wait(5))
            return create(name)

        async def held_ask(*args, **kwargs):
            started.set()
            await asyncio.wait_for(allow.wait(), 5)
            raise ValueError("offline")

        self.client.ask.side_effect = held_ask
        with patch.object(self.directory, "create", held_create):
            management = asyncio.create_task(rt.create_session("new"))
            self.assertTrue(await asyncio.to_thread(entered.wait, 5))
            a = rt.submit(self.a, "A")
            await asyncio.wait_for(started.wait(), 5)
            b = rt.submit_compact(self.b)
            shutdown = asyncio.create_task(rt.shutdown())
            await asyncio.sleep(0)
            self.assertEqual(rt.state, RuntimeState.CLOSING)
            shutdown.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await shutdown
            with self.assertRaises(RuntimeError):
                await rt.list_sessions()
            with self.assertRaises(RuntimeError):
                rt.submit(self.a, "no")
            allow.set()
            self.assertEqual((await self.wait_task(a))["status"], "failed")
            self.assertEqual((await self.wait_task(b))["status"], "completed")
            self.client.close.assert_not_awaited()
            release.set()
            await asyncio.wait_for(management, 5)
            await asyncio.wait_for(asyncio.gather(rt.shutdown(), rt.shutdown()), 5)
            self.client.close.assert_awaited_once()
            self.assertEqual(rt.state, RuntimeState.CLOSED)

    async def test_blocking_tool_and_internal_compact(self):
        entered, release = threading.Event(), self.gate(True)
        rt = await self.setup_runtime(max_concurrent_tasks=2)
        self.assertEqual((await self.wait_task(rt.submit_compact(self.a)))["status"], "completed")
        runtime = rt._sessions[self.a]

        def tool():
            entered.set()
            self.assertTrue(release.wait(5))
            return "ok"

        runtime.executor.register_tool("slow", "test", {}, tool, mode=ToolExecutionMode.BLOCKING)
        self.client.ask.side_effect = [
            response(None, calls=[tool_call("1", "slow"), tool_call("2", "compact_context")], finish="tool_calls"),
            completion("B"), completion("A"),
        ]
        a = rt.submit(self.a, "A")
        self.assertTrue(await asyncio.to_thread(entered.wait, 5))
        self.assertEqual((await self.wait_task(rt.submit(self.b, "B")))["answer"], "B")
        release.set()
        self.assertEqual((await self.wait_task(a))["answer"], "A")
        self.assertEqual([e["message"]["tool_call_id"] for e in self.events(self.a)
                          if e.get("message", {}).get("role") == "tool"], ["1", "2"])

    async def test_partial_commits_quarantine_only_the_affected_session(self):
        rt = await self.setup_runtime()
        for target, method in (("history.os", "fsync"), (Stage1Store, "record_session_activity"),
                               ("context", "append"), ("tokens", "update")):
            sid = await rt.create_session("fault")
            await self.wait_task(rt.submit_compact(sid))
            session = rt._sessions[sid].session
            fault = OSError("injected")
            if target == "history.os":
                injection = patch("history.os.fsync", side_effect=fault)
            else:
                obj = getattr(session, target) if isinstance(target, str) else target
                injection = patch.object(obj, method, side_effect=fault)
            with injection:
                result = await self.wait_task(rt.submit(sid, "A"))
            self.assertIn("injected", result["error"])
            self.assertIsNotNone(session.failure_reason)
            self.assertNotIn("turn/end", [e["type"] for e in self.events(sid)])
            with self.assertRaises(SessionIntegrityError):
                rt.submit_compact(sid)
        self.assertEqual((await self.wait_task(rt.submit(self.b, "B")))["status"], "completed")

    async def test_compact_tool_partial_replace_stops_all_writes(self):
        rt = await self.setup_runtime(max_concurrent_tasks=1)
        await self.wait_task(rt.submit(self.a, "a" * 4000))
        session = rt._sessions[self.a].session
        self.client.ask.side_effect = [
            response(None, calls=[tool_call("compact", "compact_context")], finish="tool_calls"), response("摘要"),
        ]
        with patch.object(session.context, "replace", side_effect=ValueError("replace fault")):
            result = await self.wait_task(rt.submit(self.a, "压缩" * 1000))
        self.assertIn("replace fault", result["error"])
        self.assertEqual(self.events(self.a)[-1]["type"], "compaction/summary")
        with self.assertRaises(SessionIntegrityError):
            rt.submit(self.a, "retry")

    async def test_watermark_and_open_history_admission(self):
        rt = await self.setup_runtime()
        await self.wait_task(rt.submit(self.a, "A"))
        end = self.events(self.a)[-1]["seq"]
        for value in (None, end, end + 1, end - 1):
            with closing(get_connection()) as conn, conn:
                conn.execute("UPDATE stage1_outputs SET last_completed_seq=? WHERE session_id=?", (value, self.a))
            if value is None or value == end:
                self.assertEqual(self.directory.load(self.a, self.client).last_completed_seq, end)
                with closing(get_connection()) as conn:
                    self.assertEqual(conn.execute("SELECT last_completed_seq FROM stage1_outputs WHERE session_id=?",
                                                  (self.a,)).fetchone()[0], end)
            else:
                with self.assertRaises(SessionIntegrityError):
                    self.directory.load(self.a, self.client)
        path = get_data_paths().event_log(self.b)
        store = HistoryStore(path)
        store.load("system")
        store.append([{"turn": 1, "type": "turn/start"}, {"turn": 1, "step": 1, "type": "step/start"}])
        result = await self.wait_task(rt.submit(self.b, "B"))
        self.assertIn("未闭合", result["error"])
        with self.assertRaises(SessionIntegrityError):
            rt.submit(self.b, "重开")
        with self.assertRaises(ValueError):
            EventReader(path).read(max_seq=2)

    async def test_tool_ids_rejected_live_and_at_replay(self):
        rt = await self.setup_runtime()
        for ids in (("",), ("A", "A")):
            self.client.ask.return_value = response(None, calls=[tool_call(i) for i in ids], finish="tool_calls")
            result = await self.wait_task(rt.submit(self.a, "A"))
            self.assertIn("ID", result["error"])
            self.assertFalse(any(e.get("message", {}).get("tool_calls") for e in self.events(self.a)))
        store = HistoryStore(get_data_paths().event_log(self.b))
        store.load("system")
        store.append([{"turn": 1, "type": "message", "message": {
            "role": "assistant", "tool_calls": [tool_call("A"), tool_call("B")]}}])
        for key in ("A", "A"):
            store.append([{"turn": 1, "type": "message", "message": {"role": "tool", "tool_call_id": key}}])
        self.assertIn("ID", (await self.wait_task(rt.submit_compact(self.b)))["error"])

    async def test_directory_creation_and_deletion_failures(self):
        rt = await self.setup_runtime()
        path = get_data_paths().event_log(self.a)
        with closing(get_connection()) as conn, conn:
            conn.execute("CREATE TRIGGER deny BEFORE DELETE ON sessions BEGIN SELECT RAISE(ABORT,'deny'); END")
        with self.assertRaises(sqlite3.Error):
            await rt.delete_session(self.a)
        self.assertTrue(path.exists())
        with closing(get_connection()) as conn, conn:
            conn.execute("DROP TRIGGER deny")
        with patch.object(Path, "unlink", side_effect=OSError("denied")), self.assertRaisesRegex(OSError, "日志清理失败"):
            await rt.delete_session(self.a)
        self.assertNotIn(self.a, [s["session_id"] for s in await rt.list_sessions()])
        directories = set(path.parent.parent.iterdir())
        with patch("history.os.fsync", side_effect=OSError("create")), self.assertRaises(OSError):
            await rt.create_session("failed")
        self.assertEqual(set(path.parent.parent.iterdir()), directories)

    async def test_start_lifecycle_and_loop_affinity(self):
        rt = RuntimeManager(self.directory, client_factory=lambda: self.client)
        await rt.start()
        self.runtime = rt
        await rt.start()
        async def foreign():
            rt.submit(self.a, "foreign")
        with self.assertRaisesRegex(RuntimeError, "Event Loop"):
            await asyncio.to_thread(asyncio.run, foreign())
        broken = RuntimeManager(self.directory, client_factory=unittest.mock.Mock(side_effect=RuntimeError("startup")))
        with self.assertRaisesRegex(RuntimeError, "startup"):
            await broken.start()
        self.assertEqual(broken.state, RuntimeState.CLOSED)
        self.assertFalse(hasattr(self.directory, "client"))
        self.client.close.assert_not_awaited()

    async def test_initialization_exclusive_and_failed_start_never_publishes(self):
        rt = await self.setup_runtime()
        entered, release = asyncio.Event(), self.gate()

        async def initialize(executor):
            entered.set()
            await asyncio.wait_for(release.wait(), 4)
            raise RuntimeError("initialize")

        with patch.object(ToolExecutor, "initialize", initialize), patch.object(
            self.directory, "load", wraps=self.directory.load,
        ) as loads:
            key = rt.submit_compact(self.a)
            await asyncio.wait_for(entered.wait(), 4)
            self.assertNotIn(self.a, rt._sessions)
            with self.assertRaises(RuntimeError):
                rt.submit(self.a, "重复")
            release.set()
            self.assertIn("initialize", (await self.wait_task(key))["error"])
            self.assertNotIn(self.a, rt._sessions)
            loads.assert_called_once_with(self.a, self.client)
        self.assertEqual((await self.wait_task(rt.submit_compact(self.a)))["status"], "completed")

    async def test_background_memory_drains_after_foreground(self):
        entered, release = asyncio.Event(), self.gate()
        async def phase(agent, stopped):
            entered.set()
            await asyncio.wait_for(release.wait(), 4)
            raise ValueError("offline")
        with patch("memory.memory_maneger.Phase1Agent.run", phase), self.assertLogs("memory.memory_maneger"):
            rt = await self.setup_runtime(max_concurrent_tasks=1)
            await asyncio.wait_for(entered.wait(), 4)
            self.assertEqual((await self.wait_task(rt.submit(self.a, "A")))["status"], "completed")
            shutdown = asyncio.create_task(rt.shutdown())
            await asyncio.sleep(0)
            self.client.close.assert_not_awaited()
            release.set()
            await asyncio.wait_for(shutdown, 4)
            self.client.close.assert_awaited_once()

    async def test_replay_context_and_search_do_not_block_other_session(self):
        from agent import GuideAgent
        from memory.reader import search_memory
        rt = await self.setup_runtime(max_concurrent_tasks=2)
        loop_thread = threading.get_ident()
        for name in ("load", "context", "search"):
            entered, release = threading.Event(), self.gate(True)
            def held(operation, *args, **kwargs):
                self.assertNotEqual(threading.get_ident(), loop_thread)
                entered.set()
                self.assertTrue(release.wait(4))
                return operation(*args, **kwargs)
            sid = await rt.create_session(name)
            if name == "load":
                load = self.directory.load
                injection = patch.object(self.directory, "load", side_effect=lambda s, client: held(load, s, client) if s == sid else load(s, client))
            elif name == "context":
                build = GuideAgent._request_messages
                injection = patch.object(GuideAgent, "_request_messages",
                                         lambda agent, *args: held(build, agent, *args) if agent.session.session_id == sid
                                         else build(agent, *args))
            else:
                injection = patch("memory.tools.search_memory", side_effect=lambda **kw: held(search_memory, **kw))

            async def ask(messages, **options):
                if name == "search" and messages[-1].get("content") == "search":
                    return response(None, calls=[tool_call("s", "search_memory", queries=["火车"])], finish="tool_calls")
                return completion("ok")
            self.client.ask.side_effect = ask
            with injection:
                a = rt.submit(sid, name)
                self.assertTrue(await asyncio.to_thread(entered.wait, 4))
                self.assertEqual((await self.wait_task(rt.submit(self.b, "other")))["status"], "completed")
                self.assertEqual(rt.get_task(a)["status"], "running")
                release.set()
                self.assertEqual((await self.wait_task(a))["status"], "completed")

    async def test_manual_compact_model_holds_session_and_permit(self):
        rt = await self.setup_runtime(max_concurrent_tasks=1)
        for _ in range(2):
            await self.wait_task(rt.submit(self.a, "历史" * 1200))
        entered, release = asyncio.Event(), self.gate()
        async def summarize(messages, tools=None, **options):
            self.assertIsNone(tools)
            self.assertNotIn("response_format", options)
            entered.set()
            await asyncio.wait_for(release.wait(), 4)
            return response("摘要")
        self.client.ask.side_effect = summarize
        key = rt.submit_compact(self.a)
        await asyncio.wait_for(entered.wait(), 4)
        with self.assertRaises(RuntimeError):
            rt.submit(self.a, "冲突")
        b = rt.submit_compact(self.b)
        self.assertEqual(rt.get_task(b)["status"], "queued")
        release.set()
        self.assertEqual((await self.wait_task(key))["result"].status, "compacted")
        self.assertEqual((await self.wait_task(b))["status"], "completed")

    async def test_cli_switches_with_an_active_turn_then_drains(self):
        import main
        from functools import partial
        entered, release = threading.Event(), asyncio.Event()
        loop = asyncio.get_running_loop()
        async def ask(messages, **options):
            question = messages[-1]["content"]
            if question == "A":
                entered.set()
                await asyncio.wait_for(release.wait(), 4)
            else:
                release.set()
            return completion(question)
        self.client.ask.side_effect = ask
        commands = iter(("1", "A", "/switch", "2", "B", "/tasks", "/exit"))
        def read(prompt):
            command = next(commands)
            if command == "/switch":
                self.assertTrue(entered.wait(4))
            self.assertNotEqual(threading.get_ident(), loop_thread)
            return command
        loop_thread = threading.get_ident()
        with patch("main.RuntimeManager", partial(RuntimeManager, client_factory=lambda: self.client,
                                                 executor_factory=lambda: ToolExecutor(include_amap=False))), \
                patch("builtins.input", read), patch("builtins.print") as output:
            try:
                await asyncio.wait_for(main.main(), 6)
            finally:
                loop.call_soon(release.set)
        self.client.close.assert_awaited_once()
        self.assertEqual({json.loads(self.events(sid)[-3]["message"]["content"])["answer"] for sid in (self.a,self.b)},
                         {"A", "B"})
        self.assertFalse(any(args.args == ("A",) or args.args == ("B",) for args in output.call_args_list))

    async def test_completed_log_without_sqlite_commit_does_not_publish_an_answer(self):
        rt = await self.setup_runtime()
        original = Stage1Store.record_session_activity
        def fail_watermark(store, sid, when, seq):
            if seq is not None:
                raise sqlite3.OperationalError("watermark failed")
            return original(store, sid, when, seq)
        self.client.ask.return_value = completion("已生成", "source")
        with patch.object(Stage1Store, "record_session_activity", fail_watermark), \
                patch.object(Phase2Store, "record_usage") as usage:
            result = await self.wait_task(rt.submit(self.a, "A"))
            self.assertEqual(result["status"], "failed")
            self.assertIsNone(result["answer"])
            self.assertIn("watermark failed", result["error"])
            usage.assert_not_called()
        events = self.events(self.a)
        self.assertEqual([e["reason"] for e in events if e["type"] == "turn/end"], ["completed"])
        with closing(get_connection()) as conn:
            self.assertIsNone(conn.execute("SELECT last_completed_seq FROM stage1_outputs WHERE session_id=?",
                                           (self.a,)).fetchone()[0])
        with self.assertRaises(SessionIntegrityError):
            rt.submit(self.a, "retry")

    async def test_cancel_batch_after_tool_started_pairs_remaining_ids(self):
        rt = await self.setup_runtime(max_concurrent_tasks=1)
        await self.wait_task(rt.submit_compact(self.a))
        entered = asyncio.Event()
        async def stall():
            entered.set()
            await asyncio.Event().wait()
        rt._sessions[self.a].executor.register_tool("stall", "test", {}, stall, mode=ToolExecutionMode.ASYNC)
        self.client.ask.return_value = response(None, calls=[tool_call("A", "stall"), tool_call("B", "stall")],
                                                finish="tool_calls")
        key = rt.submit(self.a, "A")
        await asyncio.wait_for(entered.wait(), 4)
        rt._tasks[key].task.cancel()
        self.assertEqual((await self.wait_task(key))["status"], "failed")
        events = self.events(self.a)
        self.assertEqual([e["message"]["tool_call_id"] for e in events if e.get("message", {}).get("role") == "tool"],
                         ["A", "B"])
        self.assertEqual(events[-1]["reason"], "error")
        self.directory.load(self.a, self.client)

    async def test_load_lock_can_be_retried_and_activity_is_reconciled(self):
        rt = await self.setup_runtime()
        def immediate_connection():
            conn = get_connection()
            conn.execute("PRAGMA busy_timeout=0")
            return conn

        with closing(get_connection()) as holder, patch("session_repository.get_connection", immediate_connection):
            holder.execute("BEGIN IMMEDIATE")
            result = await self.wait_task(rt.submit_compact(self.a))
            self.assertIn("locked", result["error"])
        path = get_data_paths().event_log(self.a)
        stamp = int(path.stat().st_mtime) + 5
        os.utime(path, (stamp, stamp))
        self.assertEqual((await self.wait_task(rt.submit_compact(self.a)))["status"], "completed")
        await self.wait_task(rt.submit_compact(self.a))
        await rt.select_session(self.a)
        with closing(get_connection()) as conn:
            row = conn.execute("SELECT updated_at,usage FROM sessions WHERE session_id=?", (self.a,)).fetchone()
        self.assertEqual(tuple(row), (stamp, 2))

    async def test_repair_failure_quarantines_even_after_file_was_truncated(self):
        rt = await self.setup_runtime()
        path = get_data_paths().event_log(self.a)
        with path.open("ab") as target:
            target.write(b'{"seq":')
        with patch("history.os.fsync", side_effect=OSError("repair")):
            self.assertIn("repair", (await self.wait_task(rt.submit_compact(self.a)))["error"])
        with self.assertRaises(SessionIntegrityError):
            rt.submit_compact(self.a)
        self.assertTrue(await rt.archive_session(self.a))
        self.assertTrue(await rt.delete_session(self.a))
        self.assertIn("SessionNotFound", (await self.wait_task(rt.submit_compact(self.a)))["error"])

    async def test_failed_session_survives_rollback_but_confirmed_delete_retires_writer(self):
        rt = await self.setup_runtime()
        await self.wait_task(rt.submit_compact(self.a))
        old = rt._sessions[self.a].session
        with patch.object(old.context, "append", side_effect=ValueError("projection")):
            self.assertIn("projection", (await self.wait_task(rt.submit(self.a, "A")))["error"])
        self.assertTrue(await rt.archive_session(self.a))
        with closing(get_connection()) as conn, conn:
            conn.execute("CREATE TRIGGER deny BEFORE DELETE ON sessions BEGIN SELECT RAISE(ABORT,'deny'); END")
        with self.assertRaises(sqlite3.Error):
            await rt.delete_session(self.a)
        with self.assertRaisesRegex(SessionIntegrityError, "projection"):
            rt.submit_compact(self.a)
        with closing(get_connection()) as conn, conn:
            conn.execute("DROP TRIGGER deny")
        with patch.object(Path, "unlink", side_effect=OSError("busy")), self.assertRaisesRegex(OSError, "日志清理失败"):
            await rt.delete_session(self.a)
        with self.assertRaisesRegex(SessionIntegrityError, "目录已删除"):
            old.commit_event({"type": "turn/start", "turn": 2})
        self.assertIn("SessionNotFound", (await self.wait_task(rt.submit_compact(self.a)))["error"])

    async def test_cancel_at_compact_commit_distinguishes_task_from_tool(self):
        rt = await self.setup_runtime(max_concurrent_tasks=1)
        for sid in (self.a, self.b):
            await self.wait_task(rt.submit(sid, "历史" * 2000))
            await self.wait_task(rt.submit(sid, "历史" * 1000))
        for sid in (self.a, self.b):
            compact = rt._sessions[sid].session.compact
            commit = compact.commit_compaction
            entered, release = threading.Event(), self.gate(True)
            def held(*args):
                result = commit(*args)
                entered.set()
                self.assertTrue(release.wait(4))
                return result
            self.client.ask.side_effect = [response("摘要")] if sid == self.a else [
                response(None, calls=[tool_call("compact", "compact_context")], finish="tool_calls"), response("摘要"),
            ]
            with patch.object(compact, "commit_compaction", held):
                key = rt.submit_compact(sid) if sid == self.a else rt.submit(sid, "请压缩")
                self.assertTrue(await asyncio.to_thread(entered.wait, 4))
                rt._tasks[key].task.cancel()
                with self.assertRaises(RuntimeError):
                    rt.submit_compact(sid)
                release.set()
                result = await self.wait_task(key)
            if sid == self.a:
                self.assertEqual(result["result"].summary_seq, self.events(sid)[-1]["seq"])
                self.assertEqual(result["status"], "completed")
            else:
                self.assertIn("CancelledError", result["error"])
                self.assertEqual(self.events(sid)[-1]["reason"], "error")
                self.directory.load(sid, self.client)

    async def test_direct_drain_cancel_preserves_completion_and_owner(self):
        rt = await self.setup_runtime()
        entered, release = asyncio.Event(), self.gate()
        async def ask(*args, **kwargs):
            entered.set()
            await asyncio.wait_for(release.wait(), 4)
            return completion("已提交")
        self.client.ask.side_effect = ask
        key = rt.submit(self.a, "A")
        await asyncio.wait_for(entered.wait(), 4)
        draining = asyncio.Event()
        drain = rt._drain
        async def observed_drain():
            draining.set()
            await drain()
        with patch.object(rt, "_drain", observed_drain):
            waiter = asyncio.create_task(rt.shutdown())
            await asyncio.wait_for(draining.wait(), 4)
        rt._closing.cancel()
        await asyncio.sleep(0)
        self.assertFalse(rt._tasks[key].completion.done())
        self.assertEqual(rt._owners[self.a], key)
        release.set()
        self.assertEqual((await self.wait_task(key))["answer"], "已提交")
        await asyncio.wait_for(waiter, 4)
        self.assertNotIn(self.a, rt._owners)
        self.assertEqual(rt.state, RuntimeState.CLOSED)
        self.client.close.assert_awaited_once()

    async def test_completed_structured_turn_needs_input_and_answer_in_step(self):
        rt = await self.setup_runtime()
        for sid, with_input in ((self.a, False), (self.b, True)):
            events = [{"turn": 1, "type": "turn/start"}]
            if with_input:
                events.append({"turn": 1, "type": "message", "message": {"role": "user", "content": "Q"}})
            events.append({"turn": 1, "step": 1, "type": "step/start"})
            answer = {"turn": 1, "type": "message", "message": {"role": "assistant", "content": "A"}}
            if not with_input:
                events.append({**answer, "step": 1})
            events.append({"turn": 1, "step": 1, "type": "step/end"})
            if with_input:
                events.append(answer)
            events.append({"turn": 1, "type": "turn/end", "reason": "completed"})
            store = HistoryStore(get_data_paths().event_log(sid))
            store.load("导游")
            store.append(events)
            self.assertIn("completed", (await self.wait_task(rt.submit_compact(sid)))["error"])
            with self.assertRaises(SessionIntegrityError):
                rt.submit(sid, "Q")

    async def test_unconfirmed_delete_keeps_log_and_blocks_cached_writer(self):
        rt = await self.setup_runtime()
        await self.wait_task(rt.submit_compact(self.a))
        path = get_data_paths().event_log(self.a)
        class FailedCommit(sqlite3.Connection):
            def commit(self):
                raise sqlite3.OperationalError("commit failed")
        attempts = iter(("transaction", "verify"))
        def connect():
            if next(attempts) == "verify":
                raise sqlite3.OperationalError("verify failed")
            conn = sqlite3.connect(get_data_paths().database, factory=FailedCommit)
            conn.row_factory = sqlite3.Row
            return conn
        with patch("session_repository.get_connection", connect), self.assertRaisesRegex(SessionIntegrityError, "无法确认"):
            await rt.delete_session(self.a)
        self.assertTrue(path.is_file())
        with self.assertRaisesRegex(SessionIntegrityError, "无法确认"):
            rt.submit(self.a, "A")
        self.assertTrue(await rt.delete_session(self.a))

    async def test_replay_program_error_does_not_quarantine_session(self):
        rt = await self.setup_runtime()
        with patch.object(Session, "_replay", side_effect=ValueError("implementation fault")):
            failed = await self.wait_task(rt.submit_compact(self.a))
        self.assertIn("ValueError: implementation fault", failed["error"])
        self.assertEqual((await self.wait_task(rt.submit(self.a, "A")))["status"], "completed")
