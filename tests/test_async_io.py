"""验证原生异步客户端和 MCP 请求级资源生命周期，不访问网络。"""

import asyncio
import threading
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from MCP.amap_mcp import AmapMCP
from client import AsyncTripClient, Trip_Client
from tool_executor import ToolExecutionMode, ToolExecutor


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_clients_share_payload_and_keep_transport_modes(self):
        config = {"DEEPSEEK_API_KEY": "local-test", "DEEPSEEK_MODEL": "test", "DEEPSEEK_TIMEOUT": 3,
                  "DEEPSEEK_BASE_URL": "https://example.invalid"}
        sync_sdk, async_sdk = MagicMock(), MagicMock()
        async_sdk.chat.completions.create = AsyncMock()
        async_sdk.close = AsyncMock()
        with patch("client.load_settings", return_value=config), \
                patch.object(Trip_Client, "sdk_class", return_value=sync_sdk) as sync_factory, \
                patch.object(AsyncTripClient, "sdk_class", return_value=async_sdk) as async_factory:
            async_client = AsyncTripClient()
            with Trip_Client() as sync:
                for tools in (None, [], [{"type": "function"}]):
                    options = dict(temperature=.3, response_format={"type": "json_object"}, max_tokens=128)
                    messages = [{"role": "user", "content": "test"}]
                    sync.ask(messages, tools, **options)
                    await async_client.ask(messages, tools, **options)
                    args = async_sdk.chat.completions.create.await_args.kwargs
                    self.assertEqual(args, sync_sdk.chat.completions.create.call_args.kwargs)
                    self.assertEqual("tool_choice" in args, bool(tools))
                    self.assertEqual("tools" in args, bool(tools))
                    self.assertFalse(args["stream"])
            await async_client.close()
            self.assertEqual(sync_factory.call_args, async_factory.call_args)
            self.assertEqual(async_factory.call_args.kwargs["max_retries"], 0)
            async_sdk.close.assert_awaited_once()

    def contexts(self, remote, trace, cleanup_gate=None):
        """记录 SDK 各层进入退出的实际 Task，取消测试可暂停最内层清理。"""
        def context(label, value):
            @asynccontextmanager
            async def open_context(*args, **kwargs):
                trace.append(("enter", label, asyncio.current_task()))
                try:
                    yield value
                finally:
                    if label == "session" and cleanup_gate:
                        entered, release = cleanup_gate
                        entered.set()
                        await asyncio.wait_for(release.wait(), 3)
                    trace.append(("exit", label, asyncio.current_task()))
            return open_context
        for target, value in (
            ("AsyncClient", context("http", object())),
            ("streamable_http_client", context("stream", (object(), object(), None))),
            ("ClientSession", context("session", remote)),
        ):
            mock = patch("MCP.amap_mcp." + target, value)
            mock.start()
            self.addCleanup(mock.stop)

    def assert_closed(self, trace):
        self.assertEqual([(op, label) for op, label, _ in trace],
                         [("enter", "http"), ("enter", "stream"), ("enter", "session"),
                          ("exit", "session"), ("exit", "stream"), ("exit", "http")])
        self.assertEqual(len({task for _, _, task in trace}), 1)

    async def test_discovery_pagination_and_handler_share_no_sessions(self):
        tool = SimpleNamespace(name="maps_text_search", description="search", inputSchema={"type": "object"})
        other = SimpleNamespace(name="excluded")
        remote = AsyncMock()
        remote.list_tools.side_effect = [SimpleNamespace(tools=[other], nextCursor="second"),
                                         SimpleNamespace(tools=[tool], nextCursor=None)]
        remote.call_tool.return_value = MagicMock()
        remote.call_tool.return_value.model_dump.return_value = {"ok": True}
        trace = []
        self.contexts(remote, trace)
        executor = ToolExecutor(include_amap=False)
        executor._amap = AmapMCP("local-test", 2)
        self.assertEqual(executor.get_all_tools(), [])
        await executor.initialize()
        self.assert_closed(trace)
        trace.clear()
        with self.assertRaises(TypeError):
            executor.call_tool(tool.name, {})
        self.assertEqual(await executor.call_tool_async(tool.name, {"q": "杭州"}), {"ok": True})
        self.assert_closed(trace)
        remote.call_tool.assert_awaited_once_with(tool.name, arguments={"q": "杭州"})
        self.assertEqual(remote.initialize.await_count, 2)

    async def test_mcp_discovery_failure_and_pagination_limits(self):
        remote = AsyncMock()
        trace = []
        self.contexts(remote, trace)
        for repeat, expected in ((True, 2), (False, 32)):
            with self.subTest(repeat=repeat):
                trace.clear()
                remote.list_tools.reset_mock()
                remote.list_tools.side_effect = [SimpleNamespace(tools=[], nextCursor="same" if repeat else str(i))
                                                 for i in range(35)]
                executor = ToolExecutor(include_amap=False)
                executor._amap = AmapMCP("secret-test", 2)
                executor.register_tool("local", "local", {}, lambda: 1)
                with self.assertLogs("tool_executor", level="WARNING") as logs:
                    await executor.initialize()
                self.assertNotIn("secret-test", str(logs.output))
                self.assertEqual(remote.list_tools.await_count, expected)
                self.assertEqual(await executor.call_tool_async("local", {}), 1)
                self.assert_closed(trace)

    async def test_timeout_and_network_failure_close_resources(self):
        trace = []
        remote = AsyncMock()
        self.contexts(remote, trace)
        async def stalled(*args, **kwargs):
            await asyncio.Event().wait()
        for operation in (stalled, OSError("secret-test")):
            trace.clear()
            remote.call_tool.side_effect = operation
            mcp = AmapMCP("secret-test", .05)
            result = await asyncio.wait_for(mcp.handler("maps_text_search")(), 3)
            self.assertTrue(result["isError"])
            self.assertNotIn("secret-test", str(result))
            self.assert_closed(trace)

    async def test_repeated_cancel_drains_mcp_context_in_own_task(self):
        trace = []
        entered, closing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        remote = AsyncMock()
        self.contexts(remote, trace, (closing, release))
        async def stalled(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()
        remote.call_tool.side_effect = stalled
        task = asyncio.create_task(AmapMCP("test", 10).handler("maps_text_search")())
        try:
            await asyncio.wait_for(entered.wait(), 3)
            task.cancel()
            await asyncio.wait_for(closing.wait(), 3)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
        finally:
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
        self.assert_closed(trace)

    async def test_blocking_handlers_have_worker_affinity(self):
        executor = ToolExecutor(include_amap=False)
        executor.register_tool("blocking", "test", {}, threading.get_ident, mode=ToolExecutionMode.BLOCKING)
        self.assertEqual(executor.call_tool("blocking", {}), threading.get_ident())
        self.assertNotEqual(await executor.call_tool_async("blocking", {}), threading.get_ident())
