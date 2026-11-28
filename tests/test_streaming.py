"""使用真实 SDK 解码离线 HTTP SSE，并观察每个响应的关闭边界。"""

import asyncio
import json
import unittest
from unittest.mock import Mock

import httpx
from openai import AsyncOpenAI

from client import AsyncTripClient


def chunk(delta=None, finish=None, usage=None):
    choices = [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]
    return dict(id="chat", object="chat.completion.chunk", created=1, model="test", choices=choices, usage=usage)


class Body(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.entered = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        for item in self.chunks:
            if isinstance(item, asyncio.Event):
                self.entered.set()
                await asyncio.wait_for(item.wait(), 5)
            elif isinstance(item, Exception):
                raise item
            else:
                yield ("data: " + json.dumps(item) + "\n\n").encode()
        yield b"data: [DONE]\n\n"

    async def aclose(self):
        self.closed = True


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bodies = {}
        self.requests = []

        def handle(request):
            options = json.loads(request.content)
            self.requests.append(options)
            return httpx.Response(200, stream=self.bodies[options["messages"][0]["content"]],
                                  headers={"content-type": "text/event-stream"})

        self.client = AsyncTripClient.__new__(AsyncTripClient)
        self.client.model = "test"
        self.client._api = AsyncOpenAI(api_key="test", max_retries=0,
                                       http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        self.addAsyncCleanup(self.client.close)

    def setup_stream(self, key, chunks, **options):
        self.bodies[key] = Body(chunks)
        return self.client.ask_stream([{"role": "user", "content": key}], **options)

    async def test_interleaved_tools_reasoning_content_usage_and_request(self):
        def part(i, args, meta=False):
            call = dict(index=i, function=dict(arguments=args))
            if meta:
                call.update(id=f"id{i}", type="function")
                call["function"]["name"] = f"f{i}"
            return call

        usage = dict(prompt_tokens=4, completion_tokens=6, total_tokens=10)
        seen = []
        result = await self.setup_stream("a", [
            chunk(dict(role="assistant", content="甲", reasoning_content="私",
                       tool_calls=[part(1, '{"n":', True), part(0, '{"n":', True)])),
            chunk(dict(content="乙", reasoning_content="密", tool_calls=[part(0, "0}", True), part(1, "1}")])),
            chunk({}, "tool_calls"), chunk(usage=usage),
        ], on_delta=seen.append, response_format={"type": "json_object"}, max_tokens=50)
        choice = result.choices[0]
        message = choice.message.model_dump(exclude_none=True)
        self.assertEqual((result.id, result.model, result.created), ("chat", "test", 1))
        self.assertEqual((message["content"], message["reasoning_content"], seen), ("甲乙", "私密", ["甲", "乙"]))
        self.assertEqual(choice.finish_reason, "tool_calls")
        self.assertEqual([(c.id, c.name, c.arguments) for c in self.client.parse_response(result).tool_calls],
                         [("id0", "f0", {"n": 0}), ("id1", "f1", {"n": 1})])
        self.assertEqual(result.usage.total_tokens, 10)
        self.assertTrue(self.bodies["a"].closed)
        self.assertEqual(self.requests, [dict(model="test", messages=[dict(role="user", content="a")], stream=True,
                                              stream_options=dict(include_usage=True), response_format=dict(type="json_object"),
                                              max_tokens=50)])

    async def test_concurrent_streams_keep_state_and_last_valid_usage(self):
        gate = asyncio.Event()
        usage = dict(prompt_tokens=2, completion_tokens=1, total_tokens=3)
        seen = []
        a = asyncio.create_task(self.setup_stream("a", [
            chunk(dict(role="assistant", content="start"), usage=usage), gate, chunk(dict(content="end"), "stop")
        ], on_delta=seen.append))
        try:
            await asyncio.wait_for(self.bodies["a"].entered.wait(), 4)
            b = await self.setup_stream("b", [chunk(dict(role="assistant", content="other"), "stop")])
            self.assertIsNone(b.usage)
            self.assertEqual(b.choices[0].message.content, "other")
            self.assertEqual(seen, ["start"])
        finally:
            gate.set()
            result = await asyncio.wait_for(a, 4)
        self.assertEqual(result.choices[0].message.content, "startend")
        self.assertEqual(result.usage.total_tokens, 3)

    async def test_incomplete_streams_and_callback_error_close_response(self):
        for finish in (None, "length", "content_filter", "tool_calls"):
            with self.subTest(finish=finish), self.assertRaises(ValueError):
                await self.setup_stream("a", [chunk(dict(role="assistant", content="partial"), finish)])
            self.assertTrue(self.bodies["a"].closed)
        with self.assertRaisesRegex(RuntimeError, "callback"):
            await self.setup_stream("a", [chunk(dict(role="assistant", content="x"), "stop")],
                                    on_delta=Mock(side_effect=RuntimeError("callback")))
        self.assertTrue(self.bodies["a"].closed)
        with self.assertRaises(httpx.ReadError):
            await self.setup_stream("a", [chunk(dict(role="assistant")), httpx.ReadError("broken")])
        self.assertTrue(self.bodies["a"].closed)
        self.assertFalse(self.client._api.is_closed())

    async def test_incomplete_tool_structure_and_invalid_argument_json(self):
        call = dict(index=0, type="function", id="id", function=dict(name="echo", arguments="{bad"))
        for field in ("type", "id", "function"):
            incomplete = {k: v for k, v in call.items() if k != field}
            with self.subTest(field=field), self.assertRaises(ValueError):
                await self.setup_stream("a", [chunk(dict(role="assistant", tool_calls=[incomplete]), "tool_calls")])
        with self.assertRaises(ValueError):
            await self.setup_stream("a", [chunk(dict(role="assistant", tool_calls=[call]), "stop")])
        result = await self.setup_stream("a", [chunk(dict(role="assistant", tool_calls=[call]), "tool_calls")])
        with self.assertRaises(json.JSONDecodeError):
            self.client.parse_response(result)

    async def test_cancel_partial_tool_response_closes_only_that_stream(self):
        task = asyncio.create_task(self.setup_stream("a", [
            chunk(dict(role="assistant", content="partial", tool_calls=[
                dict(index=0, id="pending", type="function", function=dict(name="f", arguments="{"))])),
            asyncio.Event(),
        ]))
        await asyncio.wait_for(self.bodies["a"].entered.wait(), 4)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 4)
        self.assertTrue(self.bodies["a"].closed)
        self.assertFalse(self.client._api.is_closed())
