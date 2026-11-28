import asyncio
import unittest

from runtime.events import MAX_TASK_EVENTS, TaskEventBuffer


class EventTests(unittest.IsolatedAsyncioTestCase):
    async def test_independent_cursors_and_payloads(self):
        buffer = TaskEventBuffer("a")
        other = TaskEventBuffer("b")
        data = {"nested": ["queued"]}
        for item in (buffer, other):
            item.publish("task/queued", data)
        data["nested"].clear()
        readers = [buffer.subscribe(0) for _ in range(2)]
        for reader in readers:
            event = await anext(reader)
            self.assertEqual((event.seq, event.data), (1, {"nested": ["queued"]}))
            event.data["nested"].clear()
        self.assertEqual((await anext(other.subscribe(0))).seq, 1)
        entered = [asyncio.Event(), asyncio.Event()]

        async def read(index):
            entered[index].set()
            return await anext(readers[index])

        pending = [asyncio.create_task(read(i)) for i in range(2)]
        await asyncio.wait_for(asyncio.gather(*(gate.wait() for gate in entered)), 2)
        buffer.publish("task/end", {"answer": "ok"})
        results = await asyncio.wait_for(asyncio.gather(*pending), 2)
        self.assertEqual([e.seq for e in results], [2, 2])
        for reader in readers:
            self.assertEqual([e async for e in reader], [])
        self.assertEqual([e.seq async for e in buffer.subscribe(1)], [2])
        self.assertEqual([e async for e in buffer.subscribe(2)], [])
        for cursor in (-1, 3, True, 1.0):
            with self.assertRaises(ValueError):
                await anext(buffer.subscribe(cursor))

    async def test_count_and_byte_eviction_terminal_exemption_and_gaps(self):
        buffer = TaskEventBuffer("a")
        buffer.publish("task/queued", {})
        reader = buffer.subscribe(0)
        await anext(reader)
        for i in range(MAX_TASK_EVENTS + 10):
            buffer.publish("step/start", {"step": i})
        gap = await anext(reader)
        self.assertEqual((gap.type, gap.seq, gap.data), ("stream/gap", 11, {"missing_from": 2, "missing_to": 11}))
        self.assertEqual((await anext(reader)).seq, 12)
        delta = "中" * 100000
        buffer.publish("llm/content_delta", {"delta": delta})
        buffer.publish("task/end", {"answer": delta})
        events = [e async for e in reader]
        self.assertEqual(events[0].type, "stream/gap")
        self.assertEqual(events[0].data["missing_from"], 13)
        tail = "".join(e.data["delta"] for e in events[1:-1])
        self.assertTrue(tail and delta.endswith(tail))
        self.assertLess(len(tail.encode()), 256 * 1024)
        self.assertEqual(events[-1].data["answer"], delta)
        self.assertEqual(len([e for e in events if e.type == "task/end"]), 1)
