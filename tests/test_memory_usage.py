"""通过离线模型、真实日志和临时 SQLite 验证记忆使用闭环。"""

import json
import os
import time
import unittest
from contextlib import closing
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agent import GuideAgent
from data_paths import get_data_paths
from database.phase2_store import Phase2Store
from database.sqlite import get_connection
from database.stage1_store import Stage1Store
from memory.memory_maneger import MemoryManager
from session_manager import SessionManager
from test_compact import FakeClient, completion, response, tool_call
from tool_executor import ToolExecutor


def step(*calls):
    return response(None, calls=list(calls), finish="tool_calls")


class MemoryUsageTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        env = patch.dict(os.environ, DSH_DATA_DIR=temp.name)
        env.start()
        self.addCleanup(env.stop)
        get_data_paths.cache_clear()
        self.addCleanup(get_data_paths.cache_clear)
        self.client = FakeClient()
        self.manager = SessionManager(self.client, "导游", {
            "context_window": 32000, "max_output_tokens": 500, "threshold_ratio": .85, "retain_ratio": .1,
        })
        self.session = self.manager.create("当前会话")
        self.executor = ToolExecutor(include_amap=False)
        self.agent = GuideAgent(self.client, self.executor, self.session)
        self.path = get_data_paths().memory_summary

    def seed(self, sid="A", summary="杭州游览，偏好公交", slug="hangzhou", enabled=1, allow=1):
        now = int(time.time())
        with closing(get_connection()) as conn, conn:
            conn.execute(
                "INSERT INTO sessions(session_id,event_log_path,name,created_at,memory_enabled) VALUES (?,?,?,?,?)",
                (sid, sid, sid, now, enabled),
            )
            conn.execute(
                """INSERT INTO stage1_outputs(session_id,last_completed_seq,source_max_seq,
                   rollout_summary,rollout_slug,generated_at,allow_global) VALUES (?,10,10,?,?,?,?)""",
                (sid, summary, slug, now, allow),
            )

    def usage(self, sid="A"):
        with closing(get_connection()) as conn:
            return tuple(conn.execute(
                "SELECT usage_count,last_usage FROM stage1_outputs WHERE session_id=?", (sid,),
            ).fetchone())

    def test_snapshot_retrieval_feedback_and_replay(self):
        self.seed(allow=0)
        self.seed("B", "苏州园林", "suzhou")
        self.seed("C", "只有全局记忆用到此来源", "global")
        self.path.write_text("快照甲 session-C.md")
        self.executor.register_tool("refresh", "模拟 Phase 2 更新", {"type": "object"},
                                    lambda: self.path.write_text("快照乙"))
        self.client.normal_responses = [
            step(tool_call("search", "search_memory", query="杭州"), tool_call("update", "refresh")),
            step(tool_call("read", "read_memory", session_id="A"), tool_call("search2", "search_memory", query="SUZHOU")),
            step(tool_call("read2", "read_memory", session_id="B")),
            completion("乘公交去园林", "A", "A", "B", "C"),
            completion("继续坐公交", "A"),
        ]
        with patch.object(Phase2Store, "record_usage", autospec=True, side_effect=Phase2Store.record_usage) as record:
            self.assertEqual(self.agent.ask("按我的偏好规划"), "乘公交去园林")
            self.assertEqual(record.call_args.args[1], {"A", "B", "C"})
            record.assert_called_once()
        self.assertEqual(self.usage()[0], 1)
        self.assertIsNotNone(self.usage()[1])
        self.assertEqual(self.usage("C")[0], 1)
        for messages, tools, options in self.client.requests:
            self.assertEqual([m["role"] for m in messages].count("system"), 1)
            self.assertIn("快照甲", messages[0]["content"])
            self.assertNotIn("快照乙", messages[0]["content"])
        messages = self.client.requests[-1][0]
        results = {m["tool_call_id"]: json.loads(m["content"]) for m in messages if m["role"] == "tool"}
        self.assertEqual(results["search"]["results"], [
            {"session_id": "A", "source_seq": 10, "slug": "hangzhou", "preview": "杭州游览，偏好公交"},
        ])
        self.assertEqual(results["read"]["summary"], "杭州游览，偏好公交")
        self.assertEqual(results["search2"]["results"][0]["session_id"], "B")
        self.assertEqual(results["read2"]["source_seq"], 10)
        replay = self.manager.open(self.session.session_id)
        self.assertEqual(replay.context.nodes(), self.session.context.nodes())
        surface = replay.context.build()
        calls = [call["id"] for m in surface for call in m.get("tool_calls", [])]
        self.assertEqual(calls, [m["tool_call_id"] for m in surface if m["role"] == "tool"])
        self.assertEqual(surface[-1], {"role": "assistant", "content": "乘公交去园林"})
        self.assertEqual(surface[0]["content"], "导游")
        self.assertNotIn("快照甲", self.session.history.path.read_text())
        agent = GuideAgent(self.client, ToolExecutor(include_amap=False), replay)
        self.assertEqual(agent.ask("继续"), "继续坐公交")
        self.assertIn("快照乙", self.client.requests[-1][0][0]["content"])
        self.assertIn(surface[-1], self.client.requests[-1][0])
        self.assertEqual(self.usage()[0], 2)
        self.assertEqual(self.usage("B")[0], 1)

    def test_missing_and_blank_global_memory(self):
        self.client.normal_responses = [completion("你好"), completion("你好")]
        with patch.object(Phase2Store, "record_usage") as record:
            self.assertEqual(self.agent.ask("你好"), "你好")
            self.path.write_text(" \n")
            self.assertEqual(self.agent.ask("你好"), "你好")
            record.assert_not_called()
        self.assertEqual(len(self.client.requests), 2)
        self.assertEqual(self.client.requests[0][0][0], self.client.requests[1][0][0])

    def test_protocol_conflict_executes_siblings_and_retries(self):
        self.seed()
        self.client.normal_responses = [
            step(tool_call("bad", "finish_answer", answer="忽略", used_session_ids=["A"]),
                 tool_call("good", "read_memory", session_id="A")),
            completion("完成"),
        ]
        self.assertEqual(self.agent.ask("查询"), "完成")
        messages = self.client.requests[-1][0]
        self.assertIn("error", json.loads(messages[-2]["content"]))
        self.assertEqual(json.loads(messages[-1]["content"])["session_id"], "A")
        self.assertEqual(self.usage(), (0, None))

    def test_invalid_finish_is_paired_and_never_completed(self):
        self.seed()
        malformed = tool_call("bad", "finish_answer")
        malformed["function"]["arguments"] = "{"
        for reply in [
            response("直接回答"), step(malformed), step(tool_call("bad", "finish_answer", answer="缺字段")),
            step(tool_call("bad", "finish_answer", answer="错类型", used_session_ids="A")),
            response(None, calls=[tool_call("bad", "finish_answer", answer="截断", used_session_ids=["A"])],
                     finish="length"),
        ]:
            with self.subTest(reply=reply):
                self.client.normal_responses = [reply]
                with self.assertRaises((ValueError, TypeError)):
                    self.agent.ask("测试")
                events = self.session.history.load("导游")
                self.assertEqual(events[-1]["reason"], "error")
                reopened = self.manager.open(self.session.session_id)
                self.assertEqual(reopened.compact.ensure_balance_cache(reopened).in_progress_tool_calls, 0)
        self.assertEqual(self.usage(), (0, None))

    def test_max_steps_after_real_retrieval(self):
        self.seed()
        self.agent.max_steps = 1
        self.client.normal_responses = [step(tool_call("lookup", "read_memory", session_id="A"))]
        with self.assertRaisesRegex(RuntimeError, "最大 LLM 请求次数"):
            self.agent.ask("查询")
        self.assertEqual(self.session.history.load("导游")[-1]["reason"], "max_steps")
        self.assertEqual(self.usage(), (0, None))

    def test_permissions_and_result_bounds(self):
        self.seed("disabled", enabled=0)
        self.seed("blank", summary=" \n")
        for i in range(12):
            self.seed(str(i), summary="杭州" * 4500)
        search = self.executor.call_tool("search_memory", {"query": "hangzhou", "limit": 100})
        self.assertEqual(len(search["results"]), 10)
        for item in search["results"]:
            self.assertTrue(item["session_id"].isdigit())
            self.assertEqual(len(item["preview"]), 240)
        read = self.executor.call_tool("read_memory", {"session_id": "0"})
        self.assertTrue(read["truncated"])
        self.assertEqual(len(read["summary"]), 8000)
        for sid in (None, "", "blank", "disabled", "missing"):
            self.assertIn("error", self.executor.call_tool("read_memory", {"session_id": sid}))

    def test_full_request_budget_includes_snapshot_and_output(self):
        self.path.write_text("快照" * 200)
        self.client.normal_responses = [completion("完成")]
        self.agent.ask("预算")
        messages, tools, options = self.client.requests[0]
        total = self.session.tokens.count_request(messages, tools) + options["max_tokens"]
        second = self.manager.create("切换会话")
        second.compact.context_window = total - 1
        agent = GuideAgent(self.client, ToolExecutor(include_amap=False), second)
        with self.assertRaisesRegex(RuntimeError, "输入预算"):
            agent.ask("预算")
        self.assertEqual(len(self.client.requests), 1)
        self.assertEqual(second.history.load("导游")[-1]["reason"], "error")

    def test_usage_survives_phase1_and_ranks_phase2(self):
        self.seed("A")
        self.seed("Z")
        self.client.normal_responses = [completion("完成", "Z")]
        self.agent.ask("沿用此前决定")
        store = Stage1Store()
        store.record_turn_end("Z", 11, int(time.time()))
        claim = store.claim_stage1(scan_limit=20, max_running=1, max_age_days=1, min_idle_seconds=0,
                                   lease_seconds=600, max_attempts=1, current_session_id=self.session.session_id)
        self.assertEqual(claim.session_id, "Z")
        self.assertTrue(store.commit_stage1_result("Z", 11, "更新摘要", "new", claim.ownership_token))
        self.assertEqual(self.usage("Z")[0], 1)
        claim2 = Phase2Store().claim(limit=1, active_days=1, interval=60, max_retries=1)
        self.assertEqual([(m.session_id, m.source_seq) for m in claim2.memories], [("Z", 11)])

    def test_phase2_runs_with_no_phase1_tasks(self):
        manager = MemoryManager(self.client)
        with patch.object(manager._phase1, "run", return_value={}), patch.object(
            manager._phase2, "run", return_value="unchanged",
        ) as phase2:
            self.assertEqual(manager.run(), "unchanged")
            phase2.assert_called_once_with()
