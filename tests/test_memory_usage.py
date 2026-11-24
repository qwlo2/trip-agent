"""通过离线模型、真实日志和临时 SQLite 验证记忆使用闭环。"""

import json
import os
import sqlite3
import time
import unittest
from contextlib import closing
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agent import GuideAgent, parse_final_answer
from data_paths import get_data_paths
from database.phase2_store import Phase2Store
from database.sqlite import get_connection
from database.stage1_store import Stage1Store
from memory.reader import search_memory, truncate_middle_with_token_budget, turn_memory_prompt
from memory.search import MAX_CONTENT, MAX_MATCHES, MAX_OUTPUT_BYTES
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
        self.path.write_text("快照甲" + "中" * 5000 + "session-C.md")
        self.executor.register_tool("refresh", "模拟 Phase 2 更新", {"type": "object"},
                                    lambda: self.path.write_text("快照乙"))
        self.client.normal_responses = [
            step(tool_call("search", "search_memory", queries=["杭州", "公交"]), tool_call("update", "refresh")),
            step(tool_call("read", "read_memory", session_id="A"), tool_call("search2", "search_memory", queries=["SUZHOU"])),
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
            self.assertEqual(options["response_format"], {"type": "json_object"})
            self.assertNotIn("finish_answer", [tool["function"]["name"] for tool in tools])
            self.assertEqual([m["role"] for m in messages].count("system"), 1)
            schema = next(tool["function"]["parameters"] for tool in tools if tool["function"]["name"] == "search_memory")
            self.assertEqual(schema["required"], ["queries"])
            self.assertEqual(schema["properties"]["queries"]["type"], "array")
            self.assertEqual(messages[0], self.client.requests[0][0][0])
            self.assertIn("tokens truncated…", messages[0]["content"])
            self.assertIn("快照甲", messages[0]["content"])
            self.assertNotIn("快照乙", messages[0]["content"])
        messages = self.client.requests[-1][0]
        results = {m["tool_call_id"]: json.loads(m["content"]) for m in messages if m["role"] == "tool"}
        item, = results["search"]["results"]
        self.assertEqual((item["session_id"], item["source_seq"], item["slug"]), ("A", 10, "hangzhou"))
        self.assertEqual(item["matched_queries"], ["杭州", "公交"])
        self.assertEqual(item["matches"][0]["content"], "杭州游览，偏好公交")
        self.assertEqual(results["read"]["summary"], "杭州游览，偏好公交")
        self.assertEqual(results["search2"]["results"][0]["session_id"], "B")
        self.assertEqual(results["read2"]["source_seq"], 10)
        replay = self.manager.open(self.session.session_id)
        self.assertEqual(replay.context.nodes(), self.session.context.nodes())
        surface = replay.context.build()
        calls = [call["id"] for m in surface for call in m.get("tool_calls", [])]
        self.assertEqual(calls, [m["tool_call_id"] for m in surface if m["role"] == "tool"])
        self.assertEqual(surface[-1], completion("乘公交去园林", "A", "A", "B", "C").choices[0].message.model_dump(
            exclude_none=True,
        ))
        self.assertEqual(sum(m["role"] == "assistant" for m in surface), len(self.client.requests))
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

    def test_legacy_session_replays_and_continues_without_terminal_tool(self):
        legacy = [
            {"role": "user", "content": "旧问题"},
            {"role": "assistant", "tool_calls": [tool_call("lookup", "search_memory", query="杭州")]},
            {"role": "tool", "tool_call_id": "lookup", "content": '{"results":[]}'},
            {"role": "assistant", "reasoning_content": "已完成", "tool_calls": [
                tool_call("old", "finish_answer", answer="旧答案", used_session_ids=["A"]),
            ]},
            {"role": "tool", "tool_call_id": "old", "content": '{"accepted":true}'},
            {"role": "assistant", "content": "旧答案"},
        ]
        self.session.append_history([{"type": "message", "turn": 1, "message": m} for m in legacy])
        self.session.append_history([{"type": "turn/end", "turn": 1, "reason": "completed"}])
        original = self.session.history.path.read_bytes()
        reopened = self.manager.open(self.session.session_id)
        agent = GuideAgent(self.client, ToolExecutor(include_amap=False), reopened)
        reply = completion("新答案")
        reply.choices[0].message.reasoning_content = "当前回答依据"
        self.client.normal_responses = [reply]
        with patch.object(Phase2Store, "record_usage") as usage:
            self.assertEqual(agent.ask("新问题"), "新答案")
            usage.assert_not_called()
        messages, tools, _ = self.client.requests[0]
        self.assertEqual(messages[1:-1], legacy)
        self.assertNotIn("finish_answer", [t["function"]["name"] for t in tools])
        self.assertTrue(reopened.history.path.read_bytes().startswith(original))
        projected = reopened.context.build()
        self.assertEqual(projected[len(legacy) + 2:], [reply.choices[0].message.model_dump(exclude_none=True)])
        records = reopened.history.load("导游")
        self.assertEqual(records[-3]["message"], projected[-1])
        self.assertEqual(records[-3]["usage"], reply.usage.model_dump())

    def test_final_json_requires_both_typed_fields(self):
        for content in (None, "", "   ", "{", "null", "[]", "{}", '{"answer":"好"}',
                        '{"used_session_ids":[]}', '{"answer":" ","used_session_ids":[]}',
                        '{"answer":1,"used_session_ids":[]}', '{"answer":"好","used_session_ids":"A"}',
                        '{"answer":"好","used_session_ids":[1]}'):
            with self.subTest(content=content), self.assertRaises(ValueError):
                parse_final_answer(content)

    def test_format_correction_within_loop(self):
        self.client.normal_responses = [response("   "), completion("已修正")]
        self.assertEqual(self.agent.ask("测试"), "已修正")
        self.assertIn("格式无效", self.client.requests[1][0][-1]["content"])
        self.assertEqual(self.session.context.build()[-2]["content"], "   ")
        self.assertEqual(self.session.history.load("导游")[-1]["reason"], "completed")

    def test_failed_correction_and_step_limit(self):
        for limit, reason, error in ((30, "error", ValueError), (1, "max_steps", RuntimeError)):
            with self.subTest(limit=limit), patch.object(Phase2Store, "record_usage") as usage:
                self.agent.max_steps = limit
                self.client.normal_responses = [response("bad"), response("[]")]
                with self.assertRaises(error):
                    self.agent.ask("测试")
                usage.assert_not_called()
                self.assertEqual(self.session.history.load("导游")[-1]["reason"], reason)

    def test_incomplete_responses_and_failed_batches(self):
        calls = [tool_call("a", "read_memory", session_id="A"), tool_call("b", "unknown")]
        for reason, batch in [("length", []), ("content_filter", []), ("tool_calls", []),
                              ("length", calls), ("stop", calls), ("tool_calls", list(reversed(calls)))]:
            with self.subTest(reason=reason, batch=batch), patch.object(Phase2Store, "record_usage") as usage:
                self.client.normal_responses = [response('{"answer":"忽略","used_session_ids":["A"]}',
                                                        calls=batch, finish=reason)]
                with self.assertRaises((RuntimeError, KeyError)):
                    self.agent.ask("测试")
                events = self.session.history.load("导游")
                self.assertEqual(events[-1]["reason"], "error")
                results = [e["message"] for e in events if e["turn"] == self.session.turn
                           and e.get("message", {}).get("role") == "tool"]
                self.assertEqual([m["tool_call_id"] for m in results], [c["id"] for c in batch])
                self.assertTrue(all("error" in json.loads(m["content"]) for m in results))
                usage.assert_not_called()

    def test_usage_failure_after_commit_is_only_a_warning(self):
        self.client.normal_responses = [completion("已保存", "A")]
        with patch.object(Phase2Store, "record_usage", side_effect=sqlite3.OperationalError("locked")), \
                self.assertLogs("agent", level="WARNING"):
            self.assertEqual(self.agent.ask("测试"), "已保存")
        self.assertEqual(self.session.history.load("导游")[-1]["reason"], "completed")

    def test_failed_turn_end_propagates_before_usage(self):
        append = self.session.append_history

        def fail_end(events):
            if events[0]["type"] == "turn/end":
                raise OSError("写入失败")
            return append(events)

        self.client.normal_responses = [completion("完成", "A")]
        with patch.object(self.session, "append_history", side_effect=fail_end), \
                patch.object(Phase2Store, "record_usage") as usage:
            with self.assertRaises(OSError):
                self.agent.ask("测试")
            usage.assert_not_called()

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
        search = self.executor.call_tool("search_memory", {"queries": ["杭州"], "limit": 100})
        self.assertTrue(search["truncated"])
        self.assertEqual(len(search["results"]), 10)
        for item in search["results"]:
            self.assertTrue(item["session_id"].isdigit())
            self.assertEqual(len(item["matches"][0]["content"]), MAX_CONTENT)
            self.assertTrue(item["truncated"])
        read = self.executor.call_tool("read_memory", {"session_id": "0"})
        self.assertTrue(read["truncated"])
        self.assertEqual(len(read["summary"]), 8000)
        for sid in (None, "", "blank", "disabled", "missing"):
            self.assertIn("error", self.executor.call_tool("read_memory", {"session_id": sid}))

    def test_full_request_budget_includes_snapshot_and_output(self):
        self.path.write_text("快照" * 3000)
        self.client.normal_responses = [completion("完成")]
        self.agent.ask("预算")
        messages, tools, options = self.client.requests[0]
        self.assertIn("tokens truncated…", messages[0]["content"])
        total = self.session.tokens.count_request(messages, tools, response_format=options["response_format"]) + options["max_tokens"]
        second = self.manager.create("切换会话")
        second.compact.context_window = total - 1
        agent = GuideAgent(self.client, ToolExecutor(include_amap=False), second)
        with self.assertRaisesRegex(RuntimeError, "输入预算"):
            agent.ask("预算")
        self.assertEqual(len(self.client.requests), 1)
        self.assertEqual(second.history.load("导游")[-1]["reason"], "error")

    def test_summary_byte_boundaries(self):
        instructions = turn_memory_prompt()
        for text in ('  引号"\n', '😀' * 2500, 'a' * 10000):
            self.path.write_text(text)
            self.assertEqual(json.loads(turn_memory_prompt().split("全局记忆（参考资料）\n")[1]), text)
        head = "a" * 4999
        tail = "b" * 4998
        original = head + "中" + "删除" + "😀" + tail
        self.path.write_text(original)
        prompt = turn_memory_prompt()
        summary = json.loads(prompt.split("全局记忆（参考资料）\n")[1])
        self.assertEqual(summary, head + "…3 tokens truncated…" + tail)
        self.assertLessEqual(len((head + tail).encode()), 10000)
        self.assertEqual(self.path.read_text(), original)
        self.assertTrue(prompt.startswith(instructions))
        self.assertEqual(truncate_middle_with_token_budget("中文", 0), "…2 tokens truncated…")

    def test_window_modes_and_original_context(self):
        self.seed("A", "前言\n杭州\n杭州\n中间\n住宿\n尾行", "预算")
        self.seed("B", "杭州和住宿", "other")
        self.seed("C", "住宿", "杭州")
        keys = ["杭州", "住宿"]
        self.assertEqual([r["session_id"] for r in search_memory(keys, "all_on_same_line")["results"]], ["B"])
        self.assertEqual([r["session_id"] for r in search_memory(keys, "all_within_lines", 2)["results"]], ["B"])
        results = search_memory(keys, "all_within_lines", 3)["results"]
        self.assertEqual({r["session_id"] for r in results}, {"A", "B"})
        match, = next(r["matches"] for r in results if r["session_id"] == "A")
        self.assertEqual((match["match_start_line"], match["match_end_line"], match["content_start_line"]), (3, 5, 2))
        self.assertEqual(match["content"], "杭州\n杭州\n中间\n住宿\n尾行")
        self.assertFalse(search_memory(["预算", "住宿"], "all_within_lines")["results"])

    def test_normalization_and_match_ranking(self):
        self.seed("A", "公交", "Hangzhou-Trip")
        self.seed("B", "公交", "other")
        self.seed("C", "公交", "other")
        with closing(get_connection()) as conn, conn:
            conn.execute("UPDATE stage1_outputs SET generated_at = CASE session_id WHEN 'A' THEN 1 ELSE 2 END")
        results = search_memory([" hangzhou trip ", "HANGZHOU-TRIP", "公交"], normalized=True)["results"]
        self.assertEqual([r["session_id"] for r in results], ["A", "B", "C"])
        self.assertEqual(results[0]["matched_queries"], ["hangzhou trip", "公交"])
        self.assertEqual(results[0]["matches"][-1], {"source": "slug", "content": "Hangzhou-Trip",
                                                      "matched_queries": ["hangzhou trip"]})
        with closing(get_connection()) as conn, conn:
            conn.execute("UPDATE stage1_outputs SET generated_at = 3 WHERE session_id='C'")
        self.assertEqual([r["session_id"] for r in search_memory(["公交"])["results"]], ["C", "B", "A"])
        self.assertFalse(search_memory(["hangzhou-trip"], case_sensitive=True)["results"])
        self.assertFalse(search_memory(["hangzhou trip"])["results"])

    def test_search_validates_inputs_and_unreadable_rows(self):
        for args in ({"queries": []}, {"queries": "x"}, {"queries": [1]}, {"queries": [" "]},
                     {"queries": ["-!"], "normalized": True}, {"queries": ["x"] * 9}, {"queries": ["x" * 65]},
                     {"match_mode": []}, {"line_count": False}, {"line_count": 33}, {"context_lines": -1},
                     {"context_lines": 5}, {"limit": 0}, {"normalized": 1}, {"case_sensitive": "yes"}):
            with self.subTest(args=args):
                self.assertIn("error", search_memory(**({"queries": ["x"]} | args)))
        self.seed("no-seq")
        self.seed("orphan")
        with closing(get_connection()) as conn, conn:
            conn.execute("UPDATE stage1_outputs SET source_max_seq=NULL WHERE session_id='no-seq'")
            conn.execute("DELETE FROM sessions WHERE session_id='orphan'")
        self.assertFalse(search_memory(["杭州"])["results"])

    def test_search_error_can_be_corrected_in_the_same_turn(self):
        self.seed()
        self.assertEqual(self.agent.max_steps, 30)
        self.client.normal_responses = [
            step(tool_call("bad", "search_memory", query="杭州")),
            step(tool_call("good", "search_memory", queries=["杭州"])),
            completion("乘公交", "A"),
        ]
        self.assertEqual(self.agent.ask("之前的偏好？"), "乘公交")
        self.assertIn("error", json.loads(self.client.requests[1][0][-1]["content"]))
        result = json.loads(self.client.requests[2][0][-1]["content"])
        self.assertEqual(result["results"][0]["session_id"], "A")
        self.assertEqual(self.usage()[0], 1)

    def test_snippets_and_total_response_budget(self):
        for i in range(10):
            text = "前" * 1000 + "HANG-ZHOU" + "后" * 1000
            self.seed(str(i), "\n".join([text] * 8), slug="long-lines")
        report = search_memory(["hang zhou"], normalized=True, limit=10)
        self.assertTrue(report["truncated"])
        self.assertLess(len(report["results"]), 10)
        self.assertLessEqual(len(json.dumps(report, ensure_ascii=False).encode()), MAX_OUTPUT_BYTES)
        item = report["results"][0]
        self.assertTrue(item["truncated"])
        self.assertEqual(len(item["matches"]), MAX_MATCHES)
        for index, match in enumerate(item["matches"], 1):
            self.assertEqual(match["match_start_line"], index)
            self.assertIn("HANG-ZHOU", match["content"])
            self.assertLessEqual(len(match["content"]), MAX_CONTENT)

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

    def test_memory_pipeline_keeps_its_own_response_contracts(self):
        self.client.normal_responses = [completion("乘坐公交"), response('{"action":"replace","summary":"偏好公交"}')]
        self.client.summary_response = response('{"rollout_summary":"偏好公交","rollout_slug":"travel"}')
        self.agent.ask("我偏好公交。")
        with closing(get_connection()) as conn, conn:
            conn.execute("UPDATE sessions SET updated_at = ?", (int(time.time()) - 86400,))
        self.assertEqual(MemoryManager(self.client).run(), "updated")
        self.assertEqual(self.path.read_text().strip(), "偏好公交")
        _, phase1, phase2 = self.client.requests
        self.assertIsNone(phase1[1])
        self.assertEqual({tool["function"]["name"] for tool in phase2[1]},
                         {"read_memory_workspace", "read_rollout_summary"})
        for messages, _, options in (phase1, phase2):
            self.assertEqual(options["response_format"], {"type": "json_object"})
            self.assertNotIn("used_session_ids", messages[0]["content"])

    def test_phase2_runs_with_no_phase1_tasks(self):
        manager = MemoryManager(self.client)
        with patch.object(manager._phase1, "run", return_value={}), patch.object(
            manager._phase2, "run", return_value="unchanged",
        ) as phase2:
            self.assertEqual(manager.run(), "unchanged")
            phase2.assert_called_once_with()
