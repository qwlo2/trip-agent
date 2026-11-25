"""组合历史存储和模型上下文，提供会话级操作。"""

import time
from typing import Any, Callable, Sequence

from compact import Compact, CompactResult
from context_builder import ContextBuilder, SurfaceNode
from database.stage1_store import Stage1Store
from history import HistoryStore, TurnEndReason, track_message, validate_history
from tokens import Tokens


class SessionIntegrityError(RuntimeError):
    """持久化或投影结果不确定，当前会话禁止继续写入。"""


class Session:
    """持有 History、ContextBuilder、Tokens 和 Compact，管理会话的完整生命周期。"""

    def __init__(
        self,
        history: HistoryStore,
        context: ContextBuilder,
        system_prompt: str,
        compact: Compact,
        *,
        session_id: str | None = None,
    ) -> None:
        """加载历史消息到上下文，并恢复最近的 turn 编号。

        Args:
            history: 提供事件持久化及 seq 编号的历史存储。
            context: 用于存储消息的空上下文对象。
            system_prompt: 新建或清空会话时使用的系统身份提示词。
            compact: 当前会话拥有的压缩对象，调用时由 Session 提供上下文和历史。
            session_id: SessionManager 分配的持久化标识；独立使用时可不传入。
        """
        self.session_id = session_id
        self.history = history
        self.context = context
        self.system_prompt = system_prompt
        self.compact = compact
        self.tokens = Tokens(compression_limit=compact.threshold)
        self.tools_provider: Callable[[], Sequence[dict[str, Any]]] | None = None
        self.failure_reason: str | None = None
        events = history.load(system_prompt, existing=session_id is not None)
        validate_history(events)
        self._replay(events)
        pending = set()
        for node in self.context.nodes():
            track_message(node.message, pending)
        if pending:
            raise ValueError("有效 Surface 中存在未配对的调用")

    def require_usable(self) -> None:
        """拒绝复用曾经部分提交的投影。"""
        if self.failure_reason is not None:
            raise SessionIntegrityError(self.failure_reason)

    def _failure(self, exc: Exception) -> SessionIntegrityError:
        self.failure_reason = f"会话持久化或投影失败，重启前禁止写入：{exc}"
        return SessionIntegrityError(self.failure_reason)

    def _replay(self, events: list[dict[str, Any]]) -> None:
        """按日志顺序重建上下文，并恢复最后一次正常结束的事件序号。"""
        self.turn = max(event["turn"] for event in events)
        self.last_completed_seq: int | None = None
        for event in events:
            if event["type"] == "message":
                self.append_context(event["message"], usage=event.get("usage"), seq=event["seq"])
            elif event["type"] == "compaction/summary":
                self.apply_summary(event)
            elif event["type"] == "turn/end" and event.get("reason") == TurnEndReason.COMPLETED.value:
                self.last_completed_seq = event["seq"]

    def append_history(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """先持久化事件，再同步会话活动时间及最新正常结束的序号。"""
        self.require_usable()
        try:
            records = self.history.append(events)
            completed_seq = None
            for event in records:
                self.turn = max(self.turn, event["turn"])
                if event["type"] == "turn/end" and event.get("reason") == TurnEndReason.COMPLETED.value:
                    completed_seq = event["seq"]
            if self.session_id is not None and records:
                Stage1Store().record_session_activity(self.session_id, int(time.time()), completed_seq)
            if completed_seq is not None:
                self.last_completed_seq = completed_seq
        except Exception as exc:
            raise self._failure(exc) from exc
        return records

    def commit_event(self, event: dict[str, Any]) -> None:
        """一个同步边界内提交日志、目录水位和模型可见投影。"""
        try:
            record = self.append_history([event])[0]
            if event["type"] == "message":
                self.append_context(record["message"], usage=record.get("usage"), seq=record["seq"])
        except SessionIntegrityError:
            raise
        except Exception as exc:
            raise self._failure(exc) from exc

    def close_failed_turn(self, turn: int, reason: TurnEndReason) -> None:
        """从本轮已提交的事件确定清理起点，包含取消时已落盘的调用或结果。"""
        self.require_usable()
        if self.turn != turn:
            return
        step = None
        pending = {}
        for event in self.history.turn_events(turn):
            kind = event["type"]
            if kind == "turn/end":
                return
            if kind == "step/start":
                step = event["step"]
            elif kind == "step/end":
                step = None
            elif kind == "message":
                message = event["message"]
                for call in message.get("tool_calls") or []:
                    pending[call["id"]] = None
                if message["role"] == "tool":
                    pending.pop(message["tool_call_id"])
        for key in pending:
            self.commit_event({"type": "message", "turn": turn, "step": step, "message": {
                "role": "tool", "tool_call_id": key, "content": '{"error":"本轮已中止"}',
            }})
        if step is not None:
            self.commit_event({"type": "step/end", "turn": turn, "step": step})
        self.commit_event({"type": "turn/end", "turn": turn, "reason": reason.value})

    def append_context(
        self,
        message: dict[str, Any],
        usage: dict[str, Any] | None = None,
        *,
        seq: int,
    ) -> None:
        """追加上下文消息并更新 token 缓存，usage 用于服务端总量校准。"""
        self.require_usable()
        try:
            self.context.append(message, seq)
            self.tokens.update(message, usage=usage)
        except Exception as exc:
            raise self._failure(exc) from exc

    def get_tools(self) -> Sequence[dict[str, Any]]:
        """统一获取当前工具 Schema，尚未绑定提供函数时返回空列表。"""
        return self.tools_provider() if self.tools_provider is not None else []

    def count_context_tokens(self, messages: Sequence[dict[str, Any]] | None = None) -> int:
        """计算消息及当前工具 Schema 的成本，可用于评估候选上下文，不更新缓存。"""
        if messages is None:
            messages = self.context.build()
        return self.tokens.count_request(messages, self.get_tools())

    def measure_context_tokens(self, messages: Sequence[dict[str, Any]] | None = None) -> int:
        """统一计量实际上下文并更新总 token 缓存。"""
        self.tokens.total_tokens = self.count_context_tokens(messages)
        return self.tokens.total_tokens

    async def compact_if_needed(self) -> CompactResult:
        """调用所属 Compact，自动判断当前完整上下文是否需要压缩。"""
        return await self.compact.compact_if_needed(self)

    async def compact_now(self, *, source: str = "manual") -> CompactResult:
        """手动压缩及 LLM 工具入口，向所属 Compact 临时传入当前会话。"""
        return await self.compact.compact_now(self, source=source)

    def apply_summary(self, event: dict[str, Any], *, persist: bool = False) -> dict[str, Any]:
        """验证并应用一条完整摘要 Replace；persist 时先追加单条提交事件再替换。

        相同入口用于日志重放和运行时提交，原始消息及摘要来源均保留在 History。
        日志提交后即使进程在内存替换前退出，重启也能重放同一次 Replace。
        """
        if event["type"] != "compaction/summary":
            raise ValueError("Replace 必须由 compaction/summary 事件提交")
        data, operation = event["data"], event["surfaceOp"]
        if operation["op"] != "replace" or data["shadowedRange"] != {
            "startSeq": operation["startSeq"], "endSeq": operation["endSeq"],
        }:
            raise ValueError("摘要与 Replace 边界不一致")
        if not isinstance(data["summary"], str) or not data["summary"].strip():
            raise ValueError("摘要正文为空或不是文本")
        if event["message"] != {"role": "user", "content": data["summary"]}:
            raise ValueError("模型可见摘要与持久化正文不一致")
        start, end = self.context.locate_range(
            operation["startSeq"], operation["endSeq"], data["shadowedSeqs"],
        )
        pending = set()
        for seq in data["shadowedSeqs"]:
            source = self.history.get(seq)
            if source["type"] not in {"message", "compaction/summary"}:
                raise ValueError("Replace 来源不是模型可见消息")
            track_message(source["message"], pending)
        if pending:
            raise ValueError("Replace 拆分了工具调用批次")
        if persist:
            event = self.append_history([event])[0]
        try:
            self.context.replace(start, end, SurfaceNode(event["seq"], event["message"]))
            self.measure_context_tokens()
        except Exception as exc:
            raise self._failure(exc) from exc
        return event

    def clear(self) -> None:
        """清空独立会话；已登记的会话应通过 SessionManager 创建新会话。"""
        if self.session_id is not None:
            raise ValueError("已登记的会话不能重置序号，请创建新会话")
        self.history.clear()
        self.context.clear()
        self.tokens.reset()
        self.compact.balance_cache = None
        self.compact.last_result = None
        self.failure_reason = None
        events = self.history.load(self.system_prompt)
        self._replay(events)
