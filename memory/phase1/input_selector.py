"""按优先级和新近程度筛选原始候选事件，构建 Memory 自定义角色的消息列表。"""

import json
from dataclasses import dataclass
from math import floor
from typing import Any

from memory.phase1.event_filter import render_candidate
from memory.phase1.event_reader import MemoryCandidate
from memory.settings_loader import load_config


@dataclass
class MemorySelection:
    """messages 与 candidates 一一对应，每条消息携带来源 seq 及自定义 role。"""

    candidates: list[MemoryCandidate]
    messages: list[dict]
    rendered_text: str
    token_count: int
    session_id: str
    source_max_seq: int


class InputSelector:
    """按配置的事件优先级筛选，工具调用及结果作为同优先级证据组处理。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """接收已加载的配置，默认通过统一加载器读取 memory/.config。"""
        self.config = config if config is not None else load_config()

    def select(
        self,
        candidates: list[MemoryCandidate],
        *,
        session_id: str,
        source_max_seq: int,
    ) -> MemorySelection:
        """按优先级升序、seq 降序尝试纳入，最后按 seq 升序输出。

        仅渲染待选候选项；工具按 call_id 配对，缺少调用或结果的证据组跳过。
        证据组整体计费，预算不足时整体舍弃。同一来源的正文和各次工具调用
        分别保留为独立记录，共享 seq，同 seq 按原始候选项顺序排列。
        输入预算按格式化后消息列表的 JSON 字节数计量，包含 seq、自定义 role
        等字段；默认每四个 UTF-8 字节估算一个 Token，向上取整。
        Token 预算 B=floor(context_window × input_budget_ratio)。
        """
        budget_tokens = floor(self.config["context_window"] * self.config["input_budget_ratio"])
        budget_bytes = self.config["max_rollout_bytes"]
        bytes_per_token = self.config["bytes_per_token"]
        used_bytes = len(b"[]")
        if (
            used_bytes > budget_bytes
            or (used_bytes + bytes_per_token - 1) // bytes_per_token > budget_tokens
        ):
            raise ValueError("输入预算不足以容纳空消息列表")

        indexed = list(enumerate(candidates))
        call_groups: dict[str, list[tuple[int, MemoryCandidate]]] = {}
        for index, candidate in indexed:
            if candidate.call_id is not None:
                call_groups.setdefault(candidate.call_id, []).append((index, candidate))

        selected: dict[int, tuple[MemoryCandidate, str]] = {}
        processed: set[int] = set()
        ranked = sorted(indexed, key=lambda item: (item[1].priority, -item[1].seq, item[0]))
        for index, candidate in ranked:
            if index in processed:
                continue
            group = call_groups[candidate.call_id] if candidate.call_id is not None else [(index, candidate)]
            processed.update(item_index for item_index, _ in group)
            if (
                candidate.kind in {"tool_call", "tool_result"}
                and {item.kind for _, item in group} != {"tool_call", "tool_result"}
            ):
                continue

            rendered = {
                item_index: (item, render_candidate(item, self.config))
                for item_index, item in group
            }
            # 用候选项下标区分同 seq 的记录，数组逗号也计入预算。
            commas = len(rendered) if selected else len(rendered) - 1
            trial_bytes = used_bytes + commas + sum(
                len(text.encode("utf-8")) for _, text in rendered.values()
            )
            if (
                trial_bytes > budget_bytes
                or (trial_bytes + bytes_per_token - 1) // bytes_per_token > budget_tokens
            ):
                continue
            used_bytes = trial_bytes
            selected.update(rendered)

        ordered = sorted(selected.items(), key=lambda pair: (pair[1][0].seq, pair[0]))
        rendered_text = "[" + ",".join(text for _, (_, text) in ordered) + "]"
        return MemorySelection(
            candidates=[item for _, (item, _) in ordered],
            messages=[json.loads(text) for _, (_, text) in ordered],
            rendered_text=rendered_text,
            token_count=(used_bytes + bytes_per_token - 1) // bytes_per_token,
            session_id=session_id,
            source_max_seq=source_max_seq,
        )
