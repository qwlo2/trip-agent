"""在 SQLite 可读局部记忆上执行逐行关键词匹配，并按 Session 聚合结果。"""

from collections.abc import Iterator
from sqlite3 import Row
from typing import Any

MATCH_MODES = ("any", "all_on_same_line", "all_within_lines")
MAX_QUERIES = 8
MAX_QUERY_CHARS = 64
MAX_WINDOW = 32
MAX_CONTEXT = 4
MAX_SESSIONS = 10
MAX_MATCHES = 4
MAX_CONTENT = 480
MAX_OUTPUT_BYTES = 24000


def prepare_text(text: str, case_sensitive: bool, normalized: bool) -> str:
    """对比较双方执行相同的 Unicode 小写转换及可选的字母数字筛选。"""
    if not case_sensitive:
        text = text.lower()
    if normalized:
        text = "".join(char for char in text if char.isalnum())
    return text


def _windows(hits: list[set[str]], mode: str, count: int, width: int) -> Iterator[tuple[int, int, set[str]]]:
    """按行返回有效窗口。最短结束位置单调不减，同一终点只留最晚起点。"""
    pending = None
    for start, found in enumerate(hits):
        if not found:
            continue
        if mode != "all_within_lines":
            if mode == "any" or len(found) == count:
                yield start, start, found
            continue
        combined = set()
        for end in range(start, min(start + width, len(hits))):
            combined.update(hits[end])
            if len(combined) == count:
                if pending and pending[1] != end:
                    yield pending
                pending = start, end, combined
                break
    if pending:
        yield pending


def _excerpt(lines: list[str], start: int, end: int, context: int, first_hit: int) -> tuple[int, str, bool]:
    """优先保留首个命中附近的原文，长上下文不会把命中挤出片段。"""
    left = max(0, start - context)
    right = min(len(lines), end + context + 1)
    if sum(len(lines[i]) for i in range(left, right)) + right - left - 1 <= MAX_CONTENT:
        return left + 1, "\n".join(lines[left:right]), False
    # 以首个命中为锚点，在长度允许时完整保留前文。
    offset = first_hit + sum(len(lines[i]) + 1 for i in range(left, start))
    skip = max(0, offset - MAX_CONTENT // 2)
    clipped = skip > 0
    while skip > len(lines[left]):
        skip -= len(lines[left]) + 1
        left += 1
    content = "\n".join(lines[left:right])[skip:]
    return left + 1, content[:MAX_CONTENT], clipped or len(content) > MAX_CONTENT


def session_match(row: Row, terms: dict[str, str], mode: str, width: int, context: int,
                   case_sensitive: bool, normalized: bool) -> dict[str, Any] | None:
    """独立匹配正文和 slug，统计全部命中词，限制持有的片段数量。"""
    if row["rollout_summary"].isspace():
        return None
    lines = row["rollout_summary"].splitlines()
    seen: set[str] = set()
    matches = []
    slug = row["rollout_slug"] or ""
    truncated = len(slug) > 120
    for source, originals in (("summary", lines), ("slug", [slug])):
        prepared = [prepare_text(line, case_sensitive, normalized) for line in originals]
        hits = [{key for key in terms if key in line} for line in prepared]
        for start, end, found in _windows(hits, mode, len(terms), width):
            seen.update(found)
            if len(matches) == MAX_MATCHES:
                truncated = True
                if source == "summary":
                    continue
                matches.pop()  # 为独立命中的 slug 保留一个片段。
            offset = 0
            if len(originals[start]) > MAX_CONTENT // 2:
                # 转换后的位置可能与原文不同，如连字符过滤或小写展开。
                target = min(prepared[start].index(key) for key in hits[start])
                position = 0
                for offset, char in enumerate(originals[start]):
                    position += len(prepare_text(char, case_sensitive, normalized))
                    if position > target:
                        break
            line, content, clipped = _excerpt(originals, start, end, context, offset)
            match = {"source": source, "content": content,
                     "matched_queries": [original for key, original in terms.items() if key in found]}
            if source == "summary":
                match.update(match_start_line=start + 1, match_end_line=end + 1, content_start_line=line)
            matches.append(match)
            truncated |= clipped
    if seen:
        return {"session_id": row["session_id"], "source_seq": row["source_max_seq"], "slug": slug[:120],
                "matched_queries": [original for key, original in terms.items() if key in seen],
                "matches": matches, "truncated": truncated}
    return None
