"""命令行导游问答入口。"""

from math import floor, isfinite
from pathlib import Path
from typing import Any

from agent import GuideAgent, SYSTEM_PROMPT
from client import Trip_Client
from compact import Compact
from context_builder import ContextBuilder
from history import HistoryStore
from session import Session
from settings_loader import convert_settings, load_settings
from tool_executor import ToolExecutor


def load_compact_settings(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """通过公共加载器获取容量及比例，校验业务约束并返回 Compact 的关键字参数。

    可传入配置字典进行转换；未传时读取根目录 .env，并使用 .env.example 默认值。
    """
    fields = {
        "DEEPSEEK_CONTEXT_WINDOW": ("context_window", int),
        "DEEPSEEK_MAX_OUTPUT_TOKENS": ("max_output_tokens", int),
        "COMPACT_THRESHOLD_RATIO": ("threshold_ratio", float),
        "COMPACT_RETAIN_RATIO": ("retain_ratio", float),
    }
    converters = {name: convert for name, (_, convert) in fields.items()}
    if settings is None:
        converted = load_settings(
            path=Path(__file__).with_name(".env"),
            defaults_path=Path(__file__).with_name(".env.example"),
            fields=converters,
        )
    else:
        converted = convert_settings(settings, converters)
    values = {parameter: converted[name] for name, (parameter, _) in fields.items()}
    for name in ("context_window", "max_output_tokens"):
        if values[name] <= 0:
            raise ValueError(f"{name} 必须为正整数")
    for name in ("threshold_ratio", "retain_ratio"):
        if not isfinite(values[name]) or not 0 < values[name] < 1:
            raise ValueError(f"{name} 必须在 0 和 1 之间")
    if values["retain_ratio"] >= values["threshold_ratio"]:
        raise ValueError("retain_ratio 必须小于 threshold_ratio")
    window, output = values["context_window"], values["max_output_tokens"]
    threshold = floor(min(window * values["threshold_ratio"], window - 2 * output))
    retain = floor((window - output) * values["retain_ratio"])
    if threshold <= 0:
        raise ValueError("压缩阈值 T 必须大于 0，请检查模型窗口和最大输出容量")
    if not 0 < retain < threshold:
        raise ValueError("保留预算 K 必须大于 0 且小于压缩阈值 T")
    return values


def main() -> None:
    """启动导游对话；/compact 手动压缩，/clear 清空历史，exit 或 quit 退出。"""
    executor = ToolExecutor()
    history = HistoryStore()
    context = ContextBuilder()
    compact_settings = load_compact_settings()
    with Trip_Client() as client:
        compact = Compact(client, **compact_settings)
        session = Session(history, context, SYSTEM_PROMPT, compact)
        agent = GuideAgent(client, executor, session)
        print("导游已就绪。/compact 压缩上下文，/clear 清空历史，exit 退出。")
        while True:
            question = input("你：").strip()
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break
            if question == "/clear":
                agent.clear_history()
                print("历史已清空。")
                continue
            if question == "/compact":
                result = session.compact_now()
                print(f"压缩：{result.status}（{result.reason}）")
                continue
            answer = agent.ask(question)
            print(f"导游：{answer}")


if __name__ == "__main__":
    main()
