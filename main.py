"""命令行导游问答入口。"""

from math import floor, isfinite
from pathlib import Path
from typing import Any

from agent import GuideAgent, SYSTEM_PROMPT
from client import Trip_Client
from session import Session
from session_manager import SESSION_PAGE_SIZE, SessionManager
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


def create_session(manager: SessionManager) -> Session:
    """询问会话名称并创建新的会话。"""
    name = input("会话名称（回车使用默认名称）：").strip()
    return manager.create(name or "新会话")


def select_session(manager: SessionManager) -> Session | None:
    """每页展示有限数量的会话，返回用户选择的会话，退出时返回 None。"""
    offset = 0
    while True:
        sessions = manager.get_sessions(offset=offset)
        if not sessions and offset:
            print("没有更多会话。")
            offset -= SESSION_PAGE_SIZE
            continue

        choices = {}
        print(f"会话列表 · 第 {offset // SESSION_PAGE_SIZE + 1} 页")
        for number, item in enumerate(sessions, start=1):
            choices[str(number)] = item["session_id"]
            print(f"  {number}. {item['name']}（打开 {item['usage']} 次）")
        prompt = ["0 新建", "exit 退出"]
        if offset:
            prompt.append("p 上一页")
        if len(sessions) == SESSION_PAGE_SIZE:
            prompt.append("n 下一页")
        choice = input(f"请选择会话（{'，'.join(prompt)}）：").strip().lower()
        if choice in choices:
            return manager.open(choices[choice])
        if choice == "0":
            return create_session(manager)
        if choice in {"exit", "quit"}:
            return None
        if choice == "p" and offset:
            offset -= SESSION_PAGE_SIZE
        elif choice == "n" and len(sessions) == SESSION_PAGE_SIZE:
            offset += SESSION_PAGE_SIZE
        else:
            print("请输入列表中的会话编号或命令。")


def main() -> None:
    """选择会话后启动导游对话，支持切换会话、压缩和新建。"""
    compact_settings = load_compact_settings()
    with Trip_Client() as client:
        manager = SessionManager(client, SYSTEM_PROMPT, compact_settings)
        session = select_session(manager)
        if session is None:
            return
        agent = GuideAgent(client, ToolExecutor(), session)
        print("导游已就绪。/sessions 切换会话，/clear 新建会话，/compact 压缩上下文，exit 退出。")
        while True:
            question = input("你：").strip()
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break
            if question in {"/sessions", "/clear"}:
                session = select_session(manager) if question == "/sessions" else create_session(manager)
                if session is None:
                    break
                agent = GuideAgent(client, ToolExecutor(), session)
                continue
            if question == "/compact":
                result = session.compact_now()
                print(f"压缩：{result.status}（{result.reason}）")
                continue
            answer = agent.ask(question)
            print(f"导游：{answer}")


if __name__ == "__main__":
    main()
