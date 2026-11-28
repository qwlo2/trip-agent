"""命令行导游问答入口。"""

import asyncio
from functools import partial
from math import floor, isfinite
from pathlib import Path
from typing import Any

from agent import SYSTEM_PROMPT
from runtime.manager import RuntimeManager
from runtime.blocking import blocking, wait_finished
from session_repository import SESSION_PAGE_SIZE, SessionRepository
from settings_loader import convert_settings, load_settings


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


async def create_session(create) -> str | None:
    name = (await asyncio.to_thread(input, "会话名称（回车使用默认名称）：")).strip()
    if name.lower() in {"exit", "quit", "/exit"}:
        return None
    return await create(name or "新会话")


async def select_session(list_sessions, select, create) -> str | None:
    """分页选择会话；启动目录和运行中 Runtime 共用输入流程。"""
    offset = 0
    while True:
        sessions = await list_sessions(offset=offset)
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
        choice = (await asyncio.to_thread(input, f"请选择会话（{'，'.join(prompt)}）：")).strip().lower()
        if choice in choices:
            return await select(choices[choice])
        if choice == "0":
            return await create_session(create)
        if choice in {"exit", "quit", "/exit"}:
            return None
        if choice == "p" and offset:
            offset -= SESSION_PAGE_SIZE
        elif choice == "n" and len(sessions) == SESSION_PAGE_SIZE:
            offset += SESSION_PAGE_SIZE
        else:
            print("请输入列表中的会话编号或命令。")


async def main() -> None:
    """CLI 只持有当前 Session ID，提交任务后可继续选择其他会话。"""
    settings = load_compact_settings()
    directory = await blocking(SessionRepository, SYSTEM_PROMPT, settings)
    runtime = RuntimeManager(directory)
    try:
        session_id = await select_session(partial(blocking, directory.list_sessions),
                                          partial(blocking, directory.record_open), partial(blocking, directory.create))
        if session_id is None:
            return
        await runtime.start()
        print("导游已就绪。/sessions 或 /switch 切换，/clear 新建，/compact 压缩，"
              "/tasks 列表，/status、/wait、/result、/cancel 后接任务 ID，/exit 退出。")
        while True:
            question = (await asyncio.to_thread(input, "你：")).strip()
            if not question:
                continue
            command, _, argument = question.partition(" ")
            try:
                if question.lower() in {"/exit", "exit", "quit"}:
                    break
                if command in {"/sessions", "/switch"}:
                    if command == "/switch" and argument.strip():
                        selected = await runtime.select_session(argument.strip())
                    else:
                        selected = await select_session(runtime.list_sessions, runtime.select_session, runtime.create_session)
                    if selected is None:
                        break
                    session_id = selected
                elif command == "/clear":
                    session_id = await create_session(runtime.create_session)
                    if session_id is None:
                        break
                elif command == "/compact":
                    print(runtime.submit_compact(session_id))
                elif command == "/cancel":
                    accepted = runtime.cancel(argument.strip())
                    print("已接受取消请求" if accepted else "任务已结束或正在取消")
                elif command == "/tasks":
                    for item in runtime.list_tasks():
                        print(item["task_id"], item["session_id"], item["kind"], item["status"])
                elif command in {"/status", "/result", "/wait"}:
                    key = argument.strip()
                    item = await runtime.wait(key) if command == "/wait" else runtime.get_task(key)
                    if command == "/status":
                        print(item["status"])
                    else:
                        print(item["error"] or item["answer"] or item["result"] or item["status"])
                else:
                    print(runtime.submit(session_id, question))
            except (ValueError, KeyError, RuntimeError) as exc:
                print(str(exc))
    except EOFError:
        pass
    finally:
        await wait_finished(asyncio.create_task(runtime.shutdown()))


if __name__ == "__main__":
    asyncio.run(main())
