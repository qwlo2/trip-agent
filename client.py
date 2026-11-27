"""DeepSeek Chat Completions 封装；工具的执行由 Agent 层负责。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from settings_loader import load_settings


@dataclass
class ToolCall:
    """模型请求的一次工具调用，由 Agent 根据名称和参数执行。

    Attributes:
        id: API 返回的单次工具调用标识，回传工具结果时用作 tool_call_id。
        name: 模型选择的工具名称。
        arguments: 从工具调用参数 JSON 解析得到的字典。
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """统一表示用户、模型或工具消息。

    Attributes:
        role: 消息角色，例如 system、user、assistant 或 tool。
        content: 消息正文，工具调用时可能为 None。
        tool_calls: 模型请求的工具调用列表，每条消息使用独立的列表。
        tool_call_id: 工具结果对应的调用标识，非工具消息通常为 None。
        metadata: 推理内容、结束原因、token 用量等附加信息。
    """

    role: str
    content: Any = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    """ field用来配置数据类中的成员变量,default_factory 的设计就是为了让每个实例获得独立的默认对象。"""

class AsyncTripClient:
    """前台问答和后台记忆共用的原生异步客户端。"""

    sdk_class = AsyncOpenAI

    def __init__(self, env_file: str | Path | None = None) -> None:
        """加载配置并创建 SDK 客户端，不在初始化时发送请求。

        Args:
            env_file: 配置文件路径，默认使用 client.py 同目录下的 .env。
                同名系统环境变量优先于文件配置。

        Raises:
            ValueError: 缺少必填配置或配置类型转换失败。
        """
        config = load_settings(
            path=env_file or Path(__file__).with_name(".env"),
            defaults_path=Path(__file__).with_name(".env.example"),
            fields={
                "DEEPSEEK_API_KEY": str,
                "DEEPSEEK_BASE_URL": str,
                "DEEPSEEK_MODEL": str,
                "DEEPSEEK_TIMEOUT": float,
            },
        )
        self.model = config["DEEPSEEK_MODEL"]
        self._api = self.sdk_class(
            api_key=config["DEEPSEEK_API_KEY"],
            base_url=config["DEEPSEEK_BASE_URL"],
            timeout=config["DEEPSEEK_TIMEOUT"],
            max_retries=0,
        )

    def _request(self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]] | None,
                 *, tool_choice: str | dict[str, Any] = "auto", temperature: float | None = None,
                 max_tokens: int | None = None, response_format: dict[str, Any] | None = None) -> dict[str, Any]:
        """构造 Chat Completions 参数；无工具时省略工具选项。"""
        options: dict[str, Any] = {}
        if tools:
            options["tools"] = list(tools)
            options["tool_choice"] = tool_choice
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["max_tokens"] = max_tokens
        if response_format is not None:
            options["response_format"] = response_format

        return dict(model=self.model, messages=messages, stream=False, **options)

    @staticmethod
    def parse_response(response: ChatCompletion) -> Message:
        """
        SDK返回的是对象(类)而不是dict
        将 SDK 响应的第一个候选消息解析为 Message。

        Args:
            response: OpenAI SDK 返回的非流式 ChatCompletion 对象。

        Returns:
            统一的 Message，工具参数转换为 Python 字典，其他信息放入 metadata。

        Raises:
            json.JSONDecodeError: 工具调用参数不是合法 JSON。
            ValueError: 工具调用参数的 JSON 顶层不是对象。
        """
        choice = response.choices[0]
        message = choice.message.model_dump(exclude_none=True)
        tool_calls = [AsyncTripClient.parse_tool_call(call) for call in message.get("tool_calls", [])]

        return Message(
            role=choice.message.role,
            content=choice.message.content,
            tool_calls=tool_calls,
            metadata={
                "reasoning_content": message.get("reasoning_content"),
                "finish_reason": choice.finish_reason,
                "usage": response.usage.model_dump() if response.usage is not None else None,
            },
        )

    @staticmethod
    def parse_tool_call(call: dict[str, Any]) -> ToolCall:
        """解析单个调用，允许 Agent 为同批每个调用分别记录结果。"""
        function = call["function"]
        arguments = json.loads(function["arguments"])
        if not isinstance(arguments, dict):
            raise ValueError(f"工具 {function['name']} 的参数必须为 JSON 对象")
        return ToolCall(call["id"], function["name"], arguments)

    async def ask(self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]] | None = None, *,
                  tool_choice: str | dict[str, Any] = "auto", temperature: float | None = None,
                  max_tokens: int | None = None, response_format: dict[str, Any] | None = None) -> ChatCompletion:
        options = self._request(messages, tools, tool_choice=tool_choice, temperature=temperature,
                                max_tokens=max_tokens, response_format=response_format)
        return await self._api.chat.completions.create(**options)

    async def close(self) -> None:
        await self._api.close()
