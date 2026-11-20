"""注册和执行本地工具及高德 MCP 查询工具。"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from MCP.amap_mcp import AmapMCP
from settings_loader import load_settings


class ToolExecutor:
    """统一管理工具描述与 Python 实现，按工具名称执行调用。

    配置高德 Key 后注册地点搜索、周边搜索和路线查询，也可注册其他来源的工具。
    工具执行结果直接返回，由 Agent 负责序列化并关联 tool_call_id。
    """

    def __init__(self, env_file: str | Path | None = None, *, include_amap: bool = True) -> None:
        """加载高德配置，首次获取工具列表时发现 MCP 工具。

        Args:
            env_file: 配置文件路径，默认使用当前模块同目录下的 .env。
                同名系统环境变量优先于文件配置。
            include_amap: 为 False 时只建立空工具注册表。

        未填写高德 Key 时不注册高德工具，仍可注册和执行其他工具。
        """
        self._tools: dict[str, tuple[dict[str, Any], Callable[..., Any]]] = {}
        self._amap: AmapMCP | None = None
        if not include_amap:
            return
        config = load_settings(
            path=env_file or Path(__file__).with_name(".env"),
            defaults_path=Path(__file__).with_name(".env.example"),
            fields={"AMAP_API_KEY": str, "AMAP_TIMEOUT": float},
            allow_empty=("AMAP_API_KEY",),
        )
        if config["AMAP_API_KEY"]:
            self._amap = AmapMCP(config["AMAP_API_KEY"], config["AMAP_TIMEOUT"])

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        """注册工具描述和实现，供模型选择及执行器调用。

        Args:
            name: 唯一工具名称，模型调用时通过此名称查找实现。
            description: 面向模型的工具用途说明。
            parameters: 参数的 JSON Schema；字段名应与 handler 的参数一致。
            handler: 接收关键字参数的同步函数，结果应可序列化为 JSON。

        Raises:
            ValueError: 工具名称已注册。
        """
        if name in self._tools:
            raise ValueError(f"工具已注册：{name}")
        definition = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": deepcopy(parameters),
            },
        }
        self._tools[name] = (definition, handler)

    def get_all_tools(self) -> list[dict[str, Any]]:
        """返回所有工具描述的副本，可直接传入 Trip_Client.ask(tools=...)。

        返回值不包含 Python 函数，修改返回内容不会影响已注册工具。
        """
        if self._amap is not None:
            amap, self._amap = self._amap, None
            try:
                tools = amap.list_tools()
            except RuntimeError:
                logging.getLogger(__name__).warning("无法获取高德 MCP 工具，请检查网络和 AMAP_API_KEY；本会话继续普通问答。")
            else:
                for tool in tools:
                    self.register_tool(
                        tool.name, tool.description or tool.name, tool.inputSchema,
                        amap.handler(tool.name),
                    )
        return deepcopy([definition for definition, _ in self._tools.values()])

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """按名称执行一次工具调用并原样返回结果，不自动捕获执行异常。

        Args:
            name: 模型返回的 ToolCall.name。
            arguments: 模型返回的 ToolCall.arguments，作为关键字参数传给工具。

        Raises:
            KeyError: 工具名称未注册。
            TypeError: 参数与工具函数签名不匹配。
        """
        _, handler = self._tools[name]
        return handler(**arguments)
