"""异步工具执行器与高德 Streamable HTTP MCP 的适配层。"""

import asyncio
from datetime import timedelta
from typing import Any, Awaitable, Callable
from urllib.parse import urlencode

from runtime.blocking import wait_finished
from httpx import AsyncClient
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, PaginatedRequestParams, Tool


_MAP_TOOLS = frozenset({
    "maps_text_search", "maps_around_search", "maps_search_detail",
    "maps_direction_walking", "maps_direction_driving", "maps_direction_transit_integrated",
})


class AmapMCP:
    """只向旅行 Agent 提供已允许的高德查询工具。"""

    def __init__(self, api_key: str, timeout: float) -> None:
        """使用现有 Web 服务 Key 连接官方 MCP 端点。"""
        self._url = f"https://mcp.amap.com/mcp?{urlencode({'key': api_key})}"
        self._timeout = timeout

    async def list_tools(self) -> list[Tool]:
        """获取允许注册的远程工具及其参数定义。"""
        try:
            return await self._timed_request()
        except Exception:
            raise RuntimeError("高德 MCP 工具发现失败") from None

    def handler(self, name: str) -> Callable[..., Awaitable[dict[str, Any]]]:
        """绑定远程工具名，返回可供 ToolExecutor 注册的异步函数。"""
        async def call(**arguments: Any) -> dict[str, Any]:
            """调用高德工具；网络或协议失败时向模型返回可读错误。"""
            try:
                result = await self._timed_request(name, arguments)
            except Exception:
                # 原始 HTTP 异常可能包含带 Key 的 URL。
                return {"isError": True, "error": "高德 MCP 请求失败，无法核实查询结果"}
            return result.model_dump(mode="json", by_alias=True, exclude_none=True)

        return call

    async def _timed_request(self, name: str | None = None, arguments: dict[str, Any] | None = None):
        """期限涵盖整个请求，取消一次并等待所有异步上下文实际退出。"""
        work = asyncio.create_task(self._request(name, arguments))
        try:
            done, _ = await asyncio.wait({work}, timeout=self._timeout)
            if not done:
                raise asyncio.TimeoutError
            return work.result()
        except (asyncio.TimeoutError, asyncio.CancelledError):
            work.cancel()
            try:
                await wait_finished(work, finish_on_cancel=True)
            except asyncio.CancelledError:
                pass
            raise

    async def _request(
        self, name: str | None = None, arguments: dict[str, Any] | None = None,
    ) -> list[Tool] | CallToolResult:
        """在同一异步上下文完成会话初始化、请求和清理。"""
        async with AsyncClient(timeout=self._timeout) as http_client:
            async with streamable_http_client(self._url, http_client=http_client) as (read, write, _):
                async with ClientSession(
                    read, write, read_timeout_seconds=timedelta(seconds=self._timeout),
                ) as session:
                    await session.initialize()
                    if name is not None:
                        return await session.call_tool(name, arguments=arguments)
                    tools = []
                    cursor = None
                    seen = set()
                    for _ in range(32):
                        page = await session.list_tools(params=PaginatedRequestParams(cursor=cursor))
                        tools.extend(tool for tool in page.tools if tool.name in _MAP_TOOLS)
                        cursor = page.nextCursor
                        if cursor is None:
                            return tools
                        if cursor in seen:
                            raise ValueError("高德工具分页 cursor 重复")
                        seen.add(cursor)
                    raise ValueError("高德工具发现超过 32 页")
