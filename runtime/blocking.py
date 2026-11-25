"""等待底层操作真实结束，再向独占 Session 的调用方交付结果或取消。"""

import asyncio
from functools import partial
from typing import Callable, TypeVar

T = TypeVar("T")


async def wait_finished(work: asyncio.Future[T], *, finish_on_cancel: bool = False) -> T:
    """只屏蔽完成通知，底层异常由唯一的 result() 调用收集。"""
    done = asyncio.get_running_loop().create_future()
    work.add_done_callback(lambda _: done.set_result(None))
    cancelled = False
    while not work.done():
        try:
            await asyncio.shield(done)
        except asyncio.CancelledError:
            cancelled = True
    result = work.result()
    if cancelled and not finish_on_cancel:
        raise asyncio.CancelledError
    return result


async def blocking(operation: Callable[..., T], *args, finish_on_cancel: bool = False, **kwargs) -> T:
    """使用默认执行器，取消不能中断 worker 或提前释放 Session。"""
    work = asyncio.get_running_loop().run_in_executor(None, partial(operation, *args, **kwargs))
    return await wait_finished(work, finish_on_cancel=finish_on_cancel)
