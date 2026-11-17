"""统一读取 KEY=VALUE 配置，处理默认值并按字段定义转换类型。"""

import os
from pathlib import Path
from typing import Any, Callable, Collection, Mapping

from dotenv import dotenv_values


def convert_settings(
    values: Mapping[str, Any],
    fields: Mapping[str, Callable[[Any], Any]],
    *,
    allow_empty: Collection[str] = (),
) -> dict[str, Any]:
    """转换指定字段；缺失或格式错误时只报告参数名，避免泄露配置值。

    字符串统一去除首尾空白。allow_empty 用于允许暂未填写的字段，例如
    在实际工具请求前才校验的高德 Key；业务范围与参数间关系由所属模块校验。
    """
    settings = {}
    for name, convert in fields.items():
        raw = values.get(name)
        if isinstance(raw, str):
            raw = raw.strip()
        if raw is None or raw == "":
            if name not in allow_empty:
                raise ValueError(f"缺少必填配置：{name}")
            raw = ""
        try:
            settings[name] = convert(raw)
        except (TypeError, ValueError):
            raise ValueError(f"配置格式错误：{name}") from None
    return settings


def load_settings(
    path: str | Path,
    fields: Mapping[str, Callable[[Any], Any]],
    *,
    defaults_path: str | Path | None = None,
    allow_empty: Collection[str] = (),
) -> dict[str, Any]:
    """读取配置并转换指定字段，不修改进程环境变量。

    同名系统环境变量覆盖配置文件；未设置或留空时使用 defaults_path 中的
    示例默认值。没有默认值的必填字段报错，允许为空的字段由 allow_empty 指定。
    """
    defaults = dotenv_values(defaults_path, encoding="utf-8") if defaults_path is not None else {}
    values = {**dotenv_values(path, encoding="utf-8"), **os.environ}
    for name in fields:
        raw = values.get(name)
        if raw is None or not raw.strip():
            values[name] = defaults.get(name)
    return convert_settings(values, fields, allow_empty=allow_empty)
