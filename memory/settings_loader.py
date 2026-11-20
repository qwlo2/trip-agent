"""统一加载 Memory 的 .config，转换参数类型并校验运行预算。"""

from pathlib import Path
from typing import Any

from settings_loader import load_settings


# Python 仅保留内部字段与环境变量名称的映射，默认参数由 .config.example 提供。
_ENV_KEYS = {
    "max_sessions": "MEMORY_PHASE1_MAX_SESSIONS",
    "scan_limit": "MEMORY_PHASE1_SCAN_LIMIT",
    "max_age_days": "MEMORY_PHASE1_MAX_AGE_DAYS",
    "min_idle_seconds": "MEMORY_PHASE1_MIN_IDLE_SECONDS",
    "lease_seconds": "MEMORY_PHASE1_LEASE_SECONDS",
    "max_attempts": "MEMORY_PHASE1_MAX_ATTEMPTS",
    "retry_delay_seconds": "MEMORY_PHASE1_RETRY_DELAY_SECONDS",
    "max_output_tokens": "MEMORY_PHASE1_MAX_OUTPUT_TOKENS",
    "max_read_bytes": "MEMORY_PHASE1_MAX_READ_BYTES",
    "max_record_bytes": "MEMORY_PHASE1_MAX_RECORD_BYTES",
    "max_rollout_bytes": "MEMORY_PHASE1_MAX_ROLLOUT_BYTES",
    "max_row_bytes": "MEMORY_PHASE1_MAX_ROW_BYTES",
    "tool_output_tokens": "MEMORY_PHASE1_TOOL_OUTPUT_TOKENS",
    "bytes_per_token": "MEMORY_BYTES_PER_TOKEN",
    "priority_user": "MEMORY_PRIORITY_USER",
    "priority_assistant": "MEMORY_PRIORITY_ASSISTANT",
    "priority_tool": "MEMORY_PRIORITY_TOOL",
    "phase2_max_items": "MEMORY_PHASE2_MAX_ITEMS",
    "phase2_active_days": "MEMORY_PHASE2_ACTIVE_DAYS",
    "phase2_interval_seconds": "MEMORY_PHASE2_INTERVAL_SECONDS",
    "phase2_max_retries": "MEMORY_PHASE2_MAX_RETRIES",
    "phase2_max_output_tokens": "MEMORY_PHASE2_MAX_OUTPUT_TOKENS",
    "phase2_max_steps": "MEMORY_PHASE2_MAX_STEPS",
    "phase2_tool_page_chars": "MEMORY_PHASE2_TOOL_PAGE_CHARS",
}


def load_config() -> dict[str, Any]:
    """按系统环境变量、memory/.config、.config.example 的顺序获取参数。

    使用模块所在目录定位文件，不依赖工作目录，也不修改进程环境变量。
    未设置或留空时使用模板默认值；整数参数必须为正，只有最短静置时间允许为 0。
    input_budget_ratio 校验在 (0, 1] 内，W 从根目录 .env 的模型配置读取。
    """
    directory = Path(__file__).resolve().parent
    values = load_settings(
        path=directory / ".config",
        defaults_path=directory / ".config.example",
        fields={**{name: int for name in _ENV_KEYS.values()}, "input_budget_ratio": float},
    )
    config = {}
    for key, name in _ENV_KEYS.items():
        value = values[name]
        if key == "min_idle_seconds":
            if value < 0:
                raise ValueError(f"{name} 不得小于 0")
        elif value <= 0:
            raise ValueError(f"{name} 必须为正整数")
        config[key] = value
    if config["min_idle_seconds"] > config["max_age_days"] * 86400:
        raise ValueError("Phase 1 最短静置时间不能超过会话历史上限")
    ratio = values["input_budget_ratio"]
    if not 0 < ratio <= 1:
        raise ValueError("input_budget_ratio 必须大于 0 且不超过 1")
    config["input_budget_ratio"] = ratio
    model = load_settings(
        path=directory.parent / ".env",
        defaults_path=directory.parent / ".env.example",
        fields={"DEEPSEEK_CONTEXT_WINDOW": int},
    )
    if model["DEEPSEEK_CONTEXT_WINDOW"] <= 0:
        raise ValueError("DEEPSEEK_CONTEXT_WINDOW 必须为正整数")
    config["context_window"] = model["DEEPSEEK_CONTEXT_WINDOW"]
    return config
