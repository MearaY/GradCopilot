"""
标准应用日志初始化。

日志同时输出到控制台和 logs/app.log，级别通过环境变量 LOG_LEVEL 配置。
"""
import logging
import os
from pathlib import Path


def setup_logging() -> None:
    """
    初始化全局日志配置。必须在应用启动时调用一次。

    日志级别从环境变量 LOG_LEVEL 读取，默认 INFO。
    日志同时输出到控制台和 logs/app.log。
    """
    log_level_str = os.environ.get("LOG_LEVEL", "INFO")
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    log_dir = Path(os.environ.get("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "app.log", encoding="utf-8"),
        ],
    )
