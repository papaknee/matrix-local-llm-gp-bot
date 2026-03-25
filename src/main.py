"""
src/main.py
===========
Entry point for the Matrix LLM Bot.

Single-bot mode:
    python -m src.main
    python -m src.main config/config.yaml

Multi-bot mode (auto-detected or explicit):
    python -m src.main --multi
    python -m src.main --multi bots/fleet_config.yaml

The script sets up logging, loads configuration, and starts the bot (or the
multi-bot orchestrator).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure root logger with colour support when available."""
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = []

    try:
        import colorlog  # type: ignore[import-untyped]

        colour_fmt = (
            "%(log_color)s%(asctime)s [%(levelname)s]%(reset)s "
            "%(name)s: %(message)s"
        )
        console = colorlog.StreamHandler()
        console.setFormatter(
            colorlog.ColoredFormatter(colour_fmt, datefmt=datefmt)
        )
    except ImportError:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    handlers.append(console)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )


async def _run(config_path: str) -> None:
    from src.config_manager import ConfigManager
    from src.bot import Bot

    cfg = ConfigManager(config_path)
    cfg.ensure_directories()
    setup_logging(cfg.logging.level, cfg.logging.file)

    logger = logging.getLogger("main")
    logger.info("=" * 60)
    logger.info("  Matrix LLM Bot starting up")
    logger.info("  Config: %s", config_path)
    logger.info("  Backend: %s  |  Mode: %s", cfg.llm.backend, cfg.llm.hardware_mode)
    logger.info("=" * 60)

    bot = Bot(cfg)

    # Start periodic thoughts runner before bot.start() (which blocks)
    interval = getattr(cfg.bot, 'thoughts_interval_seconds', 3600)
    thoughts_task = asyncio.create_task(run_thoughts_periodically(interval))

    try:
        await bot.start()
    finally:
        thoughts_task.cancel()
        try:
            await thoughts_task
        except asyncio.CancelledError:
            pass


async def run_thoughts_periodically(interval: int):
    while True:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "thoughts.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if stdout:
            logging.getLogger("thoughts").info(stdout.decode().strip())
        if stderr:
            logging.getLogger("thoughts").error(stderr.decode().strip())
        await asyncio.sleep(interval)


async def _run_multi(fleet_config_path: str) -> None:
    """Start the multi-bot orchestrator."""
    from src.config_manager import ConfigManager
    from src.fleet_config import FleetConfig
    from src.orchestrator import Orchestrator

    fleet = FleetConfig(fleet_config_path)
    # Use the first child bot's logging config for the fleet
    first_bot_cfg = ConfigManager(fleet.child_bots[0].config_path)
    setup_logging(first_bot_cfg.logging.level, first_bot_cfg.logging.file)

    logger = logging.getLogger("main")
    logger.info("=" * 60)
    logger.info("  Matrix LLM Bot — Multi-Bot Orchestrator")
    logger.info("  Fleet config: %s", fleet_config_path)
    logger.info("  Route-bot config: %s", fleet.route_bot_config_path)
    logger.info("  Child bots: %s", [b.name for b in fleet.child_bots])
    logger.info("=" * 60)

    orch = Orchestrator(fleet)
    await orch.start()


def main() -> None:
    # Parse arguments
    args = sys.argv[1:]
    multi_mode = False
    config_path = None

    # Check for --multi flag
    if "--multi" in args:
        multi_mode = True
        args.remove("--multi")

    if args:
        config_path = args[0]

    # Auto-detect multi-bot mode if fleet_config.yaml exists
    default_fleet_path = "bots/fleet_config.yaml"
    if not multi_mode and config_path is None and Path(default_fleet_path).exists():
        multi_mode = True

    if multi_mode:
        fleet_path = config_path or default_fleet_path
        if not Path(fleet_path).exists():
            print(
                f"\n[ERROR] Fleet config not found: {fleet_path}\n"
                "Run the setup wizard in multi-bot mode first:\n\n"
                "    python setup_wizard.py\n"
            )
            sys.exit(1)
        try:
            asyncio.run(_run_multi(fleet_path))
        except KeyboardInterrupt:
            print("\nShutting down fleet — goodbye!")
    else:
        # Single-bot mode
        single_config = config_path or "config/config.yaml"
        if not Path(single_config).exists():
            print(
                f"\n[ERROR] Config file not found: {single_config}\n"
                "Run the setup wizard first:\n\n"
                "    python setup_wizard.py\n"
            )
            sys.exit(1)
        try:
            asyncio.run(_run(single_config))
        except KeyboardInterrupt:
            print("\nShutting down — goodbye!")


if __name__ == "__main__":
    main()
