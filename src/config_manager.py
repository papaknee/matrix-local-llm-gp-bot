"""
src/config_manager.py
=====================
Loads, validates, and provides typed access to the bot's YAML configuration.

Usage
-----
    from src.config_manager import ConfigManager

    cfg = ConfigManager("config/config.yaml")
    print(cfg.matrix.homeserver)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


# ---------------------------------------------------------------------------
# Typed configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MatrixConfig:
    homeserver: str
    username: str
    password: str
    device_name: str = "LLM-Bot"
    allowed_rooms: List[str] = field(default_factory=list)
    store_path: str = "data/matrix_store"


@dataclass
class LLMConfig:
    backend: str = "llamacpp"           # "llamacpp" | "transformers"
    model_path: str = ""
    hf_model_id: str = ""
    hf_cache_dir: str = "models/hf_cache"  # local cache for HuggingFace downloads
    hardware_mode: str = "cpu"          # "cpu" | "gpu"
    n_gpu_layers: int = 0
    context_length: int = 4096
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1


@dataclass
class BotConfig:
    display_name: str = "GPBot"
    trigger_names: List[str] = field(default_factory=lambda: ["gpbot"])
    persona_file: str = "config/bot_persona.txt"
    chime_in_probability: float = 0.08
    chime_in_cooldown_messages: int = 5
    chime_in_cooldown_seconds: int = 60
    conversation_history_limit: int = 12
    dossier_token_budget: int = 512


@dataclass
class TemperatureControllerConfig:
    enabled: bool = True
    change_interval_minutes: int = 30
    min_temperature: float = 0.3
    max_temperature: float = 1.15
    randomise_chime_in: bool = True
    min_chime_in: float = 0.02
    max_chime_in: float = 0.20


@dataclass
class MemoryConfig:
    dossier_dir: str = "data/dossiers"
    archive_dir: str = "data/archives"
    max_active_entries: int = 50
    compaction_interval_hours: int = 6
    max_summary_chars: int = 800


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: Optional[str] = None


# ---------------------------------------------------------------------------
# ConfigManager
# ---------------------------------------------------------------------------


class ConfigManager:
    """Loads a YAML config file and exposes typed sub-configs as attributes.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file (e.g. ``config/config.yaml``).

    Raises
    ------
    FileNotFoundError
        If *config_path* does not exist.
    ValueError
        If required fields are missing or values are invalid.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self._path = Path(config_path)
        if not self._path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self._path}\n"
                "Copy config/config.example.yaml to config/config.yaml and fill in your details."
            )

        with self._path.open("r", encoding="utf-8") as fh:
            raw: dict = yaml.safe_load(fh) or {}

        self.matrix = self._parse_matrix(raw.get("matrix", {}))
        self.llm = self._parse_llm(raw.get("llm", {}))
        self.bot = self._parse_bot(raw.get("bot", {}))
        self.temperature_controller = self._parse_temp_ctrl(
            raw.get("temperature_controller", {})
        )
        self.memory = self._parse_memory(raw.get("memory", {}))
        self.logging = self._parse_logging(raw.get("logging", {}))

        self._validate()

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_matrix(d: dict) -> MatrixConfig:
        return MatrixConfig(
            homeserver=d.get("homeserver", ""),
            username=d.get("username", ""),
            password=d.get("password", ""),
            device_name=d.get("device_name", "LLM-Bot"),
            allowed_rooms=d.get("allowed_rooms", []) or [],
            store_path=d.get("store_path", "data/matrix_store"),
        )

    @staticmethod
    def _parse_llm(d: dict) -> LLMConfig:
        return LLMConfig(
            backend=d.get("backend", "llamacpp"),
            model_path=d.get("model_path", ""),
            hf_model_id=d.get("hf_model_id", ""),
            hf_cache_dir=d.get("hf_cache_dir", "models/hf_cache"),
            hardware_mode=d.get("hardware_mode", "cpu"),
            n_gpu_layers=int(d.get("n_gpu_layers", 0)),
            context_length=int(d.get("context_length", 4096)),
            max_new_tokens=int(d.get("max_new_tokens", 512)),
            temperature=float(d.get("temperature", 0.7)),
            top_p=float(d.get("top_p", 0.9)),
            repeat_penalty=float(d.get("repeat_penalty", 1.1)),
        )

    @staticmethod
    def _parse_bot(d: dict) -> BotConfig:
        return BotConfig(
            display_name=d.get("display_name", "GPBot"),
            trigger_names=[t.lower() for t in d.get("trigger_names", ["gpbot"])],
            persona_file=d.get("persona_file", "config/bot_persona.txt"),
            chime_in_probability=float(d.get("chime_in_probability", 0.08)),
            chime_in_cooldown_messages=int(d.get("chime_in_cooldown_messages", 5)),
            chime_in_cooldown_seconds=int(d.get("chime_in_cooldown_seconds", 60)),
            conversation_history_limit=int(d.get("conversation_history_limit", 12)),
            dossier_token_budget=int(d.get("dossier_token_budget", 512)),
        )

    @staticmethod
    def _parse_temp_ctrl(d: dict) -> TemperatureControllerConfig:
        return TemperatureControllerConfig(
            enabled=bool(d.get("enabled", True)),
            change_interval_minutes=int(d.get("change_interval_minutes", 30)),
            min_temperature=float(d.get("min_temperature", 0.3)),
            max_temperature=float(d.get("max_temperature", 1.15)),
            randomise_chime_in=bool(d.get("randomise_chime_in", True)),
            min_chime_in=float(d.get("min_chime_in", 0.02)),
            max_chime_in=float(d.get("max_chime_in", 0.20)),
        )

    @staticmethod
    def _parse_memory(d: dict) -> MemoryConfig:
        return MemoryConfig(
            dossier_dir=d.get("dossier_dir", "data/dossiers"),
            archive_dir=d.get("archive_dir", "data/archives"),
            max_active_entries=int(d.get("max_active_entries", 50)),
            compaction_interval_hours=int(d.get("compaction_interval_hours", 6)),
            max_summary_chars=int(d.get("max_summary_chars", 800)),
        )

    @staticmethod
    def _parse_logging(d: dict) -> LoggingConfig:
        return LoggingConfig(
            level=d.get("level", "INFO").upper(),
            file=d.get("file") or None,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        errors: List[str] = []

        if not self.matrix.homeserver:
            errors.append("matrix.homeserver is required")
        if not self.matrix.username:
            errors.append("matrix.username is required")
        if not self.matrix.password:
            errors.append("matrix.password is required")

        if self.llm.backend not in ("llamacpp", "transformers"):
            errors.append(
                f"llm.backend must be 'llamacpp' or 'transformers', got '{self.llm.backend}'"
            )
        if self.llm.backend == "llamacpp" and not self.llm.model_path:
            errors.append("llm.model_path is required when backend is 'llamacpp'")
        if self.llm.backend == "transformers" and not self.llm.hf_model_id:
            errors.append("llm.hf_model_id is required when backend is 'transformers'")
        if self.llm.hardware_mode not in ("cpu", "gpu"):
            errors.append(
                f"llm.hardware_mode must be 'cpu' or 'gpu', got '{self.llm.hardware_mode}'"
            )

        tc = self.temperature_controller
        if tc.min_temperature >= tc.max_temperature:
            errors.append(
                "temperature_controller.min_temperature must be less than max_temperature"
            )

        if errors:
            raise ValueError(
                "Configuration errors in {}:\n  - {}".format(
                    self._path, "\n  - ".join(errors)
                )
            )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def ensure_directories(self) -> None:
        """Create all required data directories if they don't already exist."""
        dirs = [
            self.memory.dossier_dir,
            self.memory.archive_dir,
            self.matrix.store_path,
            self.llm.hf_cache_dir if self.llm.backend == "transformers" else None,
            os.path.dirname(self.logging.file) if self.logging.file else None,
        ]
        for d in dirs:
            if d:
                Path(d).mkdir(parents=True, exist_ok=True)
