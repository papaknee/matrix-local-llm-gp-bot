"""
tests/test_config_manager.py
============================
Tests for src/config_manager.py
"""

import os
import textwrap
import tempfile
from pathlib import Path

import pytest

from src.config_manager import ConfigManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_config(tmp_path: Path, content: str) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return str(p)


VALID_YAML = """
matrix:
  homeserver: "https://example.org"
  username: "@bot:example.org"
  password: "secret"

llm:
  backend: "llamacpp"
  model_path: "models/test.gguf"
  hardware_mode: "cpu"

bot:
  display_name: "TestBot"
  trigger_names:
    - "testbot"
    - "hey bot"

temperature_controller:
  enabled: true
  change_interval_minutes: 10
  min_temperature: 0.3
  max_temperature: 0.9

memory:
  dossier_dir: "data/dossiers"
  archive_dir: "data/archives"

logging:
  level: "DEBUG"
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConfigManagerLoading:
    def test_loads_valid_config(self, tmp_path):
        path = _write_config(tmp_path, VALID_YAML)
        cfg = ConfigManager(path)

        assert cfg.matrix.homeserver == "https://example.org"
        assert cfg.matrix.username == "@bot:example.org"
        assert cfg.matrix.password == "secret"
        assert cfg.llm.backend == "llamacpp"
        assert cfg.llm.model_path == "models/test.gguf"
        assert cfg.llm.hardware_mode == "cpu"
        assert cfg.bot.display_name == "TestBot"
        assert "testbot" in cfg.bot.trigger_names
        assert "hey bot" in cfg.bot.trigger_names
        assert cfg.temperature_controller.enabled is True
        assert cfg.temperature_controller.min_temperature == pytest.approx(0.3)
        assert cfg.logging.level == "DEBUG"

    def test_trigger_names_lowercased(self, tmp_path):
        yaml = VALID_YAML.replace("testbot", "TESTBOT").replace("hey bot", "HEY BOT")
        path = _write_config(tmp_path, yaml)
        cfg = ConfigManager(path)
        for name in cfg.bot.trigger_names:
            assert name == name.lower(), f"Trigger name not lowercased: {name}"

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ConfigManager(str(tmp_path / "nonexistent.yaml"))

    def test_missing_homeserver_raises(self, tmp_path):
        bad_yaml = VALID_YAML.replace('homeserver: "https://example.org"', "")
        path = _write_config(tmp_path, bad_yaml)
        with pytest.raises(ValueError, match="homeserver"):
            ConfigManager(path)

    def test_missing_username_raises(self, tmp_path):
        bad_yaml = VALID_YAML.replace('username: "@bot:example.org"', "")
        path = _write_config(tmp_path, bad_yaml)
        with pytest.raises(ValueError, match="username"):
            ConfigManager(path)

    def test_missing_model_path_for_llamacpp_raises(self, tmp_path):
        bad_yaml = VALID_YAML.replace('model_path: "models/test.gguf"', "")
        path = _write_config(tmp_path, bad_yaml)
        with pytest.raises(ValueError, match="model_path"):
            ConfigManager(path)

    def test_bad_hardware_mode_raises(self, tmp_path):
        bad_yaml = VALID_YAML.replace('hardware_mode: "cpu"', 'hardware_mode: "tpu"')
        path = _write_config(tmp_path, bad_yaml)
        with pytest.raises(ValueError, match="hardware_mode"):
            ConfigManager(path)

    def test_bad_backend_raises(self, tmp_path):
        bad_yaml = VALID_YAML.replace('backend: "llamacpp"', 'backend: "onnx"')
        path = _write_config(tmp_path, bad_yaml)
        with pytest.raises(ValueError, match="backend"):
            ConfigManager(path)

    def test_inverted_temp_range_raises(self, tmp_path):
        bad_yaml = VALID_YAML.replace(
            "min_temperature: 0.3", "min_temperature: 1.0"
        ).replace("max_temperature: 0.9", "max_temperature: 0.5")
        path = _write_config(tmp_path, bad_yaml)
        with pytest.raises(ValueError, match="min_temperature"):
            ConfigManager(path)


class TestConfigManagerDefaults:
    def _minimal_yaml(self, tmp_path: Path) -> str:
        return _write_config(
            tmp_path,
            """
matrix:
  homeserver: "https://example.org"
  username: "@bot:example.org"
  password: "secret"

llm:
  backend: "llamacpp"
  model_path: "models/test.gguf"
""",
        )

    def test_default_hardware_mode(self, tmp_path):
        cfg = ConfigManager(self._minimal_yaml(tmp_path))
        assert cfg.llm.hardware_mode == "cpu"

    def test_default_chime_in_probability(self, tmp_path):
        cfg = ConfigManager(self._minimal_yaml(tmp_path))
        assert cfg.bot.chime_in_probability == pytest.approx(0.08)

    def test_default_allowed_rooms_empty(self, tmp_path):
        cfg = ConfigManager(self._minimal_yaml(tmp_path))
        assert cfg.matrix.allowed_rooms == []

    def test_default_logging_level_info(self, tmp_path):
        cfg = ConfigManager(self._minimal_yaml(tmp_path))
        assert cfg.logging.level == "INFO"


class TestEnsureDirectories:
    def test_creates_directories(self, tmp_path):
        yaml_content = f"""
matrix:
  homeserver: "https://example.org"
  username: "@bot:example.org"
  password: "secret"
  store_path: "{tmp_path}/matrix_store"

llm:
  backend: "llamacpp"
  model_path: "models/test.gguf"
  hardware_mode: "cpu"

memory:
  dossier_dir: "{tmp_path}/dossiers"
  archive_dir: "{tmp_path}/archives"
"""
        path = _write_config(tmp_path, yaml_content)
        cfg = ConfigManager(path)
        cfg.ensure_directories()

        assert Path(cfg.memory.dossier_dir).is_dir()
        assert Path(cfg.memory.archive_dir).is_dir()
