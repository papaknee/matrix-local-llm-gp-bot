"""
tests/test_temperature_controller.py
=====================================
Tests for src/temperature_controller.py
"""

import pytest

from src.config_manager import TemperatureControllerConfig
from src.temperature_controller import TemperatureController, _temperature_to_label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    enabled=True,
    change_interval_minutes=30,
    min_temperature=0.3,
    max_temperature=1.15,
    randomise_chime_in=True,
    min_chime_in=0.02,
    max_chime_in=0.20,
) -> TemperatureControllerConfig:
    return TemperatureControllerConfig(
        enabled=enabled,
        change_interval_minutes=change_interval_minutes,
        min_temperature=min_temperature,
        max_temperature=max_temperature,
        randomise_chime_in=randomise_chime_in,
        min_chime_in=min_chime_in,
        max_chime_in=max_chime_in,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTemperatureToLabel:
    @pytest.mark.parametrize(
        "temp, expected",
        [
            (0.1, "zen"),
            (0.4, "calm"),
            (0.65, "balanced"),
            (0.8, "chatty"),
            (0.95, "hyper"),
            (1.05, "feisty"),
            (1.2, "unhinged"),
        ],
    )
    def test_labels(self, temp, expected):
        assert _temperature_to_label(temp) == expected


class TestTemperatureControllerInit:
    def test_initial_temperature(self):
        cfg = _make_config()
        tc = TemperatureController(cfg, initial_temperature=0.7, initial_chime_in=0.1)
        assert tc.temperature == pytest.approx(0.7)

    def test_initial_chime_in(self):
        cfg = _make_config()
        tc = TemperatureController(cfg, initial_temperature=0.7, initial_chime_in=0.15)
        assert tc.chime_in_probability == pytest.approx(0.15)

    def test_initial_history_has_one_entry(self):
        cfg = _make_config()
        tc = TemperatureController(cfg, initial_temperature=0.5)
        assert len(tc.history) == 1

    def test_current_mood_label(self):
        cfg = _make_config()
        tc = TemperatureController(cfg, initial_temperature=0.65)
        assert tc.current_mood_label == "balanced"


class TestRollNewMood:
    def test_roll_changes_temperature(self):
        cfg = _make_config(min_temperature=0.3, max_temperature=1.15)
        tc = TemperatureController(cfg, initial_temperature=0.7)
        # Roll many times and verify the range
        for _ in range(50):
            tc.roll_new_mood()
            assert cfg.min_temperature <= tc.temperature <= cfg.max_temperature

    def test_roll_changes_chime_in_when_enabled(self):
        cfg = _make_config(randomise_chime_in=True, min_chime_in=0.02, max_chime_in=0.20)
        tc = TemperatureController(cfg, initial_temperature=0.7, initial_chime_in=0.1)
        for _ in range(30):
            tc.roll_new_mood()
            assert cfg.min_chime_in <= tc.chime_in_probability <= cfg.max_chime_in

    def test_roll_does_not_change_chime_in_when_disabled(self):
        cfg = _make_config(randomise_chime_in=False)
        tc = TemperatureController(cfg, initial_temperature=0.7, initial_chime_in=0.05)
        for _ in range(10):
            tc.roll_new_mood()
        assert tc.chime_in_probability == pytest.approx(0.05)

    def test_roll_adds_to_history(self):
        cfg = _make_config()
        tc = TemperatureController(cfg, initial_temperature=0.7)
        initial_len = len(tc.history)
        tc.roll_new_mood()
        assert len(tc.history) == initial_len + 1

    def test_roll_returns_snapshot_with_correct_temp(self):
        cfg = _make_config()
        tc = TemperatureController(cfg, initial_temperature=0.7)
        snap = tc.roll_new_mood()
        assert snap.temperature == tc.temperature

    def test_roll_disabled_returns_last_snapshot(self):
        cfg = _make_config(enabled=False)
        tc = TemperatureController(cfg, initial_temperature=0.5)
        original_temp = tc.temperature
        snap = tc.roll_new_mood()
        # Temperature should remain unchanged
        assert tc.temperature == pytest.approx(original_temp)


class TestSetters:
    def test_set_temperature_valid(self):
        cfg = _make_config()
        tc = TemperatureController(cfg, initial_temperature=0.7)
        tc.set_temperature(1.0)
        assert tc.temperature == pytest.approx(1.0)

    def test_set_temperature_invalid_zero(self):
        cfg = _make_config()
        tc = TemperatureController(cfg, initial_temperature=0.7)
        with pytest.raises(ValueError):
            tc.set_temperature(0.0)

    def test_set_temperature_invalid_too_high(self):
        cfg = _make_config()
        tc = TemperatureController(cfg, initial_temperature=0.7)
        with pytest.raises(ValueError):
            tc.set_temperature(2.5)

    def test_set_chime_in_valid(self):
        cfg = _make_config()
        tc = TemperatureController(cfg, initial_temperature=0.7)
        tc.set_chime_in_probability(0.5)
        assert tc.chime_in_probability == pytest.approx(0.5)

    def test_set_chime_in_invalid_negative(self):
        cfg = _make_config()
        tc = TemperatureController(cfg, initial_temperature=0.7)
        with pytest.raises(ValueError):
            tc.set_chime_in_probability(-0.1)

    def test_set_chime_in_invalid_over_one(self):
        cfg = _make_config()
        tc = TemperatureController(cfg, initial_temperature=0.7)
        with pytest.raises(ValueError):
            tc.set_chime_in_probability(1.5)


class TestHistoryBoundary:
    def test_history_bounded_at_200(self):
        cfg = _make_config()
        tc = TemperatureController(cfg, initial_temperature=0.7)
        # Roll 250 times — history should never exceed 200 (trimmed to 100)
        for _ in range(250):
            tc.roll_new_mood()
        assert len(tc.history) <= 200
