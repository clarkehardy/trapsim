"""Smoke tests for trapsim.schedule — trigger override semantics."""

import numpy as np
import pytest

from trapsim.schedule import Schedule


def _main_schedule():
    """Two electrodes, linear ramps from 0→10 V over 1000 µs."""
    t = np.linspace(0, 1000, 11)
    return {
        "time_us": t,
        "dc": {
            "plate_top": np.linspace(0, 10, 11),
            "plate_bottom": np.linspace(0, -10, 11),
        },
    }


def _trigger_a():
    """Override plate_top with a step to 50 V, leave plate_bottom alone."""
    return {
        "name": "release",
        "axis": "z",
        "threshold_mm": 100.0,
        "schedule": {
            "time_us": np.array([0, 500]),
            "dc": {"plate_top": np.array([50.0, 50.0])},
        },
    }


def _trigger_b():
    """Override plate_bottom with -50 V."""
    return {
        "name": "catch",
        "axis": "z",
        "threshold_mm": 200.0,
        "schedule": {
            "time_us": np.array([0, 500]),
            "dc": {"plate_bottom": np.array([-50.0, -50.0])},
        },
    }


class TestScheduleEvaluate:
    def test_main_only_no_triggers(self):
        sched = Schedule(_main_schedule(), [], ["plate_top", "plate_bottom"])
        ts = {}
        v = sched.evaluate(500.0, ts)
        assert v["plate_top"] == pytest.approx(5.0)
        assert v["plate_bottom"] == pytest.approx(-5.0)

    def test_main_clamps_beyond_time_axis(self):
        sched = Schedule(_main_schedule(), [], ["plate_top", "plate_bottom"])
        v = sched.evaluate(2000.0, {})
        assert v["plate_top"] == pytest.approx(10.0)
        assert v["plate_bottom"] == pytest.approx(-10.0)

    def test_single_trigger_overrides_one_electrode(self):
        sched = Schedule(_main_schedule(), [_trigger_a()],
                         ["plate_top", "plate_bottom"])
        ts = {"release": 400.0}  # fired at t=400
        v = sched.evaluate(600.0, ts)
        # plate_top: uses trigger schedule at t_local = 600 - 400 = 200
        assert v["plate_top"] == pytest.approx(50.0)
        # plate_bottom: still on main schedule at t=600
        assert v["plate_bottom"] == pytest.approx(-6.0)

    def test_unfired_trigger_has_no_effect(self):
        sched = Schedule(_main_schedule(), [_trigger_a()],
                         ["plate_top", "plate_bottom"])
        ts = {"release": None}
        v = sched.evaluate(500.0, ts)
        assert v["plate_top"] == pytest.approx(5.0)

    def test_two_triggers_most_recent_wins(self):
        """If both triggers override plate_top, the one that fired later wins."""
        trig_a = {
            "name": "first",
            "axis": "z",
            "threshold_mm": 100.0,
            "schedule": {
                "time_us": np.array([0, 500]),
                "dc": {"plate_top": np.array([10.0, 10.0])},
            },
        }
        trig_b = {
            "name": "second",
            "axis": "z",
            "threshold_mm": 200.0,
            "schedule": {
                "time_us": np.array([0, 500]),
                "dc": {"plate_top": np.array([99.0, 99.0])},
            },
        }
        sched = Schedule(_main_schedule(), [trig_a, trig_b],
                         ["plate_top", "plate_bottom"])
        ts = {"first": 100.0, "second": 300.0}
        v = sched.evaluate(400.0, ts)
        # "second" fired more recently → wins
        assert v["plate_top"] == pytest.approx(99.0)

    def test_rf_cosine(self):
        """RF produces amplitude * cos(2π·f·t·1e-6 + phase)."""
        main = {
            "time_us": np.array([0, 1000]),
            "rf": {
                "rod": {
                    "amplitude": np.array([40.0, 40.0]),
                    "frequency_hz": 3000.0,
                    "phase_deg": 0.0,
                },
            },
        }
        sched = Schedule(main, [], ["rod"])
        v = sched.evaluate(0.0, {})
        # cos(0) = 1 → V = 40
        assert v["rod"] == pytest.approx(40.0)


class TestCheckTriggers:
    def test_fires_when_threshold_crossed(self):
        sched = Schedule(_main_schedule(), [_trigger_a()],
                         ["plate_top", "plate_bottom"])
        ts = sched.initial_trigger_state()
        assert ts["release"] is None

        pos = np.array([0.0, 0.0, 150.0])
        fired = sched.check_triggers(500.0, pos, ts)
        assert "release" in fired
        assert ts["release"] == 500.0

    def test_does_not_fire_below_threshold(self):
        sched = Schedule(_main_schedule(), [_trigger_a()],
                         ["plate_top", "plate_bottom"])
        ts = sched.initial_trigger_state()
        pos = np.array([0.0, 0.0, 50.0])
        fired = sched.check_triggers(500.0, pos, ts)
        assert fired == []
        assert ts["release"] is None

    def test_does_not_refire(self):
        sched = Schedule(_main_schedule(), [_trigger_a()],
                         ["plate_top", "plate_bottom"])
        ts = {"release": 100.0}  # already fired
        pos = np.array([0.0, 0.0, 999.0])
        fired = sched.check_triggers(500.0, pos, ts)
        assert fired == []
        assert ts["release"] == 100.0  # unchanged

    def test_negative_axis_fires_below_threshold(self):
        """axis='-y' trigger fires when y <= threshold (negative-going crossing)."""
        trig = {
            "name": "fall",
            "axis": "-y",
            "threshold_mm": -5.0,
            "schedule": {"time_us": np.array([0, 100]),
                         "dc": {"plate_top": np.array([0.0, 0.0])}},
        }
        sched = Schedule(_main_schedule(), [trig], ["plate_top", "plate_bottom"])
        ts = sched.initial_trigger_state()
        # y = -10 <= -5 → should fire
        fired = sched.check_triggers(1.0, np.array([0.0, -10.0, 0.0]), ts)
        assert "fall" in fired

    def test_negative_axis_does_not_fire_above_threshold(self):
        trig = {
            "name": "fall",
            "axis": "-y",
            "threshold_mm": -5.0,
            "schedule": {"time_us": np.array([0, 100]),
                         "dc": {"plate_top": np.array([0.0, 0.0])}},
        }
        sched = Schedule(_main_schedule(), [trig], ["plate_top", "plate_bottom"])
        ts = sched.initial_trigger_state()
        # y = 0 > -5 → should not fire
        fired = sched.check_triggers(1.0, np.array([0.0, 0.0, 0.0]), ts)
        assert fired == []
