"""trapsim.schedule  –  Voltage schedule + trigger evaluation.

A `Schedule` aggregates the main voltage schedule and any number of
trigger schedules.  For an evaluation time `t_us` and a `trigger_state`
(per-particle dict mapping trigger name → fire time in µs, or None),
`evaluate(t_us, trigger_state)` returns a `{electrode_name: voltage_V}` dict.

Each trigger schedule may contain `dc` and/or `rf` blocks listing only
the electrodes it overrides.  When the trigger has fired, its overrides
apply with time = t_us − t_fire; other electrodes continue to follow
the main schedule with time = t_us.

If multiple fired triggers override the same electrode, the trigger that
fired most recently wins (its `t - t_fire` time axis is used).

Voltages on RF electrodes are evaluated as
    V(t) = dc(t) + amplitude(t) * cos(2π · frequency_hz · t · 1e-6 + phase)
where amplitude and frequency can either be scalars or arrays (interpolated
against the schedule's time_us axis).
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


def _interp(t_arr, v_arr, t):
    """Linear interpolation of v_arr at t, clamped at endpoints.  v_arr may
    be a scalar (returned as-is) or a 1-D array aligned with t_arr."""
    if np.isscalar(v_arr):
        return float(v_arr)
    if len(v_arr) == 0:
        return 0.0
    if len(v_arr) == 1:
        return float(v_arr[0])
    if t <= t_arr[0]:
        return float(v_arr[0])
    if t >= t_arr[-1]:
        return float(v_arr[-1])
    idx = int(np.searchsorted(t_arr, t, side="right")) - 1
    idx = max(0, min(idx, len(t_arr) - 2))
    t0, t1 = t_arr[idx], t_arr[idx + 1]
    frac = (t - t0) / (t1 - t0)
    return float(v_arr[idx] + frac * (v_arr[idx + 1] - v_arr[idx]))


class Schedule:
    """Resolve electrode voltages at any time, given per-particle trigger state."""

    def __init__(self,
                 main_schedule: Mapping[str, Any],
                 triggers: list[Mapping[str, Any]],
                 electrode_names: list[str]):
        self._electrode_names = list(electrode_names)
        self._main = self._normalize(main_schedule)
        self._triggers = [self._normalize_trigger(t) for t in triggers]

    @property
    def trigger_names(self) -> list[str]:
        return [t["name"] for t in self._triggers]

    @staticmethod
    def _normalize(blk: Mapping[str, Any]) -> dict[str, Any]:
        out = {
            "time_us": np.asarray(blk.get("time_us", []), dtype=float),
            "dc":  {},
            "rf":  {},
        }
        for name, v in (blk.get("dc") or {}).items():
            out["dc"][name] = np.asarray(v, dtype=float)
        for name, rf in (blk.get("rf") or {}).items():
            out["rf"][name] = {
                "amplitude":    np.asarray(rf["amplitude"], dtype=float),
                "frequency_hz": (np.asarray(rf["frequency_hz"], dtype=float)
                                  if isinstance(rf["frequency_hz"], (list, np.ndarray))
                                  else float(rf["frequency_hz"])),
                "phase_deg":    float(rf.get("phase_deg", 0.0)),
            }
        return out

    @classmethod
    def _normalize_trigger(cls, trig: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "name":         str(trig["name"]),
            "axis":         str(trig["axis"]),
            "axis_index":   {"x": 0, "y": 1, "z": 2}[trig["axis"]],
            "threshold_mm": float(trig["threshold_mm"]),
            "schedule":     cls._normalize(trig["schedule"]),
        }

    # ── Resolver ──────────────────────────────────────────────────────
    def evaluate(self, t_us: float,
                 trigger_state: Mapping[str, float | None]) -> dict[str, float]:
        """Return {electrode_name: voltage_V} at absolute time `t_us`.

        `trigger_state[name]` is the fire time in µs (absolute) or None.
        """
        # For each electrode, find the active source (main or most-recent trigger).
        # dc and rf are tracked independently because a trigger may override only one.
        dc_source = {}    # name → (time_axis, value_array, t_offset)
        rf_source = {}    # name → (time_axis, rf_dict, t_offset)

        # Start from main schedule (t_offset = 0; absolute time)
        for name, arr in self._main["dc"].items():
            dc_source[name] = (self._main["time_us"], arr, 0.0)
        for name, rf in self._main["rf"].items():
            rf_source[name] = (self._main["time_us"], rf, 0.0)

        # Apply each fired trigger in order of fire time (most-recent wins)
        fired = [(t["name"], trigger_state.get(t["name"]), t)
                 for t in self._triggers
                 if trigger_state.get(t["name"]) is not None]
        # Sort: oldest first, so later overrides win when looping
        fired.sort(key=lambda x: x[1])
        for name, t_fire, trig in fired:
            tsched = trig["schedule"]
            for elec, arr in tsched["dc"].items():
                dc_source[elec] = (tsched["time_us"], arr, t_fire)
            for elec, rf in tsched["rf"].items():
                rf_source[elec] = (tsched["time_us"], rf, t_fire)

        out = {}
        for name in self._electrode_names:
            v_dc = 0.0
            v_rf = 0.0
            if name in dc_source:
                t_arr, arr, t_off = dc_source[name]
                v_dc = _interp(t_arr, arr, t_us - t_off)
            if name in rf_source:
                t_arr, rf, t_off = rf_source[name]
                amp = _interp(t_arr, rf["amplitude"], t_us - t_off)
                f_hz = rf["frequency_hz"]
                if isinstance(f_hz, np.ndarray):
                    f_hz = _interp(t_arr, f_hz, t_us - t_off)
                # cos arg uses absolute time so the RF phase is continuous
                # across trigger boundaries.
                omega_us = 2.0 * math.pi * float(f_hz) * 1e-6   # rad/µs
                phase_r  = math.radians(rf["phase_deg"])
                v_rf = amp * math.cos(omega_us * t_us + phase_r)
            out[name] = v_dc + v_rf
        return out

    # ── Trigger-fire check ────────────────────────────────────────────
    def check_triggers(self, t_us: float, pos_world_mm,
                       trigger_state: dict[str, float | None]) -> list[str]:
        """Update `trigger_state` for any newly-firing triggers.  Returns
        the list of trigger names that fired on this call."""
        newly = []
        for trig in self._triggers:
            name = trig["name"]
            if trigger_state.get(name) is not None:
                continue
            if pos_world_mm[trig["axis_index"]] >= trig["threshold_mm"]:
                trigger_state[name] = t_us
                newly.append(name)
        return newly

    def initial_trigger_state(self) -> dict[str, float | None]:
        return {t["name"]: None for t in self._triggers}
