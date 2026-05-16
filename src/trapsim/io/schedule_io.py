"""trapsim.io.schedule_io  –  JSON snapshot of a voltage schedule.

Written alongside each run so animate.py can plot the actual voltages used,
without re-evaluating the trigger logic.  The snapshot is a resolved
timeseries: one DC and one RF amplitude/frequency per electrode at every
sample time.  Triggers contribute their own time arrays (offset = t_fire).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def _to_list(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def write_schedule_snapshot(path: str, main_schedule: dict[str, Any],
                            triggers: list[dict[str, Any]]) -> None:
    """Dump the schedule data (main + triggers) as JSON.

    All numpy arrays are converted to Python lists.  Schema:
      {
        "main": {"time_us": [...], "dc": {name: [...]},
                  "rf":   {name: {"amplitude": [...],
                                  "frequency_hz": scalar_or_list,
                                  "phase_deg": scalar}}},
        "triggers": [
          {"name": str, "axis": "x"|"y"|"z", "threshold_mm": float,
           "schedule": {"time_us": [...], "dc": {...}, "rf": {...}}},
          ...
        ]
      }
    """
    def serialize_block(blk):
        out = {"time_us": _to_list(blk.get("time_us", []))}
        if blk.get("dc"):
            out["dc"] = {name: _to_list(v) for name, v in blk["dc"].items()}
        if blk.get("rf"):
            out["rf"] = {}
            for name, rf in blk["rf"].items():
                out["rf"][name] = {
                    "amplitude":     _to_list(rf["amplitude"]),
                    "frequency_hz":  _to_list(rf.get("frequency_hz", 0.0)),
                    "phase_deg":     float(rf.get("phase_deg", 0.0)),
                }
        return out

    data = {
        "main": serialize_block(main_schedule),
        "triggers": [
            {
                "name":         t["name"],
                "axis":         t["axis"],
                "threshold_mm": float(t["threshold_mm"]),
                "schedule":     serialize_block(t["schedule"]),
            }
            for t in triggers
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def read_schedule_snapshot(path: str) -> dict[str, Any]:
    """Load a previously-written schedule snapshot.  Time arrays and
    voltages are returned as numpy arrays for plotting convenience."""
    with open(path) as f:
        data = json.load(f)

    def deserialize_block(blk):
        out = {"time_us": np.asarray(blk.get("time_us", []), dtype=float)}
        if "dc" in blk:
            out["dc"] = {n: np.asarray(v, dtype=float)
                         for n, v in blk["dc"].items()}
        if "rf" in blk:
            out["rf"] = {}
            for n, rf in blk["rf"].items():
                f_hz = rf["frequency_hz"]
                out["rf"][n] = {
                    "amplitude":    np.asarray(rf["amplitude"], dtype=float),
                    "frequency_hz": (np.asarray(f_hz, dtype=float)
                                     if isinstance(f_hz, list) else float(f_hz)),
                    "phase_deg":    float(rf["phase_deg"]),
                }
        return out

    return {
        "main":     deserialize_block(data["main"]),
        "triggers": [
            {
                "name":         t["name"],
                "axis":         t["axis"],
                "threshold_mm": float(t["threshold_mm"]),
                "schedule":     deserialize_block(t["schedule"]),
            }
            for t in data["triggers"]
        ],
    }
