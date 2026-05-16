"""trapsim.config  –  YAML geometry loader + experiment.py loader.

`load_geometry(path)` returns a `GeometryConfig` (dataclass) with the parsed
and validated YAML.  Each electrode is assigned an integer `electrode_id` in
declaration order (1..N).

`load_experiment(path)` imports the user's experiment.py without putting it on
sys.path and returns an `ExperimentConfig` populated from its module-level
names (particle, particles, physics, integrator, main_schedule, triggers).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import yaml


# ── STL path resolution ───────────────────────────────────────────────────────
# During the transition, STL files may live at the repo root or under stl/.
# The resolver tries the path as written, then the basename under the repo
# root, then the basename under stl/.  Raises FileNotFoundError on miss.

def _resolve_stl(path: str, base_dir: str) -> str:
    candidates = [
        path if os.path.isabs(path) else os.path.join(base_dir, path),
        os.path.join(base_dir, os.path.basename(path)),
        os.path.join(base_dir, "stl", os.path.basename(path)),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        f"STL not found.  Tried:\n  " + "\n  ".join(candidates))


# ── Geometry ──────────────────────────────────────────────────────────────────

@dataclass
class Electrode:
    name: str
    stls: list[str]               # resolved absolute paths
    electrode_id: int             # assigned 1..N in declaration order
    color: tuple[float, float, float] | None = None    # None → auto-assigned
    opacity: float = 0.40


@dataclass
class Dielectric:
    name: str
    stl: str                      # resolved absolute path
    epsilon_r: float
    color: tuple[float, float, float] = (0.50, 0.90, 0.95)
    opacity: float = 0.30


@dataclass
class Decoration:
    name: str
    stl: str                      # resolved absolute path
    color: tuple[float, float, float] = (0.5, 0.5, 0.5)
    opacity: float = 0.30


@dataclass
class Grid:
    dx_mm: float
    bounds_mm: dict[str, tuple[float, float]]   # {"x": (lo,hi), "y": ..., "z": ...}

    @property
    def world_offset_mm(self) -> tuple[float, float, float]:
        """Offset added to a GEM index*dx to get a Fusion-world coord."""
        return (self.bounds_mm["x"][0],
                self.bounds_mm["y"][0],
                self.bounds_mm["z"][0])

    @property
    def shape(self) -> tuple[int, int, int]:
        """(NX, NY, NZ).  Endpoint-inclusive: floor((hi-lo)/dx)+1."""
        out = []
        for axis in ("x", "y", "z"):
            lo, hi = self.bounds_mm[axis]
            out.append(int(round((hi - lo) / self.dx_mm)) + 1)
        return tuple(out)


@dataclass
class GeometryConfig:
    grid: Grid
    electrodes: list[Electrode]
    dielectrics: list[Dielectric] = field(default_factory=list)
    decoration: list[Decoration] = field(default_factory=list)
    source_path: str = ""

    @property
    def n_electrodes(self) -> int:
        return len(self.electrodes)

    def electrode_by_name(self, name: str) -> Electrode:
        for e in self.electrodes:
            if e.name == name:
                return e
        raise KeyError(f"No electrode named {name!r} in {self.source_path}")

    def electrode_names(self) -> list[str]:
        return [e.name for e in self.electrodes]


def load_geometry(path: str) -> GeometryConfig:
    """Load and validate geometry.yaml.  Returns a GeometryConfig."""
    path = os.path.abspath(path)
    base_dir = os.path.dirname(path)

    with open(path) as f:
        raw = yaml.safe_load(f)

    # Grid
    if "grid" not in raw:
        raise ValueError(f"{path}: missing required 'grid' block")
    g = raw["grid"]
    if "dx_mm" not in g or "bounds_mm" not in g:
        raise ValueError(f"{path}: grid must have 'dx_mm' and 'bounds_mm'")
    bounds = {}
    for axis in ("x", "y", "z"):
        if axis not in g["bounds_mm"]:
            raise ValueError(f"{path}: grid.bounds_mm.{axis} is missing")
        lo, hi = g["bounds_mm"][axis]
        if not (hi > lo):
            raise ValueError(
                f"{path}: grid.bounds_mm.{axis} must be ascending; got [{lo}, {hi}]")
        bounds[axis] = (float(lo), float(hi))
    grid = Grid(dx_mm=float(g["dx_mm"]), bounds_mm=bounds)

    # Electrodes
    if not raw.get("electrodes"):
        raise ValueError(f"{path}: at least one electrode required")
    seen = set()
    electrodes = []
    for i, e in enumerate(raw["electrodes"], start=1):
        if "name" not in e or "stls" not in e:
            raise ValueError(f"{path}: electrode #{i} must have 'name' and 'stls'")
        name = str(e["name"])
        if name in seen:
            raise ValueError(f"{path}: duplicate electrode name {name!r}")
        seen.add(name)
        stls = [_resolve_stl(s, base_dir) for s in e["stls"]]
        if not stls:
            raise ValueError(f"{path}: electrode {name!r} has empty 'stls'")
        color = tuple(e["color"]) if "color" in e else None
        electrodes.append(Electrode(
            name=name, stls=stls, electrode_id=i,
            color=color, opacity=float(e.get("opacity", 0.40))))

    # Dielectrics (optional)
    dielectrics = []
    for i, d in enumerate(raw.get("dielectrics") or [], start=1):
        if "name" not in d or "stl" not in d or "epsilon_r" not in d:
            raise ValueError(
                f"{path}: dielectric #{i} must have 'name', 'stl', 'epsilon_r'")
        color = tuple(d.get("color", (0.50, 0.90, 0.95)))
        dielectrics.append(Dielectric(
            name=str(d["name"]),
            stl=_resolve_stl(d["stl"], base_dir),
            epsilon_r=float(d["epsilon_r"]),
            color=color,
            opacity=float(d.get("opacity", 0.30)),
        ))

    # Decoration (optional)
    decoration = []
    for i, dec in enumerate(raw.get("decoration") or [], start=1):
        if "name" not in dec or "stl" not in dec:
            raise ValueError(
                f"{path}: decoration #{i} must have 'name' and 'stl'")
        color = tuple(dec.get("color", (0.5, 0.5, 0.5)))
        decoration.append(Decoration(
            name=str(dec["name"]),
            stl=_resolve_stl(dec["stl"], base_dir),
            color=color,
            opacity=float(dec.get("opacity", 0.30)),
        ))

    return GeometryConfig(
        grid=grid,
        electrodes=electrodes,
        dielectrics=dielectrics,
        decoration=decoration,
        source_path=path,
    )


# ── Experiment ────────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    particle: dict[str, Any]
    particles: dict[str, Any]
    physics: list[Any]
    integrator: dict[str, Any]
    main_schedule: dict[str, Any]
    triggers: list[dict[str, Any]]
    source_path: str = ""
    module: Any = None             # the imported module, for advanced access


def _import_path(path: str, module_name: str = "_trapsim_user_experiment"):
    """Import a Python file by path without polluting sys.path."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    # Register so that classes defined in it pickle correctly (needed when
    # passed across multiprocessing.Pool workers — but only matters under
    # spawn, not fork; we register anyway so behaviour is identical).
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_experiment(path: str, geometry: GeometryConfig | None = None
                    ) -> ExperimentConfig:
    """Import the user's experiment.py and validate against `geometry`.

    If `geometry` is given, the loader checks that:
      - every electrode name in main_schedule.dc / main_schedule.rf exists in
        geometry.electrodes
      - every electrode name in each trigger.schedule.dc / .rf exists
      - every trigger.axis is one of {x, y, z}
    """
    path = os.path.abspath(path)
    mod = _import_path(path)

    required = ("particle", "particles", "physics", "integrator",
                "main_schedule", "triggers")
    for name in required:
        if not hasattr(mod, name):
            raise AttributeError(f"{path}: missing required name {name!r}")

    cfg = ExperimentConfig(
        particle=mod.particle,
        particles=mod.particles,
        physics=list(mod.physics),
        integrator=mod.integrator,
        main_schedule=mod.main_schedule,
        triggers=list(mod.triggers),
        source_path=path,
        module=mod,
    )

    if geometry is not None:
        _validate_against_geometry(cfg, geometry, path)

    return cfg


def _validate_against_geometry(exp: ExperimentConfig,
                               geo: GeometryConfig,
                               source: str) -> None:
    valid_names = set(geo.electrode_names())

    def check_block(block, where):
        for kind in ("dc", "rf"):
            for elec_name in (block.get(kind) or {}).keys():
                if elec_name not in valid_names:
                    raise ValueError(
                        f"{source}: {where} references electrode "
                        f"{elec_name!r}, not defined in "
                        f"{geo.source_path}.  Known: {sorted(valid_names)}")

    if "time_us" not in exp.main_schedule:
        raise ValueError(f"{source}: main_schedule.time_us is missing")
    check_block(exp.main_schedule, "main_schedule")

    for i, trig in enumerate(exp.triggers, start=1):
        for required in ("name", "axis", "threshold_mm", "schedule"):
            if required not in trig:
                raise ValueError(
                    f"{source}: trigger #{i} missing {required!r}")
        if trig["axis"] not in ("x", "y", "z"):
            raise ValueError(
                f"{source}: trigger #{i} ({trig['name']}) axis must be "
                f"one of x/y/z; got {trig['axis']!r}")
        if "time_us" not in trig["schedule"]:
            raise ValueError(
                f"{source}: trigger #{i} ({trig['name']}) schedule missing "
                f"'time_us'")
        check_block(trig["schedule"], f"trigger {trig['name']!r}")
