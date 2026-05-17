# trapsim

Geometry-agnostic particle integrator for electrode-trap simulations. Pure-Python pipeline (C++ Laplace solver + Python integrator) driven by a single `geometry.yaml` and an `experiment.py`. Designed to replace SIMION-based workflows on macOS/Linux.

The default example geometry — a 400 mm RF guide loading an optical Paul trap across a gate valve — is available as a reference at [github.com/clarkehardy/rf-guide-sim](https://github.com/clarkehardy/rf-guide-sim).

---

## Install

```
pip install "trapsim[viz] @ git+https://github.com/clarkehardy/trapsim.git"
```

For development:

```
git clone https://github.com/clarkehardy/trapsim ~/Code/trapsim
cd ~/Code/trapsim
pip install -e .[dev,all]
```

Extras:

| Extra | Brings in | When you need it |
|-------|-----------|------------------|
| `[refine]` | `trimesh` | only if calling `trapsim.voxelize` (STL → voxel masks) |
| `[viz]` | `matplotlib`, `pyvista` | only if calling `trapsim.viz.*` |
| `[all]` | both of the above | usually fine to install upfront |
| `[dev]` | `pytest`, `pytest-cov` | running the test suite |

The C++ Laplace solver compiles on first `trapsim.refine` call directly from the bundled package source (needs Xcode CLT on macOS or `build-essential` on Linux). The binary is written to `<CWD>/solver/laplace`. To hack the SOR loop without forking the package, copy `laplace.cpp` from `site-packages/trapsim/_solver/` into your project's `solver/` folder — the build picks up a local source preferentially. Override the build via `CXX`, `CXXFLAGS`, `LDFLAGS` env vars.

---

## Quickstart

The recommended layout is **one folder per simulation**. Each simulation directory holds its own `geometry.yaml` (electrodes and dielectrics), `experiment.py` (particles, schedule, triggers, physics), STL files, and a thin `run.py` shim:

```
~/sims/paul_trap/
  geometry.yaml
  experiment.py
  run.py
  stl/
    plate_top.stl
    plate_bottom.stl
```

The `run.py` shim is three lines:

```python
# run.py
from trapsim.run import main
if __name__ == "__main__":
    main()
```

Then from inside the simulation folder:

```
python run.py                 # refine if needed → fly → animate → visualize
python run.py --run 2         # writes trajectories_2.csv, schedule_2.json
python run.py --no-animate --no-visualize
```

All output (`field.pa*`, `trajectories_<N>.csv`, `schedule_<N>.json`, `solver/`) is written into the simulation folder, so you can have as many independent simulations as you want without collisions — they don't share anything.

Minimal `geometry.yaml`:

```yaml
grid:
  dx_mm: 0.5
  bounds_mm:
    x: [-10.0, 10.0]
    y: [-10.0, 10.0]
    z: [-20.0, 20.0]

electrodes:
  - name: plate_top
    stls: [stl/plate_top.stl]
  - name: plate_bottom
    stls: [stl/plate_bottom.stl]
```

Minimal `experiment.py`:

```python
import numpy as np
from trapsim.physics import Electrostatic, Gravity

particle  = {"radius_m": 83e-9, "density_kgm3": 2200, "charge_e": 100}
particles = {"n": 1,
             "starts": [{"position_mm": [0, 0, -10], "ke_ev": 0,
                         "direction": [0, 0, 1], "sigma_mm": [0, 0, 0]}]}
physics = [Electrostatic(), Gravity()]
integrator = {"dt_init_us": 1.0, "dt_min_us": 0.01, "dt_max_us": 25.0,
              "atol": 1e-3, "rtol": 1e-4,
              "v_stop_mm_us": 1e-6, "record_stride": 20}
t = np.linspace(0, 1e5, 500)
main_schedule = {"time_us": t, "dc": {"plate_top":  10*np.ones_like(t),
                                       "plate_bottom": -10*np.ones_like(t)}}
triggers = []
```

`pip install "trapsim[viz] @ git+..."`, drop in your STL files, then `python run.py`. The solver, voxelizer, integrator, animator, and 3D viewer all work without further setup.

---

## `geometry.yaml` schema

```yaml
grid:
  dx_mm: 0.5
  bounds_mm:
    x: [xmin, xmax]                  # Fusion-world mm
    y: [ymin, ymax]
    z: [zmin, zmax]

electrodes:                          # one entry per independent voltage source
  - name: <unique name>              # referenced from the schedule and physics
    stls: [path1.stl, path2.stl]     # all STLs listed get the same voltage
    color: [r, g, b]                 # optional, 0..1, for visualizations
    opacity: 0.40                    # optional

dielectrics:                         # bodies that distort the Laplace solve
  - name: <unique name>
    stl: path.stl
    epsilon_r: 3.0

decoration:                          # drawn by visualize.py, no field contribution
  - name: <unique name>
    stl: path.stl
    color: [r, g, b]
```

Each electrode is assigned an integer `electrode_id` (1, 2, …) in declaration order. STL paths are resolved against the YAML's directory, then the repo root, then `stl/` — so `rod.stl` and `stl/rod.stl` both work.

Output files (all written into the simulation folder):

| File | Path |
|------|------|
| Potential array (per electrode) | `field.pa<electrode_id>` |
| Trajectories (per run) | `trajectories_<run>.csv` |
| Schedule snapshot (per run) | `schedule_<run>.json` |
| Solver work dir (masks, ε, grid, binary) | `solver/` |

---

## Exporting STL files from CAD

Each rigid body that you want as an independent voltage source, dielectric, or decoration needs its own binary-STL export. Bodies wired together (e.g. four rods on the same RF supply) get listed under the same electrode `name` in `geometry.yaml`; the voxelizer takes the union of their meshes.

The simulation volume (`grid.bounds_mm`) must enclose every body. The grid spacing (`grid.dx_mm`) sets the accuracy/memory tradeoff — at 0.5 mm a 130×90×850 grid uses ~80 MB per electrode.

### Autodesk Fusion

1. In the canvas, right-click the component → **Find in Browser**.
2. Expand to the **Body**, right-click → **Isolate**.
3. Right-click the top-level assembly → **Save As Mesh**.
4. **Format:** STL (Binary), **Unit Type:** Millimeter, **Structure:** One File, **Refinement:** High.
5. Save to `stl/<body_name>.stl`.

### SolidWorks

1. Open the assembly file in SolidWorks.
2. Select the part(s) you want to export.
3. Right-click → **Invert Selection**.
4. Right-click any of the inverted selection → **Hide Components**.
5. **File → Save As → Save as type: STL → Options**:
   - **Unit:** Millimeters
   - **Do not translate STL output data to positive space** (keeps the part in assembly coordinates)
   - **Save all components of an assembly in a single file**
   - → **OK → Save**
6. Undo **Hide Components**.
7. Repeat for each independent body.

Both workflows give you binary STL files in assembly-world mm coordinates — that's what `geometry.yaml`'s `bounds_mm` should be expressed in, too.

---

## `experiment.py` shape

Edit the six blocks below for a new run; all electrode names must match `geometry.yaml`.

```python
import numpy as np
from trapsim.physics import Electrostatic, Gravity, EpsteinDrag, Langevin

particle  = {"radius_m": ..., "density_kgm3": ..., "charge_e": ...}
particles = {"n": N,
             "starts": [{"position_mm": [x, y, z], "ke_ev": ...,
                         "direction": [...], "sigma_mm": [...]}]}

physics = [Electrostatic(), Gravity(),
           EpsteinDrag(pressure_pa=..., temperature_k=..., gas_mass_amu=28.0),
           Langevin(temperature_k=...)]

integrator = {"dt_init_us": ..., "dt_min_us": ..., "dt_max_us": ...,
              "atol": ..., "rtol": ...,
              "v_stop_mm_us": ..., "record_stride": ...}

main_schedule = {
    "time_us": np.linspace(...),
    "dc": {electrode_name: voltage_array, ...},
    "rf": {electrode_name: {"amplitude": ..., "frequency_hz": ..., "phase_deg": ...}, ...},
}

triggers = [
    {"name": <unique>, "axis": "z", "threshold_mm": ...,
     "schedule": {"time_us": ..., "dc": {...}, "rf": {...}}},
    ...
]
```

Triggers fire when `pos[axis] >= threshold`. The trigger's `schedule` then overrides only the listed electrodes from `t_fire` onward — every other electrode keeps following `main_schedule`. Each trigger has its own time axis (measured from the trigger's fire time).

---

## Custom physics modules

A physics module overrides any subset of three hooks. Drop it into `experiment.py`'s `physics = [...]` list — no registration needed.

```python
from trapsim.physics import Physics
import numpy as np

class HarmonicAxialTrap(Physics):
    def __init__(self, omega_us, z0_mm):
        self.omega2 = omega_us ** 2
        self.z0 = z0_mm

    def accel(self, t_us, pos_mm, vel_mm_us, env):
        return np.array([0, 0, -self.omega2 * (pos_mm[2] - self.z0)])
```

The hooks:

| Hook | Returns | When |
|------|---------|------|
| `accel(t, pos, vel, env)`         | 3-vec acceleration [mm/µs²] | every RK4/5 stage |
| `damping_rate(t, pos, vel, env)`  | scalar γ [1/µs]             | once per accepted step (summed) |
| `kick(dt, t, pos, vel, env)`      | 3-vec Δv [mm/µs]            | once per accepted step          |

`env` exposes:
- `env.particle` — dict with `mass_kg`, `charge_C`, `radius_m`, `charge_e`
- `env.voltages` — `{electrode_name: V}` at the current time
- `env.field(pos_mm)` — total `(Ex, Ey, Ez)` in V/mm at `pos_mm` (Fusion world)
- `env.trigger_state` — `{trigger_name: t_fire_µs or None}` for this particle
- `env.total_damping_rate` — γ summed across all physics modules (used by Langevin)
- `env.rng` — `numpy.random.Generator` seeded per particle

The integrator special-cases `damping_rate`: it sums all contributions and applies them via the exact factor `v ← exp(−γ·dt)·v` after each accepted step. This is more accurate than computing `accel = −γv` for large dt.

Built-in classes (in `trapsim.physics`):

- `Electrostatic()` — `q·E/m` from `env.field`
- `Gravity(g_mm_us2=9.81e-9, axis="-y")`
- `EpsteinDrag(pressure_pa, temperature_k, gas_mass_amu, pressure_ramp=None, scale=1.0)` — `pressure_ramp = {"trigger": "release", "p_final_pa": 100.0, "duration_us": 5e5}` ramps pressure linearly starting at the named trigger's fire time
- `Langevin(temperature_k)` — FDT noise scaled to `env.total_damping_rate`

---

## CLI reference

| Command | Equivalent | Notes |
|---------|-----------|-------|
| `python run.py` | `python -m trapsim.run` | Full pipeline orchestrator (run.py is your project's thin shim) |
| `python -m trapsim.refine` | | Voxelize + Laplace solve only |
| `python -m trapsim.fly` | | Particle integration only |
| `python -m trapsim.viz.animate` | | 2-D matplotlib animation |
| `python -m trapsim.viz.visualize` | | 3-D PyVista viewer (with flythrough) |
| `python -m trapsim.viz.plot_field` | | 2-D field cross-section |

All commands default to `./geometry.yaml`, `./experiment.py`, and write PA / trajectory / schedule files in the current directory. Pass `--help` on any of them for the full flag list.

---

## Output file formats

### `field.pa<electrode_id>` — SIMION potential array (binary)

56-byte header (flags, scale_ref, NX, NY, NZ, dx) followed by `NX·NY·NZ` float64 in `[k][j][i]` order. Electrode-surface voxels are encoded with sign-bit or `>1.5·scale_ref`. Free-space voxels store φ/scale_ref where φ is the unit-drive potential. Read via:

```python
from trapsim.io.pa import read_pa
phi, NX, NY, NZ, dx = read_pa("field.pa1")
```

### `trajectories_<N>.csv`

```
ion_id,t_us,x_mm,y_mm,z_mm
1,0.0000,0.00000,19.00000,-98.04495
1,456.0000,-0.00463,19.02678,-98.04464
…
```

Coordinates are in Fusion-world mm. Recorded every `record_stride` accepted integrator steps, plus start and end.

### `schedule_<N>.json`

```json
{
  "main": {
    "time_us": [...],
    "dc": {"plate_top": [...], ...},
    "rf": {"rf_loading": {"amplitude": [...], "frequency_hz": 2000, "phase_deg": 0}}
  },
  "triggers": [
    {"name": "release", "axis": "z", "threshold_mm": 272.0,
     "schedule": {"time_us": [...], "dc": {...}}}
  ]
}
```

A snapshot of the schedule actually used for the run. The animator reads this to plot voltage traces.

---

## Package layout

```
src/trapsim/
  __init__.py            public API: load_geometry, load_experiment, …
  config.py              YAML loader + validation
  voxelize.py            STL → voxel masks (driven by GeometryConfig)
  refine.py              orchestrates voxelize + C++ Laplace solve
  fly.py                 Dormand-Prince RK4/5 integrator + workers
  schedule.py            Schedule + trigger resolution
  run.py                 full-pipeline orchestrator (invoked via your project's run.py shim)
  physics/               pluggable physics modules
  io/                    PA, trajectory, schedule readers/writers
  viz/                   animate, visualize, plot_field
  _solver/               bundled C++ source (package data)
    laplace.cpp
tests/                   pytest unit tests
pyproject.toml
```

---

## Contributing

```
pip install -e .[dev,all]
pytest -q
```

Smoke tests cover `config.py` validation, `Schedule.evaluate` trigger semantics, and the closed-form values of `EpsteinDrag.damping_rate` and `Langevin.kick` variance. The full integrator and refine pipeline are exercised end-to-end in the [rf-guide-sim](https://github.com/clarkehardy/rf-guide-sim) example repo rather than here, since they need a real geometry and C++ toolchain.

PRs welcome.
