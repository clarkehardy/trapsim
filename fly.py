"""trapsim.fly  –  Particle integrator (geometry-agnostic).

Replaces the legacy fly.py.  Reads geometry.yaml and experiment.py, loads
PA files for every electrode, and integrates each particle's trajectory
using Dormand-Prince RK4/5 with the physics list, exact damping, and
stochastic kicks defined in experiment.py.

Trajectory output: <out_dir>/trajectories_<run>.csv  (Fusion-world mm).
Schedule snapshot: <out_dir>/schedule_<run>.json     (for animate.py).
"""

from __future__ import annotations

import argparse
import math
import multiprocessing
import os
import sys
import time
from typing import Any

import numpy as np

from .config import GeometryConfig, ExperimentConfig, load_geometry, load_experiment
from .io.pa import load_phi_stack
from .io.trajectory import write_trajectories
from .io.schedule_io import write_schedule_snapshot
from .schedule import Schedule

# ── Physical constants ──────────────────────────────────────────────────────
E_C = 1.602176634e-19   # Coulombs per elementary charge

# ── Worker globals (set in main() before fork; inherited copy-on-write) ─────
_W_phi_stack: np.ndarray | None = None
_W_grid: dict[str, Any] | None  = None
_W_geometry: GeometryConfig | None = None
_W_experiment: ExperimentConfig | None = None
_W_schedule: Schedule | None    = None


# ── Trilinear φ-gradient (vectorised over all electrodes) ───────────────────
def _phi_grad_all(phi_stack, NX, NY, NZ, dx, px, py, pz):
    """Return (Ex_per_elec, Ey_per_elec, Ez_per_elec), each shape (N_elec,).

    px, py, pz are grid-index coords (i.e. (world - world_offset) / dx).
    """
    fx = max(0.0, min(px, NX - 1.0001))
    fy = max(0.0, min(py, NY - 1.0001))
    fz = max(0.0, min(pz, NZ - 1.0001))
    i0 = int(fx);  wx = fx - i0;  i0 = min(i0, NX - 2)
    j0 = int(fy);  wy = fy - j0;  j0 = min(j0, NY - 2)
    k0 = int(fz);  wz = fz - k0;  k0 = min(k0, NZ - 2)
    ox, oy, oz = 1.0 - wx, 1.0 - wy, 1.0 - wz

    c000 = phi_stack[:, k0,   j0,   i0  ]
    c100 = phi_stack[:, k0,   j0,   i0+1]
    c010 = phi_stack[:, k0,   j0+1, i0  ]
    c110 = phi_stack[:, k0,   j0+1, i0+1]
    c001 = phi_stack[:, k0+1, j0,   i0  ]
    c101 = phi_stack[:, k0+1, j0,   i0+1]
    c011 = phi_stack[:, k0+1, j0+1, i0  ]
    c111 = phi_stack[:, k0+1, j0+1, i0+1]

    dphi_dx = (oy*oz*(c100-c000) + wy*oz*(c110-c010) +
               oy*wz*(c101-c001) + wy*wz*(c111-c011)) / dx
    dphi_dy = (ox*oz*(c010-c000) + wx*oz*(c110-c100) +
               ox*wz*(c011-c001) + wx*wz*(c111-c101)) / dx
    dphi_dz = (ox*oy*(c001-c000) + wx*oy*(c101-c100) +
               ox*wy*(c011-c010) + wx*wy*(c111-c110)) / dx
    # E = -∇φ. Returned in axis order (x, y, z) for natural use.
    return -dphi_dx, -dphi_dy, -dphi_dz


# ── Per-worker environment (passed to every Physics call) ───────────────────
class _Env:
    """Lightweight context object handed to each physics module each step."""
    __slots__ = ("particle", "voltages", "trigger_state",
                 "total_damping_rate", "rng", "_phi_stack", "_grid",
                 "_world_offset", "_voltage_vec_buf", "_electrode_names")

    def __init__(self, particle, electrode_names, phi_stack, grid_dict,
                 world_offset, rng):
        self.particle      = particle
        self.voltages      = {}
        self.trigger_state = {}
        self.total_damping_rate = 0.0
        self.rng           = rng
        self._phi_stack    = phi_stack
        self._grid         = grid_dict
        self._world_offset = world_offset
        self._electrode_names = electrode_names
        self._voltage_vec_buf = np.zeros(len(electrode_names))

    def _voltage_vec(self):
        # Mutate the persistent buffer so we don't allocate per field-eval.
        for i, name in enumerate(self._electrode_names):
            self._voltage_vec_buf[i] = self.voltages[name]
        return self._voltage_vec_buf

    def field(self, pos_world_mm):
        NX = self._grid["NX"]; NY = self._grid["NY"]; NZ = self._grid["NZ"]
        dx = self._grid["dx"]
        wox, woy, woz = self._world_offset
        px = (pos_world_mm[0] - wox) / dx
        py = (pos_world_mm[1] - woy) / dx
        pz = (pos_world_mm[2] - woz) / dx
        ex_per, ey_per, ez_per = _phi_grad_all(
            self._phi_stack, NX, NY, NZ, dx, px, py, pz)
        v = self._voltage_vec()
        return (float(np.dot(v, ex_per)),
                float(np.dot(v, ey_per)),
                float(np.dot(v, ez_per)))


# ── Integrator ──────────────────────────────────────────────────────────────
def _rhs(t_us, y, env, physics):
    """dy/dt = [vx, vy, vz, ax, ay, az].  Sums accel from every physics."""
    pos = y[:3]
    vel = y[3:6]
    a = np.zeros(3)
    for p in physics:
        a += p.accel(t_us, pos, vel, env)
    return np.array([vel[0], vel[1], vel[2], a[0], a[1], a[2]])


def integrate_particle(ion_id, y0, t_start, t_max_us, intg, physics, schedule,
                       env, world_offset_mm, grid_shape, grid_dx,
                       record_stride, v_stop, verbose=True):
    """Run one Dormand-Prince RK4/5 integration with split-operator damping."""
    # Dormand-Prince coefficients
    a21 = 1/5
    a31, a32 = 3/40, 9/40
    a41, a42, a43 = 44/45, -56/15, 32/9
    a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    b1, b3, b4, b5, b6 = 35/384, 500/1113, 125/192, -2187/6784, 11/84
    e1, e3, e4, e5, e6, e7 = 71/57600, -71/16695, 71/1920, -17253/339200, 22/525, -1/40

    NX, NY, NZ = grid_shape
    dx = grid_dx
    atol, rtol = intg["atol"], intg["rtol"]
    dt_min, dt_max = intg["dt_min_us"], intg["dt_max_us"]
    dt = intg["dt_init_us"]

    y = y0.copy()
    t = float(t_start)
    rows = []
    n_acc = n_rej = step_counter = consec_rej = 0
    reason = "max_time"
    t_wall_start = time.perf_counter()
    t_wall_last  = t_wall_start

    def _refresh_voltages(t_now):
        env.voltages = schedule.evaluate(t_now, env.trigger_state)
        env.total_damping_rate = sum(p.damping_rate(t_now, y[:3], y[3:6], env)
                                     for p in physics)

    def _record(t_now, y_now):
        rows.append(f"{ion_id},{t_now:.4f},{y_now[0]:.5f},{y_now[1]:.5f},{y_now[2]:.5f}")

    def _outside_grid(pos_world):
        wox, woy, woz = world_offset_mm
        return (pos_world[0] < wox or pos_world[0] > wox + (NX - 1) * dx or
                pos_world[1] < woy or pos_world[1] > woy + (NY - 1) * dx or
                pos_world[2] < woz or pos_world[2] > woz + (NZ - 1) * dx)

    # FSAL: compute k1 once
    _refresh_voltages(t)
    k1 = _rhs(t, y, env, physics)

    if record_stride > 0:
        _record(t, y)

    while t < t_max_us:
        dt = min(dt, t_max_us - t)
        if dt < dt_min:
            dt = dt_min

        # ── Stages 2..6 ─────────────────────────────────────────────────
        y2 = y + dt * (a21 * k1)
        _refresh_voltages(t + dt/5)
        k2 = _rhs(t + dt/5, y2, env, physics)

        y3 = y + dt * (a31*k1 + a32*k2)
        _refresh_voltages(t + 3*dt/10)
        k3 = _rhs(t + 3*dt/10, y3, env, physics)

        y4 = y + dt * (a41*k1 + a42*k2 + a43*k3)
        _refresh_voltages(t + 4*dt/5)
        k4 = _rhs(t + 4*dt/5, y4, env, physics)

        y5 = y + dt * (a51*k1 + a52*k2 + a53*k3 + a54*k4)
        _refresh_voltages(t + 8*dt/9)
        k5 = _rhs(t + 8*dt/9, y5, env, physics)

        y6 = y + dt * (a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5)
        _refresh_voltages(t + dt)
        k6 = _rhs(t + dt, y6, env, physics)

        # 5th-order solution
        y_new = y + dt * (b1*k1 + b3*k3 + b4*k4 + b5*k5 + b6*k6)

        # k7 (FSAL) + error estimate
        _refresh_voltages(t + dt)
        k7 = _rhs(t + dt, y_new, env, physics)
        err_vec = dt * (e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6 + e7*k7)
        sc = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = float(np.max(np.abs(err_vec) / sc))

        if err_norm <= 1.0:
            # ── Accept ───────────────────────────────────────────────────
            t += dt
            y  = y_new
            k1 = k7      # FSAL
            n_acc += 1
            step_counter += 1
            consec_rej = 0

            # Progress
            if verbose:
                now = time.perf_counter()
                if now - t_wall_last >= 10.0:
                    t_wall_last = now
                    print(f"  [ion {ion_id}] t={t:.0f}/{t_max_us:.0f} µs  "
                          f"pos=({y[0]:.1f}, {y[1]:.1f}, {y[2]:.1f})  "
                          f"dt={dt:.2g} µs  acc={n_acc} rej={n_rej}",
                          flush=True)

            # ── Triggers ─────────────────────────────────────────────────
            newly = schedule.check_triggers(t, y[:3], env.trigger_state)
            for name in newly:
                print(f"  [ion {ion_id}] trigger {name!r} fired at "
                      f"t={t:.1f} µs, pos=({y[0]:.2f}, {y[1]:.2f}, {y[2]:.2f})",
                      flush=True)
            if newly:
                _refresh_voltages(t)
                k1 = _rhs(t, y, env, physics)

            # ── Damping (exact factor) + stochastic kicks ────────────────
            gamma_total = env.total_damping_rate
            if gamma_total * dt > 1e-12:
                decay = math.exp(-gamma_total * dt)
                y[3] *= decay
                y[4] *= decay
                y[5] *= decay
            for p in physics:
                dv = p.kick(dt, t, y[:3], y[3:6], env)
                y[3] += dv[0]; y[4] += dv[1]; y[5] += dv[2]

            # ── Boundary + termination checks ────────────────────────────
            if _outside_grid(y[:3]):
                reason = "lost"
                if record_stride > 0:
                    _record(t, y)
                break
            speed = math.sqrt(y[3]**2 + y[4]**2 + y[5]**2)
            if v_stop > 0 and speed < v_stop:
                reason = "trapped"
                if record_stride > 0:
                    _record(t, y)
                break

            # ── Record ───────────────────────────────────────────────────
            if record_stride > 0 and step_counter % record_stride == 0:
                _record(t, y)

            # ── Adapt dt ─────────────────────────────────────────────────
            factor = 0.9 * err_norm**(-0.2) if err_norm > 1e-10 else 5.0
            dt = min(dt * min(5.0, factor), dt_max)
            dt = max(dt, dt_min)

        else:
            # ── Reject ───────────────────────────────────────────────────
            factor = max(0.1, 0.9 * err_norm**(-0.2))
            dt = max(dt * factor, dt_min)
            n_rej     += 1
            consec_rej += 1
            if consec_rej >= 200_000:
                print(f"  [ion {ion_id}] WARNING stuck at t={t:.2f} µs — "
                      f"forced Euler step", flush=True)
                t += dt_min
                y  = y + dt_min * k1
                _refresh_voltages(t)
                k1 = _rhs(t, y, env, physics)
                consec_rej = 0

    return rows, {
        "steps_accepted": n_acc,
        "steps_rejected": n_rej,
        "t_sim_us":       t,
        "reason":         reason,
    }


# ── Worker ──────────────────────────────────────────────────────────────────
def _worker(args):
    ion_id, y0, t_start, t_max_us, integrator, world_off, particle, rng_seed = args
    schedule  = _W_schedule
    physics   = _W_experiment.physics
    grid      = _W_grid
    grid_shape = (grid["NX"], grid["NY"], grid["NZ"])
    grid_dx    = grid["dx"]
    rng        = np.random.default_rng(rng_seed)

    env = _Env(particle, _W_geometry.electrode_names(), _W_phi_stack, grid,
               world_off, rng)
    env.trigger_state = schedule.initial_trigger_state()

    record_stride = integrator.get("record_stride", 20)
    v_stop        = integrator.get("v_stop_mm_us", 1e-6)

    print(f"  [ion {ion_id}] start at pos=({y0[0]:.2f}, {y0[1]:.2f}, {y0[2]:.2f}) mm, "
          f"vel=({y0[3]:.3g}, {y0[4]:.3g}, {y0[5]:.3g}) mm/µs", flush=True)
    t0 = time.perf_counter()
    rows, info = integrate_particle(
        ion_id, y0, t_start, t_max_us, integrator, physics, schedule, env,
        world_off, grid_shape, grid_dx, record_stride, v_stop, verbose=True)
    elapsed = time.perf_counter() - t0
    print(f"  [ion {ion_id}] done: {info['steps_accepted']} acc / "
          f"{info['steps_rejected']} rej steps, t_sim={info['t_sim_us']:.1f} µs, "
          f"reason={info['reason']}, wall={elapsed:.1f} s", flush=True)
    return ion_id, rows, info


# ── Particle starts ─────────────────────────────────────────────────────────
def _build_ion_starts(particles_cfg, mass_kg, charge_C, master_seed=42):
    starts = particles_cfg.get("starts", [])
    if not starts:
        raise ValueError("experiment.py: particles.starts is empty")
    n = int(particles_cfg.get("n", 1))
    rng = np.random.default_rng(master_seed)

    ions = []
    for i in range(n):
        s = starts[i % len(starts)]
        pos = np.array(s["position_mm"], dtype=float)
        sig = np.array(s.get("sigma_mm", [0, 0, 0]), dtype=float)
        pos = pos + sig * rng.standard_normal(3)

        ke_ev = float(s.get("ke_ev", 0.0))
        if ke_ev == 0.0:
            vel = np.zeros(3)
        else:
            v_mag = math.sqrt(2.0 * ke_ev * E_C / mass_kg) * 1e-3  # mm/µs
            direction = np.array(s.get("direction", [0, 0, 1]), dtype=float)
            direction = direction / np.linalg.norm(direction)
            vel = v_mag * direction

        y0 = np.concatenate([pos, vel])
        ions.append((i + 1, y0))
    return ions


# ── Top-level orchestrator ──────────────────────────────────────────────────
def fly(geometry: GeometryConfig, experiment: ExperimentConfig, *,
        base_dir: str,
        run_number: int = 1,
        workers: int | None = None) -> dict:
    """Run the full fly pipeline and write outputs.  Returns a summary dict."""
    global _W_phi_stack, _W_grid, _W_geometry, _W_experiment, _W_schedule

    n_workers = workers or os.cpu_count() or 1
    print(f"━━━ trapsim.fly  run={run_number}  workers={n_workers} ━━━\n")

    # Derived particle properties
    r_m     = float(experiment.particle["radius_m"])
    rho     = float(experiment.particle["density_kgm3"])
    q_e     = float(experiment.particle["charge_e"])
    mass_kg = (4.0/3.0) * math.pi * r_m**3 * rho
    charge_C = q_e * E_C
    particle = {
        "mass_kg":  mass_kg,
        "charge_C": charge_C,
        "radius_m": r_m,
        "charge_e": q_e,
    }
    print(f"Particle: r={r_m*1e9:.0f} nm  ρ={rho} kg/m³  m={mass_kg:.3e} kg  "
          f"q={q_e}e = {charge_C:.3e} C")

    # Schedule
    schedule = Schedule(experiment.main_schedule, experiment.triggers,
                        geometry.electrode_names())
    main_t = experiment.main_schedule["time_us"]
    t_max_us = float(np.asarray(main_t)[-1])
    print(f"Simulation time: 0 → {t_max_us:.0f} µs")

    # Physics description
    print("Physics:")
    for p in experiment.physics:
        print(f"  - {p.__class__.__name__}")

    # PA files
    print(f"\nLoading PA files from {base_dir} …")
    t0 = time.perf_counter()
    phi_stack, grid_dict = load_phi_stack(geometry, base_dir, verbose=True)
    print(f"  PA stack: shape {phi_stack.shape}  "
          f"({phi_stack.nbytes/1e6:.0f} MB)  in {time.perf_counter()-t0:.1f} s")

    # Set worker globals BEFORE forking
    _W_phi_stack = phi_stack
    _W_grid      = grid_dict
    _W_geometry  = geometry
    _W_experiment = experiment
    _W_schedule  = schedule

    # Build particle args
    ions = _build_ion_starts(experiment.particles, mass_kg, charge_C)
    print(f"\nPreparing {len(ions)} ions …")
    world_off = geometry.grid.world_offset_mm
    intg = experiment.integrator
    task_args = [
        (ion_id, y0, 0.0, t_max_us, intg, world_off, particle, ion_id)
        for ion_id, y0 in ions
    ]

    # Fork-based pool: workers inherit phi_stack via copy-on-write
    print(f"\nRunning {len(task_args)} particles on {n_workers} workers …")
    t0 = time.perf_counter()
    rows_per_ion = {}
    summaries    = {}
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(processes=n_workers) as pool:
        for ion_id, rows, info in pool.imap_unordered(_worker, task_args):
            rows_per_ion[ion_id] = rows
            summaries[ion_id]    = info
    elapsed = time.perf_counter() - t0
    print(f"\nAll particles done in {elapsed:.1f} s wall")

    # Write outputs
    traj_path = os.path.join(base_dir, f"trajectories_{run_number}.csv")
    n_rows = write_trajectories(traj_path, rows_per_ion)
    print(f"Wrote {n_rows} rows to {traj_path}")

    sched_path = os.path.join(base_dir, f"schedule_{run_number}.json")
    write_schedule_snapshot(sched_path,
                            experiment.main_schedule,
                            experiment.triggers)
    print(f"Wrote schedule snapshot to {sched_path}")

    return {
        "n_particles":  len(ions),
        "wall_seconds": elapsed,
        "trajectory":   traj_path,
        "schedule":     sched_path,
        "summaries":    summaries,
    }


def main():
    ap = argparse.ArgumentParser(description="trapsim particle integrator.")
    cwd = os.getcwd()
    ap.add_argument("--geometry",   default=os.path.join(cwd, "geometry.yaml"))
    ap.add_argument("--experiment", default=os.path.join(cwd, "experiment.py"))
    ap.add_argument("--run",        type=int, default=1)
    ap.add_argument("--workers",    type=int, default=None)
    ap.add_argument("--base-dir",   default=cwd,
                    help="Directory for paulTrap.pa* and output files")
    args = ap.parse_args()

    geo = load_geometry(args.geometry)
    exp = load_experiment(args.experiment, geo)
    fly(geo, exp,
        base_dir=args.base_dir,
        run_number=args.run,
        workers=args.workers)


if __name__ == "__main__":
    main()
