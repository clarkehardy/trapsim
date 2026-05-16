"""trapsim.viz.plot_field  –  2-D cross-section of |E| or potential.

Reads geometry.yaml, the PA files for every electrode, and a schedule
snapshot.  Renders a cross-section of the total potential or field
magnitude at a chosen time.

Usage:
    python -m trapsim.viz.plot_field
    python -m trapsim.viz.plot_field --slice y=19 --time 150000
    python -m trapsim.viz.plot_field --quantity E --slice z=276
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from ..config import load_geometry
from ..io.pa import load_phi_stack
from ..io.schedule_io import read_schedule_snapshot
from ..schedule import Schedule


def _parse_slice(s: str):
    """'y=19'  →  ('y', 19.0)."""
    if "=" not in s:
        sys.exit("--slice must look like 'y=19'")
    a, v = s.split("=", 1)
    a = a.strip().lower()
    if a not in ("x", "y", "z"):
        sys.exit(f"--slice axis must be x/y/z; got {a!r}")
    return a, float(v)


def main():
    cwd = os.getcwd()
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default=os.path.join(cwd, "geometry.yaml"))
    ap.add_argument("--schedule", default=os.path.join(cwd, "schedule_1.json"),
                    help="schedule_N.json; --time is interpreted against this.")
    ap.add_argument("--pa-dir",   default=cwd,
                    help="Directory containing paulTrap.pa* files")
    ap.add_argument("--time",     type=float, default=0.0,
                    help="Time (µs) at which to resolve voltages; default 0.")
    ap.add_argument("--slice",    default="y=19",
                    help="Cross-section plane, e.g. 'y=19' or 'z=276'")
    ap.add_argument("--quantity", default="E", choices=("phi", "E"),
                    help="phi = total potential, E = |electric field|")
    ap.add_argument("--cmap",     default="viridis")
    ap.add_argument("--save",     default=None)
    args = ap.parse_args()

    geo = load_geometry(args.geometry)
    print(f"Loading PA files for {geo.n_electrodes} electrodes …")
    phi_stack, grid = load_phi_stack(geo, args.pa_dir, verbose=False)
    NX, NY, NZ = grid["NX"], grid["NY"], grid["NZ"]
    dx = grid["dx"]
    wox, woy, woz = geo.grid.world_offset_mm

    # Resolve voltages
    if os.path.exists(args.schedule):
        snap = read_schedule_snapshot(args.schedule)
        sched = Schedule(snap["main"], snap["triggers"], geo.electrode_names())
        ts = {t["name"]: None for t in snap["triggers"]}
        # If --time is past a trigger's threshold-time, we don't know the
        # fire time without a trajectory.  Use the main schedule only.
        voltages = sched.evaluate(args.time, ts)
    else:
        print(f"  [warn] {args.schedule} not found; using 1 V per electrode.")
        voltages = {e.name: 1.0 for e in geo.electrodes}

    # Build total potential
    print(f"Computing total φ at t={args.time} µs …")
    v_vec = np.array([voltages[e.name] for e in geo.electrodes])
    phi_total = np.tensordot(v_vec, phi_stack, axes=([0], [0]))  # (NZ, NY, NX)

    # Slice
    axis, val = _parse_slice(args.slice)
    if axis == "x":
        idx = int(round((val - wox) / dx))
        idx = max(0, min(idx, NX - 1))
        slab = phi_total[:, :, idx]                 # (NZ, NY)
        h_label, v_label = "Y (mm)", "Z (mm)"
        h_axis = np.arange(NY) * dx + woy
        v_axis = np.arange(NZ) * dx + woz
        slab_disp = slab     # already (NZ, NY) → imshow shows Y across, Z down (transposed below)
    elif axis == "y":
        idx = int(round((val - woy) / dx))
        idx = max(0, min(idx, NY - 1))
        slab = phi_total[:, idx, :]                 # (NZ, NX)
        h_label, v_label = "X (mm)", "Z (mm)"
        h_axis = np.arange(NX) * dx + wox
        v_axis = np.arange(NZ) * dx + woz
        slab_disp = slab
    else:  # z
        idx = int(round((val - woz) / dx))
        idx = max(0, min(idx, NZ - 1))
        slab = phi_total[idx, :, :]                 # (NY, NX)
        h_label, v_label = "X (mm)", "Y (mm)"
        h_axis = np.arange(NX) * dx + wox
        v_axis = np.arange(NY) * dx + woy
        slab_disp = slab

    if args.quantity == "phi":
        Z = slab_disp
        title = f"φ at {axis}={val:.1f} mm, t={args.time:.0f} µs"
        cbar_label = "Potential (V)"
    else:
        # |E| via central differences in the slice plane
        gy, gx = np.gradient(slab_disp, dx, dx)
        Z = np.sqrt(gx**2 + gy**2)
        title = f"|E| at {axis}={val:.1f} mm, t={args.time:.0f} µs"
        cbar_label = "|E| (V/mm)"

    fig, ax = plt.subplots(figsize=(11, 7))
    extent = (h_axis[0], h_axis[-1], v_axis[0], v_axis[-1])
    im = ax.imshow(Z, origin="lower", extent=extent, cmap=args.cmap,
                   aspect="auto")
    ax.set_xlabel(h_label)
    ax.set_ylabel(v_label)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=130)
        print(f"Saved: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
