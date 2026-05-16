"""trapsim.viz.visualize  –  3D PyVista viewer driven by geometry.yaml.

Renders every body declared in the geometry file (electrodes + dielectrics
+ decoration) plus a trajectory CSV, colored by time-of-flight.

Usage:
    python -m trapsim.viz.visualize
    python -m trapsim.viz.visualize --geometry geometry.yaml --traj trajectories_1.csv
    python -m trapsim.viz.visualize --animation flythrough.mp4 --duration 12
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

import numpy as np

try:
    import pyvista as pv
except ImportError:
    sys.exit("PyVista not found.  Install: pip install pyvista")

from ..config import GeometryConfig, load_geometry
from ._common import auto_color, bbox_union, load_trajectories, read_stl_bbox


def add_geometry(plotter: pv.Plotter, geo: GeometryConfig,
                 show_labels: bool = True) -> None:
    """Add every electrode / dielectric / decoration body to the plotter."""
    n_elec = geo.n_electrodes
    for i, e in enumerate(geo.electrodes):
        if e.color is None:
            color = auto_color(i, n_elec)
        else:
            color = e.color
        for stl in e.stls:
            plotter.add_mesh(pv.read(stl), color=color, opacity=e.opacity,
                             smooth_shading=True,
                             label=e.name if show_labels and stl == e.stls[0] else None)
    for d in geo.dielectrics:
        plotter.add_mesh(pv.read(d.stl), color=d.color, opacity=d.opacity,
                         smooth_shading=True,
                         label=f"{d.name} (ε={d.epsilon_r})" if show_labels else None)
    for dec in geo.decoration:
        plotter.add_mesh(pv.read(dec.stl), color=dec.color, opacity=dec.opacity,
                         smooth_shading=True,
                         label=dec.name if show_labels else None)


def compute_global_bbox(geo: GeometryConfig):
    """Union STL bbox of every body in the geometry."""
    boxes = []
    for e in geo.electrodes:
        for stl in e.stls:
            boxes.append(read_stl_bbox(stl))
    for d in geo.dielectrics:
        boxes.append(read_stl_bbox(d.stl))
    for dec in geo.decoration:
        boxes.append(read_stl_bbox(dec.stl))
    return bbox_union(*boxes)


def add_trajectories(plotter: pv.Plotter, ions, cmap: str = "plasma") -> None:
    if not ions:
        print("  No trajectory data.")
        return

    t_min = min(d["t"][0]  for d in ions.values())
    t_max = max(d["t"][-1] for d in ions.values())
    clim  = [t_min, t_max]

    bar_added = False
    for ion_id, d in sorted(ions.items()):
        if len(d["t"]) < 2:
            continue
        pts = np.column_stack([d["x"], d["y"], d["z"]])
        line = pv.lines_from_points(pts)
        line["Time (µs)"] = d["t"]
        plotter.add_mesh(
            line, scalars="Time (µs)", cmap=cmap, clim=clim, line_width=2.0,
            show_scalar_bar=not bar_added,
            scalar_bar_args=dict(
                title="Time (µs)", title_font_size=14, label_font_size=12,
                width=0.55, height=0.06, position_x=0.22, position_y=0.02,
                n_labels=5, fmt="%.0f", color="black",
            ),
        )
        bar_added = True
    n_pts = sum(len(d["t"]) for d in ions.values())
    print(f"  {len(ions)} ion(s), {n_pts} points, t = {t_min:.0f} – {t_max:.0f} µs")


# ── Camera flythrough ────────────────────────────────────────────────────────
def _smoothstep(s):
    s = max(0.0, min(1.0, float(s)))
    return s * s * (3.0 - 2.0 * s)


def render_flythrough(plotter, output_path, bbox, *,
                       fps: int = 30, duration: float = 10.0,
                       orbit_radius: float = 50.0, orbit_height: float = 100.0,
                       axis: str = "auto",
                       hold_frac: float = 0.10, fly_frac: float = 0.40) -> None:
    """Generic flythrough: hold near one end of `axis`, fly along it, orbit far end.

    `bbox` is ((xlo,xhi),(ylo,yhi),(zlo,zhi)) from compute_global_bbox().
    `axis` is "x"/"y"/"z" or "auto" (longest bbox dimension).
    """
    if bbox is None:
        sys.exit("ERROR: empty bounding box — cannot compute camera path.")
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = bbox
    spans = {"x": xhi - xlo, "y": yhi - ylo, "z": zhi - zlo}
    if axis == "auto":
        axis = max(spans, key=spans.__getitem__)
    if axis not in ("x", "y", "z"):
        sys.exit(f"--axis must be x/y/z/auto; got {axis!r}")
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    transverse = [c for c in (0, 1, 2) if c != axis_idx]

    lo = (xlo, ylo, zlo)[axis_idx]
    hi = (xhi, yhi, zhi)[axis_idx]
    length = hi - lo

    # Centre of the transverse cross-section
    centre = [0.0, 0.0, 0.0]
    centre[transverse[0]] = 0.5 * ((xlo, ylo, zlo)[transverse[0]] +
                                   (xhi, yhi, zhi)[transverse[0]])
    centre[transverse[1]] = 0.5 * ((xlo, ylo, zlo)[transverse[1]] +
                                   (xhi, yhi, zhi)[transverse[1]])

    def at(axis_val, t1_off=0.0, t2_off=0.0):
        p = list(centre)
        p[axis_idx] = axis_val
        p[transverse[0]] += t1_off
        p[transverse[1]] += t2_off
        return np.array(p, dtype=float)

    # Phase 1: held wide view from behind the "low" end of the axis.
    P1_pos   = at(lo - 1.7 * length, t2_off=orbit_height * 0.7)
    P1_focal = at(lo + 0.5 * length)

    # Phase 3: orbit around the "high" end.
    P3_focal     = at(hi)
    P3_pos_start = at(hi, t1_off=orbit_radius, t2_off=orbit_height)

    # view_up = + along whichever transverse axis is closer to "y"
    view_up = [0.0, 0.0, 0.0]
    view_up[transverse[1]] = 1.0

    ext = os.path.splitext(output_path)[1].lower()
    if ext in (".mp4", ".mov", ".avi"):
        plotter.open_movie(output_path, framerate=fps, quality=7)
    elif ext == ".gif":
        plotter.open_gif(output_path, fps=fps)
    else:
        sys.exit(f"ERROR: unknown animation extension {ext!r}.  Use .mp4 or .gif.")

    n_frames = max(2, int(round(fps * duration)))
    print(f"Camera axis: {axis}  (length {length:.1f} mm)")
    print(f"Rendering {n_frames} frames at {fps} fps ({duration:.1f} s) → {output_path}")

    for frame in range(n_frames):
        t = frame / (n_frames - 1)
        if t < hold_frac:
            pos, focal = P1_pos, P1_focal
        elif t < hold_frac + fly_frac:
            s = (t - hold_frac) / fly_frac
            ss = _smoothstep(s)
            pos   = P1_pos   * (1 - ss) + P3_pos_start * ss
            focal = P1_focal * (1 - ss) + P3_focal     * ss
        else:
            s = (t - hold_frac - fly_frac) / max(1e-9, 1.0 - hold_frac - fly_frac)
            theta = 2.0 * np.pi * s
            pos = list(centre)
            pos[axis_idx] = hi
            pos[transverse[0]] += orbit_radius * np.cos(theta)
            pos[transverse[1]] += orbit_height
            # Add a radial bobble along the *other* transverse axis for visual interest
            # (kept minimal so the orbit is still clearly a rotation)
            pos = np.array(pos)
            focal = P3_focal
        plotter.camera_position = [list(pos), list(focal), view_up]
        plotter.write_frame()
        if (frame + 1) % max(1, fps) == 0:
            print(f"  frame {frame + 1}/{n_frames}")
    plotter.close()
    print(f"Saved: {output_path}")


def main():
    cwd = os.getcwd()
    ap = argparse.ArgumentParser(description="trapsim 3D viewer.")
    ap.add_argument("--geometry",   default=os.path.join(cwd, "geometry.yaml"))
    ap.add_argument("--traj",       default=os.path.join(cwd, "trajectories_1.csv"))
    ap.add_argument("--screenshot", default=None)
    ap.add_argument("--animation",  default=None,
                    help="Save cinematic flythrough to .mp4 or .gif")
    ap.add_argument("--axis",       default="auto",
                    help="Flythrough axis: x, y, z, or auto (longest dim)")
    ap.add_argument("--duration",   type=float, default=10.0)
    ap.add_argument("--fps",        type=int,   default=30)
    ap.add_argument("--orbit-radius", type=float, default=50.0)
    ap.add_argument("--orbit-height", type=float, default=100.0)
    ap.add_argument("--resolution", type=int, nargs=2, default=None,
                    metavar=("WIDTH", "HEIGHT"))
    ap.add_argument("--cmap",       default="plasma")
    args = ap.parse_args()

    geo = load_geometry(args.geometry)
    off_screen = bool(args.screenshot) or bool(args.animation)
    plotter = pv.Plotter(off_screen=off_screen, title="trapsim — 3D view")
    plotter.set_background("white")

    if args.resolution is not None:
        plotter.window_size = list(args.resolution)
    elif args.animation:
        plotter.window_size = [1920, 1080]

    print(f"Loading geometry from {args.geometry} …")
    add_geometry(plotter, geo, show_labels=not bool(args.animation))

    print(f"Loading trajectories from {args.traj} …")
    ions = load_trajectories(args.traj)
    add_trajectories(plotter, ions, cmap=args.cmap)

    plotter.add_axes(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")
    if not args.animation:
        plotter.add_legend(bcolor="white", border=True, size=(0.16, 0.18))
    plotter.camera_position = "iso"

    if args.animation:
        bbox = compute_global_bbox(geo)
        render_flythrough(plotter, args.animation, bbox,
                          fps=args.fps, duration=args.duration,
                          orbit_radius=args.orbit_radius,
                          orbit_height=args.orbit_height,
                          axis=args.axis)
    elif args.screenshot:
        plotter.show(screenshot=args.screenshot, auto_close=True)
        print(f"Saved: {args.screenshot}")
    else:
        plotter.show()


if __name__ == "__main__":
    main()
