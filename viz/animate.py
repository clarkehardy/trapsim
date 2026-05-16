"""trapsim.viz.animate  –  2-D animated view, driven by geometry.yaml.

Three panels (top → bottom):
  - Top view  (Z horizontal, X vertical) with ion trails.
  - Side view (Z horizontal, Y vertical) with ion trails.
  - Voltage panel: resolved DC voltages from the schedule snapshot, with
    a vertical cursor tracking the current animation frame.

The geometry is drawn directly from `geometry.yaml`: each electrode
becomes one or more filled rectangles in the projection, using the colour
declared in the YAML.  No electrode names are hardcoded.

Usage:
    python -m trapsim.viz.animate
    python -m trapsim.viz.animate --traj trajectories_2.csv --schedule schedule_2.json
    python -m trapsim.viz.animate --save animation.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation

from ..config import load_geometry
from ..io.schedule_io import read_schedule_snapshot
from ..schedule import Schedule
from ._common import bbox_union, load_trajectories, read_stl_bbox


def compute_electrode_bboxes(geo):
    """Per-electrode (xlo,xhi,ylo,yhi,zlo,zhi) from STL bboxes (union over STLs)."""
    out = []
    for e in geo.electrodes:
        boxes = [read_stl_bbox(s) for s in e.stls]
        u = bbox_union(*boxes)
        if u is None:
            continue
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = u
        out.append((e.name, e.color, xlo, xhi, ylo, yhi, zlo, zhi))
    return out


def draw_geometry(ax, electrode_bboxes, view, view_lims):
    """view = "side" (Y vs Z) or "top" (X vs Z)."""
    for name, color, xlo, xhi, ylo, yhi, zlo, zhi in electrode_bboxes:
        if view == "side":
            v0, v1 = ylo, yhi
        else:
            v0, v1 = xlo, xhi
        rect = mpatches.Rectangle(
            (zlo, v0), zhi - zlo, v1 - v0,
            facecolor=color if color else (0.6, 0.6, 0.6),
            edgecolor="none", alpha=0.35, zorder=1)
        ax.add_patch(rect)
    if view_lims:
        ax.set_xlim(view_lims["z"])
        ax.set_ylim(view_lims["side" if view == "side" else "top"])
    ax.set_xlabel("Z (mm)")
    ax.set_ylabel("Y (mm)" if view == "side" else "X (mm)")
    ax.set_title("Side view (Z, Y)" if view == "side" else "Top view (Z, X)")
    ax.grid(True, alpha=0.25)


def compute_fire_times(ions, triggers):
    """For each trigger, return the abs time at which the first ion crossed
    its threshold along its axis (or None if no ion ever did)."""
    axis_to_key = {"x": "x", "y": "y", "z": "z"}
    out = []
    for trig in triggers:
        key = axis_to_key[trig["axis"]]
        t_fire = np.inf
        for d in ions.values():
            mask = d[key] >= trig["threshold_mm"]
            if mask.any():
                t_fire = min(t_fire, float(d["t"][mask][0]))
        out.append(t_fire if np.isfinite(t_fire) else None)
    return out


def resolve_voltage_timeline(schedule: Schedule, fire_times, n_samples=1000,
                             t_max=None):
    """Sample schedule.evaluate at uniformly-spaced absolute times.

    `fire_times` is a list aligned with schedule._triggers giving the
    absolute fire time for each trigger (or None).  Returns
    (time_axis, dict {electrode_name: voltage array}).
    """
    # Build trigger_state keyed by name
    trigger_state = {}
    for trig, t_fire in zip(schedule._triggers, fire_times):
        trigger_state[trig["name"]] = t_fire
    main_t = schedule._main["time_us"]
    if t_max is None:
        t_max = float(main_t[-1]) if len(main_t) else 0.0
    t_axis = np.linspace(0, t_max, n_samples)

    # Per-electrode voltages over time, evaluating with the *eventual* trigger
    # state after each fire time.  Before a trigger's fire time, that trigger
    # contributes nothing (state = None).
    voltages = {name: np.zeros(n_samples) for name in schedule._electrode_names}
    for i, t in enumerate(t_axis):
        ts = {name: (tf if (tf is not None and t >= tf) else None)
              for name, tf in trigger_state.items()}
        v = schedule.evaluate(float(t), ts)
        for name, val in v.items():
            voltages[name][i] = val
    return t_axis, voltages


def main():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry",   default=os.path.join(base, "geometry.yaml"))
    ap.add_argument("--traj",       default=os.path.join(base, "trajectories_1.csv"))
    ap.add_argument("--schedule",   default=os.path.join(base, "schedule_1.json"))
    ap.add_argument("--fps",        type=float, default=30.0)
    ap.add_argument("--speed",      type=float, default=None,
                    help="µs of sim time per wall-second (default: 20 s total)")
    ap.add_argument("--save",       default=None,
                    help="Output .mp4 / .gif file (needs ffmpeg / imageio)")
    ap.add_argument("--samples",    type=int, default=2000,
                    help="Voltage-timeline samples (default: 2000)")
    args = ap.parse_args()

    geo = load_geometry(args.geometry)

    ions = load_trajectories(args.traj)
    if not ions:
        sys.exit(f"No trajectory data in {args.traj}")

    have_sched = os.path.exists(args.schedule)
    if not have_sched:
        print(f"  [skip] schedule snapshot not found at {args.schedule}; "
              "voltage panel omitted.")

    # Build the Schedule for trigger logic + voltage resolution
    sched = trigger_data = None
    if have_sched:
        snap = read_schedule_snapshot(args.schedule)
        sched = Schedule(snap["main"], snap["triggers"], geo.electrode_names())
        trigger_data = snap["triggers"]

    fire_times = compute_fire_times(ions, trigger_data or [])

    # Electrode rectangles for the 2D plots
    elec_bboxes = compute_electrode_bboxes(geo)

    # View limits from the union bounding box
    global_bb = bbox_union(*[read_stl_bbox(s) for e in geo.electrodes
                              for s in e.stls])
    if global_bb is None:
        sys.exit("No STL bounding boxes available for geometry rendering.")
    (xl, xh), (yl, yh), (zl, zh) = global_bb
    view_lims = {
        "z":    (zl - 10, zh + 10),
        "side": (yl -  2, yh +  2),
        "top":  (xl -  2, xh +  2),
    }

    # ── Layout ────────────────────────────────────────────────────────
    all_t = np.concatenate([d["t"] for d in ions.values()])
    t_min, t_max = float(all_t.min()), float(all_t.max())
    duration = t_max - t_min
    speed    = args.speed if args.speed else duration / 20.0
    n_frames = max(2, int(args.fps * duration / speed))
    times    = np.linspace(t_min, t_max, n_frames)

    if have_sched:
        fig = plt.figure(figsize=(13, 9.5))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.3, 1.3, 1.0], hspace=0.45)
        ax_xz  = fig.add_subplot(gs[0])
        ax_yz  = fig.add_subplot(gs[1], sharex=ax_xz)
        ax_bot = fig.add_subplot(gs[2])
    else:
        fig, (ax_xz, ax_yz) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                            gridspec_kw={"hspace": 0.40})
        ax_bot = None

    draw_geometry(ax_xz, elec_bboxes, view="top",  view_lims=view_lims)
    draw_geometry(ax_yz, elec_bboxes, view="side", view_lims=view_lims)

    # Trigger overlays
    _TRIG_PALETTE = ["darkorchid", "tomato", "mediumseagreen", "saddlebrown"]
    for i, trig in enumerate(trigger_data or []):
        c = _TRIG_PALETTE[i % len(_TRIG_PALETTE)]
        # Vertical line at the threshold *only* in the panel whose axis matches
        if trig["axis"] == "z":
            for ax in (ax_xz, ax_yz):
                ax.axvline(trig["threshold_mm"], color=c, lw=1.5,
                           ls=(0, (4, 2)), alpha=0.85,
                           label=f"trig {trig['name']}: z={trig['threshold_mm']:.0f}")

    # Ion trails
    cmap = plt.cm.tab10 if len(ions) <= 10 else plt.cm.viridis
    ion_ids = sorted(ions.keys())
    colors  = {iid: cmap(i / max(1, len(ions) - 1)) for i, iid in enumerate(ion_ids)}
    trails_xz, dots_xz = {}, {}
    trails_yz, dots_yz = {}, {}
    for iid in ion_ids:
        c = colors[iid]
        tr_xz, = ax_xz.plot([], [], lw=1.3, color=c, alpha=0.85, zorder=3)
        do_xz, = ax_xz.plot([], [], "o", ms=5, color=c, zorder=4)
        tr_yz, = ax_yz.plot([], [], lw=1.3, color=c, alpha=0.85, zorder=3)
        do_yz, = ax_yz.plot([], [], "o", ms=5, color=c, zorder=4)
        trails_xz[iid] = tr_xz; dots_xz[iid] = do_xz
        trails_yz[iid] = tr_yz; dots_yz[iid] = do_yz

    time_label = ax_xz.text(
        0.995, 0.97, "", transform=ax_xz.transAxes,
        va="top", ha="right", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.75))

    for ax in (ax_xz, ax_yz):
        ax.legend(loc="lower left", fontsize=7, ncol=3, framealpha=0.8)

    # ── Voltage panel ────────────────────────────────────────────────
    vcursor = None
    if have_sched and ax_bot is not None:
        t_axis, voltages = resolve_voltage_timeline(
            sched, fire_times, n_samples=args.samples, t_max=t_max)
        # Plot each electrode in its YAML color (or auto if unset)
        for e in geo.electrodes:
            v = voltages.get(e.name, None)
            if v is None or np.allclose(v, v[0]):
                # constant traces just look like a horizontal line — still plot for completeness
                pass
            color = e.color if e.color else None
            ax_bot.plot(t_axis, voltages[e.name], lw=1.0,
                        color=color, label=e.name)
        ax_bot.set_xlim(t_axis[0], t_axis[-1])
        ax_bot.set_xlabel("Time (µs)")
        ax_bot.set_ylabel("Voltage (V)")
        ax_bot.set_title("Resolved electrode voltages (DC + RF carrier)")
        ax_bot.grid(True, alpha=0.3)
        ax_bot.legend(loc="upper right", fontsize=7, ncol=3, framealpha=0.85)
        # Trigger fire-time markers
        for i, (trig, tf) in enumerate(zip(trigger_data, fire_times)):
            if tf is None:
                continue
            c = _TRIG_PALETTE[i % len(_TRIG_PALETTE)]
            ax_bot.axvline(tf, color=c, lw=1.0, ls=(0, (4, 2)), alpha=0.7)
        vcursor = ax_bot.axvline(t_min, color="black", lw=1.2, alpha=0.6)

    # ── Animation ───────────────────────────────────────────────────────
    def frame(i):
        t = times[i]
        time_label.set_text(f"t = {t:.0f} µs")
        for iid in ion_ids:
            d = ions[iid]
            mask = d["t"] <= t
            trails_xz[iid].set_data(d["z"][mask], d["x"][mask])
            trails_yz[iid].set_data(d["z"][mask], d["y"][mask])
            if mask.any():
                dots_xz[iid].set_data([d["z"][mask][-1]], [d["x"][mask][-1]])
                dots_yz[iid].set_data([d["z"][mask][-1]], [d["y"][mask][-1]])
            else:
                dots_xz[iid].set_data([], []); dots_yz[iid].set_data([], [])
        if vcursor is not None:
            vcursor.set_xdata([t, t])
        artists = (list(trails_xz.values()) + list(dots_xz.values()) +
                   list(trails_yz.values()) + list(dots_yz.values()) +
                   [time_label])
        if vcursor is not None:
            artists.append(vcursor)
        return artists

    interval = 1000.0 / args.fps
    anim = animation.FuncAnimation(fig, frame, frames=len(times),
                                    interval=interval, blit=True)
    if args.save:
        ext = os.path.splitext(args.save)[1].lower()
        if ext == ".gif":
            anim.save(args.save, writer="pillow", fps=args.fps)
        else:
            anim.save(args.save, writer="ffmpeg", fps=args.fps)
        print(f"Saved animation: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
