"""trapsim.run  –  Top-level pipeline orchestrator.

    python -m trapsim.run                  # refine if needed, fly, animate, visualize
    python -m trapsim.run --run 2          # output → trajectories_2.csv etc.
    python -m trapsim.run --no-animate
    python -m trapsim.run --refine         # force a full refine before flying
    python -m trapsim.run --no-fly         # re-use existing trajectories_N.csv

The same flags are accepted by trapsim.run.run() if called as a function.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from .config import load_geometry, load_experiment
from . import refine as refine_mod
from . import fly as fly_mod


def run(geometry_path: str,
        experiment_path: str, *,
        base_dir: str,
        run_number: int = 1,
        do_refine: bool = False,
        do_fly: bool = True,
        do_animate: bool = True,
        do_visualize: bool = True,
        workers: int | None = None) -> None:
    """Drive the full pipeline: refine → fly → animate → visualize."""

    print(f"━━━ trapsim.run  geometry={os.path.basename(geometry_path)}  "
          f"experiment={os.path.basename(experiment_path)}  run={run_number} ━━━\n")

    geo = load_geometry(geometry_path)
    exp = load_experiment(experiment_path, geo)

    # ── Refine ───────────────────────────────────────────────────────────
    # Auto-refine only when a PA file is missing.  STL-mtime-based staleness
    # is too aggressive (e.g. a `git mv` updates mtime without changing
    # geometry) — the user should pass --refine explicitly after editing STLs.
    pa_missing = [e.electrode_id for e in geo.electrodes
                  if not os.path.exists(os.path.join(
                      base_dir, f"paulTrap.pa{e.electrode_id}"))]
    if do_refine or pa_missing:
        if pa_missing and not do_refine:
            print(f"── Refine: missing PA files for electrodes {pa_missing} ──")
        else:
            print("── Refine: voxelize + Laplace solve (forced) ──")
        refine_mod.refine(geo, out_dir=base_dir,
                          force_voxelize=do_refine)
        print()
    else:
        print("All PA files present — skipping refine  "
              "(use --refine to force).\n")

    # ── Fly ──────────────────────────────────────────────────────────────
    if do_fly:
        print("── Fly: integrate particles ──")
        fly_mod.fly(geo, exp, base_dir=base_dir,
                    run_number=run_number, workers=workers)
        print()
    else:
        traj = os.path.join(base_dir, f"trajectories_{run_number}.csv")
        if not os.path.exists(traj):
            sys.exit(f"--no-fly given but {traj} does not exist.")
        print(f"Skipping fly — using existing {traj}\n")

    # ── Animate ──────────────────────────────────────────────────────────
    if do_animate:
        print("── Animate ──")
        cmd = [sys.executable, "-m", "trapsim.viz.animate",
               "--geometry", geometry_path,
               "--traj", os.path.join(base_dir, f"trajectories_{run_number}.csv"),
               "--schedule", os.path.join(base_dir, f"schedule_{run_number}.json")]
        subprocess.run(cmd, check=False)
        print()

    # ── Visualize ────────────────────────────────────────────────────────
    if do_visualize:
        print("── Visualize ──")
        cmd = [sys.executable, "-m", "trapsim.viz.visualize",
               "--geometry", geometry_path,
               "--traj", os.path.join(base_dir, f"trajectories_{run_number}.csv")]
        subprocess.run(cmd, check=False)

    print("━━━ Done ━━━")


def main():
    cwd = os.getcwd()
    ap = argparse.ArgumentParser(description="trapsim full-pipeline runner.")
    ap.add_argument("--geometry",     default=os.path.join(cwd, "geometry.yaml"))
    ap.add_argument("--experiment",   default=os.path.join(cwd, "experiment.py"))
    ap.add_argument("--base-dir",     default=cwd)
    ap.add_argument("--run",          type=int, default=1)
    ap.add_argument("--workers",      type=int, default=None)
    ap.add_argument("--refine",       action="store_true",
                    help="Force a full refine (voxelize + solve) before flying.")
    ap.add_argument("--no-fly",       action="store_true")
    ap.add_argument("--no-animate",   action="store_true")
    ap.add_argument("--no-visualize", action="store_true")
    args = ap.parse_args()

    run(args.geometry, args.experiment,
        base_dir=args.base_dir,
        run_number=args.run,
        do_refine=args.refine,
        do_fly=not args.no_fly,
        do_animate=not args.no_animate,
        do_visualize=not args.no_visualize,
        workers=args.workers)


if __name__ == "__main__":
    main()
