"""trapsim.viz._common  –  Helpers shared by visualize, animate, plot_field."""

from __future__ import annotations

import os
import struct
import numpy as np


def read_stl_bbox(path: str):
    """Return ((xmin,xmax),(ymin,ymax),(zmin,zmax)) for a binary STL, or None."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        f.read(80)
        (n,) = struct.unpack("<I", f.read(4))
        raw = f.read(n * 50)
    if n == 0 or len(raw) < n * 50:
        return None
    arr   = np.frombuffer(raw, dtype=np.uint8).reshape(n, 50)
    verts = np.frombuffer(arr[:, 12:48].tobytes(), dtype="<f4").reshape(-1, 3)
    return ((float(verts[:, 0].min()), float(verts[:, 0].max())),
            (float(verts[:, 1].min()), float(verts[:, 1].max())),
            (float(verts[:, 2].min()), float(verts[:, 2].max())))


def bbox_union(*boxes):
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return ((min(b[0][0] for b in boxes), max(b[0][1] for b in boxes)),
            (min(b[1][0] for b in boxes), max(b[1][1] for b in boxes)),
            (min(b[2][0] for b in boxes), max(b[2][1] for b in boxes)))


def auto_color(idx: int, n_total: int):
    """Return an [r,g,b] color for the idx-th body out of n_total."""
    # tab10 for small N, viridis-like for larger.
    if n_total <= 10:
        tab10 = [
            (0.12, 0.47, 0.71), (1.00, 0.50, 0.05), (0.17, 0.63, 0.17),
            (0.84, 0.15, 0.16), (0.58, 0.40, 0.74), (0.55, 0.34, 0.29),
            (0.89, 0.47, 0.76), (0.50, 0.50, 0.50), (0.74, 0.74, 0.13),
            (0.09, 0.75, 0.81),
        ]
        return tab10[idx % len(tab10)]
    # Simple HSV → RGB cycle for many bodies
    h = (idx / n_total) % 1.0
    s, v = 0.55, 0.85
    i = int(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    return [(v, t, p), (q, v, p), (p, v, t),
            (p, q, v), (t, p, v), (v, p, q)][i % 6]


def load_trajectories(path: str):
    """Return dict {ion_id: {"t": np.array, "x": ..., "y": ..., "z": ...}}.

    Reads the trapsim trajectory CSV format
        ion_id,t_us,x_mm,y_mm,z_mm
    """
    if not os.path.exists(path):
        print(f"[skip] trajectory file not found: {path}")
        return {}
    ions = {}
    with open(path) as f:
        f.readline()  # header
        for line in f:
            parts = line.split(",")
            if len(parts) < 5:
                continue
            ion_id = int(parts[0])
            entry  = ions.setdefault(ion_id, {"t": [], "x": [], "y": [], "z": []})
            entry["t"].append(float(parts[1]))
            entry["x"].append(float(parts[2]))
            entry["y"].append(float(parts[3]))
            entry["z"].append(float(parts[4]))
    return {k: {kk: np.array(vv) for kk, vv in v.items()} for k, v in ions.items()}
