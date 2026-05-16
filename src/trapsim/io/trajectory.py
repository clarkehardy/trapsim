"""trapsim.io.trajectory  –  Trajectory CSV writer/reader.

The output CSV has one header line and one row per recorded sample:

    ion_id, t_us, x_mm, y_mm, z_mm

Positions are in **Fusion-world** coordinates (the same coordinate system
as the geometry.yaml grid.bounds_mm).
"""

from __future__ import annotations

import os


HEADER = "ion_id,t_us,x_mm,y_mm,z_mm\n"


def write_trajectories(path: str, rows_per_ion: dict[int, list[str]]) -> int:
    """Write a trajectory CSV.  `rows_per_ion` maps ion_id → list of CSV row
    strings (without trailing newline).  Returns the total number of rows."""
    n_rows = 0
    with open(path, "w") as f:
        f.write(HEADER)
        for ion_id in sorted(rows_per_ion):
            for r in rows_per_ion[ion_id]:
                f.write(r + "\n")
                n_rows += 1
    return n_rows
