"""Binary-STL sanity checks used by FusionExportSTL after each export.

No `adsk` imports — this file is also loaded by trapsim's test suite via a
direct file-path import, exactly like yaml_subset.py.

The check exists because Fusion's export pipeline has silently produced
meshes in the wrong coordinate frame before (body-local instead of
assembly-world).  Comparing the written file's bounding box against the
body's world bounding box turns that silent failure into a loud one.
"""

from __future__ import annotations

import struct

_TRI_RECORD = struct.Struct("<12fH")   # normal + 3 vertices + attribute count


def read_binary_stl_bbox(path):
    """Return ((minx, miny, minz), (maxx, maxy, maxz), n_triangles) of a
    binary STL, in the file's own units.  Raises ValueError on ASCII,
    empty, or truncated files."""
    with open(path, "rb") as f:
        header = f.read(80)
        if len(header) < 80:
            raise ValueError("truncated STL: header shorter than 80 bytes")
        if header.lstrip().startswith(b"solid") and b"facet" in header:
            raise ValueError("ASCII STL where binary was expected")
        raw_count = f.read(4)
        if len(raw_count) < 4:
            raise ValueError("truncated STL: missing triangle count")
        (n,) = struct.unpack("<I", raw_count)
        body = f.read(n * _TRI_RECORD.size)
    if n == 0:
        raise ValueError("STL contains no triangles")
    if len(body) < n * _TRI_RECORD.size:
        raise ValueError(
            f"truncated STL: header claims {n} triangles, "
            f"payload holds {len(body) // _TRI_RECORD.size}")

    mins = [float("inf")] * 3
    maxs = [float("-inf")] * 3
    for rec in _TRI_RECORD.iter_unpack(body):
        for v in range(3):
            for a in range(3):
                c = rec[3 + 3 * v + a]      # skip the normal (first 3 floats)
                if c < mins[a]:
                    mins[a] = c
                if c > maxs[a]:
                    maxs[a] = c
    return tuple(mins), tuple(maxs), n


def boxes_match(box_a, box_b, center_tol,
                extent_rel_tol=0.05, extent_abs_tol=0.5):
    """True if two ((min3), (max3)) boxes agree in centre (within
    `center_tol` per axis) and extent (within abs + rel tolerance per axis).

    The centre check is the coordinate-frame guard: a mesh exported in
    body-local coordinates lands centred near its own origin, far from the
    body's assembly-world position.  The extent check additionally catches
    unit mix-ups (cm vs mm) and wrong-body exports.
    """
    (a_min, a_max) = box_a
    (b_min, b_max) = box_b
    for i in range(3):
        centre_a = 0.5 * (a_min[i] + a_max[i])
        centre_b = 0.5 * (b_min[i] + b_max[i])
        if abs(centre_a - centre_b) > center_tol:
            return False
        extent_a = a_max[i] - a_min[i]
        extent_b = b_max[i] - b_min[i]
        if abs(extent_a - extent_b) > (extent_abs_tol
                                       + extent_rel_tol * max(extent_a, extent_b)):
            return False
    return True
