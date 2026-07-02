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


def scale_binary_stl(path, factor):
    """Scale all vertex coordinates (not normals) in-place by `factor`.
    Used to convert a cm-unit export (Fusion's internal unit, which the API
    STL writer emits) into the mm the rest of the pipeline expects."""
    with open(path, "rb") as f:
        head = f.read(84)
        body = bytearray(f.read())
    if len(head) < 84:
        raise ValueError("truncated STL: shorter than header + count")
    (n,) = struct.unpack("<I", head[80:84])
    if len(body) < n * _TRI_RECORD.size:
        raise ValueError("truncated STL: payload shorter than triangle count")
    for t in range(n):
        base = t * _TRI_RECORD.size + 12        # skip the 3 normal floats
        for k in range(9):
            off = base + 4 * k
            (v,) = struct.unpack_from("<f", body, off)
            struct.pack_into("<f", body, off, v * factor)
    with open(path, "wb") as f:
        f.write(head)
        f.write(body)


def diagnose_frame(file_box, world_box_mm, local_box_mm):
    """Classify an exported mesh's coordinate frame and units.

    `file_box` is the mesh bounding box in the file's raw units;
    `world_box_mm` / `local_box_mm` are the body's bounding box in the
    assembly frame and in its own component frame, both in mm.

    Returns one of:
      'ok'       — world frame, mm; nothing to do
      'cm'       — world frame, cm; caller should scale the file by 10
      'local'    — component-local frame (occurrence transform not applied)
      'local_cm' — component-local frame AND cm units
      'unknown'  — matches nothing recognisable
    """
    def tol(box):
        diag = sum((box[1][i] - box[0][i]) ** 2 for i in range(3)) ** 0.5
        return max(1.0, 0.02 * diag)

    def scaled(box, s):
        return (tuple(c * s for c in box[0]), tuple(c * s for c in box[1]))

    checks = (("ok",       world_box_mm, 1.0),
              ("cm",       world_box_mm, 10.0),
              ("local",    local_box_mm, 1.0),
              ("local_cm", local_box_mm, 10.0))
    for status, expected, s in checks:
        if boxes_match(scaled(file_box, s), expected, tol(expected)):
            return status
    return "unknown"


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
