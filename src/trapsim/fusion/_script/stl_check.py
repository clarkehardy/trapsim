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


MAT4_IDENTITY = (1.0, 0.0, 0.0, 0.0,
                 0.0, 1.0, 0.0, 0.0,
                 0.0, 0.0, 1.0, 0.0,
                 0.0, 0.0, 0.0, 1.0)


def mat4_multiply(a, b):
    """Row-major 4x4 (length-16) matrix product a·b."""
    out = [0.0] * 16
    for r in range(4):
        for c in range(4):
            out[4 * r + c] = (a[4 * r + 0] * b[0 + c] + a[4 * r + 1] * b[4 + c]
                              + a[4 * r + 2] * b[8 + c] + a[4 * r + 3] * b[12 + c])
    return out


def transform_points(m, coords):
    """Apply a row-major 4x4 matrix to a flat [x0,y0,z0, x1,...] list."""
    out = []
    for i in range(0, len(coords), 3):
        x, y, z = coords[i], coords[i + 1], coords[i + 2]
        out.append(m[0] * x + m[1] * y + m[2] * z + m[3])
        out.append(m[4] * x + m[5] * y + m[6] * z + m[7])
        out.append(m[8] * x + m[9] * y + m[10] * z + m[11])
    return out


def bbox_of_points(coords):
    """((min3), (max3)) of a flat [x0,y0,z0, x1,...] coordinate list."""
    xs, ys, zs = coords[0::3], coords[1::3], coords[2::3]
    if not xs:
        raise ValueError("no points")
    return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))


def write_binary_stl(path, coords, indices, scale=1.0):
    """Write a binary STL from flat vertex coords and triangle node indices
    (3 per triangle).  Coordinates are multiplied by `scale` on the way out
    (Fusion tessellates in cm; trapsim wants mm → scale=10).  Facet normals
    are computed from the triangle winding order."""
    n = len(indices) // 3
    if n == 0:
        raise ValueError("no triangles")
    with open(path, "wb") as f:
        f.write(b"trapsim FusionExportSTL".ljust(80, b"\0"))
        f.write(struct.pack("<I", n))
        for t in range(n):
            v = []
            for idx in indices[3 * t:3 * t + 3]:
                v.append((coords[3 * idx] * scale,
                          coords[3 * idx + 1] * scale,
                          coords[3 * idx + 2] * scale))
            e1 = (v[1][0] - v[0][0], v[1][1] - v[0][1], v[1][2] - v[0][2])
            e2 = (v[2][0] - v[0][0], v[2][1] - v[0][1], v[2][2] - v[0][2])
            nx = e1[1] * e2[2] - e1[2] * e2[1]
            ny = e1[2] * e2[0] - e1[0] * e2[2]
            nz = e1[0] * e2[1] - e1[1] * e2[0]
            norm = (nx * nx + ny * ny + nz * nz) ** 0.5
            if norm > 0.0:
                nx, ny, nz = nx / norm, ny / norm, nz / norm
            f.write(_TRI_RECORD.pack(nx, ny, nz,
                                     *v[0], *v[1], *v[2], 0))


def placement_plausible(mesh_box, brep_box_mm, center_tol, pad=0.5):
    """True if a tight tessellated-mesh bounding box sits at a (possibly
    loose) BRep reference box: centres agree per axis within `center_tol`
    and the mesh box is contained in the reference box padded by `pad`.

    Fusion's BRepBody.boundingBox is conservative for curved faces — it is
    computed from NURBS control points, which e.g. overhang by ~√2 across a
    45°-rotated cylinder — so extent equality is the wrong test against a
    tessellation whose vertices lie exactly on the surface.  Containment
    plus centre agreement is the right one: a mesh in the wrong frame lands
    with the wrong centre, and a mis-rotated mesh pokes outside the box."""
    (m_lo, m_hi) = mesh_box
    (r_lo, r_hi) = brep_box_mm
    for i in range(3):
        centre_m = 0.5 * (m_lo[i] + m_hi[i])
        centre_r = 0.5 * (r_lo[i] + r_hi[i])
        if abs(centre_m - centre_r) > center_tol:
            return False
        if m_lo[i] < r_lo[i] - pad or m_hi[i] > r_hi[i] + pad:
            return False
    return True


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
