"""Tests for the bundled stl_check module used by FusionExportSTL.

Like yaml_subset.py, stl_check.py ships inside the Fusion Scripts folder and
is deliberately not a trapsim submodule — we load it by file path so the
tests exercise the same code that runs inside Fusion.
"""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import pytest


# ── module loader ─────────────────────────────────────────────────────────────

def _load_stl_check():
    path = (Path(__file__).resolve().parents[1]
            / "src" / "trapsim" / "fusion" / "_script" / "stl_check.py")
    spec = importlib.util.spec_from_file_location("stl_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sc():
    return _load_stl_check()


# ── synthetic binary STL writer ───────────────────────────────────────────────

def _write_stl(path, triangles):
    """triangles: list of 3-vertex tuples, each vertex an (x, y, z)."""
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(triangles)))
        for tri in triangles:
            f.write(struct.pack("<3f", 0.0, 0.0, 1.0))     # normal
            for v in tri:
                f.write(struct.pack("<3f", *v))
            f.write(struct.pack("<H", 0))


# ── read_binary_stl_bbox ──────────────────────────────────────────────────────

class TestReadBBox:
    def test_bbox_of_known_triangles(self, sc, tmp_path):
        p = tmp_path / "t.stl"
        _write_stl(p, [
            ((0, 0, 0), (10, 0, 0), (0, 5, 0)),
            ((-2, 1, 3), (4, 4, 4), (0, 0, 7)),
        ])
        lo, hi, n = sc.read_binary_stl_bbox(p)
        assert n == 2
        assert lo == pytest.approx((-2.0, 0.0, 0.0))
        assert hi == pytest.approx((10.0, 5.0, 7.0))

    def test_zero_triangles_rejected(self, sc, tmp_path):
        p = tmp_path / "empty.stl"
        _write_stl(p, [])
        with pytest.raises(ValueError, match="no triangles"):
            sc.read_binary_stl_bbox(p)

    def test_truncated_payload_rejected(self, sc, tmp_path):
        p = tmp_path / "trunc.stl"
        _write_stl(p, [((0, 0, 0), (1, 0, 0), (0, 1, 0))])
        data = p.read_bytes()
        p.write_bytes(data[:-10])
        with pytest.raises(ValueError, match="truncated"):
            sc.read_binary_stl_bbox(p)

    def test_truncated_header_rejected(self, sc, tmp_path):
        p = tmp_path / "short.stl"
        p.write_bytes(b"\0" * 40)
        with pytest.raises(ValueError, match="header"):
            sc.read_binary_stl_bbox(p)

    def test_ascii_stl_rejected(self, sc, tmp_path):
        p = tmp_path / "ascii.stl"
        p.write_text("solid part\n facet normal 0 0 1\n" + " " * 100)
        with pytest.raises(ValueError, match="ASCII"):
            sc.read_binary_stl_bbox(p)


# ── scale_binary_stl ──────────────────────────────────────────────────────────

class TestScaleBinarySTL:
    def test_scales_vertices_not_normals(self, sc, tmp_path):
        p = tmp_path / "s.stl"
        _write_stl(p, [((1, 2, 3), (4, 5, 6), (7, 8, 9))])
        sc.scale_binary_stl(p, 10.0)
        lo, hi, n = sc.read_binary_stl_bbox(p)
        assert n == 1
        assert lo == pytest.approx((10.0, 20.0, 30.0))
        assert hi == pytest.approx((70.0, 80.0, 90.0))
        # normal (0, 0, 1) untouched
        raw = p.read_bytes()
        assert struct.unpack_from("<3f", raw, 84) == pytest.approx((0.0, 0.0, 1.0))

    def test_truncated_rejected(self, sc, tmp_path):
        p = tmp_path / "t.stl"
        _write_stl(p, [((0, 0, 0), (1, 0, 0), (0, 1, 0))])
        p.write_bytes(p.read_bytes()[:-10])
        with pytest.raises(ValueError, match="truncated"):
            sc.scale_binary_stl(p, 10.0)


# ── diagnose_frame ────────────────────────────────────────────────────────────

class TestDiagnoseFrame:
    # a 10×20×300 mm rod sitting at x ∈ [100, 110] in the assembly,
    # but at the origin in its own component frame
    WORLD = ((100.0, 0.0, 0.0), (110.0, 20.0, 300.0))
    LOCAL = ((0.0, -10.0, -150.0), (10.0, 10.0, 150.0))

    @staticmethod
    def _scaled(box, s):
        return (tuple(c * s for c in box[0]), tuple(c * s for c in box[1]))

    def test_world_mm_is_ok(self, sc):
        assert sc.diagnose_frame(self.WORLD, self.WORLD, self.LOCAL) == "ok"

    def test_world_cm_detected(self, sc):
        file_box = self._scaled(self.WORLD, 0.1)
        assert sc.diagnose_frame(file_box, self.WORLD, self.LOCAL) == "cm"

    def test_local_mm_detected(self, sc):
        assert sc.diagnose_frame(self.LOCAL, self.WORLD, self.LOCAL) == "local"

    def test_local_cm_detected(self, sc):
        file_box = self._scaled(self.LOCAL, 0.1)
        assert sc.diagnose_frame(file_box, self.WORLD, self.LOCAL) == "local_cm"

    def test_garbage_is_unknown(self, sc):
        junk = ((-500.0, -500.0, -500.0), (500.0, 500.0, 500.0))
        assert sc.diagnose_frame(junk, self.WORLD, self.LOCAL) == "unknown"

    def test_body_at_origin_prefers_ok(self, sc):
        # world == local (unrotated body at origin): 'ok' wins by precedence
        box = ((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
        assert sc.diagnose_frame(box, box, box) == "ok"


# ── boxes_match ───────────────────────────────────────────────────────────────

class TestBoxesMatch:
    BOX = ((0.0, 0.0, 0.0), (10.0, 20.0, 300.0))

    def test_identical_boxes_match(self, sc):
        assert sc.boxes_match(self.BOX, self.BOX, center_tol=1.0)

    def test_small_deviation_within_tolerance(self, sc):
        near = ((0.2, -0.1, 0.3), (10.1, 20.2, 299.8))
        assert sc.boxes_match(self.BOX, near, center_tol=1.0)

    def test_shifted_center_fails(self, sc):
        # body-local export symptom: same extents, translated position
        shifted = ((50.0, 0.0, 0.0), (60.0, 20.0, 300.0))
        assert not sc.boxes_match(self.BOX, shifted, center_tol=1.0)

    def test_wrong_units_fail(self, sc):
        # cm-vs-mm symptom: extents (and centre) 10x off
        cm = ((0.0, 0.0, 0.0), (1.0, 2.0, 30.0))
        assert not sc.boxes_match(self.BOX, cm, center_tol=1.0)

    def test_wrong_extent_fails_even_with_same_center(self, sc):
        fat = ((-5.0, 0.0, 0.0), (15.0, 20.0, 300.0))
        assert not sc.boxes_match(self.BOX, fat, center_tol=100.0)
