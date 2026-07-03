"""Tests for trapsim.io.pa.load_splat_mask — the labelled splat mask.

Uses a tiny synthetic geometry (SimpleNamespace stands in for
GeometryConfig: only .grid.shape and .electrodes[].electrode_id/.name are
touched) and hand-written mask_<id>.raw / dielectric_mask.raw files.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from trapsim.io.pa import load_splat_mask


def _geometry(n_electrodes=2, shape=(4, 3, 2)):
    electrodes = [
        SimpleNamespace(electrode_id=i + 1, name=f"elec_{i + 1}")
        for i in range(n_electrodes)
    ]
    return SimpleNamespace(grid=SimpleNamespace(shape=shape),
                           electrodes=electrodes)


def _write_mask(solver_dir, filename, shape, true_voxels):
    """Write a uint8 mask file; true_voxels are (iz, iy, ix) node indices."""
    NX, NY, NZ = shape
    m = np.zeros((NZ, NY, NX), dtype=np.uint8)
    for iz, iy, ix in true_voxels:
        m[iz, iy, ix] = 1
    m.ravel().tofile(str(solver_dir / filename))


class TestLoadSplatMask:
    def test_labels_and_names(self, tmp_path):
        geo = _geometry()
        shape = geo.grid.shape
        _write_mask(tmp_path, "mask_1.raw", shape, [(0, 0, 0)])
        _write_mask(tmp_path, "mask_2.raw", shape, [(1, 2, 3)])

        labels, names = load_splat_mask(geo, str(tmp_path))
        assert labels.shape == (shape[2], shape[1], shape[0])
        assert labels[0, 0, 0] == 1
        assert labels[1, 2, 3] == 2
        assert int(np.count_nonzero(labels)) == 2
        assert names == {1: "elec_1", 2: "elec_2"}

    def test_dielectric_gets_reserved_label(self, tmp_path):
        geo = _geometry()
        shape = geo.grid.shape
        _write_mask(tmp_path, "mask_1.raw", shape, [(0, 0, 0)])
        _write_mask(tmp_path, "mask_2.raw", shape, [])
        _write_mask(tmp_path, "dielectric_mask.raw", shape, [(1, 1, 1)])

        labels, names = load_splat_mask(geo, str(tmp_path))
        assert labels[1, 1, 1] == 3
        assert names[3] == "dielectric"

    def test_missing_electrode_mask_returns_none(self, tmp_path):
        geo = _geometry()
        _write_mask(tmp_path, "mask_1.raw", geo.grid.shape, [])
        assert load_splat_mask(geo, str(tmp_path)) is None

    def test_union_matches_old_boolean_behaviour(self, tmp_path):
        """Nonzero labels must equal the OR of all input masks, even where
        electrodes overlap (later declaration wins the label)."""
        geo = _geometry()
        shape = geo.grid.shape
        _write_mask(tmp_path, "mask_1.raw", shape, [(0, 0, 0), (1, 1, 1)])
        _write_mask(tmp_path, "mask_2.raw", shape, [(1, 1, 1)])

        labels, _names = load_splat_mask(geo, str(tmp_path))
        assert labels[0, 0, 0] == 1
        assert labels[1, 1, 1] == 2       # overlap: later electrode wins
        assert int(np.count_nonzero(labels)) == 2

    def test_wrong_size_mask_raises(self, tmp_path):
        geo = _geometry(n_electrodes=1)
        np.zeros(3, dtype=np.uint8).tofile(str(tmp_path / "mask_1.raw"))
        try:
            load_splat_mask(geo, str(tmp_path))
        except IOError:
            pass
        else:
            raise AssertionError("expected IOError for truncated mask")
