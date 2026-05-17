"""Smoke tests for trapsim.config — YAML geometry loader + validation."""

import os
import tempfile
import textwrap

import pytest

from trapsim.config import load_geometry


def _write_geometry(tmp_path, yaml_text, stl_names=None):
    """Write geometry.yaml + dummy STL files into a temp directory."""
    geo_path = os.path.join(tmp_path, "geometry.yaml")
    with open(geo_path, "w") as f:
        f.write(textwrap.dedent(yaml_text))
    for name in (stl_names or []):
        stl_path = os.path.join(tmp_path, name)
        os.makedirs(os.path.dirname(stl_path), exist_ok=True)
        with open(stl_path, "w") as f:
            f.write("")  # empty placeholder
    return geo_path


class TestLoadGeometry:
    def test_minimal_two_electrodes(self, tmp_path):
        geo_path = _write_geometry(str(tmp_path), """
            grid:
              dx_mm: 0.5
              bounds_mm:
                x: [-10.0, 10.0]
                y: [-10.0, 10.0]
                z: [-20.0, 20.0]

            electrodes:
              - name: plate_top
                stls: [plate_top.stl]
              - name: plate_bottom
                stls: [plate_bottom.stl]
        """, stl_names=["plate_top.stl", "plate_bottom.stl"])

        geo = load_geometry(geo_path)
        assert geo.n_electrodes == 2
        assert geo.electrode_names() == ["plate_top", "plate_bottom"]
        assert geo.electrodes[0].electrode_id == 1
        assert geo.electrodes[1].electrode_id == 2
        assert geo.grid.shape == (41, 41, 81)
        assert geo.grid.dx_mm == 0.5
        assert geo.grid.world_offset_mm == (-10.0, -10.0, -20.0)

    def test_stl_resolution_under_stl_subdir(self, tmp_path):
        geo_path = _write_geometry(str(tmp_path), """
            grid:
              dx_mm: 1.0
              bounds_mm:
                x: [0, 10]
                y: [0, 10]
                z: [0, 10]
            electrodes:
              - name: ring
                stls: [ring.stl]
        """, stl_names=["stl/ring.stl"])

        geo = load_geometry(geo_path)
        assert geo.electrodes[0].name == "ring"

    def test_missing_grid_raises(self, tmp_path):
        geo_path = _write_geometry(str(tmp_path), """
            electrodes:
              - name: a
                stls: [a.stl]
        """, stl_names=["a.stl"])
        with pytest.raises(ValueError, match="missing required 'grid'"):
            load_geometry(geo_path)

    def test_non_ascending_bounds_raises(self, tmp_path):
        geo_path = _write_geometry(str(tmp_path), """
            grid:
              dx_mm: 1.0
              bounds_mm:
                x: [10.0, -10.0]
                y: [0, 10]
                z: [0, 10]
            electrodes:
              - name: a
                stls: [a.stl]
        """, stl_names=["a.stl"])
        with pytest.raises(ValueError, match="ascending"):
            load_geometry(geo_path)

    def test_duplicate_electrode_name_raises(self, tmp_path):
        geo_path = _write_geometry(str(tmp_path), """
            grid:
              dx_mm: 1.0
              bounds_mm:
                x: [0, 10]
                y: [0, 10]
                z: [0, 10]
            electrodes:
              - name: ring
                stls: [a.stl]
              - name: ring
                stls: [b.stl]
        """, stl_names=["a.stl", "b.stl"])
        with pytest.raises(ValueError, match="duplicate electrode name"):
            load_geometry(geo_path)

    def test_missing_stl_raises(self, tmp_path):
        geo_path = _write_geometry(str(tmp_path), """
            grid:
              dx_mm: 1.0
              bounds_mm:
                x: [0, 10]
                y: [0, 10]
                z: [0, 10]
            electrodes:
              - name: ring
                stls: [nonexistent.stl]
        """)
        with pytest.raises(FileNotFoundError):
            load_geometry(geo_path)

    def test_no_electrodes_raises(self, tmp_path):
        geo_path = _write_geometry(str(tmp_path), """
            grid:
              dx_mm: 1.0
              bounds_mm:
                x: [0, 10]
                y: [0, 10]
                z: [0, 10]
            electrodes: []
        """)
        with pytest.raises(ValueError, match="at least one electrode"):
            load_geometry(geo_path)
