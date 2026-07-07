"""Tests for the bundled yaml_subset parser used by FusionExportSTL.

`yaml_subset.py` is intentionally NOT a trapsim submodule — it is shipped
inside the Fusion Scripts folder, imported at runtime by Fusion.  We
load it by file path here so the test suite exercises the same code that
runs inside Fusion.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


# ── module loader ─────────────────────────────────────────────────────────────

def _load_yaml_subset():
    path = (Path(__file__).resolve().parents[1]
            / "src" / "trapsim" / "fusion" / "_script" / "yaml_subset.py")
    spec = importlib.util.spec_from_file_location("yaml_subset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def y():
    return _load_yaml_subset()


# ── scalars ───────────────────────────────────────────────────────────────────

class TestScalars:
    def test_int(self, y):
        assert y.parse("k: 3") == {"k": 3}

    def test_float(self, y):
        assert y.parse("k: 3.14") == {"k": 3.14}

    def test_negative_float(self, y):
        assert y.parse("k: -0.5") == {"k": -0.5}

    def test_bool(self, y):
        assert y.parse("k: true\nj: false") == {"k": True, "j": False}

    def test_null(self, y):
        assert y.parse("k: null") == {"k": None}
        assert y.parse("k: ~")    == {"k": None}

    def test_string_bare(self, y):
        assert y.parse("k: hello") == {"k": "hello"}

    def test_string_double_quoted(self, y):
        assert y.parse('k: "hello world"') == {"k": "hello world"}

    def test_string_single_quoted(self, y):
        assert y.parse("k: 'x:y+z:1'") == {"k": "x:y+z:1"}

    def test_string_with_hash_inside_quotes(self, y):
        assert y.parse('k: "has # hash"') == {"k": "has # hash"}


# ── inline lists ──────────────────────────────────────────────────────────────

class TestInlineLists:
    def test_empty(self, y):
        assert y.parse("k: []") == {"k": []}

    def test_ints(self, y):
        assert y.parse("k: [1, 2, 3]") == {"k": [1, 2, 3]}

    def test_floats(self, y):
        assert y.parse("k: [-10.0, 10.0]") == {"k": [-10.0, 10.0]}

    def test_strings_bare(self, y):
        assert y.parse("k: [a, b, c]") == {"k": ["a", "b", "c"]}

    def test_strings_quoted_with_comma_inside(self, y):
        assert y.parse('k: ["a, b", c]') == {"k": ["a, b", "c"]}


# ── nested dicts and list-of-dicts (the geometry.yaml shape) ─────────────────

class TestNested:
    def test_geometry_yaml_shape(self, y):
        text = """\
grid:
  dx_mm: 0.5
  bounds_mm:
    x: [-10.0, 10.0]
    y: [-10.0, 10.0]
    z: [-20.0, 20.0]

electrodes:
  - name: plate_top
    stls: [stl/plate_top.stl]
  - name: plate_bottom
    stls: [stl/plate_bottom.stl, stl/plate_bottom_extra.stl]

dielectrics:
  - name: gate_insulator
    stl: stl/insulator.stl
    epsilon_r: 3.9
"""
        r = y.parse(text)
        assert r["grid"]["dx_mm"] == 0.5
        assert r["grid"]["bounds_mm"] == {
            "x": [-10.0, 10.0], "y": [-10.0, 10.0], "z": [-20.0, 20.0]}
        assert len(r["electrodes"]) == 2
        assert r["electrodes"][0] == {
            "name": "plate_top", "stls": ["stl/plate_top.stl"]}
        assert r["electrodes"][1]["stls"] == [
            "stl/plate_bottom.stl", "stl/plate_bottom_extra.stl"]
        assert r["dielectrics"][0]["epsilon_r"] == 3.9

    def test_comments_and_blank_lines_ignored(self, y):
        text = """\
# top-of-file comment
grid:
  dx_mm: 0.5   # inline comment

  bounds_mm:
    x: [-1, 1]
"""
        r = y.parse(text)
        assert r == {"grid": {"dx_mm": 0.5, "bounds_mm": {"x": [-1, 1]}}}


# ── fusion_map.yaml round-trip ────────────────────────────────────────────────

class TestFusionMapRoundtrip:
    def test_dump_then_parse(self, y):
        mappings = [
            {"stl": "stl/rod_1.stl",
             "occurrence": "trap_stage:1+rod v1:1",
             "body": "Body1"},
            {"stl": "stl/rod_2.stl",
             "occurrence": "trap_stage:1+rod v1:2",
             "body": "Body1"},
            {"stl": "stl/plate.stl", "occurrence": "", "body": "Body1"},
        ]
        text = y.dump_mapping_file("OpticalPaulTrapMkIV", mappings)
        parsed = y.parse(text)
        assert parsed["fusion_design_name"] == "OpticalPaulTrapMkIV"
        assert parsed["mappings"] == mappings

    def test_all_bodies_sentinel_roundtrips(self, y):
        # body: "*" marks an all-bodies (whole occurrence) mapping; '*' is
        # outside the bare-scalar charset so it must be quoted and parse back
        mappings = [
            {"stl": "stl/gate_valve.stl",
             "occurrence": "gate valve v3:1",
             "body": "*"},
        ]
        text = y.dump_mapping_file("RF guide assembly", mappings)
        assert 'body: "*"' in text
        parsed = y.parse(text)
        assert parsed["mappings"] == mappings

    def test_all_bodies_flag_parses_as_bool(self, y):
        parsed = y.parse(
            "decoration:\n"
            "  - name: gate_valve\n"
            "    stl: stl/gate_valve.stl\n"
            "    all_bodies: true\n")
        assert parsed["decoration"][0]["all_bodies"] is True

    def test_double_quotes_survive_specials(self, y):
        # spaces, colons, plus, hash-inside-body-name all need quoting
        mappings = [
            {"stl": "stl/weird.stl",
             "occurrence": "outer:1+inner asm:2",
             "body": "Body 1"},
        ]
        text = y.dump_mapping_file("Design With Spaces", mappings)
        parsed = y.parse(text)
        assert parsed["fusion_design_name"] == "Design With Spaces"
        assert parsed["mappings"] == mappings


# ── rejections ────────────────────────────────────────────────────────────────

class TestRejections:
    def test_tab_indent_rejected(self, y):
        with pytest.raises(ValueError, match="tab"):
            y.parse("k:\n\tv: 1")

    def test_missing_colon_rejected(self, y):
        with pytest.raises(ValueError):
            y.parse("just a bare line")

    def test_unterminated_inline_list_rejected(self, y):
        with pytest.raises(ValueError, match="unterminated"):
            y.parse("k: [1, 2")
