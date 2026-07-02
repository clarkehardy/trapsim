"""Tests for `trapsim.fusion` — install helper for the FusionExportSTL script.

We drive `install()` with `dest=tmp_path` so nothing is written to the real
Fusion Scripts folder.  This exercises the entire copy / symlink / force /
is_installed logic without depending on Fusion being present on the host.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

import pytest

from trapsim import fusion as trapsim_fusion


# ── install() ─────────────────────────────────────────────────────────────────

class TestInstall:
    def test_fresh_install_copies_files(self, tmp_path):
        target = trapsim_fusion.install(dest=tmp_path)
        assert target == tmp_path / "FusionExportSTL"
        assert (target / "FusionExportSTL.py").is_file()
        assert (target / "FusionExportSTL.manifest").is_file()
        assert (target / "yaml_subset.py").is_file()
        assert (target / "stl_check.py").is_file()
        assert not (target / "__pycache__").exists()

    def test_reinstall_without_force_raises(self, tmp_path):
        trapsim_fusion.install(dest=tmp_path)
        with pytest.raises(FileExistsError):
            trapsim_fusion.install(dest=tmp_path)

    def test_force_overwrites(self, tmp_path):
        target = trapsim_fusion.install(dest=tmp_path)
        (target / "FusionExportSTL.py").write_text("junk\n")
        target2 = trapsim_fusion.install(dest=tmp_path, force=True)
        assert target == target2
        assert (target2 / "FusionExportSTL.py").read_text() != "junk\n"

    def test_force_overwrites_symlink(self, tmp_path):
        target = trapsim_fusion.install(dest=tmp_path, symlink=True)
        assert target.is_symlink()
        target2 = trapsim_fusion.install(dest=tmp_path, force=True)
        assert not target2.is_symlink()
        assert (target2 / "FusionExportSTL.py").is_file()

    def test_symlink_points_at_shipped_source(self, tmp_path):
        target = trapsim_fusion.install(dest=tmp_path, symlink=True)
        assert target.is_symlink()
        # dereferencing the symlink yields the real _script/ inside trapsim
        real = target.resolve()
        assert real.name == "_script"
        assert (real / "FusionExportSTL.py").is_file()

    def test_creates_dest_dir_if_missing(self, tmp_path):
        nested = tmp_path / "does" / "not" / "exist" / "yet"
        target = trapsim_fusion.install(dest=nested)
        assert (target / "FusionExportSTL.py").is_file()


# ── is_installed() ────────────────────────────────────────────────────────────

class TestIsInstalled:
    def test_false_when_empty(self, tmp_path):
        assert trapsim_fusion.is_installed(dest=tmp_path) is False

    def test_true_after_install(self, tmp_path):
        trapsim_fusion.install(dest=tmp_path)
        assert trapsim_fusion.is_installed(dest=tmp_path) is True


# ── fusion_scripts_dir() ──────────────────────────────────────────────────────

class TestScriptsDir:
    def test_mac_or_windows_returns_reasonable_path(self):
        system = platform.system()
        if system not in ("Darwin", "Windows"):
            with pytest.raises(RuntimeError):
                trapsim_fusion.fusion_scripts_dir()
            return
        p = trapsim_fusion.fusion_scripts_dir()
        assert p.name == "Scripts"
        assert "Autodesk Fusion 360" in str(p)


# ── CLI smoke ────────────────────────────────────────────────────────────────

class TestCLI:
    def test_status_prints_scripts_dir(self):
        r = subprocess.run(
            [sys.executable, "-m", "trapsim.fusion", "status"],
            capture_output=True, text=True, check=False)
        # Non-Darwin/Windows platforms error cleanly with rc=2
        if platform.system() not in ("Darwin", "Windows"):
            assert r.returncode == 2
            return
        assert r.returncode == 0, r.stderr
        assert "Fusion Scripts folder" in r.stdout
        assert "FusionExportSTL" in r.stdout
