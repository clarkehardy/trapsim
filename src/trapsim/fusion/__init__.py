"""trapsim.fusion  –  install helper for the FusionExportSTL script.

The script itself lives in `trapsim/fusion/_script/` and is written to run
inside Fusion 360's embedded Python.  This module (which runs in the user's
system Python) copies the shipped script into Fusion's Scripts folder so it
appears in Tools → Scripts and Add-Ins.

    python -m trapsim.fusion install
    python -m trapsim.fusion status

See README.md for the full workflow.
"""

from __future__ import annotations

import os
import platform
import shutil
from importlib import resources
from pathlib import Path

SCRIPT_NAME = "FusionExportSTL"


# ── Platform-specific Fusion paths ────────────────────────────────────────────

def fusion_scripts_dir() -> Path:
    """Return the platform-specific Fusion 360 Scripts folder."""
    system = platform.system()
    if system == "Darwin":
        return (Path.home() / "Library" / "Application Support"
                / "Autodesk" / "Autodesk Fusion 360" / "API" / "Scripts")
    if system == "Windows":
        appdata = os.environ.get("APPDATA")
        if not appdata:
            raise RuntimeError("APPDATA environment variable is not set")
        return Path(appdata) / "Autodesk" / "Autodesk Fusion 360" / "API" / "Scripts"
    raise RuntimeError(
        f"Fusion 360 is not officially supported on {system}; "
        "no known Scripts folder location.")


# ── Install / status ─────────────────────────────────────────────────────────

def install(*, force: bool = False,
            symlink: bool = False,
            dest: Path | None = None) -> Path:
    """Copy (or symlink) the shipped script into Fusion's Scripts folder.

    Returns the path to the installed `FusionExportSTL/` folder.
    """
    if dest is None:
        dest = fusion_scripts_dir()
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / SCRIPT_NAME

    if target.exists() or target.is_symlink():
        if not force:
            raise FileExistsError(
                f"{target} already exists.  Re-run with --force to overwrite.")
        _remove(target)

    src_pkg = resources.files(__package__).joinpath("_script")
    with resources.as_file(src_pkg) as src_dir:
        if symlink:
            target.symlink_to(Path(src_dir).resolve())
        else:
            shutil.copytree(src_dir, target,
                            ignore=shutil.ignore_patterns("__pycache__"))

    return target


def is_installed(dest: Path | None = None) -> bool:
    if dest is None:
        dest = fusion_scripts_dir()
    p = Path(dest) / SCRIPT_NAME / f"{SCRIPT_NAME}.py"
    return p.exists()


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)
