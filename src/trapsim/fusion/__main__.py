"""CLI wrapper: `python -m trapsim.fusion {install|status}`."""

from __future__ import annotations

import argparse
import sys

from . import (
    SCRIPT_NAME,
    fusion_scripts_dir,
    install,
    is_installed,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m trapsim.fusion",
        description="Install or inspect the FusionExportSTL script for trapsim.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_install = sub.add_parser(
        "install",
        help="Copy the script into Fusion 360's Scripts folder.")
    p_install.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing installation.")
    p_install.add_argument(
        "--symlink", action="store_true",
        help="Symlink instead of copy (nice for editing in-place).")

    sub.add_parser(
        "status",
        help="Show whether the script is installed and where.")

    args = p.parse_args(argv)

    try:
        scripts_dir = fusion_scripts_dir()
    except RuntimeError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 2

    if args.cmd == "install":
        try:
            target = install(force=args.force, symlink=args.symlink)
        except FileExistsError as ex:
            print(f"error: {ex}", file=sys.stderr)
            return 1
        print(f"Installed {SCRIPT_NAME} → {target}")
        print("In Fusion 360:  Tools → Scripts and Add-Ins → "
              f"{SCRIPT_NAME} → Run")
        return 0

    if args.cmd == "status":
        print(f"Fusion Scripts folder: {scripts_dir}")
        installed = is_installed(scripts_dir)
        print(f"{SCRIPT_NAME}: "
              f"{'installed' if installed else 'not installed'}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
