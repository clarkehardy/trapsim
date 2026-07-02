"""FusionExportSTL — trapsim STL exporter for Autodesk Fusion 360.

Runs inside Fusion's embedded Python (invoked via
Tools → Scripts and Add-Ins → FusionExportSTL → Run).

Flow (all in one shot):
  1. Prompt for the simulation folder (containing geometry.yaml).
  2. Read geometry.yaml → list of (stl_path, category, entity_name) targets.
  3. Read fusion_map.yaml if present; resolve each mapping to a live body proxy.
     Any mapping that no longer resolves (renamed / deleted / moved) is
     treated as unmapped.
  4. For each target still unmapped, prompt the user to click the body in the
     viewport (adsk.core.UserInterface.selectEntity — this returns a body proxy
     with .assemblyContext set to the specific Occurrence clicked, so
     ':1' and ':2' copies are naturally distinguished).
  5. Write fusion_map.yaml.
  6. Export each mapped body via Design.exportManager.createSTLExportOptions
     using the body PROXY (not the raw body) so the occurrence transform is
     applied automatically — same effect as the manual "isolate + save from
     top-level" workflow.

The script has no state between runs beyond fusion_map.yaml + a
last-used-folder preference stored in Fusion's user prefs.
"""

import os
import sys
import traceback

import adsk.core
import adsk.fusion

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import yaml_subset  # noqa: E402  (bundled next to this file)


GEOMETRY_FILENAME  = "geometry.yaml"
STL_MAP_FILENAME   = "fusion_map.yaml"
LAST_FOLDER_ATTR   = "trapsim_last_folder"


# ── Entry point ───────────────────────────────────────────────────────────────

def run(_context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui  = app.userInterface
        design = adsk.fusion.Design.cast(app.activeProduct)
        if design is None:
            ui.messageBox("No active Fusion design.  Open a design first.")
            return

        design_name = app.activeDocument.name
        root = design.rootComponent

        folder = _pick_folder(app, ui)
        if folder is None:
            return

        try:
            targets = _load_targets(folder)
        except Exception as ex:                                     # noqa: BLE001
            ui.messageBox(f"Failed to read {GEOMETRY_FILENAME}:\n{ex}")
            return
        if not targets:
            ui.messageBox(
                f"{GEOMETRY_FILENAME} has no electrodes / dielectrics / decoration.")
            return

        stored_map, stored_design = _load_map(folder)
        if stored_design and stored_design != design_name:
            r = ui.messageBox(
                f"{STL_MAP_FILENAME} was made for design '{stored_design}', "
                f"but the active design is '{design_name}'.\n\n"
                "Re-use the existing map anyway?",
                "Design mismatch",
                adsk.core.MessageBoxButtonTypes.YesNoButtonType)
            if r != adsk.core.DialogResults.DialogYes:
                stored_map = {}

        resolved, unmapped, stale = _resolve_all(root, targets, stored_map)

        if stale:
            ui.messageBox(
                "The following mappings no longer resolve and will be re-picked:\n\n" +
                "\n".join(f"  {s}" for s in stale),
                "Stale mappings")

        if unmapped:
            r = ui.messageBox(
                f"{len(unmapped)} of {len(targets)} bodies need mapping.\n\n"
                "You'll be asked to click each one in the viewport.\n"
                "Escape at a prompt to skip that body.\n\n"
                "Continue?",
                "Map bodies",
                adsk.core.MessageBoxButtonTypes.YesNoButtonType)
            if r != adsk.core.DialogResults.DialogYes:
                return
            for stl, category, entity_name in unmapped:
                picked = _prompt_pick(ui, category, entity_name, stl)
                if picked is None:
                    r = ui.messageBox(
                        f"Skipped '{stl}'.\n\nContinue with the next body?",
                        "Skipped",
                        adsk.core.MessageBoxButtonTypes.YesNoButtonType)
                    if r != adsk.core.DialogResults.DialogYes:
                        return
                    continue
                occ_path, body_name = picked
                stored_map[stl] = (occ_path, body_name)
                body_proxy = _resolve_body(root, occ_path, body_name)
                if body_proxy is not None:
                    resolved[stl] = body_proxy

        _save_map(folder, design_name, stored_map)

        if not resolved:
            ui.messageBox("Nothing to export.")
            return

        r = ui.messageBox(
            f"Ready to export {len(resolved)} of {len(targets)} STL files.\n"
            f"Destination: {folder}\n\n"
            "Existing files will be overwritten.  Proceed?",
            "Export",
            adsk.core.MessageBoxButtonTypes.YesNoButtonType)
        if r != adsk.core.DialogResults.DialogYes:
            return

        exported, errors = _export_all(design, folder, resolved)

        msg = f"Exported {exported} of {len(resolved)} STL files."
        if errors:
            msg += "\n\nErrors:\n" + "\n".join(f"  {e}" for e in errors)
        ui.messageBox(msg, "Export complete")

    except Exception:                                               # noqa: BLE001
        if ui:
            ui.messageBox("Unhandled error:\n\n" + traceback.format_exc())


# ── Folder selection ──────────────────────────────────────────────────────────

def _pick_folder(app, ui):
    dlg = ui.createFolderDialog()
    dlg.title = "Select simulation folder (contains geometry.yaml)"
    last = _get_last_folder(app)
    if last and os.path.isdir(last):
        dlg.initialDirectory = last
    if dlg.showDialog() != adsk.core.DialogResults.DialogOK:
        return None
    folder = dlg.folder
    if not os.path.isfile(os.path.join(folder, GEOMETRY_FILENAME)):
        ui.messageBox(
            f"{folder}\n\ndoes not contain {GEOMETRY_FILENAME}.  Aborting.")
        return None
    _set_last_folder(app, folder)
    return folder


def _get_last_folder(app):
    try:
        attrs = app.activeDocument.attributes
        a = attrs.itemByName("trapsim", LAST_FOLDER_ATTR)
        return a.value if a else None
    except Exception:                                                # noqa: BLE001
        return None


def _set_last_folder(app, folder):
    try:
        app.activeDocument.attributes.add("trapsim", LAST_FOLDER_ATTR, folder)
    except Exception:                                                # noqa: BLE001
        pass


# ── geometry.yaml → target list ───────────────────────────────────────────────

def _load_targets(folder):
    with open(os.path.join(folder, GEOMETRY_FILENAME)) as f:
        raw = yaml_subset.parse(f.read()) or {}
    targets = []
    for e in raw.get("electrodes") or []:
        name = e.get("name", "?")
        for s in e.get("stls") or []:
            targets.append((s, "electrode", name))
    for d in raw.get("dielectrics") or []:
        s = d.get("stl")
        if s:
            targets.append((s, "dielectric", d.get("name", "?")))
    for dec in raw.get("decoration") or []:
        s = dec.get("stl")
        if s:
            targets.append((s, "decoration", dec.get("name", "?")))
    return targets


# ── fusion_map.yaml I/O ───────────────────────────────────────────────────────

def _load_map(folder):
    path = os.path.join(folder, STL_MAP_FILENAME)
    if not os.path.isfile(path):
        return {}, None
    with open(path) as f:
        raw = yaml_subset.parse(f.read()) or {}
    design_name = raw.get("fusion_design_name")
    mp = {}
    for m in raw.get("mappings") or []:
        stl = m.get("stl")
        if stl:
            mp[stl] = (m.get("occurrence") or "", m.get("body") or "")
    return mp, design_name


def _save_map(folder, design_name, mp):
    mappings = [{"stl": stl, "occurrence": occ, "body": body}
                for stl, (occ, body) in mp.items()]
    with open(os.path.join(folder, STL_MAP_FILENAME), "w") as f:
        f.write(yaml_subset.dump_mapping_file(design_name, mappings))


# ── Occurrence-path → body proxy resolution ───────────────────────────────────

def _resolve_all(root_component, targets, stored_map):
    """Return (resolved: {stl: body_proxy},
                unmapped: [(stl, category, name), ...],
                stale:    [stl, ...])."""
    resolved = {}
    unmapped = []
    stale    = []
    for stl, category, entity_name in targets:
        if stl in stored_map:
            occ_path, body_name = stored_map[stl]
            body_proxy = _resolve_body(root_component, occ_path, body_name)
            if body_proxy is not None:
                resolved[stl] = body_proxy
                continue
            stale.append(stl)
        unmapped.append((stl, category, entity_name))
    return resolved, unmapped, stale


def _resolve_body(root_component, occurrence_path, body_name):
    """Walk `occurrence_path` (Fusion's `+`-separated fullPathName) from the
    root component and return the named body proxy, or None if missing."""
    if occurrence_path:
        parts = occurrence_path.split("+")
        container = root_component.occurrences
        occurrence = None
        for part in parts:
            found = None
            for i in range(container.count):
                o = container.item(i)
                if o.name == part:
                    found = o
                    break
            if found is None:
                return None
            occurrence = found
            container = occurrence.childOccurrences
        bodies = occurrence.bRepBodies
    else:
        bodies = root_component.bRepBodies
    for i in range(bodies.count):
        b = bodies.item(i)
        if b.name == body_name:
            return b
    return None


# ── Interactive body pick ─────────────────────────────────────────────────────

def _prompt_pick(ui, category, entity_name, stl_path):
    """Ask the user to click a body in the viewport.  Returns
    (occurrence_full_path, body_name) or None on cancel."""
    prompt = (f"Click the body for {stl_path}    "
              f"({category}: {entity_name})")
    try:
        selection = ui.selectEntity(prompt, "SolidBodies")
    except Exception:                                                # noqa: BLE001
        # User pressed Escape → SDK raises.  Treat as skip.
        return None
    if selection is None:
        return None
    body = selection.entity
    if not isinstance(body, adsk.fusion.BRepBody):
        return None
    ctx = body.assemblyContext
    occ_path = ctx.fullPathName if ctx is not None else ""
    return (occ_path, body.name)


# ── STL export ────────────────────────────────────────────────────────────────

def _export_all(design, folder, resolved):
    exported = 0
    errors = []
    mgr = design.exportManager
    for stl_rel, body_proxy in resolved.items():
        out = stl_rel if os.path.isabs(stl_rel) else os.path.join(folder, stl_rel)
        try:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            opts = mgr.createSTLExportOptions(body_proxy, out)
            opts.meshRefinement = (
                adsk.fusion.MeshRefinementSettings.MeshRefinementHigh)
            opts.sendToPrintUtility = False
            opts.isBinaryFormat = True
            mgr.execute(opts)
            exported += 1
        except Exception as ex:                                      # noqa: BLE001
            errors.append(f"{stl_rel}: {ex}")
    return exported, errors
