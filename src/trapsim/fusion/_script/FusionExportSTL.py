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
  6. For each mapped body, programmatically reproduce the manual export
     recipe: hide every occurrence and body (what Isolate does), show only
     the target body and its occurrence chain, then export the TOP-LEVEL
     assembly via Design.exportManager.  Exporting from the top level is
     what bakes the occurrence transforms in — exporting the body (even its
     proxy) directly writes body-LOCAL coordinates and stacks every part at
     the origin.  Visibility state is restored afterwards.
  7. Verify each written STL: its bounding box must match the body's
     assembly-world bounding box, so a wrong-frame export fails loudly.

The script has no state between runs beyond fusion_map.yaml + a
last-used-folder preference stored in Fusion's user prefs.
"""

import importlib
import os
import sys
import time
import traceback

import adsk.core
import adsk.fusion

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import stl_check    # noqa: E402  (bundled next to this file)
import yaml_subset  # noqa: E402  (bundled next to this file)

# Fusion keeps its embedded interpreter alive between script runs, so a
# previously imported copy of the bundled helpers would shadow an updated
# install (the main script is reloaded each run, its imports are not).
# Reload them every run so they always match what's on disk.
stl_check = importlib.reload(stl_check)
yaml_subset = importlib.reload(yaml_subset)


GEOMETRY_FILENAME   = "geometry.yaml"
STL_MAP_FILENAME    = "fusion_map.yaml"
EXPORT_LOG_FILENAME = "fusion_export_log.txt"
LAST_FOLDER_ATTR    = "trapsim_last_folder"


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
            "Each body is temporarily isolated during its export "
            "(visibility is restored afterwards).  "
            "Existing files will be overwritten.  Proceed?",
            "Export",
            adsk.core.MessageBoxButtonTypes.YesNoButtonType)
        if r != adsk.core.DialogResults.DialogYes:
            return

        exported, errors = _export_all(design, folder, resolved, stored_map)

        msg = f"Exported {exported} of {len(resolved)} STL files."
        if errors:
            msg += "\n\nErrors:\n" + "\n".join(f"  {e}" for e in errors)
        msg += f"\n\nFull report: {os.path.join(folder, EXPORT_LOG_FILENAME)}"
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
    with open(os.path.join(folder, GEOMETRY_FILENAME), encoding="utf-8") as f:
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
    with open(path, encoding="utf-8") as f:
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
    with open(os.path.join(folder, STL_MAP_FILENAME), "w",
              encoding="utf-8") as f:
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


def _resolve_chain(root_component, occurrence_path):
    """Walk `occurrence_path` (Fusion's `+`-separated fullPathName) from the
    root component.  Returns the list of occurrences root→leaf ([] for a
    root-level body), or None if any path segment is missing."""
    chain = []
    if occurrence_path:
        container = root_component.occurrences
        for part in occurrence_path.split("+"):
            found = None
            for i in range(container.count):
                o = container.item(i)
                if o.name == part:
                    found = o
                    break
            if found is None:
                return None
            chain.append(found)
            container = found.childOccurrences
    return chain


def _resolve_body(root_component, occurrence_path, body_name):
    """Return the named body proxy at `occurrence_path`, or None if missing."""
    chain = _resolve_chain(root_component, occurrence_path)
    if chain is None:
        return None
    bodies = chain[-1].bRepBodies if chain else root_component.bRepBodies
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


# ── Visibility bookkeeping ────────────────────────────────────────────────────
# Body light bulbs are a property of the NATIVE body: hiding "Body1" of a
# component hides it in every occurrence of that component.  So isolating one
# copy of a 4×-instanced rod means: show the native body, show the occurrence
# chain leading to the wanted copy, and keep the sibling copies' occurrences
# hidden.  Occurrence bulbs are per-occurrence, which is what makes this work.

def _record_visibility(design):
    root = design.rootComponent
    occ_states, folder_states, body_states = [], [], []
    occs = root.allOccurrences
    for i in range(occs.count):
        o = occs.item(i)
        occ_states.append((o, o.isLightBulbOn))
    comps = design.allComponents
    for i in range(comps.count):
        c = comps.item(i)
        folder_states.append((c, c.isBodiesFolderLightBulbOn))
        bodies = c.bRepBodies
        for j in range(bodies.count):
            b = bodies.item(j)
            body_states.append((b, b.isLightBulbOn))
    return occ_states, folder_states, body_states


def _restore_visibility(state):
    occ_states, folder_states, body_states = state
    for o, on in occ_states:
        try:
            o.isLightBulbOn = on
        except Exception:                                            # noqa: BLE001
            pass
    for c, on in folder_states:
        try:
            c.isBodiesFolderLightBulbOn = on
        except Exception:                                            # noqa: BLE001
            pass
    for b, on in body_states:
        try:
            b.isLightBulbOn = on
        except Exception:                                            # noqa: BLE001
            pass


def _isolate(design, chain, body_proxy):
    """Reproduce the manual Isolate: hide every occurrence and body, then
    show only `body_proxy` and the occurrence chain leading to it."""
    root = design.rootComponent
    occs = root.allOccurrences
    for i in range(occs.count):
        occs.item(i).isLightBulbOn = False
    comps = design.allComponents
    for i in range(comps.count):
        bodies = comps.item(i).bRepBodies
        for j in range(bodies.count):
            bodies.item(j).isLightBulbOn = False
    for occ in chain:
        occ.isLightBulbOn = True
    body_proxy.parentComponent.isBodiesFolderLightBulbOn = True
    body_proxy.isLightBulbOn = True


# ── STL export ────────────────────────────────────────────────────────────────

def _export_all(design, folder, resolved, stored_map):
    """Export each mapped body by isolating it and exporting the top-level
    assembly — the only path through Fusion's export pipeline that applies
    occurrence transforms, i.e. produces assembly-world coordinates.

    Writes a per-run report to fusion_export_log.txt in the simulation
    folder so results survive the summary dialog being dismissed."""
    exported = 0
    errors = []
    log = [time.strftime("FusionExportSTL run  %Y-%m-%d %H:%M:%S")]
    mgr = design.exportManager
    root = design.rootComponent
    state = _record_visibility(design)
    try:
        for stl_rel, body_proxy in resolved.items():
            out = (stl_rel if os.path.isabs(stl_rel)
                   else os.path.join(folder, stl_rel))
            try:
                os.makedirs(os.path.dirname(out), exist_ok=True)
                occ_path = stored_map.get(stl_rel, ("", ""))[0]
                chain = _resolve_chain(root, occ_path) or []
                _isolate(design, chain, body_proxy)
                adsk.doEvents()
                opts = mgr.createSTLExportOptions(root, out)
                opts.meshRefinement = (
                    adsk.fusion.MeshRefinementSettings.MeshRefinementHigh)
                opts.sendToPrintUtility = False
                opts.isBinaryFormat = True
                mgr.execute(opts)
                note = _verify_export(out, body_proxy)
                exported += 1
                log.append(f"OK    {stl_rel}"
                           + (f"  [{note}]" if note else ""))
            except Exception as ex:                                  # noqa: BLE001
                errors.append(f"{stl_rel}: {ex}")
                log.append(f"FAIL  {stl_rel}: {ex}")
    finally:
        _restore_visibility(state)
        try:
            with open(os.path.join(folder, EXPORT_LOG_FILENAME), "a",
                      encoding="utf-8") as f:
                f.write("\n".join(log) + "\n\n")
        except OSError:
            pass
    return exported, errors


def _bbox_mm(bounding_box):
    """Fusion BoundingBox3D (internal cm) → ((min3), (max3)) in mm."""
    lo = bounding_box.minPoint
    hi = bounding_box.maxPoint
    return ((lo.x * 10.0, lo.y * 10.0, lo.z * 10.0),
            (hi.x * 10.0, hi.y * 10.0, hi.z * 10.0))


def _verify_export(path, body_proxy):
    """Check the written STL's bounding box against the body's world
    bounding box (proxy geometry queries return root-context coordinates).

    Fusion's API STL writer emits internal units (cm) — there is no unit
    option like the manual dialog has — so a mesh that matches the world
    box at 10x scale is rescaled to mm in place.  A mesh matching the
    body-LOCAL box means the occurrence transform was not applied.

    Returns a short note for the log ('' or 'rescaled cm → mm'), raises
    on anything unfixable.
    """
    if not os.path.isfile(path):
        raise RuntimeError("export wrote no file")
    lo, hi, _ntri = stl_check.read_binary_stl_bbox(path)
    file_box = (lo, hi)

    world_box = _bbox_mm(body_proxy.boundingBox)
    native = (body_proxy.nativeObject
              if body_proxy.assemblyContext is not None else body_proxy)
    local_box = _bbox_mm(native.boundingBox)

    status = stl_check.diagnose_frame(file_box, world_box, local_box)
    if status == "ok":
        return ""
    if status == "cm":
        stl_check.scale_binary_stl(path, 10.0)
        return "rescaled cm → mm"
    if status in ("local", "local_cm"):
        raise RuntimeError(
            "mesh is in body-local coordinates — the occurrence transform "
            "was not applied"
            + (" (and units are cm)" if status == "local_cm" else "")
            + f": file spans {lo}..{hi}, body sits at "
            f"{world_box[0]}..{world_box[1]} mm in the assembly")
    raise RuntimeError(
        f"bounding-box mismatch: file spans {lo}..{hi} (raw units), "
        f"expected world {world_box[0]}..{world_box[1]} mm, "
        f"body-local {local_box[0]}..{local_box[1]} mm")
