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
  6. For each mapped body, tessellate it in-process (MeshCalculator — the
     same tessellator the export dialog uses) and write the binary STL
     directly.  Fusion's exportManager is NOT used: empirically its STL
     writer emits body-LOCAL coordinates no matter what it is handed
     (body, proxy, or isolated top-level component), silently discarding
     occurrence transforms.
  7. The assembly-world placement is chosen by verification, not trust:
     the mesh is tried as-returned and under the composed occurrence-chain
     transforms, and the candidate whose bounding box matches the body
     proxy's world bounding box (a reliable proxy query) is written.  If
     no candidate matches, nothing is written and the error reports every
     candidate box.

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
            "Meshes are computed in-process; nothing in your design is "
            "modified.  Existing files will be overwritten.  Proceed?",
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


# ── STL export ────────────────────────────────────────────────────────────────

def _export_all(design, folder, resolved, stored_map):
    """Tessellate each mapped body in-process and write its STL directly,
    placing the triangles in assembly-world coordinates (see _export_body).

    Writes a per-run report to fusion_export_log.txt in the simulation
    folder so results survive the summary dialog being dismissed."""
    exported = 0
    errors = []
    log = [time.strftime("FusionExportSTL run  %Y-%m-%d %H:%M:%S")]
    root = design.rootComponent
    for stl_rel, body_proxy in resolved.items():
        out = (stl_rel if os.path.isabs(stl_rel)
               else os.path.join(folder, stl_rel))
        try:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            occ_path = stored_map.get(stl_rel, ("", ""))[0]
            chain = _resolve_chain(root, occ_path) or []
            note = _export_body(out, body_proxy, chain)
            exported += 1
            log.append(f"OK    {stl_rel}" + (f"  [{note}]" if note else ""))
        except Exception as ex:                                      # noqa: BLE001
            errors.append(f"{stl_rel}: {ex}")
            log.append(f"FAIL  {stl_rel}: {ex}")
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


def _export_body(out_path, body_proxy, chain):
    """Tessellate `body_proxy`, place the mesh in the assembly frame, and
    write the binary STL in mm.

    Placement is chosen by verification: the tessellation is tried
    as-returned and under the composed occurrence-chain transforms, and
    the first candidate whose bounding box matches the body proxy's
    assembly-world bounding box wins.  Whatever coordinate convention this
    Fusion version's MeshCalculator / transform2 actually follows, the
    written file provably lands where the body sits in the assembly.

    Returns a short placement note for the log.
    """
    calc = body_proxy.meshManager.createMeshCalculator()
    calc.setQuality(
        adsk.fusion.TriangleMeshQualityOptions.HighQualityTriangleMesh)
    mesh = calc.calculate()
    if mesh is None:
        raise RuntimeError("tessellation failed")
    coords_cm = list(mesh.nodeCoordinatesAsDouble)
    indices = list(mesh.nodeIndices)
    if not indices:
        raise RuntimeError("tessellation produced no triangles")

    world_box = _bbox_mm(body_proxy.boundingBox)
    diag = sum((world_box[1][i] - world_box[0][i]) ** 2
               for i in range(3)) ** 0.5
    tol = max(1.0, 0.02 * diag)

    tried = []
    for label, m in _placement_candidates(chain):
        pts_cm = (coords_cm if m is None
                  else stl_check.transform_points(m, coords_cm))
        lo, hi = stl_check.bbox_of_points(pts_cm)
        box_mm = (tuple(c * 10.0 for c in lo), tuple(c * 10.0 for c in hi))
        if stl_check.boxes_match(box_mm, world_box, tol):
            stl_check.write_binary_stl(out_path, pts_cm, indices, scale=10.0)
            return "" if label == "as-returned" else label
        tried.append((label, box_mm))
    raise RuntimeError(
        "could not place mesh in the assembly frame: body world box "
        f"{world_box[0]}..{world_box[1]} mm; candidates: "
        + "; ".join(f"{lbl} → {b[0]}..{b[1]}" for lbl, b in tried))


def _placement_candidates(chain):
    """Yield (label, matrix-or-None) placements to try, in order.

    'as-returned' covers a MeshCalculator that already returns root-context
    coordinates (proxy behaviour).  The transform candidates cover a
    calculator that returns component-local coordinates: occurrence
    transforms compose parent→child down the chain (transform2 is
    parent-relative), and the raw leaf transform is tried last in case
    this Fusion version bakes the full context into proxy transforms."""
    yield ("as-returned", None)
    if chain:
        composed = list(stl_check.MAT4_IDENTITY)
        for occ in chain:
            native = getattr(occ, "nativeObject", None) or occ
            composed = stl_check.mat4_multiply(
                composed, list(native.transform2.asArray()))
        yield ("chain transform", composed)
        if len(chain) > 1:
            yield ("leaf transform", list(chain[-1].transform2.asArray()))
