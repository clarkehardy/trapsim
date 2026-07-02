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
  6. Copy each mapped body into a throwaway direct-design document at its
     assembly-world position and export it there with Fusion's STANDARD
     STL writer (high refinement, binary).  TemporaryBRepManager.copy of a
     body PROXY bakes the occurrence transform into the copy — Autodesk's
     recommended workaround for the known ExportManager limitation that
     its STL writer emits body-local coordinates for anything living under
     an occurrence, and the same technique the ExportIt add-in uses.  The
     user's design is never modified; the temp document is closed unsaved.
  7. Every placement is verified twice.  Before export: the copied body's
     bounding box must match the proxy's world bounding box (like-for-like
     BRep boxes; if the plain copy is misplaced, composed occurrence-chain
     transforms are tried as fallbacks).  After export: the written file's
     mesh bounding box must sit at the world box — centre agreement plus
     containment, because Fusion BRep boxes are loose for curved faces and
     extent equality would false-fail e.g. 45°-rotated cylindrical rods.

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
    """Export each mapped body through a throwaway direct-design document
    (see module docstring, steps 6-7).  The user's design is not modified.

    Writes a per-run report to fusion_export_log.txt in the simulation
    folder so results survive the summary dialog being dismissed."""
    exported = 0
    errors = []
    log = [time.strftime("FusionExportSTL run  %Y-%m-%d %H:%M:%S")]
    root = design.rootComponent
    app = adsk.core.Application.get()
    temp_doc = None
    try:
        temp_doc, temp_root, temp_mgr = _create_temp_document(app)
        for stl_rel, body_proxy in resolved.items():
            out = (stl_rel if os.path.isabs(stl_rel)
                   else os.path.join(folder, stl_rel))
            try:
                os.makedirs(os.path.dirname(out), exist_ok=True)
                occ_path = stored_map.get(stl_rel, ("", ""))[0]
                chain = _resolve_chain(root, occ_path) or []
                note = _export_via_temp_doc(
                    temp_root, temp_mgr, out, body_proxy, chain)
                exported += 1
                log.append(f"OK    {stl_rel}"
                           + (f"  [{note}]" if note else ""))
            except Exception as ex:                                  # noqa: BLE001
                errors.append(f"{stl_rel}: {ex}")
                log.append(f"FAIL  {stl_rel}: {ex}")
    finally:
        if temp_doc is not None:
            try:
                temp_doc.close(False)                # never save
            except Exception:                                        # noqa: BLE001
                pass
        try:
            with open(os.path.join(folder, EXPORT_LOG_FILENAME), "a",
                      encoding="utf-8") as f:
                f.write("\n".join(log) + "\n\n")
        except OSError:
            pass
    return exported, errors


def _create_temp_document(app):
    """Open a scratch direct-design document to export from.  Direct design
    means bodies can be added without the baseFeature edit dance."""
    doc = app.documents.add(
        adsk.core.DocumentTypes.FusionDesignDocumentType)
    design = adsk.fusion.Design.cast(
        doc.products.itemByProductType("DesignProductType"))
    design.designType = adsk.fusion.DesignTypes.DirectDesignType
    design.fusionUnitsManager.distanceDisplayUnits = (
        adsk.fusion.DistanceUnits.MillimeterDistanceUnits)
    return doc, design.rootComponent, design.exportManager


def _bbox_mm(bounding_box):
    """Fusion BoundingBox3D (internal cm) → ((min3), (max3)) in mm."""
    lo = bounding_box.minPoint
    hi = bounding_box.maxPoint
    return ((lo.x * 10.0, lo.y * 10.0, lo.z * 10.0),
            (hi.x * 10.0, hi.y * 10.0, hi.z * 10.0))


def _export_via_temp_doc(temp_root, temp_mgr, out_path, body_proxy, chain):
    """Copy `body_proxy` at its assembly-world position into the temp
    document, export it with Fusion's standard STL writer, verify, and
    remove the copy.  Returns a short placement note for the log."""
    brep_mgr = adsk.fusion.TemporaryBRepManager.get()
    world_box = _bbox_mm(body_proxy.boundingBox)

    # 1. world-positioned temporary copy.  Copying a proxy is documented to
    #    bake the occurrence transform; trust it only after checking its
    #    BRep box against the proxy's (identical computation → tight match),
    #    and fall back to explicit chain transforms if it is misplaced.
    note = ""
    copied = brep_mgr.copy(body_proxy)
    if not _brep_boxes_close(_bbox_mm(copied.boundingBox), world_box):
        native = (body_proxy.nativeObject
                  if body_proxy.assemblyContext is not None else body_proxy)
        copied = None
        tried = []
        for label, cells in _matrix_candidates(chain):
            candidate = brep_mgr.copy(native)
            matrix = adsk.core.Matrix3D.create()
            matrix.setWithArray(list(cells))
            brep_mgr.transform(candidate, matrix)
            cand_box = _bbox_mm(candidate.boundingBox)
            if _brep_boxes_close(cand_box, world_box):
                copied = candidate
                note = label
                break
            tried.append((label, cand_box))
        if copied is None:
            raise RuntimeError(
                "could not place a copy of the body at its assembly "
                f"position {world_box[0]}..{world_box[1]} mm; candidates: "
                + "; ".join(f"{lbl} → {b[0]}..{b[1]}" for lbl, b in tried))

    # 2. add to the temp document (root level, identity frame) and export
    #    with the standard writer.
    added = temp_root.bRepBodies.add(copied)
    try:
        opts = temp_mgr.createSTLExportOptions(added, out_path)
        opts.meshRefinement = (
            adsk.fusion.MeshRefinementSettings.MeshRefinementHigh)
        opts.sendToPrintUtility = False
        opts.isBinaryFormat = True
        temp_mgr.execute(opts)
    finally:
        try:
            added.deleteMe()
        except Exception:                                            # noqa: BLE001
            pass

    # 3. verify the file: tight mesh box vs loose BRep box.
    lo, hi, _ntri = stl_check.read_binary_stl_bbox(out_path)
    diag = sum((world_box[1][i] - world_box[0][i]) ** 2
               for i in range(3)) ** 0.5
    center_tol = max(1.0, 0.02 * diag)
    if not stl_check.placement_plausible((lo, hi), world_box, center_tol):
        raise RuntimeError(
            f"exported file spans {lo}..{hi} mm but the body sits at "
            f"{world_box[0]}..{world_box[1]} mm in the assembly")
    return note


def _brep_boxes_close(box_a, box_b, tol_mm=0.2):
    """Two BRep bounding boxes of the same body under the same placement
    are the same computation — compare per-component with a small slack."""
    for lo_a, lo_b in zip(box_a[0], box_b[0]):
        if abs(lo_a - lo_b) > tol_mm:
            return False
    for hi_a, hi_b in zip(box_a[1], box_b[1]):
        if abs(hi_a - hi_b) > tol_mm:
            return False
    return True


def _matrix_candidates(chain):
    """Yield (label, 16-float row-major matrix) world-placement candidates
    for a body whose plain proxy copy came out misplaced.  transform2 is
    parent-relative, so the chain product maps component-local to world;
    the raw leaf transform is tried in case this Fusion version bakes the
    full context into proxy transforms."""
    if not chain:
        yield ("identity", list(stl_check.MAT4_IDENTITY))
        return
    composed = list(stl_check.MAT4_IDENTITY)
    for occ in chain:
        native = getattr(occ, "nativeObject", None) or occ
        composed = stl_check.mat4_multiply(
            composed, list(native.transform2.asArray()))
    yield ("chain transform", composed)
    if len(chain) > 1:
        yield ("leaf transform", list(chain[-1].transform2.asArray()))
