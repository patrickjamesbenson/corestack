import streamlit as st
def _safe_rerun():
    try:
        return st.rerun()
    except Exception:
        return None
import streamlit as st
def _safe_set_page_config(*a, **k):
    try:
        return st.set_page_config(*a, **k)
    except Exception:
        return None
# =========================================
# File: app.py — full pipeline UI (final workflow wired)
# =========================================
from __future__ import annotations

import os, re, json
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

# -----------------------------------------------------
# Page config FIRST
# -----------------------------------------------------
_safe_set_page_config(page_title="ies Prep", layout="wide")

# -----------------------------------------------------
# Imports (guard optional backends)
# -----------------------------------------------------
try:
    from polar import build_polar_figure, figure_png_bytes, compute_polar_metrics
    HAVE_PLOT = True
except Exception:
    HAVE_PLOT = False

try:
    from ugr_table_v2 import (
        Photometry as UPhot,
        build_ugr_tables,
        to_text as ugr_to_text,
        to_csv as ugr_to_csv,
        DEFAULT_REFLECTANCE_SETS,
    )
    HAVE_UGR = True
except Exception:
    HAVE_UGR = False

from ies_prep import (
    parse_ies_input,
    interpolate_candela_matrix,
    apply_geometry_overrides,
    _load_metadata_schema_from_excel,
    apply_metadata_from_schema,
    inject_calculated_metadata,
    exchange_hemispheres,
    make_scaled_copy_for_export,
    calculate_luminous_flux,
    build_filename_from_metadata,
    GEOMETRY_SCHEMA,
    merge_one_mm_json,
    scale_one_mm_json,
    build_ies_text,
    FileGenFlags,
    compute_file_generation_type,
    file_generation_title,
    inject_file_generation_type,
    _load_metadata_schema_from_google, 
    _load_metadata_schema_from_json_bytes
)

# -----------------------------------------------------
# Constants
# -----------------------------------------------------
LINEAR_XLSX_PATH = "Linear_Data.xlsx"
LINEAR_SHEET = "master_metadata_console"
DEFAULT_V_STEPS = 181
DEFAULT_H_STEPS = 25
DEFAULT_EXPORT_LEN_M = 0.001

# -----------------------------------------------------
# Helpers
# -----------------------------------------------------

LINKS_JSON_PATH = os.path.join("assets", "links.json")
LINKS_TXT_PATH  = os.path.join("assets", "links.txt")

def _normalize_sheet_id(s: str) -> str:
    s = (s or "").strip()
    m = re.search(r"/spreadsheets/d/([A-Za-z0-9-_]+)", s)
    return m.group(1) if m else s
    
def ensure_one_mm(data: dict, target_len: float = 0.001) -> dict:
    """Return a deep-copied payload scaled to 1 mm using ies_prep.make_scaled_copy_for_export."""
    from copy import deepcopy as _deepcopy
    from ies_prep import make_scaled_copy_for_export
    try:
        L = float((data.get("geometry", {}) or {}).get("G8", 0.0) or 0.0)
    except Exception:
        L = 0.0
    if abs(L - target_len) <= 1e-12:
        return _deepcopy(data)
    return make_scaled_copy_for_export(_deepcopy(data), target_len=float(target_len))

def _ensure_assets_dir() -> None:
    try:
        os.makedirs("assets", exist_ok=True)
    except Exception:
        pass

def _normalize_url(u: str) -> str:
    u = (u or "").strip()
    if u and not re.match(r"^https?://", u, re.I):
        u = "https://" + u
    return u

def _load_saved_links() -> List[Dict[str, str]]:
    items: Dict[str, Dict[str, str]] = {}
    # JSON first
    try:
        with open(LINKS_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for it in data:
                name = str(it.get("name", "")).strip()
                url = _normalize_url(str(it.get("url", "")).strip())
                if name and url:
                    items[name] = {"name": name, "url": url}
    except Exception:
        pass
    # TXT: Name|URL per line (last one wins)
    try:
        with open(LINKS_TXT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "|" in line:
                    name, url = line.split("|", 1)
                    name = name.strip()
                    url = _normalize_url(url.strip())
                    if name and url:
                        items[name] = {"name": name, "url": url}
    except Exception:
        pass
    return list(items.values())

def _save_links_legacy(urls: List[Dict[str, str]]) -> None:
    # legacy placeholder retained to avoid syntax errors from older block
    pass

def _effective_hemi_mode(choice: Optional[str], data: Optional[Dict[str, Any]]) -> str:
    """Resolve hemisphere mode.
    - Explicit: "Direct" → "direct", "Indirect" → "indirect", "None" → "none".
    - Auto: use metadata [LIGHT_DIRECTION] if present, else guess from intensity.
    """
    choice = (choice or "").strip()
    if choice.lower().startswith("direct"):
        return "direct"
    if choice.lower().startswith("indirect"):
        return "indirect"
    if not choice.lower().startswith("auto"):
        return "none"
    md = ((data or {}).get("metadata") or {})
    md_mode = str(md.get("[LIGHT_DIRECTION]", "")).strip().lower()
    if md_mode.startswith("direct"):
        return "direct"
    if md_mode.startswith("indirect"):
        return "indirect"
    # Guess: compare average intensity in 0–90 vs 90–180
    try:
        V = np.asarray((data or {}).get("vertical_angles", []) or [], dtype=float)
        H = np.asarray((data or {}).get("horizontal_angles", []) or [], dtype=float)
        I = np.asarray((data or {}).get("candela_values", []) or [], dtype=float)
        if I.ndim == 1 and V.size * H.size == I.size:
            I = I.reshape(H.size, V.size)
        if I.ndim == 2 and V.size > 1:
            prof = np.nanmean(I, axis=0)
            mask_dir = V <= 90.0
            sum_dir = float(np.nansum(prof[mask_dir]))
            sum_ind = float(np.nansum(prof[~mask_dir]))
            if sum_dir > sum_ind:
                return "direct"
            if sum_ind > sum_dir:
                return "indirect"
    except Exception:
        pass
    return "none"

def _head_tail(arr: List[float], n: int = 5) -> Dict[str, List[float]]:
    try:
        a = list(arr)
    except Exception:
        a = []
    if len(a) <= n:
        return {"first": a, "last": []}
    return {"first": a[:n], "last": a[-n:]}

def _flux_lm_safe(data: Dict[str, Any]) -> float:
    """Robust luminous flux (lm).
    Tries ies_prep.calculate_luminous_flux first; if unavailable or returns 0, uses a
    discrete integral over the Type C grid: Φ ≈ Σ I(θ,φ) * (cos θ_i − cos θ_{i+1}) * Δφ.
    """
    try:
        val = float(calculate_luminous_flux(data))
        if np.isfinite(val) and val > 0:
            return val
    except Exception:
        pass

    V = np.asarray(data.get("vertical_angles", []) or [], dtype=float)
    H = np.asarray(data.get("horizontal_angles", []) or [], dtype=float)
    I = np.asarray(data.get("candela_values", []) or [], dtype=float)
    if V.size < 2 or H.size < 1 or I.size == 0:
        return 0.0
    # shape to (H, V)
    if I.ndim == 1 and I.size == H.size * V.size:
        I = I.reshape(H.size, V.size)
    if I.ndim != 2 or I.shape != (H.size, V.size):
        return 0.0

    Vrad = np.deg2rad(V)
    cosV = np.cos(Vrad)
    dcos = cosV[:-1] - cosV[1:]  # length V-1

    Hdeg = H.astype(float)
    Hnext = np.roll(Hdeg, -1)
    dphi = np.deg2rad((Hnext - Hdeg) % 360.0)  # length H

    flux = 0.0
    for j in range(H.size):
        for i in range(V.size - 1):
            I_cell = 0.5 * (I[j, i] + I[j, i + 1])
            flux += I_cell * dcos[i] * dphi[j]
    return float(flux)

def _clean_cell(x):
    """"""
    try:
        if x is None:
            return ""
        if isinstance(x, float):
            if np.isnan(x) or np.isinf(x):
                return ""
            if abs(x - int(x)) < 1e-9:  # render 12.0 as "12"
                return str(int(x))
            return str(x)
        s = str(x)
        if s.strip().lower() in {"none", "nan"}:
            return ""
        return s
    except Exception:
        return "" if x is None else str(x)

def _auto_from_angle_coverage(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decide default hemi mode & clamping from the ORIGINAL V coverage.
    Returns a dict: {'mode': 'direct'|'indirect'|'none', 'clamp': bool, 'reason': str}
    """
    import numpy as _np
    V = _np.asarray((data or {}).get("vertical_angles", []) or [], dtype=float)
    if V.size == 0:
        return {"mode": "direct", "clamp": True, "reason": "No V angles; default to DIRECT + clamp"}
    vmin = float(_np.nanmin(V)); vmax = float(_np.nanmax(V)); eps = 1e-6
    if vmax <= 90.0 + eps:
        return {"mode": "direct",   "clamp": True,  "reason": "Original had no V>90° → clamp upper hemi"}
    if vmin >= 90.0 - eps:
        return {"mode": "indirect", "clamp": True,  "reason": "Original had only V≥90° → clamp lower hemi"}
    return {"mode": "none", "clamp": False, "reason": "Original spans both hemispheres → no clamp by default"}


def _geometry_table_from_schema(geom: Dict[str, Any], schema: Dict[str, Any]):
    import pandas as pd
    keys = sorted(schema.keys(), key=lambda k: int(k[1:]) if isinstance(k, str) and k[1:].isdigit() else 999)
    rows = []
    for k in keys:
        row = {
            "Field": k,
            "Label": (schema.get(k, {}).get("label") or schema.get(k, {}).get("name") or ""),
            "Value": (geom or {}).get(k, ""),
        }
        rows.append(row)
    return pd.DataFrame(rows)
def _angles_tables(data: Dict[str, Any]):
    """Two tiny DataFrames for V and H (first/last 5 with cd values)."""
    import pandas as pd
    V = list(map(float, (data or {}).get("vertical_angles", []) or []))
    H = list(map(float, (data or {}).get("horizontal_angles", []) or []))
    I = np.asarray((data or {}).get("candela_values", []) or [], dtype=float)
    if I.ndim == 1 and len(V) * len(H) == I.size:
        I = I.reshape(len(H), len(V))
    empty = pd.DataFrame([{"deg": "", "cd": ""}])
    if len(V) == 0 or len(H) == 0 or I.ndim != 2:
        return empty, empty
    h0, v0 = 0, 0
    v_first = [{"set":"V first 5","deg": float(V[i]), "cd": float(I[h0, i])} for i in range(0, min(5, len(V)))]
    v_last  = [{"set":"V last 5","deg": float(V[i]), "cd": float(I[h0, i])} for i in range(max(0, len(V)-5), len(V))]
    h_first = [{"set":"H first 5","deg": float(H[j]), "cd": float(I[j, v0])} for j in range(0, min(5, len(H)))]
    h_last  = [{"set":"H last 5","deg": float(H[j]), "cd": float(I[j, v0])} for j in range(max(0, len(H)-5), len(H))]
    return pd.DataFrame(v_first + v_last), pd.DataFrame(h_first + h_last)

def _render_step_tables(container, data: Dict[str, Any], *, height: int = 170) -> None:
    """Render the compact tables side-by-side in any container/column."""
    dfV, dfH = _angles_tables(data)
    c1, c2 = container.columns(2)
    c1.dataframe(dfV, use_container_width=True, height=height)
    c2.dataframe(dfH, use_container_width=True, height=height)

def _ensure_matrix_shape(d: Dict[str, Any]) -> Dict[str, Any]:
    """Make sure candela_values is shaped [H, V] before export/build."""
    V = list(map(float, d.get("vertical_angles", []) or []))
    H = list(map(float, d.get("horizontal_angles", []) or []))
    I = np.asarray(d.get("candela_values", []) or [], dtype=float)
    if I.ndim == 1 and I.size == len(H) * len(V) and len(H) > 0 and len(V) > 0:
        d["candela_values"] = I.reshape(len(H), len(V)).tolist()
    return d

def _step_json_report(data: Dict[str, Any]) -> Dict[str, Any]:
    """Short JSON object you can show behind a checkbox if you want."""
    return {
        "metadata": _json_sanitize((data or {}).get("metadata", {})),
        "geometry": _json_sanitize((data or {}).get("geometry", {})),
        "angles+cd": _samples_with_values(data) if " _samples_with_values" in globals() else {},
    }

def _samples_with_values(data: Dict[str, Any]) -> Dict[str, list]:
    """
    Build first/last-5 angle/value lists for both V and H.
    V-values at H=first available; H-values at V=first available.
    """
    V = list(map(float, (data or {}).get("vertical_angles", []) or []))
    H = list(map(float, (data or {}).get("horizontal_angles", []) or []))
    I = np.asarray((data or {}).get("candela_values", []) or [], dtype=float)
    if I.ndim == 1 and len(V) * len(H) == I.size:
        I = I.reshape(len(H), len(V))

    out = {"V first 5": [], "V last 5": [], "H first 5": [], "H last 5": []}
    if len(V) == 0 or len(H) == 0 or I.ndim != 2:
        return out

    h0 = 0  # take values along the first H-plane for V lists
    v0 = 0  # take values along the first V-index for H lists

    v_first_idx = list(range(0, min(5, len(V))))
    v_last_idx  = list(range(max(0, len(V)-5), len(V)))
    out["V first 5"] = [{"deg": float(V[i]), "cd": float(I[h0, i])} for i in v_first_idx]
    out["V last 5"]  = [{"deg": float(V[i]), "cd": float(I[h0, i])} for i in v_last_idx]

    h_first_idx = list(range(0, min(5, len(H))))
    h_last_idx  = list(range(max(0, len(H)-5), len(H)))
    out["H first 5"] = [{"deg": float(H[j]), "cd": float(I[j, v0])} for j in h_first_idx]
    out["H last 5"]  = [{"deg": float(H[j]), "cd": float(I[j, v0])} for j in h_last_idx]
    return out

def _plane_options_from(data: Optional[Dict[str, Any]]) -> List[float]:
    try:
        H = list(map(float, (data or {}).get("horizontal_angles", []) or []))
        return H if H else [0.0, 90.0]
    except Exception:
        return [0.0, 90.0]

def _json_sanitize(obj: Any):
    import numpy as _np
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, _np.generic):
        return obj.item()
    if isinstance(obj, _np.ndarray):
        return [_json_sanitize(v) for v in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in list(obj)]
    if isinstance(obj, set):
        return [_json_sanitize(v) for v in sorted(list(obj), key=lambda z: str(z))]
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    return str(obj)

def _to_ugr_photometry(data: Dict[str, Any]) -> "UPhot":
    g = data.get("geometry", {}) if isinstance(data, dict) else {}
    width = float(g.get("G7", 0.1) or 0.1)
    length = float(g.get("G8", 0.1) or 0.1)
    height = float(g.get("G9", 0.1) or 0.1)
    V = list(map(float, data.get("vertical_angles", []) or []))
    H = list(map(float, data.get("horizontal_angles", []) or []))
    I = np.array(data.get("candela_values", []) or [], dtype=float).tolist()
    ptype = int(float(g.get("G5", 3) or 3))
    return UPhot(v_angles=V, h_angles=H, candela=I, units="m", width=width, length=length, height=height, photometric_type=ptype)


# -----------------------------------------------------
# UI blocks — Geometry & Metadata
# -----------------------------------------------------

def render_geometry_block_all_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Geometry editor: compact 3-row grid, collapsible, tiny labels (G# · Label), with 'Only editable' filter.
    Also hosts LM‑63 File Generation flags and an in-expander Apply button.
    Returns possibly-updated data with overrides applied when user clicks Apply.
    Locks after apply; scales to 1mm.
    """
    geom = dict(data.get("geometry", {}))

    with st.expander("Geometric Value Editor (G0–G12+G13/14)", expanded=False):
        applied = st.session_state.get("geom_applied", False)
        only_edit = st.checkbox("Only editable", value=True, key="geom_only_edit", disabled=applied)

        def _kord(k: str) -> Tuple[int, str]:
            try:
                return (int(k[1:]), k)
            except Exception:
                return (999, k)

        all_keys = sorted(GEOMETRY_SCHEMA.keys(), key=_kord)

        # Group into three logical rows: G0–G4, G5–G9, G10–G14 (rest appended to last)
        rows: List[List[str]] = [[], [], []]
        for gk in all_keys:
            try:
                idx = int(gk[1:])
            except Exception:
                idx = 999
            if idx <= 4:
                rows[0].append(gk)
            elif idx <= 9:
                rows[1].append(gk)
            else:
                rows[2].append(gk)

        edits: Dict[str, Any] = {}
        for row_keys in rows:
            if not row_keys:
                continue
            # Filter for editable view and center within 5 columns
            filtered = []
            for _gk in row_keys:
                spec0 = GEOMETRY_SCHEMA.get(_gk, {})
                if only_edit and not bool(spec0.get("editable", False)):
                    continue
                filtered.append(_gk)
            if not filtered:
                continue
            slots = 5
            left_pad = max(0, (slots - len(filtered)) // 2)
            cols = st.columns(slots)
            for i, gk in enumerate(filtered):
                spec = GEOMETRY_SCHEMA.get(gk, {})
                editable = bool(spec.get("editable", False)) and (not applied)
                label = f"{gk} · {spec.get('label', gk)}"
                tip = spec.get("tooltip", "")
                val = geom.get(gk, "")
                col = cols[left_pad + i]
                col.caption(label)  # compact label above widget
                try:
                    val_num = float(val)
                    if editable:
                        new_val = col.number_input(label, value=float(val_num), help=tip, key=f"geom_edit_{gk}", label_visibility="collapsed", disabled=applied)
                        edits[gk] = new_val
                    else:
                        col.text_input(label, value=str(val), help=tip, key=f"geom_view_{gk}", disabled=True, label_visibility="collapsed")
                except Exception:
                    if editable:
                        new_val = col.text_input(label, value=str(val), help=tip, key=f"geom_edit_{gk}", label_visibility="collapsed", disabled=applied)
                        edits[gk] = new_val
                    else:
                        col.text_input(label, value=str(val), help=tip, key=f"geom_view_{gk}", disabled=True, label_visibility="collapsed")

        # LM‑63 flags live here (roll up/down with geometry)
        st.markdown("---")
        st.caption("LM‑63‑2019 — File Generation Flags")
        fcols = st.columns(5)
        fcols[0].checkbox("Accredited",   value=st.session_state.get("f_accredited", True),   key="f_accredited",   disabled=applied)
        fcols[1].checkbox("Interpolated", value=st.session_state.get("f_interpolated", True),  key="f_interpolated", disabled=applied)
        fcols[2].checkbox("Scaled",       value=st.session_state.get("f_scaled", True),        key="f_scaled",       disabled=applied)
        fcols[3].checkbox("Simulated",    value=st.session_state.get("f_simulated", False),    key="f_simulated",    disabled=applied)
        fcols[4].checkbox("Undefined",    value=st.session_state.get("f_undefined", False),    key="f_undefined",    disabled=applied)
        # Echo LM-63 code/title
        try:
            _flags_obj = (_FLG := FileGenFlags(
                accredited=st.session_state.get("f_accredited", False)); setattr(_FLG, 'lm63_2019', True); _FLG,
                interpolated=st.session_state.get("f_interpolated", True),
                scaled=st.session_state.get("f_scaled", True),
                simulated=st.session_state.get("f_simulated", False),
                undefined=st.session_state.get("f_undefined", False),
            )
            _code = compute_file_generation_type(_flags_obj)
            _title = file_generation_title(_code)
            st.caption(f"Type: {_code} — {_title}")
        except Exception:
            pass

        # Centered Apply button inside expander
        pads = st.columns([1,1,1])
        if not applied:
            if pads[1].button("Apply Geometry Edits", key="apply_geom_btn"):
                try:
                    data = apply_geometry_overrides(data, edits)
                    st.session_state["geom_applied"] = True
                    # create 1mm copy for downstream merge/export
                    try:
                        st.session_state["one_mm_down"] = ensure_one_mm(data, target_len=DEFAULT_EXPORT_LEN_M)
                    except Exception as e:
                        st.warning(f"Scale to 1 mm failed (continuing): {e}")
                    pads[1].markdown("<small style='color:#1aa251'>Applied ✓</small>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Geometry apply failed: {e}")
        else:
            pads[1].button("Applied ✓", key="apply_geom_done", disabled=True)
    return data

# -----------------------------------------------------
# UI blocks — Metadata editor (Proposed vs IES)
# -----------------------------------------------------

def render_metadata_selector(
    data: Dict[str, Any],
    excel_path: str,
    sheet: str,
    ies_original_md: Dict[str, Any],
) -> Dict[str, Any]:
    """Metadata editor in its own expander.
    Columns: Field | Proposed (editable) | IES (read‑only).
    Proposed pre-fills from Excel schema (IES_PROPOSED/IES_PROPOSED_KEYWORD/PROPOSED) and is editable
    unless row is derived or bound to geometry. Commit writes Proposed into data['metadata'] and
    injects calculated fields, then locks the editor.
    """
    ss = st.session_state

    def _asbool(x: Any) -> bool:
        s = str(x).strip().lower()
        if s in {"1", "true", "yes", "y", "t", "on"}:
            return True
        if s in {"0", "false", "no", "n", "f", "off", "", "nan", "none"}:
            return False
        try:
            return bool(int(s))
        except Exception:
            return bool(x)

    # === Source selector + on-demand loader (Excel or Google) ===
    src = st.radio(
        "Metadata source",
        ["", "", "JSON (novon_workflow.json)"],
        index=0,
        horizontal=True,
        key="md_source",
    )

    # JSON uploader for metadata (new mode) + auto-load from assets
    md_json_bytes = None
    if src == "JSON (novon_workflow.json)":
        _json_up =         do_refresh = g_cols[2].button("Refresh now", key="md_refresh_now")

        # show cache timestamp if exists
        try:
            import os, time
            cache_path = os.path.join("assets", "linear_data_cache.json")
            if os.path.exists(cache_path):
                ts = os.path.getmtime(cache_path)
                st.caption(f"Cache: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))} — assets/linear_data_cache.json")
        except Exception:
            pass

    
    # Key to invalidate when source/settings change
    if src == "JSON (novon_workflow.json)":
        path_key = ("json", "novon_workflow")
    elif src == "":
        path_key = (
            "gsheet",
            _normalize_sheet_id(st.session_state.get("g_sheet_id", "")),
            st.session_state.get("g_sheet_ws", "master_metadata_console"),
        )
    else:
        path_key = ("excel", excel_path, sheet)

    # Load metadata schema now
    try:
        if src == "JSON (novon_workflow.json)":
            _b = st.session_state.get("md_json_bytes", b"") or (md_json_bytes or b"")
            if not _b:
                raise SystemExit("Upload a NOVON Workflow JSON file first or place it at assets/novon_workflow.json")
            schema = _load_metadata_schema_from_json_bytes(_b)
        elif src == "":
            schema = _load_metadata_schema_from_google(
                service_account_path="service_account.json",
                cache_json=os.path.join("assets", "linear_data_cache.json"),
                refresh=bool(do_refresh),
            )
        else:
            schema = _load_metadata_schema_from_excel(excel_path, sheet)

        ss["md_schema"] = schema
        ss["md_schema_path"] = path_key

    except SystemExit as e:
        st.error(str(e))
        return data
    except Exception as e:
        st.exception(e)
        return data

    schema = ss["md_schema"]
    schema = ss["md_schema"]

    locked = ss.get("meta_applied", False)
    with st.expander("Metadata — Proposed vs IES", expanded=not locked):
        # precompute base Proposed from schema rules
        try:
            _base = apply_metadata_from_schema(deepcopy(data), schema, user_overrides=None)
            _base_md = (_base or {}).get("metadata", {}) or {}
        except Exception:
            _base_md = {}

        # compact inputs CSS
        st.markdown(
            """
            <style>
            .small-input input {padding-top:2px !important; padding-bottom:2px !important; height: 1.6em !important;}
            .small-md .markdown-text-container p {margin-bottom: 0.15rem !important;}
            </style>
            """,
            unsafe_allow_html=True,
        )

        header = st.columns([3, 5, 5])
        header[0].markdown("<small><b>Field</b></small>", unsafe_allow_html=True)
        header[1].markdown("<small><b>Proposed (editable)</b></small>", unsafe_allow_html=True)
        header[2].markdown("<small><b>IES value (read-only)</b></small>", unsafe_allow_html=True)

        ss.setdefault("md_proposed_edit", {})

        for row in schema:
            field = (row.get("field") or "").strip()
            if not field:
                continue

            derived  = _asbool(row.get("derived", False))
            geom_key = row.get("geom_key")
            tip      = row.get("tooltip", "")

            # Default 'Proposed' from Excel — prefer IES_PROPOSED, then IES_PROPOSED_KEYWORD, then other fallbacks
            proposed_val = (
                row.get("IES_PROPOSED")
                or row.get("IES_PROPOSED_KEYWORD")
                or row.get("PROPOSED")
                or row.get("Proposed")
                or row.get("proposed")
                or ""
            )

            ies_val = data.get("metadata", {}).get(field, ies_original_md.get(field, ""))

            cols = st.columns([3, 5, 5])
            cols[0].markdown(f"<small><b>{field}</b></small>", unsafe_allow_html=True)

            if derived or geom_key:
                # read-only Proposed for derived/geometry fields (calculated later)
                cols[1].text_input(
                    "",
                    value=str(_clean_cell(proposed_val)),
                    disabled=True,
                    help=(tip + " — Read-only (derived/geometry)."),
                    key=f"md_prop_ro_{field}",
                    label_visibility="collapsed",
                )
            else:
                # editable Proposed, seeded from computed base (which already uses Excel proposed columns)
                slug = re.sub(r"[^A-Za-z0-9_]+", "_", field)
                default_prop = ss["md_proposed_edit"].get(field, _clean_cell(_base_md.get(field, proposed_val)))
                edited = cols[1].text_input(
                    "",
                    value=default_prop,
                    help=tip,
                    key=f"md_prop_edit_{slug}",
                    label_visibility="collapsed",
                    disabled=locked,
                )
                ss["md_proposed_edit"][field] = edited

            cols[2].text_input(
                "",
                value=str(ies_val),
                disabled=True,
                key=f"md_ies_ro_{field}",
                label_visibility="collapsed",
                help="IES-provided value (read-only).",
            )

        # Diagnostics panel (optional)
        show_diag = st.checkbox("Show metadata diagnostics", value=False, key="md_diag")
        if show_diag:
            try:
                import pandas as _pd
                diag_rows = []
                for _row in schema:
                    _f = (_row.get("field") or "").strip()
                    if not _f:
                        continue
                    diag_rows.append({
                        "field": _f,
                        "IES_FUNC": _row.get("IES_FUNC"),
                        "derived": _row.get("derived"),
                        "geom_key": _row.get("geom_key"),
                        "base": _base_md.get(_f, ""),
                        "ies": ies_original_md.get(_f, ""),
                    })
                st.dataframe(_pd.DataFrame(diag_rows), use_container_width=True)
            except Exception as e:
                st.info(f"Diagnostics unavailable: {e}")

        # Commit / Reset
        c = st.columns([1, 1, 6])
        if c[0].button("Commit", key="commit_md", disabled=locked):
            md_out: Dict[str, Any] = {}
            try:
                base = apply_metadata_from_schema(deepcopy(data), schema, user_overrides=None)
            except Exception:
                base = deepcopy(data)

            for row in schema:
                field = (row.get("field") or "").strip()
                if not field:
                    continue

                if bool(row.get("geom_key")) or _asbool(row.get("derived", False)):
                    # keep calculated/geometry from base
                    md_out[field] = base.get("metadata", {}).get(field, "")
                else:
                    # take user's edited Proposed (or Excel proposed fallback if untouched)
                    md_out[field] = ss["md_proposed_edit"].get(
                        field,
                        (
                            row.get("IES_PROPOSED")
                            or row.get("IES_PROPOSED_KEYWORD")
                            or row.get("PROPOSED")
                            or row.get("Proposed")
                            or row.get("proposed")
                            or ""
                        ),
                    )

            data = deepcopy(data)
            data["metadata"] = md_out

            # Recompute calculated fields *after* geometry + metadata present (feeds derived values)
            try:
                data = inject_calculated_metadata(data)
            except Exception:
                pass

            st.session_state["meta_applied"] = True
            c[0].markdown("<small style='color:#1aa251'>Committed ✓</small>", unsafe_allow_html=True)

        if c[1].button("Reset edits", key="reset_md", disabled=locked):
            ss["md_proposed_edit"] = {}
            try:
                _safe_rerun()
            except Exception:
                st.rerun()

    return data

# -----------------------------------------------------
# UI blocks — Polar
# -----------------------------------------------------

def render_polar_blocks(step0: Optional[Dict[str, Any]], step1: Dict[str, Any], step3: Optional[Dict[str, Any]] = None) -> None:
    """Dynamic Polar Plot and Settings (expander contains settings + plot).
    Also includes 'Exchange hemispheres before export' toggle.
    """
    if not HAVE_PLOT or not step1:
        return

    ss = st.session_state
    payload_label = "Merged" if ss.get("cc_one_mm_merged") else ("Down-only" if ss.get("one_mm_down") else "Original")
    base_data = ss.get("cc_one_mm_merged") or ss.get("one_mm_down") or (step3 or step1)
    H_all = list(map(float, base_data.get("horizontal_angles", []) or []))

    with st.expander(f"Dynamic Polar Plot and Settings [{payload_label}]", expanded=True):
        # Optional exchange hemispheres inside expander
        ex_col = st.columns([1,3])[0]
        do_exchange = ex_col.checkbox("Exchange hemispheres before export", value=False, key="polar_exchange")
        data_for_plot = deepcopy(base_data)
        if do_exchange:
            try:
                data_for_plot = exchange_hemispheres(data_for_plot)
            except Exception as _e:
                st.warning(f"Exchange hemispheres failed: {_e}")

        # Planes
        plane_a = st.selectbox("Plane A (deg)", options=H_all if H_all else [0.0, 90.0], index=0, key="polar_plane_a")
        idx_b = (len(H_all)//2 if H_all else 1)
        plane_b = st.selectbox("Plane B (deg)", options=H_all if H_all else [0.0, 90.0], index=min(idx_b, max(0, (len(H_all)-1))), key="polar_plane_b")

        # Style
        cols = st.columns(3)
        color_a = cols[0].color_picker("Line A color", value="#1f77b4", key="polar_color_a")
        color_b = cols[1].color_picker("Line B color", value="#d32f2f", key="polar_color_b")
        fill_color = cols[2].color_picker("Fill color", value="#FFF59D", key="polar_fill_color")

        cols = st.columns(3)
        grid_color = cols[0].color_picker("Grid color", value="#BDBDBD", key="polar_grid_color")
        line_width = cols[1].slider("Line width", min_value=0.2, max_value=4.0, value=1.8, step=0.1, key="polar_line_width")
        fill_alpha = cols[2].slider("Fill alpha", min_value=0.0, max_value=1.0, value=0.25, step=0.05, key="polar_fill_alpha")

        cols = st.columns(3)
        gamma_step = cols[0].slider("Gamma labels every (deg)", min_value=5, max_value=45, value=15, step=5, key="polar_gamma_step")
        rmax_in = cols[1].number_input("Radial max (cd/klm, 0 = auto)", min_value=0.0, value=0.0, step=50.0, key="polar_rmax")
        duplicate_labels = cols[2].checkbox("Duplicate radial labels", value=True, key="polar_dup_labels")

        # Figure size (shrink/expand plot on screen)
        st.slider("Figure size (in)", min_value=3.0, max_value=8.0, value=st.session_state.get("polar_fig_inches", 5.0), step=0.5, key="polar_fig_inches")

        st.text_input("Title", value="Polar", key="polar_title")

        cols = st.columns(5)
        font_title = cols[0].number_input("Font title", min_value=2, max_value=48, value=14, step=1, key="polar_font_title")
        font_tick = cols[1].number_input("Font tick", min_value=2, max_value=48, value=9, step=1, key="polar_font_tick")
        font_radial = cols[2].number_input("Font radial", min_value=2, max_value=48, value=9, step=1, key="polar_font_radial")
        font_legend = cols[3].number_input("Font legend", min_value=2, max_value=48, value=10, step=1, key="polar_font_legend")
        font_summary = cols[4].number_input("Font summary", min_value=2, max_value=48, value=10, step=1, key="polar_font_summary")

        # Compute summary from Plane A for the text card
        try:
            hemi = str((data_for_plot.get("metadata", {}) or {}).get("[HEMI_MODE]", "B")).upper()
            summary = compute_polar_metrics(data_for_plot, plane_for_beam=float(plane_a), hemi_mode=hemi)
            # Add 'Direction' line for summary table (D/I/B)
            try:
                _dir = {"D": "Direct", "I": "Indirect"}.get(hemi[:1].upper(), "Both")
                if isinstance(summary, dict):
                    summary = dict(summary)
                    summary["Direction"] = _dir
            except Exception:
                pass
        except Exception:
            summary = None

        # Build plot
        try:
            fig = build_polar_figure(
                data_for_plot,
                title=st.session_state.get("polar_title", "Polar"),
                planes=(float(st.session_state.get("polar_plane_a", 0.0)), float(st.session_state.get("polar_plane_b", 90.0))),
                rmax=(None if st.session_state.get("polar_rmax", 0.0) <= 0 else float(st.session_state.get("polar_rmax"))),
                smart_hemi=True,
                fig_inches=float(st.session_state.get("polar_fig_inches", 5.0)),
                line_width=float(st.session_state.get("polar_line_width", 1.8)),
                fill_alpha=float(st.session_state.get("polar_fill_alpha", 0.25)),
                fill_color=st.session_state.get("polar_fill_color", "#FFF59D"),
                color_a=st.session_state.get("polar_color_a", "#1f77b4"),
                color_b=st.session_state.get("polar_color_b", "#d32f2f"),
                duplicate_radial_labels=bool(st.session_state.get("polar_dup_labels", True)),
                zoom_if_directional=True,
                summary=summary,
                summary_outside_left=True,
                gamma_grid_step=int(st.session_state.get("polar_gamma_step", 15)),
                font_title=int(st.session_state.get("polar_font_title", 14)),
                font_tick=int(st.session_state.get("polar_font_tick", 9)),
                font_radial=int(st.session_state.get("polar_font_radial", 9)),
                font_legend=int(st.session_state.get("polar_font_legend", 10)),
                font_summary=int(st.session_state.get("polar_font_summary", 10)),
            )
            # Apply grid color from UI even if polar.py doesn't support it natively
            try:
                ax = fig.axes[0]
                ax.grid(True, color=st.session_state.get("polar_grid_color", "#BDBDBD"), alpha=0.6)
            except Exception:
                pass
            st.pyplot(fig, clear_figure=False)
            st.download_button("Download Polar (PNG)", figure_png_bytes(fig, dpi=300, transparent=False), file_name="polar.png", mime="image/png")
        except Exception as e:
            st.error(f"Polar render failed: {e}")

# -----------------------------------------------------
# Health check helper
# -----------------------------------------------------

def run_health_check_ui(step0: Optional[Dict[str, Any]], step1: Optional[Dict[str, Any]], step2: Optional[Dict[str, Any]]) -> None:
    """Run a few invariants and show pass/fail."""
    ok = True
    # 1) Flux invariance with renorm
    try:
        if step0 is not None:
            flux0 = _flux_lm_safe(step0)
            interp = interpolate_candela_matrix(
                deepcopy(step0),
                v_steps=st.session_state.get("v_steps", DEFAULT_V_STEPS),
                h_steps=st.session_state.get("h_steps", DEFAULT_H_STEPS),
                hemi_mode=_effective_hemi_mode(st.session_state.get("hemi_choice", "Auto (from [LIGHT_DIRECTION])"), step0),
                clamp_dark=st.session_state.get("clamp_dark", True),
                renorm=True,
            )
            flux1 = _flux_lm_safe(interp)
            diff = abs(flux1 - flux0)
            tol = max(1e-6, 0.005 * max(flux0, 1.0))  # ~0.5%
            if diff <= tol:
                st.success(f"Flux renorm OK: raw={flux0:.1f} lm, interp+renorm={flux1:.1f} lm (Δ={diff:.2f})")
            else:
                ok = False
                st.error(f"Flux renorm FAIL: raw={flux0:.1f} lm, interp+renorm={flux1:.1f} lm (Δ={diff:.2f})")
    except Exception as e:
        ok = False
        st.error(f"Flux check error: {e}")

    # 2) Matrix shape check
    try:
        for label, dat in (("step0", step0), ("step1", step1), ("step2", step2)):
            if not dat:
                continue
            V = dat.get("vertical_angles", []) or []
            H = dat.get("horizontal_angles", []) or []
            I = np.asarray(dat.get("candela_values", []) or [], dtype=float)
            target = len(V) * len(H)
            flat = I.size
            if target == flat and target > 0:
                st.success(f"{label}: I size OK ({flat} == {len(H)}×{len(V)})")
            else:
                ok = False
                st.error(f"{label}: I size MISMATCH ({flat} vs {len(H)}×{len(V)}={target})")
    except Exception as e:
        ok = False
        st.error(f"Shape check error: {e}")

    # 3) LM-63 flags/code
    try:
        flags = (_FLG := FileGenFlags(
            accredited=st.session_state.get("f_accredited", False)); setattr(_FLG, 'lm63_2019', True); _FLG,
            interpolated=st.session_state.get("f_interpolated", True),
            scaled=st.session_state.get("f_scaled", True),
            simulated=st.session_state.get("f_simulated", False),
            undefined=st.session_state.get("f_undefined", False),
        )
        code = compute_file_generation_type(flags)
        title = file_generation_title(code)
        st.info(f"LM-63 type: {code} — {title}")
    except Exception as e:
        ok = False
        st.error(f"LM-63 check error: {e}")

    if ok:
        st.balloons()

# -----------------------------------------------------
# App
# -----------------------------------------------------

st.title("Engineering Tool")

# ===== Main: Load area right under the header =====
st.markdown("### Photometry Preparation (IES LM 63-2019)")
ies_up = st.file_uploader("Drag & drop IES file here", type=["ies","IES"], accept_multiple_files=False)
step2 = st.session_state.get('step2')

# STEP 3 — Merge up-light (optional) + Polar (opens after metadata commit)
if step2 and st.session_state.get("meta_applied", False):
    with st.expander("Add up-light (optional)", expanded=False):
        pass
    st.caption("Scale to 1 mm for datasheet exports; LM-63 text builder with File Generation Type.")
    export_len_m = st.number_input("Export length (m)", min_value=0.0001, value=DEFAULT_EXPORT_LEN_M, step=0.0005, format="%f")

    # Read flags from session (UI lives in Geometry editor)
    flags = (_FLG := FileGenFlags(
        accredited=st.session_state.get("f_accredited", False)); setattr(_FLG, 'lm63_2019', True); _FLG,
        interpolated=st.session_state.get("f_interpolated", True),
        scaled=st.session_state.get("f_scaled", True),
        simulated=st.session_state.get("f_simulated", False),
        undefined=st.session_state.get("f_undefined", False),
    )

    ss = st.session_state
    payload = ss.get("cc_one_mm_merged") or ss.get("one_mm_down") or step2
    payload_label = "Merged" if ss.get("cc_one_mm_merged") else ("Down-only" if ss.get("one_mm_down") else "Original")
    st.caption(f"Export payload: {payload_label}")

    # Ensure scaled copy and **correct [H,V] candela shape** before building IES
    scaled = make_scaled_copy_for_export(deepcopy(payload), target_len=float(export_len_m))
    scaled = _ensure_matrix_shape(scaled)  # important guard (prevents shape mismatch)
    scaled = inject_file_generation_type(scaled, flags)

# Down-only (1 mm) — compact tables
if st.session_state.get("one_mm_down"):
    with st.expander("Down-only (1 mm) — Summary", expanded=False):
        _render_step_tables(st, st.session_state["one_mm_down"])

# Merged (Down + Up) — compact tables
if st.session_state.get("cc_one_mm_merged"):
    with st.expander("Merged (Down + Up) — Summary", expanded=False):
        _render_step_tables(st, st.session_state["cc_one_mm_merged"])

    ies_text = build_ies_text(scaled)
    fname = build_filename_from_metadata(scaled.get("metadata", {}), scaled.get("geometry", {})) or "export"

    with st.expander("Export", expanded=True):
        left, right = st.columns([1, 1])
        left.download_button(
            "Download IES",
            data=ies_text,
            file_name=f"{fname}.ies",
            mime="text/plain",
            key="dl_ies",
        )
        right.download_button(
            "Download JSON",
            data=json.dumps(scaled, indent=2, ensure_ascii=False),
            file_name=f"{fname}.json",
            mime="application/json",
            key="dl_json",
        )

        show_hdr = st.checkbox("Show header preview", value=False, key="show_hdr_prev")
        if show_hdr:
            header_preview = chr(10).join(ies_text.splitlines()[:60])
            st.code(header_preview, language="text")

# Optional UGR tables (after metadata commit)
if HAVE_UGR and step2 and st.session_state.get("meta_applied", False):
    with st.expander("UGR tables (optional)", expanded=False):
        try:
            phot = _to_ugr_photometry(step2)
            sets = DEFAULT_REFLECTANCE_SETS
            tables = build_ugr_tables(phot, reflectance_sets=sets)
            st.text_area("UGR (text)", value=ugr_to_text(tables), height=180)
        except Exception as e:
            st.warning(f"UGR failed: {e}")

# Health check button at bottom
with st.expander("Health check", expanded=False):
    if st.button("Run health check"):
        run_health_check_ui(st.session_state.get('step0'), st.session_state.get('step1'), st.session_state.get('step2'))
