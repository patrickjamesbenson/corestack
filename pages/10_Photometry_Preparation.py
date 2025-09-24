from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
import streamlit as st
from utils.ui_nav import render_sidebar
render_sidebar("ADMIN")

from utils.paths import ies_repo_current_path
from utils.photometry_engine_adapter import (
    # engine / export
    scale_to_length_mm,
    export_repo_json,
    export_files_lm63,         # available if you later want file-based IES export
    build_repo_filename,
    build_ies_text,
    inject_calculated_metadata,
    # photometry + metadata
    parse_ies,
    ordered_metadata_rows,
    geometry_rows_ordered,
    luminous_flux_with_note,
    power_watts,
    lumens_per_watt,
    beam_angle_with_note,
    apply_geometry,
)

st.set_page_config(page_title="Photometry Preparation", layout="wide")
render_sidebar("Photometry Preparation")

ss = st.session_state
ss.setdefault("working", None)
ss.setdefault("geometry_locked", False)
ss.setdefault("loaded_file_id", None)

st.title("Photometry Preparation")

# ---------- Load IES (memoized so locking persists) ----------
file = st.file_uploader(
    "Drag & drop IES file (LM-63)",
    type=["ies", "txt"],
    accept_multiple_files=False,
    key="uploader",
)
if file is not None:
    try:
        file_bytes = file.getvalue()
        loaded_id = (
            getattr(file, "name", None),
            getattr(file, "size", None),
            getattr(file, "type", None),
            hashlib.md5(file_bytes).hexdigest(),
        )
        if ss.get("loaded_file_id") != loaded_id:
            ss["working"] = parse_ies(file_bytes)
            ss["geometry_locked"] = False
            ss["loaded_file_id"] = loaded_id
            st.toast("IES loaded & parsed.", icon="✅")
    except Exception as ex:
        st.error(f"Parse failed: {ex}")

w = ss.get("working")
if not w:
    st.info("Load an IES file to begin.")
    st.stop()

V = w["photometry"]["v_angles"]
H = w["photometry"]["h_angles"]
cand = np.array(w["photometry"]["candela"], dtype=float)

lm, lm_note = luminous_flux_with_note(w)
pw = power_watts(w)
lpw = lumens_per_watt(w)
ba, ba_note = beam_angle_with_note(w)

def _fmt(x, d=1):
    if x is None:
        return ""
    try:
        return f"{float(x):.{d}f}"
    except Exception:
        return str(x)

# ---------- Snapshot row ----------
c1, c2, c3, c4, c5 = st.columns([1.2, 1.0, 1.0, 1.0, 0.9], gap="small")
with c1:
    st.metric("Luminous Flux (lm)", _fmt(lm, 1))
    if lm_note:
        st.caption(lm_note)  # tiny caption e.g. "half-azimuth → symmetry ×2"
with c2:
    st.metric("Power (W)", _fmt(pw, 1))
with c3:
    st.metric("Lm/W", _fmt(lpw, 1))
with c4:
    st.metric("Beam angle (°)", _fmt(ba, 1))
    if ba_note:
        st.caption(ba_note)  # tiny caption e.g. "2×θ@50%"
with c5:
    st.metric("V/H count", f"{len(V)}/{len(H)}")

st.divider()

# ---------- Raw geometry & angles ----------
st.subheader("Raw Geometry & Angles")

# Geometry table: only Key + Value (no 'Geometry' index column)
g_rows = []
labels = {
    0:"G0 • # lamps", 1:"G1 • lumens per lamp", 2:"G2 • candela multiplier",
    3:"G3 • V-count", 4:"G4 • H-count", 5:"G5 • photometric type (1=C,2=B,3=A)",
    6:"G6 • units (1=ft,2=m)", 7:"G7 • width (mm)", 8:"G8 • length (mm)",
    9:"G9 • height (mm)", 10:"G10 • ballast factor", 11:"G11 • file generation type",
    12:"G12 • input/system watts", 13:"G13 • raw board lumens", 14:"G14 • circuit watts",
}
for (_glabel, gkey, val) in geometry_rows_ordered(w):
    try:
        idx = int(_glabel[1:])
    except Exception:
        continue
    g_rows.append({"Key": labels.get(idx, gkey), "Value": val})

cG, cV, cH = st.columns([1.0, 1.0, 1.0])
with cG:
    st.caption("Raw geometry detail")
    st.dataframe(pd.DataFrame(g_rows), use_container_width=True, hide_index=True)
with cV:
    st.caption("V-angles (°)")
    st.dataframe(pd.DataFrame({"V angle (°)": V}), use_container_width=True, hide_index=True, height=320)
with cH:
    st.caption("H-angles (°)")
    st.dataframe(pd.DataFrame({"H angle (°)": H}), use_container_width=True, hide_index=True, height=320)

st.divider()

# ---------- Geometry Edit & Flags ----------
st.subheader("Geometry Edit & Flags")

# Lock/unlock panel (just above Apply button)
lc1, lc2, lc3 = st.columns([1.6, 1.0, 1.0])
with lc1:
    st.markdown(f"**Lock status:** {'🔒 Locked' if ss['geometry_locked'] else '🔓 Unlocked'}")
with lc2:
    st.button(
        "Lock geometry",
        type="primary",
        disabled=ss["geometry_locked"],
        on_click=lambda: ss.update(geometry_locked=True),
    )
with lc3:
    st.button(
        "Unlock geometry",
        disabled=not ss["geometry_locked"],
        on_click=lambda: ss.update(geometry_locked=False),
    )

with st.form("geom_form", clear_on_submit=False):
    locked = bool(ss["geometry_locked"])
    g = w.get("geometry", {})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        width_mm = st.number_input("G7 width (mm)", 0.0, value=float(g.get("G7", 0.0)), step=0.01, disabled=locked)
    with c2:
        length_mm = st.number_input("G8 length (mm)", 0.0, value=float(g.get("G8", 0.0)), step=0.01, disabled=locked)
    with c3:
        height_mm = st.number_input("G9 height (mm)", 0.0, value=float(g.get("G9", 0.0)), step=0.01, disabled=locked)
    with c4:
        gen_type = st.number_input("G11 File Gen Type", 0.0, value=float(g.get("G11", 0.0) or 0.0), step=0.01, disabled=locked)

    c5, c6, c7 = st.columns(3)
    with c5:
        g12_input = st.number_input("G12 input watts", 0.0, value=float(g.get("G12", 0.0)), step=0.01, disabled=locked)
    with c6:
        g13_raw   = st.number_input("G13 raw board lumens", 0.0, value=float(g.get("G13", 0.0)), step=1.0, disabled=locked)
    with c7:
        g14_circ  = st.number_input("G14 circuit watts", 0.0, value=float(g.get("G14", 0.0)), step=0.01, disabled=locked)

    # Flag toggles (visible and editable when unlocked)
    fc1, fc2, fc3, fc4, fc5 = st.columns(5)
    b1 = fc1.toggle("Absolute", True,  disabled=locked)
    b2 = fc2.toggle("Interpolated", True, disabled=locked)
    b3 = fc3.toggle("Scaled", True,     disabled=locked)
    b4 = fc4.toggle("Simulated", False, disabled=locked)
    b5 = fc5.toggle("Other", False,     disabled=locked)
    flag_code = ".".join("1" if x else "0" for x in (b1, b2, b3, b4, b5))
    st.caption(f"Flags: {flag_code}")

    left, right = st.columns(2)
    apply_btn  = left.form_submit_button(
        "Apply Geometry Edits", type="primary", use_container_width=True, disabled=locked
    )
    unlock_btn = right.form_submit_button("Unlock geometry edits", use_container_width=True, disabled=not locked)

    if apply_btn and not locked:
        ss["working"] = apply_geometry(
            ss["working"],
            {
                "G7": width_mm, "G8": length_mm, "G9": height_mm,
                "G11": gen_type, "G12": g12_input, "G13": g13_raw, "G14": g14_circ,
                "file_generation_flags": flag_code,  # mirrored into G11 by engine
            },
        )
        ss["geometry_locked"] = True
        st.toast("Geometry + flags applied. Proceed to Metadata.", icon="✅")
        st.experimental_rerun()

    if unlock_btn and locked:
        ss["geometry_locked"] = False
        st.toast("Geometry unlocked.", icon="🔓")
        st.experimental_rerun()

st.divider()

# ---------- Metadata (opens ONLY after lock) ----------
st.subheader("Metadata")
if not ss["geometry_locked"]:
    st.info("Lock geometry to proceed to Metadata.")
else:
    st.dataframe(
        pd.DataFrame(ordered_metadata_rows(ss["working"]), columns=["Item", "Value"]),
        use_container_width=True,
        hide_index=True,
    )

# ============================ Export ============================
st.divider()
st.subheader("Export")

repo_dir = ies_repo_current_path()

colA, colB, colC = st.columns([1.2, 1.2, 2.0])

with colA:
    if st.button("Send to repo (1 mm JSON)", type="primary", use_container_width=True, disabled=(w is None)):
        try:
            one_mm = scale_to_length_mm(w, 1.0)
            outp = export_repo_json(one_mm, repo_dir)
            st.success(f"Saved to repo: {outp}")
        except Exception as ex:
            st.error(f"Repo export failed: {ex}")

with colB:
    # Build IES text in-memory for download
    try:
        ies_name = build_repo_filename(w, ext=".ies")
        ies_text = build_ies_text(inject_calculated_metadata(w))
    except Exception as ex:
        ies_name, ies_text = "export.ies", ""
        st.warning(f"Could not prepare IES text: {ex}")

    st.download_button(
        "Download IES (LM-63)",
        data=ies_text.encode("utf-8"),
        file_name=ies_name,
        mime="text/plain",
        use_container_width=True,
        disabled=(not ies_text),
    )

with colC:
    st.caption(f"Repo folder: {repo_dir}")
    try:
        preview_name = build_repo_filename(w, ext=".json")
        st.caption(f"Preview filename: {preview_name}")
    except Exception:
        pass
st.divider()

st.caption("© 2025 LB-Lighting P/L.")