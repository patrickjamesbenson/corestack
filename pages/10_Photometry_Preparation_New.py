# pages/10_Photometry_Preparation_New.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from utils.paths import DB_JSON, ies_repo_current_path
from utils.photometry_engine_adapter import (
    parse_ies,
    ordered_metadata_rows,
    geometry_rows_ordered,
    luminous_flux_with_note,
    power_watts,
    lumens_per_watt,
    beam_angle_with_note,
    apply_geometry,
    pack_flags,
    resample_photometry,
    source_is_0_to_90,
    build_metadata_from_photom_layout,
    export_ies_lm63,
    scale_working_length_and_flux,
)

st.set_page_config(page_title="Photometry Preparation", layout="wide")
ss = st.session_state
ss.setdefault("working", None)
ss.setdefault("geometry_locked", False)
ss.setdefault("loaded_file_id", None)
ss.setdefault("flags", (True, True, True, False, False))  # Absolute, Interpolated, Scaled, Simulated, Other
ss.setdefault("interp_target_v", 181)
ss.setdefault("interp_target_h", 25)
ss.setdefault("auto_zero_by_source", True)         # NEW: if source is 0–90, auto-zero >90
ss.setdefault("manual_zero_beyond_90", False)      # optional manual override
ss.setdefault("flip_vertical", False)
ss.setdefault("meta_rows", None)                   # editable copy presented to user
ss.setdefault("scale_target_length_mm", None)      # for scaling step

st.title("Photometry Preparation")

# ---------------- Load IES (memoized) ----------------
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
            ss["meta_rows"] = None
            st.toast("IES loaded & parsed.", icon="✅")
    except Exception as ex:
        st.error(f"Parse failed: {ex}")

w = ss.get("working")
if not w:
    st.info("Load an IES file to begin.")
    st.stop()

ph = w.get("photometry") or {}
V = ph.get("v_angles") or []
H = ph.get("h_angles") or []
I = np.array(ph.get("candela") or [], dtype=float)

# ---------------- Snapshot row ----------------
lm, lm_note = luminous_flux_with_note(w)
pw = power_watts(w)
lpw = lumens_per_watt(w)
ba, ba_note = beam_angle_with_note(w)

def _fmt(x, d=1):
    return "" if x is None else f"{float(x):.{d}f}"

cols = st.columns([1.1, 1.0, 1.0, 1.1, 1.0])
with cols[0]:
    st.metric("Luminous Flux (lm)", _fmt(lm, 1))
    if lm_note:
        st.caption(lm_note)
with cols[1]:
    st.metric("Power (W)", _fmt(pw, 1))
with cols[2]:
    st.metric("Lm/W", _fmt(lpw, 1))
with cols[3]:
    st.metric("Beam angle (°)", _fmt(ba, 1))
    if ba_note:
        st.caption(ba_note)
with cols[4]:
    st.metric("V/H count", f"{len(V)}/{len(H)}")

st.divider()

# ---------------- Raw geometry + angles (side-by-side) ----------------
st.subheader("Raw Geometry Detail • V-angles (°) • H-angles (°)")
g_rows = []
labels = {
    0:"G0 • # lamps", 1:"G1 • lumens per lamp", 2:"G2 • candela multiplier",
    3:"G3 • V-count", 4:"G4 • H-count", 5:"G5 • photometric type (1=C,2=B,3=A)",
    6:"G6 • units (1=ft,2=m)", 7:"G7 • width (mm)", 8:"G8 • length (mm)",
    9:"G9 • height (mm)", 10:"G10 • ballast factor", 11:"G11 • file generation type",
    12:"G12 • input/system watts"
}
for (g_label, g_key, val) in geometry_rows_ordered(w):
    try:
        gi = int(g_label[1:])
    except Exception:
        continue
    if gi <= 12:
        g_rows.append({"Key": labels.get(gi, g_key), "Value": val})

c1, c2, c3 = st.columns([1.2, 1.0, 1.0])
with c1:
    st.dataframe(pd.DataFrame(g_rows), use_container_width=True, hide_index=True, height=300)
with c2:
    st.dataframe(pd.DataFrame({"V angle (°)": V}), use_container_width=True, hide_index=True, height=300)
with c3:
    st.dataframe(pd.DataFrame({"H angle (°)": H}), use_container_width=True, hide_index=True, height=300)

st.divider()

# ---------------- Geometry Edit & Flags ----------------
st.subheader("Geometry Edit & Flags")
st.markdown(f"**Lock status:** {'🔒 Locked' if ss['geometry_locked'] else '🔓 Unlocked'}")

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
        g10_bf = st.number_input("G10 ballast factor", 0.0, value=float(g.get("G10", 0.0)), step=0.01, disabled=locked)

    c5, c6, c7 = st.columns(3)
    with c5:
        g12_input = st.number_input("G12 input watts", 0.0, value=float(g.get("G12", 0.0)), step=0.01, disabled=locked)
    with c6:
        g13_raw = st.number_input("G13 raw board lumens", 0.0, value=float(g.get("G13", 0.0)), step=1.0, disabled=locked)
    with c7:
        g14_circ = st.number_input("G14 circuit watts", 0.0, value=float(g.get("G14", 0.0)), step=0.01, disabled=locked)

    st.caption("LM-63 provenance flags")
    fc1, fc2, fc3, fc4, fc5 = st.columns(5)
    f_abs = fc1.toggle("Absolute", ss["flags"][0], disabled=locked)
    f_int = fc2.toggle("Interpolated", ss["flags"][1], disabled=locked)
    f_scl = fc3.toggle("Scaled", ss["flags"][2], disabled=locked)
    f_sim = fc4.toggle("Simulated", ss["flags"][3], disabled=locked)
    f_oth = fc5.toggle("Other", ss["flags"][4], disabled=locked)
    flag_code = pack_flags((f_abs, f_int, f_scl, f_sim, f_oth))
    st.caption(f"Flags: {flag_code}")

    left, right = st.columns(2)
    apply_lock_btn = left.form_submit_button(
        "Apply Geometry Edits & Lock (proceed to Metadata)",
        type="primary",
        use_container_width=True,
        disabled=locked,
    )
    unlock_btn = right.form_submit_button("Unlock geometry edits", use_container_width=True, disabled=not locked)

    if apply_lock_btn and not locked:
        ss["flags"] = (f_abs, f_int, f_scl, f_sim, f_oth)
        ss["working"] = apply_geometry(
            ss["working"],
            {
                "G7": width_mm, "G8": length_mm, "G9": height_mm,
                "G10": g10_bf,  "G12": g12_input, "G13": g13_raw, "G14": g14_circ,
                "file_generation_flags": flag_code,  # also mirrors to G11
            },
        )
        ss["geometry_locked"] = True
        ss["meta_rows"] = None  # force rebuild in Metadata
        st.toast("Geometry + flags applied. Proceeding to Metadata…", icon="✅")
        st.experimental_rerun()

    if unlock_btn and locked:
        ss["geometry_locked"] = False
        st.toast("Geometry unlocked.", icon="🔓")
        st.experimental_rerun()

st.divider()

# ---------------- Metadata ----------------
st.subheader("Metadata")
if not ss["geometry_locked"]:
    st.info("Lock geometry to proceed to Metadata.")
else:
    try:
        db = json.loads(Path(DB_JSON).read_text(encoding="utf-8"))
    except Exception:
        db = {}
    if ss["meta_rows"] is None:
        ss["meta_rows"] = build_metadata_from_photom_layout(ss["working"], db)

    df_meta = pd.DataFrame(ss["meta_rows"])
    editable_mask = df_meta["IES_FUNC"].str.lower().str.startswith("editable")

    df_meta_edit = st.data_editor(
        df_meta,
        disabled=~editable_mask,
        use_container_width=True,
        hide_index=True,
        height=380,
    )
    ss["meta_rows"] = df_meta_edit.to_dict(orient="records")

st.divider()

# ---------------- Interpolation / Normalise Angle Counts ----------------
st.subheader("Normalise Angle Counts")

c1, c2, c3, c4 = st.columns([1,1,1,1])
with c1:
    ss["interp_target_v"] = st.number_input("Target V count", min_value=3, max_value=721, value=int(ss["interp_target_v"]), step=2, help="Default 181")
with c2:
    ss["interp_target_h"] = st.number_input("Target H count", min_value=1, max_value=361, value=int(ss["interp_target_h"]), step=1, help="Default 25")
with c3:
    ss["auto_zero_by_source"] = st.checkbox("Auto-zero >90° if source is 0–90", value=bool(ss["auto_zero_by_source"]))
with c4:
    ss["flip_vertical"] = st.checkbox("Flip hemispheres (vertical)", value=bool(ss["flip_vertical"]))

# manual zero (only used if auto is off)
if not ss["auto_zero_by_source"]:
    ss["manual_zero_beyond_90"] = st.checkbox("Force zero beyond 90° (manual)", value=bool(ss["manual_zero_beyond_90"]))

force_zero = (source_is_0_to_90(w) if ss["auto_zero_by_source"] else ss["manual_zero_beyond_90"])

# live preview tables
try:
    Vt, Ht, It = resample_photometry(
        V, H, I, ss["interp_target_v"], ss["interp_target_h"],
        force_zero_beyond_90=bool(force_zero),
        flip_vertical=bool(ss["flip_vertical"]),
    )
    pv1, pv2 = st.columns(2)
    with pv1:
        st.caption("Preview • H-plane 0")
        h0 = 0
        st.dataframe(pd.DataFrame({"V (°)": Vt, "I[H=0] cd": It[h0, :]}), use_container_width=True, hide_index=True, height=260)
    with pv2:
        st.caption("Preview • V-sample 0")
        v0 = 0
        st.dataframe(pd.DataFrame({"H (°)": Ht, "I[V=0] cd": It[:, v0]}), use_container_width=True, hide_index=True, height=260)

    cmt1, cmt2 = st.columns([1,1])
    if cmt1.button("Commit interpolated photometry", type="primary", use_container_width=True):
        ss["working"]["photometry"] = {"v_angles": Vt, "h_angles": Ht, "candela": It.tolist()}
        st.toast("Interpolated photometry committed.", icon="✅")
        st.experimental_rerun()
    if cmt2.button("Revert previews (do not commit)", use_container_width=True):
        st.experimental_rerun()
except Exception as ex:
    st.error(f"Interpolation preview failed: {ex}")

st.divider()

# ---------------- Scaling (optional) ----------------
st.subheader("Scaling (length/flux/power)")
geom = w.get("geometry", {}) or {}
cur_len = float(geom.get("G8", 0.0))
if ss["scale_target_length_mm"] is None:
    ss["scale_target_length_mm"] = cur_len if cur_len > 0 else 1000.0

sc1, sc2 = st.columns([2,1])
with sc1:
    ss["scale_target_length_mm"] = st.number_input("Target length (mm)", min_value=1.0, value=float(ss["scale_target_length_mm"]), step=1.0)
with sc2:
    if st.button("Preview scale", use_container_width=True):
        preview = scale_working_length_and_flux(ss["working"], ss["scale_target_length_mm"])
        lm_p, _ = luminous_flux_with_note(preview)
        pw_p = power_watts(preview)
        st.success(f"Preview → Lm: {lm_p:.1f}  |  Power: {pw_p:.1f}")

ap1, ap2 = st.columns([1,1])
with ap1:
    if st.button("Apply scaling (commit)", type="primary", use_container_width=True):
        ss["working"] = scale_working_length_and_flux(ss["working"], ss["scale_target_length_mm"])
        st.toast("Scaling applied.", icon="✅")
        st.experimental_rerun()
with ap2:
    st.caption("Scaling multiplies G8, G12, G14, G13 and candela by the same length ratio (linear lamp).")

st.divider()

# ---------------- Export / Repo ----------------
st.subheader("Export")
exp_c1, exp_c2, exp_c3 = st.columns([1,1,1])
with exp_c1:
    fname_json = st.text_input("Sandbox JSON filename", value="photometry_working.json")
    if st.button("Download sandbox JSON", use_container_width=True):
        st.download_button(
            "Click to download JSON",
            data=json.dumps(ss["working"], ensure_ascii=False, indent=2),
            file_name=fname_json,
            mime="application/json",
        )
with exp_c2:
    fname_ies = st.text_input("Sandbox IES filename", value="photometry_export.ies")
    if st.button("Download sandbox IES", use_container_width=True):
        meta_rows = ss.get("meta_rows") or []
        ies_text = export_ies_lm63(ss["working"], meta_rows)
        st.download_button(
            "Click to download IES",
            data=ies_text.encode("utf-8"),
            file_name=fname_ies,
            mime="text/plain",
        )
with exp_c3:
    repo_btn = st.button("Save IES to Repo", type="primary", use_container_width=True)
    if repo_btn:
        try:
            meta_rows = ss.get("meta_rows") or []
            ies_text = export_ies_lm63(ss["working"], meta_rows)
            repo_dir = ies_repo_current_path()
            Path(repo_dir).mkdir(parents=True, exist_ok=True)
            outp = Path(repo_dir) / fname_ies
            outp.write_text(ies_text, encoding="utf-8")
            st.success(f"Saved → {outp}")
        except Exception as ex:
            st.error(f"Repo save failed: {ex}")
