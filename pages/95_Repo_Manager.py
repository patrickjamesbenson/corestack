# pages/95_Repo_Manager.py
from __future__ import annotations
import os, json
from pathlib import Path
import streamlit as st
import pandas as pd

from utils.provenance import (
    DB_DISPLAY_NAME,
    ASSETS, IES_INCOMING, IES_REPO, IES_RETIRED, RPT_PHOT, RPT_LED, RPT_CAL, RPT_THERM,
    ensure_dirs, make_record, add_record, list_records, update_record, retire_record, move_to_repo,
    write_bytes, destination_dir, audit_all, make_id
)

st.set_page_config(page_title="Repo & Provenance Manager", layout="wide")
ensure_dirs()

st.title("Repo & Provenance Manager")

# Prefill from query params (e.g., ?manager_label=DNX6022840&state=repo)
q = st.query_params if hasattr(st, "query_params") else {}
prefill_label = (q.get("manager_label") or [""])[0] if isinstance(q.get("manager_label"), list) else (q.get("manager_label") or "")
prefill_state = (q.get("state") or ["incoming"])[0] if isinstance(q.get("state"), list) else (q.get("state") or "incoming")
prefill_state = "repo" if str(prefill_state).lower().startswith("repo") else "incoming"

# Photometry page may have left a working dict
ss = st.session_state
working = ss.get("working")

# --- Section: File current working IES into repo/sandbox ---
st.subheader("File current IES")
colA, colB, colC = st.columns([1.2,1,1])
with colA:
    label = st.text_input("Label (e.g. LUMCAT or friendly name)", value=prefill_label)
with colB:
    dest = st.selectbox("Destination", ["incoming (Sandbox)","repo (Publish)"], index=(1 if prefill_state=="repo" else 0))
with colC:
    state = "incoming" if dest.startswith("incoming") else "repo"
    st.metric("State", state.upper())

st.caption(f"DB: {DB_DISPLAY_NAME}")

a1,a2,a3 = st.columns(3)
with a1:
    phot_pdf = st.file_uploader("Photometry report (PDF)", type=["pdf"], key="phot_pdf")
    lm80_pdf = st.file_uploader("LED LM-80 (PDF)", type=["pdf"], key="lm80_pdf")
with a2:
    tm21_pdf = st.file_uploader("LED TM-21 (PDF)", type=["pdf"], key="tm21_pdf")
    gcal_pdf = st.file_uploader("Goniometer calibration (PDF)", type=["pdf"], key="gcal_pdf")
with a3:
    scal_pdf = st.file_uploader("Spectro calibration (PDF)", type=["pdf"], key="scal_pdf")
    therm_pdf= st.file_uploader("Thermal report (PDF)", type=["pdf"], key="therm_pdf")

save_btn = st.button("Create provenance record", type="primary", use_container_width=True, disabled=(not label))
if save_btn:
    ensure_dirs()
    ies_json_path = None
    ies_txt_path = None
    if isinstance(working, dict):
        out_json = json.dumps(working, indent=2)
        dst = destination_dir(state) / f"{label}.ies.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(out_json, encoding="utf-8")
        ies_json_path = str(dst)

    def _save(uploader, base_dir: Path, fname: str):
        if uploader is not None:
            p = base_dir / f"{label}_{fname}"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(uploader.getvalue())
            return str(p)
        return None

    att_paths = {
        "photometry_pdf": _save(phot_pdf, RPT_PHOT, "photometry.pdf"),
        "lm80_pdf":       _save(lm80_pdf,  RPT_LED,  "lm80.pdf"),
        "tm21_pdf":       _save(tm21_pdf,  RPT_LED,  "tm21.pdf"),
        "goni_cal_pdf":   _save(gcal_pdf,  RPT_CAL,  "goni_cal.pdf"),
        "spectro_cal_pdf":_save(scal_pdf,  RPT_CAL,  "spectro_cal.pdf"),
        "thermal_pdf":    _save(therm_pdf, RPT_THERM,"thermal.pdf"),
        "other": None,
    }

    geo = (working or {}).get("geometry", {}) or {}
    snap = {
        "G7": geo.get("G7"), "G8": geo.get("G8"), "G9": geo.get("G9"),
        "G11": geo.get("G11"), "G12": geo.get("G12"), "G13": geo.get("G13"), "G14": geo.get("G14"),
    }

    rec = make_record(
        rec_id=make_id(label),
        label=label,
        state=state,
        ies_json_path=ies_json_path,
        ies_txt_path=ies_txt_path,
        meta=snap,
        attachments=att_paths,
    )
    add_record(rec)
    st.success(f"Record created: {rec['id']}")

st.divider()

# --- Section: Browse & maintain records ---
st.subheader("Browse records")
scope = st.selectbox("Filter", ["all","incoming","repo","retired"], index=(1 if prefill_state=="repo" else 0))
flt = None if scope=="all" else scope
recs = list_records(flt)
if recs:
    df = pd.DataFrame([
        {
            "ID": r.get("id"), "Label": r.get("label"), "State": r.get("state"),
            "Updated": r.get("updated"),
            "Has JSON": bool(r.get("ies_json_path")),
        } for r in recs
    ])
    st.dataframe(df, use_container_width=True, hide_index=True, height=300)

    st.caption("Select an ID to update or retire:")
    selected = st.selectbox("Record ID", [""] + [r.get("id") for r in recs], index=0)
    if selected:
        colx, coly, colz = st.columns([1,1,1])
        if colx.button("Move to REPO", use_container_width=True):
            move_to_repo(selected)
            st.success("Moved to repo.")
            st.rerun()
        if coly.button("Retire", use_container_width=True):
            retire_record(selected)
            st.success("Retired.")
            st.rerun()
        if colz.button("Refresh list", use_container_width=True):
            st.rerun()
else:
    st.info("No records yet.")

st.divider()

# --- Section: Audit ---
st.subheader("Audit (completeness check)")
scope2 = st.selectbox("Audit scope", ["incoming","repo","retired","all"], index=1)
flt2 = None if scope2=="all" else scope2
report = audit_all(flt2)
if report:
    dfA = pd.DataFrame([{"ID": r["id"], "Label": r["label"], "Score": r["score"], "Missing": ", ".join(r["missing"])} for r in report])
    st.dataframe(dfA, use_container_width=True, hide_index=True, height=260)
else:
    st.info("Nothing to audit in this scope.")
