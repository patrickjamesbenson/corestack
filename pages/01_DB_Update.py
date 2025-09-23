from __future__ import annotations
import json
from pathlib import Path
import streamlit as st

from utils.ui_nav import render_sidebar
from utils.paths import ensure_assets, db_key_default_path, workflow_current_path, ies_repo_current_path

render_sidebar("Admin / DB & Diagnostics")
st.title("Admin → DB & Diagnostics")

ensure_assets()

# Prefills
DEFAULT_SHEET_ID = "1tvu_RGbwBb62dj8GqDnCoS1fJPmx8AaGA66VsVcyqvw"
DEFAULT_KEY_PATH = db_key_default_path()
DEFAULT_DB_PATH  = workflow_current_path()
DEFAULT_IES_DIR  = ies_repo_current_path()

# Collapsed “prefill/state” region
with st.expander("Prefills & Paths (click to view)", expanded=False):
    st.success(f"Google Sheet ID: {DEFAULT_SHEET_ID}")
    st.success(f"Service Account Key (JSON): {DEFAULT_KEY_PATH}")
    st.success(f"DB JSON Output: {DEFAULT_DB_PATH}")
    st.success(f"IES Repo: {DEFAULT_IES_DIR}")

# Editable inputs (persist via session_state so they survive restarts if unchanged)
key_path = st.text_input("Service Account Key JSON", value=str(DEFAULT_KEY_PATH))
sheet_id = st.text_input("Google Sheet ID", value=DEFAULT_SHEET_ID)
out_path = st.text_input("Output DB JSON", value=str(DEFAULT_DB_PATH))

# Quick link to the Google Sheet
sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit"
st.link_button("Open Google Sheet →", sheet_url)

st.divider()

# --- DB Update action
def do_pull():
    try:
        from utils.gsheets_pull import pull_all_sheets
    except Exception:
        st.error("Update failed: No module named 'utils.gsheets_pull' (install or keep the helper present).")
        return

    try:
        data = pull_all_sheets(sheet_id, Path(key_path))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        st.success(f"DB updated → {out_path}")
    except Exception as ex:
        st.error(f"Update failed: {ex}")

st.button("Connect → Download all worksheets → Save to repo", type="primary", on_click=do_pull)

st.divider()

# --- Diagnostics section (rolled into this page)
st.subheader("Diagnostics")

db_path = Path(out_path)
ies_dir = Path(DEFAULT_IES_DIR)

if db_path.exists():
    st.success(f"DB found at: {db_path}")
    try:
        data = json.loads(db_path.read_text(encoding="utf-8"))
        # Required tables check
        ok_msgs, warn_msgs = [], []
        if "PHOTOM_LAYOUT" in data:
            ok_msgs.append("PHOTOM_LAYOUT sheet present.")
        else:
            warn_msgs.append("PHOTOM_LAYOUT not loaded.")
        if "one_sheet_schema_template" in data:
            ok_msgs.append("one_sheet_schema_template present.")
        else:
            warn_msgs.append("one_sheet_schema_template missing.")
        if ok_msgs:
            st.success("DB checks: " + " ".join(ok_msgs))
        if warn_msgs:
            st.warning("DB checks: " + " ".join(warn_msgs))
    except Exception as ex:
        st.error(f"DB JSON unreadable: {ex}")
else:
    st.error(f"DB not found at {db_path}")

if ies_dir.exists():
    ies_files = list(ies_dir.glob("*.ies")) + list(ies_dir.glob("*.IES")) + list(ies_dir.glob("*.txt")) + list(ies_dir.glob("*.json"))
    st.info(f"IES Repo at {ies_dir} — {len(ies_files)} files detected")
else:
    st.error(f"IES Repo missing at {ies_dir}")

st.link_button("Open Provenance Files Manager →", "pages/90_Provenance_Manager.py")
