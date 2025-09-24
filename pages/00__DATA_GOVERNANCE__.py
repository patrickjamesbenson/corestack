# pages/00__DATA_GOVERNANCE__.py
from __future__ import annotations

import json
from pathlib import Path
import datetime as _dt
import pandas as pd
import streamlit as st

from utils.paths import (
    ensure_assets,
    ASSETS_DIR, DB_DIR, IES_REPO_DIR, LOG_DIR, KEYS_DIR,
    workflow_current_path,
    ies_repo_current_path,
    db_key_default_path,
)

st.set_page_config(page_title="Data Governance", layout="wide")
st.title("Data Governance")

ensure_assets()

DEFAULT_SHEET_ID = "1tvu_RGbwBb62dj8GqDnCoS1fJPmx8AaGA66VsVcyqvw"
DEFAULT_KEY_PATH = db_key_default_path()
DEFAULT_DB_PATH  = workflow_current_path()

db_path  = workflow_current_path()
repo_dir = ies_repo_current_path()

# ------------------------------------------------------------------
# Minimal CSS helpers (non-invasive)
# ------------------------------------------------------------------
st.markdown("""
<style>
.badge-open {
  display:inline-block; padding:2px 8px; border-radius:999px;
  background:#d9534f; color:#fff; font-size:12px; font-weight:600; margin:4px 0;
}
.badge-green {
  display:inline-block; padding:2px 8px; border-radius:999px;
  background:#2e7d32; color:#fff; font-size:12px; font-weight:600; margin-left:8px;
}
.btn-green a {
  display:inline-block; text-decoration:none; text-align:center;
  padding:0.5rem 0.75rem; border-radius:0.375rem;
  background:#2e7d32; color:#fff; font-weight:600;
}
.btn-green a:hover { filter:brightness(0.95); }
.pill-yellow {
  display:inline-block; padding:4px 10px; border-radius:999px;
  background:#ffe08a; color:#6b5300; font-weight:600; font-size:12px;
}
.pill-red {
  display:inline-block; padding:4px 10px; border-radius:999px;
  background:#d9534f; color:#fff; font-weight:600; font-size:12px;
}
</style>
""", unsafe_allow_html=True)

st.divider()

# ----------------------------- Prefills & Paths (stacked fields) -----------------------------
st.subheader("Ref_Data Prefills & Paths")

key_path = st.text_input(
    "Service Account Key JSON (optional)",
    value="",
    placeholder=str(DEFAULT_KEY_PATH),
    help="Leave empty to use GOOGLE_APPLICATION_CREDENTIALS or Streamlit secrets. "
         "Typical local path example: C:\\Users\\YourName\\novon_db_key.json",
)
sheet_id = st.text_input(
    "Google Sheet ID",
    value=DEFAULT_SHEET_ID,
    help="The long ID in the Google Sheets URL between /d/ and /edit.",
)
out_path = st.text_input(
    "Output DB JSON",
    value=str(DEFAULT_DB_PATH),
    help="Where to save the pulled reference data JSON (e.g., assets\\db\\novon_db.json).",
)

st.divider()

# ----------------------------- Ref Data Edit & Sync (3 equal columns) -----------------------------
c1, c2, c3 = st.columns(3, gap="small")
with c1:
    st.subheader("Ref_Data Edit & Sync")

with c2:
    # Green link button (exact label kept)
    st.markdown(
        f'<div class="btn-green"><a href="https://docs.google.com/spreadsheets/d/{sheet_id}" target="_blank">'
        'Open Ref_Data (Google Sheet) →</a></div>',
        unsafe_allow_html=True,
    )

with c3:
    # Sync button + last-sync badge
    def _p(s: str) -> Path:
        return Path(s).expanduser().resolve()

    def do_pull():
        if not out_path:
            st.error("Update failed: Output DB JSON path is empty.")
            return
        out_file = _p(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        key_file: Path | None = None
        if key_path.strip():
            kp = _p(key_path)
            if not kp.exists():
                st.error(f"Update failed: Key file not found → {kp}")
                return
            key_file = kp

        try:
            from utils.gsheets_pull import pull_all_sheets
        except Exception:
            st.error("Update failed: No module named 'utils.gsheets_pull'")
            return

        try:
            data = pull_all_sheets(sheet_id.strip(), key_file)
            out_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            st.session_state["refdata_last_sync_ts"] = _dt.datetime.now().strftime("%a %d %b %Y %H:%M:%S")
            st.success(f"DB updated → {out_file}")
        except Exception as ex:
            st.error(f"Update failed: {ex}")

    cols_btn = st.columns([1,1], gap="small")
    with cols_btn[0]:
        st.button(
            "Sync Ref_Data (JSON)",
            type="primary",
            use_container_width=True,
            on_click=do_pull,
            help="Pulls every sheet from Google and writes a single JSON file to the path above.",
        )
    with cols_btn[1]:
        ts = st.session_state.get("refdata_last_sync_ts", "")
        if ts:
            st.markdown(f'<span class="badge-green">{ts}</span>', unsafe_allow_html=True)
        else:
            st.empty()

st.divider()

# ----------------------------- Ref_Data Sheet Status (pills + single-click preview) -----------------------------
st.subheader("Ref_Data Sheet Status")

if db_path.exists():
    st.success(f"DB ✓ {db_path}")
    try:
        data = json.loads(db_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data:
            sel_key = "dg_preview_sheet"
            st.session_state.setdefault(sel_key, "")

            sheet_names = sorted(data.keys())
            cols_per_row = 4
            rows = (len(sheet_names) + cols_per_row - 1) // cols_per_row
            idx = 0
            for _ in range(rows):
                cols = st.columns(cols_per_row, gap="small")
                for c in cols:
                    if idx >= len(sheet_names):
                        break
                    sname = sheet_names[idx]
                    rows_count = 0
                    try:
                        val = data[sname]
                        if isinstance(val, list):
                            rows_count = len(val)
                    except Exception:
                        pass

                    pill_label = f"{sname} ({rows_count})"
                    c.success(pill_label)

                    selected = (st.session_state.get(sel_key) == sname)

                    if not selected:
                        # Single-click open: set state then immediate rerun
                        if c.button("Preview", key=f"pv_{sname}", use_container_width=True,
                                    help="Show a live preview of this sheet"):
                            st.session_state[sel_key] = sname
                            st.rerun()
                    else:
                        c.markdown('<span class="badge-open" title="Preview is open for this sheet.">Preview open</span>',
                                   unsafe_allow_html=True)
                        if c.button("Close preview", key=f"pv_close_{sname}", use_container_width=True,
                                    type="primary", help="Close the live preview for this sheet"):
                            st.session_state[sel_key] = ""
                            st.rerun()

                    idx += 1

            target = st.session_state.get(sel_key) or ""
            if target and target in data:
                st.markdown("---")
                st.subheader(f"Preview: {target}")
                try:
                    records = data[target]
                    if isinstance(records, list):
                        df = pd.DataFrame(records)
                    elif isinstance(records, dict):
                        try:
                            df = pd.DataFrame(records)
                            if df.empty:
                                df = pd.DataFrame(list(records.items()), columns=["Key", "Value"])
                        except Exception:
                            df = pd.DataFrame(list(records.items()), columns=["Key", "Value"])
                    else:
                        df = pd.DataFrame({"value": [records]})
                    st.dataframe(df, use_container_width=True, height=420)
                except Exception as ex:
                    st.error(f"Could not render preview for '{target}': {ex}")
        else:
            st.warning("DB loaded, but no sheets found.")
    except Exception as ex:
        st.error(f"DB unreadable: {ex}")
else:
    st.error(f"DB not found at {db_path}")

st.divider()

# ----------------------------- Repo checks -----------------------------
st.subheader("Photometry Repo")
if repo_dir.exists():
    n_json = len(list(repo_dir.glob("*.json")))
    n_ies  = len(list(repo_dir.glob("*.ies"))) + len(list(repo_dir.glob("*.IES")))
    if (n_json + n_ies) > 0:
        st.success(f"Photometry Repo at {repo_dir} — {n_json} JSON, {n_ies} IES")
    else:
        st.info(f"Photometry Repo at {repo_dir} (empty)")
else:
    st.error(f"Photometry Repo missing at {repo_dir}")

st.divider()

# ----------------------------- Assets Summary -----------------------------
st.subheader("Assets folder is initialised")
st.code(
    "\n".join(
        [
            f"assets:           {ASSETS_DIR}",
            f"assets/db:        {DB_DIR}",
            f"assets/ies_repo:  {IES_REPO_DIR}",
            f"assets/logs:      {LOG_DIR}",
            f"assets/keys:      {KEYS_DIR}",
        ]
    ),
    language="text",
)

st.divider()

# ----------------------------- Legacy path scan (one line: title + yellow pill + button) -----------------------------
st.session_state.setdefault("legacy_scan_ts", _dt.datetime.now().isoformat(timespec="seconds"))
st.session_state.setdefault("legacy_hits", [])
st.session_state.setdefault("legacy_scanning", False)

legacy_tokens = (
    "assets/performance",
    "assets/provenance",
    "ies_prep/IES Repo",
    "../ies_prep",
)

def _scan_legacy() -> list[str]:
    hits: list[str] = []
    for p in Path(".").rglob("*.py"):
        try:
            s = p.read_text(encoding="utf-8", errors="ignore")
            if any(tok in s for tok in legacy_tokens):
                hits.append(str(p))
        except Exception:
            pass
    return sorted(set(hits))

# If a scan was requested, perform it early in this run so the UI updates once.
if st.session_state["legacy_scanning"]:
    with st.spinner("Scanning for legacy paths..."):
        st.session_state["legacy_hits"] = _scan_legacy()
        st.session_state["legacy_scan_ts"] = _dt.datetime.now().isoformat(timespec="seconds")
        st.session_state["legacy_scanning"] = False
    st.rerun()

row = st.columns(3, gap="small")
with row[0]:
    st.subheader("Legacy path scan")
with row[1]:
    if st.session_state["legacy_scanning"]:
        st.markdown('<span class="pill-red">Scanning…</span>', unsafe_allow_html=True)
    else:
        ts = _dt.datetime.fromisoformat(st.session_state["legacy_scan_ts"]).strftime("%a %d %b %Y %H:%M:%S")
        count = len(st.session_state["legacy_hits"])
        st.markdown(f'<span class="pill-yellow">{count} file(s) — {ts}</span>', unsafe_allow_html=True)
with row[2]:
    if st.button("Rescan now", use_container_width=True, help="Re-scan the codebase for legacy asset paths"):
        st.session_state["legacy_scanning"] = True
        st.rerun()

hits = st.session_state["legacy_hits"]
if hits:
    st.warning("Files still referencing legacy locations:")
    for h in hits:
        st.text(h)
else:
    st.success("No references to legacy asset paths detected.")

st.divider()
st.caption("© 2025 LB-Lighting P/L.")
