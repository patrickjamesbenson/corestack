# pages/90_Provenance_Manager.py
from __future__ import annotations
import json, os, sys
from pathlib import Path
import pandas as pd
import streamlit as st
from utils.ui_nav import render_sidebar
render_sidebar("ADMIN")

from utils.paths import ies_repo_current_path

render_sidebar("Admin / Repo")
st.title("Repo Manager")

repo = ies_repo_current_path()
st.caption(f"Repo folder: {repo}")

# Open folder (Windows-only will use explorer, mac uses open, linux uses xdg-open)
def _open_folder(p: Path) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(p))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f'open "{p}"')
        else:
            os.system(f'xdg-open "{p}"')
    except Exception as ex:
        st.error(f"Could not open folder: {ex}")

c1, c2 = st.columns([1.0, 2.5])
with c1:
    st.button("Open repo folder", on_click=lambda: _open_folder(repo), use_container_width=True)

# List JSONs (your workflow repo items) and basic summary
rows = []
for p in sorted(repo.glob("*.json")):
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        name   = data.get("repo_name") or p.stem
        length = data.get("geometry", {}).get("G8", "")
        vcnt   = len((data.get("photometry") or {}).get("v_angles") or [])
        hcnt   = len((data.get("photometry") or {}).get("h_angles") or [])
        rows.append({"file": p.name, "name": name, "length_mm": length, "V/H": f"{vcnt}/{hcnt}"})
    except Exception:
        rows.append({"file": p.name, "name": "", "length_mm": "", "V/H": ""})

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.info("Later we’ll add associations (original IES, lab PDF, LM80/TM21, calibration) to each repo item.")

st.divider()

st.caption("© 2025 LB-Lighting P/L.")