from __future__ import annotations
import json
from pathlib import Path
import streamlit as st

from utils.ui_nav import render_sidebar
from utils.paths import workflow_current_path

render_sidebar("Workflow / Outputs & Scenes")
st.title("Outputs & Scenes")

db_path = workflow_current_path()
if not db_path.exists():
    st.error(f"DB not found at {db_path}. Update it in Admin → DB & Diagnostics.")
    st.stop()

try:
    data = json.loads(Path(db_path).read_text(encoding="utf-8"))
except Exception as ex:
    st.error(f"DB JSON unreadable: {ex}")
    st.stop()

# Require the template sheet you mentioned
if "one_sheet_schema_template" not in data:
    st.error("JSON missing tables → one_sheet_schema_template")
    st.stop()

st.success(f"Loaded DB: {db_path.name}")
st.caption("Template sheet found: one_sheet_schema_template")

# Show a peek so you can validate
rows = data.get("one_sheet_schema_template") or []
st.write(f"Rows in template: {len(rows)}")
if rows[:1]:
    st.dataframe(rows[:50], use_container_width=True)

st.divider()

st.caption("© 2025 LB-Lighting P/L.")