from __future__ import annotations
import json
from pathlib import Path
import streamlit as st

from utils.ui_nav import render_sidebar
from utils.paths import ASSETS_DIR

render_sidebar("Admin / Baseline Equipment")
st.title("Baseline Equipment Scheduler (stub)")

base_dir = ASSETS_DIR / "baselines"
reg_path = base_dir / "register.json"
st.code(str(base_dir))

# Ensure folders exist
for p in [
    base_dir,
    base_dir / "goni" / "reports", base_dir / "goni" / "snapshots",
    base_dir / "spectrum_analyzer" / "reports", base_dir / "spectrum_analyzer" / "snapshots",
    base_dir / "thermal" / "reports", base_dir / "thermal" / "snapshots",
    base_dir / "leds" / "lm80", base_dir / "leds" / "tm21", base_dir / "leds" / "snapshots",
    base_dir / "ecg" / "reports", base_dir / "ecg" / "snapshots",
    base_dir / "boards" / "reports", base_dir / "boards" / "snapshots",
]:
    p.mkdir(parents=True, exist_ok=True)

# Very small register viewer
if reg_path.exists():
    try:
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
    except Exception as ex:
        st.error(f"register.json unreadable: {ex}")
        reg = {}
else:
    reg = {}
    reg_path.write_text(json.dumps({"items": []}, indent=2), encoding="utf-8")

st.success(f"Register at: {reg_path}")
st.json(reg)
