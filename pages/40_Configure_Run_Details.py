# pages/40_Build_Luminaire.py
from __future__ import annotations
from pathlib import Path
import streamlit as st

from utils.ui_nav import render_sidebar
render_sidebar("Workflow / Build Luminaire")

st.title("Build Luminaire")

APP_ROOT = Path(__file__).resolve().parents[1]
candidates = [
    APP_ROOT / "legacy_src" / "segment_batch_ui.py",   # <- your current file (had BOM)
    APP_ROOT / "legacy_src" / "Segment_UI" / "segment_batch_ui.py",
]

def _load_and_exec(p: Path) -> None:
    # Read with utf-8-sig to eliminate BOM (U+FEFF) that caused the SyntaxError
    src = p.read_text(encoding="utf-8-sig")
    glb = {"__name__": "__legacy__"}
    exec(compile(src, str(p), "exec"), glb, glb)

target = next((p for p in candidates if p.exists()), None)

if target is None:
    st.error(
        "Legacy UI not found. "
        "Looked for:\n" + "\n".join(f"- {p.as_posix()}" for p in candidates)
    )
else:
    st.caption(f"Loaded legacy UI from: **{target}**")
    try:
        _load_and_exec(target)
    except Exception as ex:
        st.exception(ex)

st.divider()

st.caption("© 2025 LB-Lighting P/L.")