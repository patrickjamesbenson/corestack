
from __future__ import annotations
from pathlib import Path
import streamlit as st
import json
from .paths import workflow_current_path, ies_repo_current_path

def header_banner(app_root: Path) -> None:
    wf = workflow_current_path(app_root)
    ies = ies_repo_current_path(app_root)
    bits = []
    bits.append(f"WF ✓ {wf.name}" if wf.exists() else "WF ✗ missing")
    if ies.exists():
        try:
            data = json.loads(ies.read_text(encoding="utf-8"))
            n = len((data.get("records") or [])) if isinstance(data.get("records"), list) else None
            bits.append(f"IES ✓ {n} records" if n is not None else "IES ✓")
        except Exception:
            bits.append("IES ! unreadable")
    else:
        bits.append("IES ✗ missing")
    st.markdown(" • ".join(bits))

