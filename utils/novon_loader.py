# utils/novon_loader.py
from __future__ import annotations
import os, json, re
from pathlib import Path
from typing import Any, Dict
import streamlit as st

def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def _manifest_workflow_rel(app_root: Path) -> str | None:
    mp = app_root / "manifest.yml"
    if not mp.exists():
        return None
    txt = mp.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r'^\s*workflow_json\s*:\s*"?([^"\r\n]+)"?\s*$', txt, flags=re.M)
    return m.group(1).strip() if m else None

def _canonical_workflow_path(app_root: Path) -> Path | None:
    # 1) explicit env var
    env = os.getenv("NOVON_WORKFLOW_PATH")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p.resolve()
    # 2) manifest mapping (inside legacy_src/)
    rel = _manifest_workflow_rel(app_root)
    if rel:
        p = (app_root / "legacy_src" / rel).resolve()
        if p.exists():
            return p
    # 3) old sibling fallback (may not exist; last resort)
    return app_root / "novon_db_updater" / "assets" / "novon_workflow.json"

def _load_from_disk(app_root: Path) -> Dict[str, Any]:
    p = _canonical_workflow_path(app_root)
    if not p or not p.exists():
        raise FileNotFoundError(f"Workflow JSON not found. Set NOVON_WORKFLOW_PATH or manifest.workflow_json. Tried: {p}")
    data = _read_json(p)
    st.session_state["NOVON_WORKFLOW_PATH"] = str(p)
    st.session_state["NOVON_WORKFLOW_MTIME_NS"] = p.stat().st_mtime_ns
    st.session_state["NOVON_WORKFLOW"] = data
    return data

def load_workflow_json() -> Dict[str, Any]:
    app_root = Path(__file__).resolve().parents[1]
    path = Path(st.session_state.get("NOVON_WORKFLOW_PATH", _canonical_workflow_path(app_root) or ""))
    if not path.exists():
        return _load_from_disk(app_root)
    try:
        mtime = path.stat().st_mtime_ns
    except FileNotFoundError:
        return _load_from_disk(app_root)
    if "NOVON_WORKFLOW" not in st.session_state:
        return _load_from_disk(app_root)
    if mtime != st.session_state.get("NOVON_WORKFLOW_MTIME_NS"):
        return _load_from_disk(app_root)
    return st.session_state["NOVON_WORKFLOW"]

def reload_workflow_json() -> Dict[str, Any]:
    app_root = Path(__file__).resolve().parents[1]
    return _load_from_disk(app_root)
