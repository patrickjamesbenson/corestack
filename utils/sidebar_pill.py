# utils/sidebar_pill.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import streamlit as st

def _pill(text: str, ok: bool | None = None) -> None:
    # Keep pills minimal and robust (why: tolerate mixed callers)
    if ok is None: bg, fg = "#EEE", "#333"
    elif ok:       bg, fg = "#E8FAEF", "#0B634B"
    else:          bg, fg = "#FDEEEE", "#8A1C1C"
    st.markdown(
        f'<span style="display:inline-block;margin:2px 6px 2px 0; padding:6px 10px; '
        f'border-radius:16px; background:{bg}; color:{fg}; font-weight:600; font-size:12px;">{text}</span>',
        unsafe_allow_html=True,
    )

def _mtime_label(p: Path) -> str:
    try:
        return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "?"

def wf_pill(wf_path: Path) -> None:
    if not wf_path or not Path(wf_path).exists():
        _pill("Workflow ✗ missing", ok=False)
        return
    p = Path(wf_path)
    _pill(f"Workflow ✓ {p.name} — {_mtime_label(p)}", ok=True)

def repo_pill(repo_json_path: Path) -> None:
    p = Path(repo_json_path) if repo_json_path else None
    if not p or not p.exists():
        _pill("IES Repo ✗ missing", ok=False)
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        # support either {"records":[...]} or {"files":[...]}
        records = data.get("records")
        if isinstance(records, list):
            n = len(records)
        else:
            files = data.get("files")
            n = len(files) if isinstance(files, list) else None
        label = f"IES Repo ✓ {n} items — {_mtime_label(p)}" if n is not None else f"IES Repo ✓ — {_mtime_label(p)}"
        _pill(label, ok=True)
    except Exception:
        _pill("IES Repo ! unreadable JSON", ok=False)

def provenance_pill(prov_json_path: Path) -> None:
    p = Path(prov_json_path) if prov_json_path else None
    if not p or not p.exists():
        _pill("Provenance ✗ (creates on first write)", ok=False)
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        n = len(data.get("records", []))
        _pill(f"Provenance ✓ {n} record(s) — {_mtime_label(p)}", ok=True)
    except Exception:
        _pill("Provenance ! unreadable JSON", ok=False)

def render_sanity_pills(*args, **kwargs) -> None:
    """
    Backward compatible with old usage (why: avoid refactors across pages):
      render_sanity_pills(workflow_path, repo_path)
      render_sanity_pills(wf_path=..., repo_json_path=..., prov_json_path=...)
    """
    wf = kwargs.get("wf_path")
    repo = kwargs.get("repo_json_path")
    prov = kwargs.get("prov_json_path")
    if wf is None and len(args) >= 1: wf = args[0]
    if repo is None and len(args) >= 2: repo = args[1]
    if prov is None and len(args) >= 3: prov = args[2]
    if wf: wf_pill(Path(wf))
    if repo: repo_pill(Path(repo))
    if prov: provenance_pill(Path(prov))
