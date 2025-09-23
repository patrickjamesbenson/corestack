from __future__ import annotations
from typing import Any, Dict, List, Tuple

def _flatten(d: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = d
    return out

def diff_dicts(old: Dict[str, Any], new: Dict[str, Any]) -> tuple[list[str], list[tuple[str, Any, Any]], list[str]]:
    f_old = _flatten(old)
    f_new = _flatten(new)
    adds, changes, dels = [], [], []
    for k in f_new:
        if k not in f_old:
            adds.append(k)
        elif f_old[k] != f_new[k]:
            changes.append((k, f_old[k], f_new[k]))
    for k in f_old:
        if k not in f_new:
            dels.append(k)
    return adds, changes, dels

def render_changelog_md(build_id: str, adds: list[str], changes: list[tuple[str, Any, Any]], dels: list[str]) -> str:
    lines = [f"# NOVON DB Changelog — {build_id}", ""]
    if changes:
        lines.append("## ❌ Changes")
        for k, a, b in changes:
            lines.append(f"- **{k}**: `{a}` → `{b}`")
        lines.append("")
    if adds:
        lines.append("## ✅ Additions")
        for k in adds:
            lines.append(f"- {k}")
        lines.append("")
    if dels:
        lines.append("## ⚠️ Deletions")
        for k in dels:
            lines.append(f"- {k}")
        lines.append("")
    if not (adds or changes or dels):
        lines.append("_No differences; structure unchanged._")
    return "\n".join(lines) + "\n"
