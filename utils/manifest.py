# utils/manifest.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

def _naive_yaml_parse(text: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):  # comments/blank
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        key = k.strip()
        val = v.strip().strip('"').strip("'")
        data[key] = val
    return data

def load_manifest(app_root: Path) -> Dict[str, Any]:
    """Load manifest.yml. Tries PyYAML; falls back to naive parser (no deps)."""
    p = app_root / "manifest.yml"
    if not p.exists():
        return {}
    txt = p.read_text(encoding="utf-8", errors="ignore")
    try:
        import yaml  # optional
        return (yaml.safe_load(txt) or {}) if txt else {}
    except Exception:
        return _naive_yaml_parse(txt)

def resolve_legacy_path(app_root: Path, rel: str | None) -> Path | None:
    """Resolve a file path inside legacy_src/ given a manifest-relative path."""
    if not rel:
        return None
    return (app_root / "legacy_src" / rel).resolve()
