from __future__ import annotations
import json, hashlib, csv
from pathlib import Path
from typing import Any, Dict, Tuple
from utils.paths import ensure_assets_tree
from utils.changelog import diff_dicts, render_changelog_md

def _hash_obj(obj: Dict[str, Any]) -> str:
    b = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:8]

def _timestamp() -> str:
    import datetime as _dt
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")

def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def _maybe_read_json(p: Path) -> Dict[str, Any]:
    return _read_json(p) if p.exists() else {}

def _read_csv_to_json(p: Path) -> Dict[str, Any]:
    rows = []
    with p.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append({k: (v if v != "" else None) for k, v in r.items()})
    return {"rows": rows, "meta": {"source": str(p)}}

def _load_source(source: str) -> Dict[str, Any]:
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"Source not found: {p}")
    if p.suffix.lower() == ".json":
        return _read_json(p)
    if p.suffix.lower() == ".csv":
        return _read_csv_to_json(p)
    raise ValueError(f"Unsupported source type: {p.suffix}")

def build_database(app_root: Path, source: str) -> Tuple[Path, Path, Path, str]:
    paths = ensure_assets_tree(app_root)
    current = paths["wf_current"]
    history_dir = paths["wf_history"]

    new_obj = _load_source(source)

    ts = _timestamp()
    content_hash = _hash_obj(new_obj)
    build_id = f"{ts}__{content_hash}"
    meta = new_obj.get("meta", {})
    meta.update({"build_id": build_id, "built_at": ts})
    new_obj["meta"] = meta

    old_obj = _maybe_read_json(current)
    adds, changes, dels = diff_dicts(old_obj, new_obj)
    changelog = render_changelog_md(build_id, adds, changes, dels)

    hist_json = history_dir / f"{build_id}.json"
    hist_md   = history_dir / f"{build_id}__changelog.md"

    hist_json.write_text(json.dumps(new_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    hist_md.write_text(changelog, encoding="utf-8")
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text(json.dumps(new_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    # latest build id (easy to read elsewhere)
    (current.parent / "latest.txt").write_text(build_id, encoding="utf-8")

    return current, hist_json, hist_md, build_id
