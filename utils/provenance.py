# utils/provenance.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import os, json, hashlib, datetime
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]

# Display name for DB banners
DB_DISPLAY_NAME = "Novon Product and Performance DB"

ASSETS = APP_ROOT / "assets"
IES_INCOMING = ASSETS / "ies" / "incoming"
IES_REPO     = ASSETS / "ies" / "repo"
IES_RETIRED  = ASSETS / "ies" / "retired"

RPT_PHOT     = ASSETS / "reports" / "photometry"
RPT_LED      = ASSETS / "reports" / "led"
RPT_CAL      = ASSETS / "reports" / "calibration"
RPT_THERM    = ASSETS / "reports" / "thermal"

INDEX_DIR    = ASSETS / "index"
INDEX_PATH   = INDEX_DIR / "ies_index.json"

REQUIRED_DIRS = [
    ASSETS, IES_INCOMING, IES_REPO, IES_RETIRED,
    RPT_PHOT, RPT_LED, RPT_CAL, RPT_THERM,
    INDEX_DIR
]

def ensure_dirs() -> None:
    for d in REQUIRED_DIRS:
        d.mkdir(parents=True, exist_ok=True)

def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")

def _hash_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def load_index() -> Dict[str, Any]:
    ensure_dirs()
    if INDEX_PATH.exists():
        try:
            return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"version": 1, "records": []}

def save_index(idx: Dict[str, Any]) -> None:
    ensure_dirs()
    INDEX_PATH.write_text(json.dumps(idx, indent=2), encoding="utf-8")

def make_id(label: str) -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = "".join(c for c in label if c.isalnum() or c in ("-","_")).strip("_")
    return f"{safe}_{stamp}" if safe else f"item_{stamp}"

def make_record(
    *,
    rec_id: str,
    label: str,
    state: str,  # "incoming" | "repo" | "retired"
    ies_json_path: Optional[str],
    ies_txt_path: Optional[str],
    meta: Dict[str, Any],
    attachments: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    return {
        "id": rec_id,
        "label": label,
        "state": state,
        "created": _now_iso(),
        "updated": _now_iso(),
        "ies_json_path": ies_json_path,
        "ies_txt_path": ies_txt_path,
        "meta": meta,
        "attachments": {
            "photometry_pdf": attachments.get("photometry_pdf"),
            "lm80_pdf":       attachments.get("lm80_pdf"),
            "tm21_pdf":       attachments.get("tm21_pdf"),
            "goni_cal_pdf":   attachments.get("goni_cal_pdf"),
            "spectro_cal_pdf":attachments.get("spectro_cal_pdf"),
            "thermal_pdf":    attachments.get("thermal_pdf"),
            "other":          attachments.get("other"),
        }
    }

def add_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    idx = load_index()
    idx["records"].append(rec)
    save_index(idx)
    return rec

def list_records(state: Optional[str] = None) -> List[Dict[str, Any]]:
    idx = load_index()
    recs = idx.get("records", [])
    if state:
        recs = [r for r in recs if r.get("state") == state]
    recs.sort(key=lambda r: r.get("created",""), reverse=True)
    return recs

def get_record(rec_id: str) -> Optional[Dict[str, Any]]:
    idx = load_index()
    for r in idx.get("records", []):
        if r.get("id") == rec_id:
            return r
    return None

def get_record_by_label_latest(label: str) -> Optional[Dict[str, Any]]:
    recs = [r for r in list_records(None) if str(r.get("label","")).strip().lower() == str(label).strip().lower()]
    return recs[0] if recs else None

def update_record(rec_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    idx = load_index()
    changed = None
    for r in idx.get("records", []):
        if r.get("id") == rec_id:
            r.update(patch)
            r["updated"] = _now_iso()
            changed = r
            break
    if changed:
        save_index(idx)
    return changed

def retire_record(rec_id: str) -> Optional[Dict[str, Any]]:
    return update_record(rec_id, {"state": "retired"})

def move_to_repo(rec_id: str) -> Optional[Dict[str, Any]]:
    return update_record(rec_id, {"state": "repo"})

def audit_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    required = {
        "photometry_pdf": "Photometry lab report (PDF)",
        "goni_cal_pdf":   "Goniophotometer calibration (PDF)",
        "spectro_cal_pdf":"Spectroradiometer calibration (PDF)",
    }
    missing = []
    for k, label in required.items():
        path = (rec.get("attachments") or {}).get(k)
        if not path or not Path(path).exists():
            missing.append(label)
    score = 100 - int(100 * len(missing) / max(1, len(required)))
    return {"id": rec.get("id"), "label": rec.get("label"), "score": score, "missing": missing}

def audit_all(state: Optional[str] = None) -> List[Dict[str, Any]]:
    return [audit_record(r) for r in list_records(state)]

def write_bytes(p: Path, data: bytes) -> str:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return str(p)

def destination_dir(kind: str) -> Path:
    if kind == "incoming": return IES_INCOMING
    if kind == "repo":     return IES_REPO
    if kind == "retired":  return IES_RETIRED
    return IES_INCOMING

# -------- Helpers for Photometry page (export/load) --------

def export_working_json(working: Dict[str, Any], *, label: str, state: str) -> Dict[str, Any]:
    """
    Writes current working dict to assets\ies\<state>\<label>.ies.json and creates index record.
    """
    ensure_dirs()
    rec_id = make_id(label)
    dst_dir = destination_dir(state)
    dst = dst_dir / f"{label}.ies.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(working, indent=2), encoding="utf-8")

    # snapshot a few numbers (if present)
    geo = (working or {}).get("geometry", {}) or {}
    snap = {
        "G7": geo.get("G7"), "G8": geo.get("G8"), "G9": geo.get("G9"),
        "G11": geo.get("G11"), "G12": geo.get("G12"), "G13": geo.get("G13"), "G14": geo.get("G14"),
    }

    rec = make_record(
        rec_id=rec_id,
        label=label,
        state=state,
        ies_json_path=str(dst),
        ies_txt_path=None,
        meta=snap,
        attachments={},
    )
    add_record(rec)
    return rec

def load_working_from_record(rec_id: str) -> Optional[Dict[str, Any]]:
    rec = get_record(rec_id)
    if not rec: return None
    p = rec.get("ies_json_path")
    if not p or not Path(p).exists(): return None
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None
