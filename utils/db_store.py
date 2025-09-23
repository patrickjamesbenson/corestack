# utils/db_store.py
from __future__ import annotations
from typing import Any, Dict
from pathlib import Path
import json
import re
import time

# One fixed “book beside the bed” location
DB_PATH = Path("assets/db/select_output_and_attribute.json")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def save_local_db(payload: Dict[str, Any]) -> Path:
    DB_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return DB_PATH

def load_local_db() -> Dict[str, Any]:
    if DB_PATH.exists():
        return json.loads(DB_PATH.read_text(encoding="utf-8"))
    return {}

def spreadsheet_id_from(url_or_id: str) -> str:
    s = (url_or_id or "").strip()
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", s)
    return m.group(1) if m else s

def stamp_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
