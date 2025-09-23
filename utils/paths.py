# utils/paths.py
from __future__ import annotations
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent

ASSETS_DIR = APP_ROOT / "assets"
DB_DIR     = ASSETS_DIR / "db"
IES_DIR = ASSETS_DIR / "performance" / "json_workflow_repo"
LOG_DIR    = ASSETS_DIR / "logs"
KEYS_DIR   = ASSETS_DIR / "db"

for p in (ASSETS_DIR, DB_DIR, IES_DIR, LOG_DIR, KEYS_DIR):
    p.mkdir(parents=True, exist_ok=True)

DB_FILENAME         = "novon_db.json"
GOOGLE_KEY_FILENAME = "novon_db_key.json"

def workflow_current_path(_app_root: Path | None = None) -> Path:
    return DB_DIR / DB_FILENAME

def ies_repo_current_path(_app_root: Path | None = None) -> Path:
    return IES_DIR

def db_key_default_path(_app_root: Path | None = None) -> Path:
    return KEYS_DIR / GOOGLE_KEY_FILENAME

def ensure_assets() -> None:
    for p in (ASSETS_DIR, DB_DIR, IES_DIR, LOG_DIR, KEYS_DIR):
        p.mkdir(parents=True, exist_ok=True)

DB_JSON = workflow_current_path()
