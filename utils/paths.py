# utils/paths.py
from __future__ import annotations
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent

ASSETS_DIR    = APP_ROOT / "assets"
DB_DIR        = ASSETS_DIR / "db"
IES_REPO_DIR  = ASSETS_DIR / "ies_repo"
BASELINES_DIR = ASSETS_DIR / "baselines"
DOCS_DIR      = ASSETS_DIR / "docs"   # NEW
LOG_DIR       = ASSETS_DIR / "logs"
KEYS_DIR      = ASSETS_DIR / "keys"

DB_FILENAME         = "novon_db.json"
GOOGLE_KEY_FILENAME = "novon_db_key.json"

def ensure_assets() -> None:
    for p in (ASSETS_DIR, DB_DIR, IES_REPO_DIR, BASELINES_DIR, DOCS_DIR, LOG_DIR, KEYS_DIR):
        p.mkdir(parents=True, exist_ok=True)

def workflow_current_path() -> Path:
    return DB_DIR / DB_FILENAME

def ies_repo_current_path() -> Path:
    return IES_REPO_DIR

def baselines_current_path() -> Path:
    return BASELINES_DIR

def docs_current_path() -> Path:
    return DOCS_DIR

def db_key_default_path() -> Path:
    return KEYS_DIR / GOOGLE_KEY_FILENAME
