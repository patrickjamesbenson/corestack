# utils/paths.py
from __future__ import annotations
from pathlib import Path

# App root
APP_ROOT = Path(__file__).resolve().parent.parent

# Assets layout (single source of truth)
ASSETS_DIR     = APP_ROOT / "assets"
DB_DIR         = ASSETS_DIR / "db"
IES_REPO_DIR   = ASSETS_DIR / "ies_repo"
BASELINES_DIR  = ASSETS_DIR / "baselines"
DOCS_DIR       = ASSETS_DIR / "docs"
LOG_DIR        = ASSETS_DIR / "logs"
KEYS_DIR       = ASSETS_DIR / "keys"

# Canonical file names (keep stable)
DB_FILENAME          = "novon_db.json"        # ref-data JSON produced from Google Sheets
GOOGLE_KEY_FILENAME  = "novon_db_key.json"    # service account key (local, not committed)

def ensure_assets() -> None:
    """Create the full assets tree (idempotent)."""
    for p in (ASSETS_DIR, DB_DIR, IES_REPO_DIR, BASELINES_DIR, DOCS_DIR, LOG_DIR, KEYS_DIR):
        p.mkdir(parents=True, exist_ok=True)

# Convenience getters (kept paramless so callers don’t need to pass roots)
def workflow_current_path() -> Path:
    """Where the ref-data JSON (novon_db.json) lives."""
    return DB_DIR / DB_FILENAME

def ies_repo_current_path() -> Path:
    """Where 1-mm JSON repo artifacts live."""
    return IES_REPO_DIR

def baselines_current_path() -> Path:
    """Where baselines/datums live."""
    return BASELINES_DIR

def docs_current_path() -> Path:
    """Where reference HTML lives."""
    return DOCS_DIR

def db_key_default_path() -> Path:
    """Default place you keep the Google service key (you can override in UI)."""
    return KEYS_DIR / GOOGLE_KEY_FILENAME
