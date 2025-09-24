# tools/overwrite_paths.py
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UTILS = ROOT / "utils"
UTILS.mkdir(parents=True, exist_ok=True)
TARGET = UTILS / "paths.py"

CONTENT = r'''from __future__ import annotations
from pathlib import Path

def app_root() -> Path:
    # utils/ is one level under app root
    return Path(__file__).resolve().parents[1]

def assets_root(root: Path | None = None) -> Path:
    r = root or app_root()
    return (r / "legacy_src" / "novon_db_update_and_build" / "assets").resolve()

def workflow_current_path(root: Path | None = None) -> Path:
    ar = assets_root(root)
    return (ar / "workflow" / "current" / "novon_workflow.json").resolve()

def ies_repo_current_path(root: Path | None = None) -> Path:
    ar = assets_root(root)
    # canonical location for current repo JSON
    return (ar / "ies" / "ies_repo.json").resolve()

def ies_originals_dir(root: Path | None = None) -> Path:
    ar = assets_root(root)
    p = (ar / "ies" / "originals").resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p

def ensure_assets_tree(root: Path | None = None) -> None:
    ar = assets_root(root)
    for sub in ["ies/originals", "workflow/current", "provenance/reports"]:
        (ar / sub).mkdir(parents=True, exist_ok=True)
'''

TARGET.write_text(CONTENT, encoding="utf-8")
print(f"[OK] Wrote {TARGET}")

