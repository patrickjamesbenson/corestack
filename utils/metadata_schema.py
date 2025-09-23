# utils/metadata_schema.py
from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, Any

def load_schema_from_json(path: Path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Schema JSON must be an object at top level.")
    return data
