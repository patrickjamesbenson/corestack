# utils/gsheets_pull.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import json

def pull_all_sheets(sheet_id: str, service_key_path: Path) -> Dict[str, Any]:
    """
    Minimal puller: tries google-api-python-client, falls back to a clear error.
    Returns a dict-of-sheets {sheet_title: rows_as_dicts}.
    """
    try:
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build
    except Exception:
        raise RuntimeError("google-api-python-client not installed. pip install google-api-python-client")

    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_file(str(service_key_path), scopes=scopes)
    svc = build("sheets", "v4", credentials=creds)

    meta = svc.spreadsheets().get(spreadsheetId=sheet_id).execute()
    sheets = meta.get("sheets", [])
    out: Dict[str, Any] = {}

    for s in sheets:
        title = s["properties"]["title"]
        # Read all values; first row as header
        resp = svc.spreadsheets().values().get(spreadsheetId=sheet_id, range=title).execute()
        values = resp.get("values", [])
        if not values:
            out[title] = []
            continue
        header = values[0]
        rows = []
        for row in values[1:]:
            rec = {header[i]: (row[i] if i < len(row) else "") for i in range(len(header))}
            rows.append(rec)
        out[title] = rows

    return out
