# utils/google_sheets_pull.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import json

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build  # type: ignore
from utils.credentials import apply_google_creds

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

def _client():
    key_path = Path(apply_google_creds())  # env/secrets win
    creds = Credentials.from_service_account_file(str(key_path), scopes=SCOPES)
    return build("sheets", "v4", credentials=creds).spreadsheets()

def pull_all_sheets(sheet_id: str, _unused=None) -> Dict[str, Any]:
    svc = _client()
    meta = svc.get(spreadsheetId=sheet_id, includeGridData=False).execute()
    titles = [s["properties"]["title"] for s in meta.get("sheets", [])]
    out: Dict[str, Any] = {}
    for t in titles:
        resp = svc.values().get(spreadsheetId=sheet_id, range=t).execute()
        out[t] = resp.get("values", [])
    return out
