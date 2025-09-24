# utils/gsheets_pull.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional

def pull_all_sheets(sheet_id: str, service_key_path: Optional[Path]) -> Dict[str, Any]:
    """
    Download all sheets into a single JSON object.
    If service_key_path is None, tries:
      - GOOGLE_APPLICATION_CREDENTIALS env
      - Streamlit secrets (if present)
    """
    # Lazy imports so the rest of the app doesn't require google libs
    try:
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build
    except Exception as ex:
        raise RuntimeError("google-api-python-client not installed. pip install google-api-python-client") from ex

    creds = None

    # 1) If explicit key path provided, use it
    if service_key_path is not None:
        scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
        creds = Credentials.from_service_account_file(str(service_key_path), scopes=scopes)

    # 2) Else: try env var GOOGLE_APPLICATION_CREDENTIALS
    if creds is None:
        import os
        gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        if gac:
            scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
            creds = Credentials.from_service_account_file(gac, scopes=scopes)

    # 3) Else: try Streamlit secrets (optional)
    if creds is None:
        try:
            import streamlit as st  # type: ignore
            key_json = st.secrets.get("google_service_key_json", "")
            if key_json:
                import json
                from google.oauth2.service_account import Credentials as C
                info = json.loads(key_json)
                scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
                creds = C.from_service_account_info(info, scopes=scopes)
        except Exception:
            pass

    if creds is None:
        raise RuntimeError("No credentials. Provide a key file, set GOOGLE_APPLICATION_CREDENTIALS, or secrets.")

    svc = build("sheets", "v4", credentials=creds)
    meta = svc.spreadsheets().get(spreadsheetId=sheet_id).execute()
    sheets = meta.get("sheets", [])
    out: Dict[str, Any] = {}

    for s in sheets:
        title = s["properties"]["title"]
        resp = svc.spreadsheets().values().get(spreadsheetId=sheet_id, range=title).execute()
        values = resp.get("values", []) or []
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

