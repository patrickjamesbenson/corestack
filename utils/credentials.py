# utils/credentials.py
from __future__ import annotations
from pathlib import Path
import os, json, tempfile

def _from_streamlit_secrets() -> str | None:
    try:
        import streamlit as st
    except Exception:
        return None
    blob = st.secrets.get("google_service_account")
    if not blob:
        return None
    if isinstance(blob, dict):
        return json.dumps(blob)
    try:
        json.loads(str(blob))
        return str(blob)
    except Exception:
        return None

def apply_google_creds() -> str:
    """
    Order:
      1) Streamlit secrets 'google_service_account' (cloud) → writes temp file, sets env
      2) $GOOGLE_APPLICATION_CREDENTIALS (local path)
    Admin/UI field is intentionally ignored.
    Returns path set in GOOGLE_APPLICATION_CREDENTIALS.
    """
    js = _from_streamlit_secrets()
    if js:
        tmp = Path(tempfile.gettempdir()) / "novon_sa.json"
        tmp.write_text(js, encoding="utf-8")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(tmp)
        return str(tmp)

    envp = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if envp and Path(envp).is_file():
        return envp

    raise FileNotFoundError(
        "Google key not found. In cloud: add secret 'google_service_account' (paste JSON). "
        "Local: set $GOOGLE_APPLICATION_CREDENTIALS to your key file."
    )
