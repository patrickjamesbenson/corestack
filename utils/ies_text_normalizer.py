# utils/ies_text_normalizer.py
from __future__ import annotations
from typing import Union

def normalize_ies_text(data: Union[str, bytes]) -> str:
    """Normalize raw IES text without changing meaning."""
    if isinstance(data, bytes):
        try:
            s = data.decode("utf-8")
        except UnicodeDecodeError:
            s = data.decode("latin-1", errors="replace")
    else:
        s = data
    if s.startswith("\ufeff"):
        s = s.lstrip("\ufeff")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u00A0", " ").replace("\t", " ")
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    return s
