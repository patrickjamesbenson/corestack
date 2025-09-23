# utils/metadata_photonlayout.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
import json, math, uuid

def load_photom_layout_json(path: Path) -> Optional[List[Dict[str, Any]]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    rows = data.get("PHOTOM_LAYOUT")
    if not isinstance(rows, list):
        return None
    cleaned: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict): continue
        cleaned.append({
            "FIELD": str(r.get("FIELD","")).strip(),
            "FORMAT": str(r.get("FORMAT","")).strip(),
            "IES_ORDER": _toi(r.get("IES_ORDER","")),
            "IES_FUNC": str(r.get("IES_FUNC","")).strip(),
            "IES_PROPOSED": r.get("IES_PROPOSED",""),
            "IES_TOOLTIP": str(r.get("IES_TOOLTIP","")).strip(),
        })
    cleaned = [x for x in cleaned if x["FIELD"]]
    cleaned.sort(key=lambda x: (x["IES_ORDER"] if x["IES_ORDER"] is not None else 9999))
    return cleaned

def derive_metadata_value(field: str, parsed: Dict[str, Any]) -> Optional[Any]:
    from utils.photometry_engine_adapter import luminous_flux_with_note, power_watts, curve_type_parts
    geom = parsed.get("geometry") or {}
    lm, _ = luminous_flux_with_note(parsed)
    pw_primary = power_watts(parsed)
    length_mm = _ff(geom.get("length"))
    length_m  = (length_mm/1000.0) if length_mm is not None else None
    g13_raw_lm = _ff(geom.get("G13"))  # Raw lumens
    g14_circ_w = _ff(geom.get("G14"))  # Circuit watts
    c_main, _ = curve_type_parts(parsed)

    f = field.strip().upper()
    if f.startswith("IESNA"): return "IESNA: LM-63-2019"
    if f == "TILT=NONE":      return "TILT=NONE"
    if f == "[_PHOTOMETRIC_TYPE]": return c_main
    if f == "[_LENGTH_M]":    return length_mm
    if f == "[_LUMINOUS_FLUX_REF]": return _rr(lm)
    if f == "[_LUMENS_PER_W_REF]":
        if lm is None or pw_primary in (None,0): return None
        return _rr(lm/pw_primary)
    if f == "[_LUMENS_PER_M_REF]":
        if lm is None or length_m in (None,0): return None
        return _rr(lm/length_m)
    if f == "[_WATTS_PER_M_REF]":
        if pw_primary in (None,0) or length_m in (None,0): return None
        return _rr(pw_primary/length_m)
    if f == "[_OPTICAL_EFFICIENCY_REF]":
        if lm is None or g13_raw_lm in (None,0): return None
        return _rr(lm/g13_raw_lm)
    if f == "[_ECG_EFFICIENCY_REF]":
        if g14_circ_w in (None,0) or pw_primary in (None,0): return None
        return _rr(g14_circ_w/pw_primary)
    if f == "[_LUMINAIRE_EFFICIENCY_REF]":
        opt = None if (lm is None or g13_raw_lm in (None,0)) else (lm/g13_raw_lm)
        drv = None if (g14_circ_w in (None,0) or pw_primary in (None,0)) else (g14_circ_w/pw_primary)
        if opt is None or drv is None: return None
        return _rr(opt*drv)
    return None

def generate_session_id() -> str:
    return uuid.uuid4().hex

def _toi(x: Any) -> Optional[int]:
    try: return int(str(x).strip())
    except: return None

def _ff(x: Any) -> Optional[float]:
    try:
        if x in (None,""): return None
        return float(x)
    except: return None

def _rr(x: Optional[float], nd: int = 4) -> Optional[float]:
    if x is None or (isinstance(x,float) and (math.isnan(x) or math.isinf(x))): return None
    try: return round(float(x), nd)
    except: return None
