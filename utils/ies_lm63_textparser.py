# utils/ies_lm63_textparser.py
from __future__ import annotations
from typing import Any, Dict, List
import re
from utils.ies_text_normalizer import normalize_ies_text

class IESParseError(ValueError):
    pass

def _is_header(line: str) -> bool:
    s = line.strip()
    return s.startswith("IES:LM-63-") or s.startswith("IESNA:")

def _tokenize_numbers(lines: List[str]) -> List[str]:
    tokens: List[str] = []
    for ln in lines:
        parts = re.split(r"[,\s]+", ln.strip())
        tokens.extend([p for p in parts if p])
    return tokens

def _to_float(tok: str) -> float:
    return float(tok.replace(",", ""))

def parse_ies_text(raw_text: str) -> Dict[str, Any]:
    """Robust LM-63 parser: header → keywords (ordered) → TILT → numeric blocks → angles → candela."""
    text = normalize_ies_text(raw_text)
    lines = text.split("\n")
    if not lines or not _is_header(lines[0]):
        raise IESParseError("Invalid or missing IES header line.")
    header = lines[0].strip()

    # Keywords until TILT=
    i = 1
    keywords: Dict[str, str] = {}
    keywords_seq: List[Dict[str, str]] = []  # preserve file order
    while i < len(lines):
        ln = lines[i].strip()
        if not ln:
            i += 1; continue
        if ln.upper().startswith("TILT="):
            break
        if ln.startswith("[") and "]" in ln:
            end = ln.find("]")
            key_full = ln[: end + 1].strip()
            key_core = key_full.strip("[]")
            val = ln[end + 1 :].strip()
            keywords[key_full.upper()] = val
            keywords_seq.append({"key": key_core, "value": val})
        else:
            # Non-bracketed line before TILT — store as OTHER in order too
            val = ln
            keywords.setdefault("[OTHER]", "")
            keywords["[OTHER]"] = (keywords["[OTHER]"] + (" " if keywords["[OTHER]"] else "") + val).strip()
            keywords_seq.append({"key": "OTHER", "value": val})
        i += 1
    if i >= len(lines):
        raise IESParseError("Missing TILT= line.")

    tilt_line = lines[i].strip()
    if not tilt_line.upper().startswith("TILT="):
        raise IESParseError("Malformed TILT= line.")
    tilt_mode = tilt_line.split("=", 1)[1].strip().upper()
    i += 1

    # Handle TILT=INCLUDE (optional block)
    tilt: Dict[str, Any]
    if tilt_mode == "INCLUDE":
        if i + 3 >= len(lines):
            raise IESParseError("Incomplete TILT=INCLUDE block.")
        try:
            lamp_geom = int(lines[i].strip()); i += 1
            n_tilt = int(lines[i].strip()); i += 1
        except Exception as e:
            raise IESParseError(f"Invalid TILT header values: {e}") from e
        ang_tokens = _tokenize_numbers([lines[i].strip()]); i += 1
        mul_tokens = _tokenize_numbers([lines[i].strip()]); i += 1
        k = i
        while len(ang_tokens) < n_tilt and k < len(lines):
            extra = lines[k].strip()
            if extra.upper().startswith("TILT="): break
            if extra: ang_tokens += _tokenize_numbers([extra])
            k += 1
        i = k
        k = i
        while len(mul_tokens) < n_tilt and k < len(lines):
            extra = lines[k].strip()
            if extra.upper().startswith("TILT="): break
            if extra: mul_tokens += _tokenize_numbers([extra])
            k += 1
        i = k
        if len(ang_tokens) != n_tilt or len(mul_tokens) != n_tilt:
            raise IESParseError("TILT angles/multipliers count mismatch.")
        tilt = {
            "mode": "INCLUDE",
            "lamp_to_lum_geom": lamp_geom,
            "angles": [ _to_float(x) for x in ang_tokens ],
            "multipliers": [ _to_float(x) for x in mul_tokens ],
        }
    else:
        tilt = {"mode": tilt_mode}

    # Remaining numeric data
    rest = [ln for ln in lines[i:] if ln.strip()]
    if not rest:
        raise IESParseError("Missing photometric numeric blocks.")
    it = iter(_tokenize_numbers(rest))

    def take_nums(n: int) -> List[float]:
        out: List[float] = []
        for _ in range(n):
            try:
                out.append(_to_float(next(it)))
            except StopIteration as e:
                raise IESParseError("Unexpected end of numeric data.") from e
            except Exception:
                raise IESParseError("Invalid numeric token.")
        return out

    # Block 1 (10)
    b1 = take_nums(10)
    (num_lamps, lumens_per_lamp, multiplier, n_v, n_h,
     photometric_type, units_type, width, length, height) = b1
    # Block 2 (3)
    b2 = take_nums(3)
    ballast_factor, file_gen_type, input_watts = b2

    v_angles = take_nums(int(n_v))
    h_angles = take_nums(int(n_h))

    candela: List[List[float]] = []
    for _h in range(int(n_h)):
        candela.append(take_nums(int(n_v)))

    meta: Dict[str, Any] = {
        "header": header,
        "keywords_order": keywords_seq,  # preserve on-disk order
        "file_generation_type": file_gen_type,
        "ballast_factor": ballast_factor,
        "input_watts": input_watts,
    }
    # also keep quick access to some common fields
    for req in ("[TEST]","[TESTLAB]","[ISSUEDATE]","[MANUFAC]"):
        if req in keywords:
            meta[req.strip("[]").lower()] = keywords[req]

    geometry: Dict[str, Any] = {
        "num_lamps": int(num_lamps),
        "lumens_per_lamp": lumens_per_lamp,
        "candela_multiplier": multiplier,
        "v_count": int(n_v),
        "h_count": int(n_h),
        "photometric_type": int(photometric_type),
        "units_type": int(units_type),
        "width": width, "length": length, "height": height,
        "ballast_factor": ballast_factor,
        "input_watts": input_watts,
        "file_generation_type": file_gen_type,
    }

    photometry: Dict[str, Any] = {
        "vertical_angles": v_angles,
        "horizontal_angles": h_angles,
        "candela": candela,  # shape H x V
    }

    return {"meta": meta, "tilt": tilt, "geometry": geometry, "photometry": photometry}

