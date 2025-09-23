# utils/photometry_engine_adapter.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
from math import isfinite, exp
import numpy as np


# ----------------------------- Parsing -----------------------------

def parse_ies(data: bytes) -> Dict[str, Any]:
    txt = data.decode("utf-8", errors="ignore")
    txt = txt.replace("\r\n", "\n").replace("\r", "\n").strip("\ufeff")
    raw_lines = [ln.rstrip() for ln in txt.split("\n")]
    lines = [ln for ln in raw_lines if ln.strip()]

    if not lines or not lines[0].upper().startswith("IESNA"):
        raise ValueError("Missing IESNA header")
    header = lines[0]

    # keywords until TILT=...
    i = 1
    keywords: List[Tuple[str, str]] = []
    tilt_mode = None
    while i < len(lines):
        ln = lines[i]
        if ln.upper().startswith("TILT"):
            tilt_mode = ln.split("=", 1)[-1].strip().upper()
            i += 1
            break
        if ln.startswith("[") and "]" in ln:
            k = ln[: ln.index("]") + 1]
            v = ln[ln.index("]") + 1 :].strip()
            keywords.append((k, v))
        else:
            keywords.append((ln, ""))  # keep raw line
        i += 1
    if tilt_mode is None:
        raise ValueError("TILT line not found")

    def _nums(s: str) -> List[float]:
        out: List[float] = []
        for p in s.replace(",", " ").split():
            try:
                out.append(float(p))
            except Exception:
                pass
        return out

    tokens: List[float] = []
    for k in range(i, len(lines)):
        tokens.extend(_nums(lines[k]))
    if len(tokens) < 10:
        raise ValueError("Not enough numbers for geometry (G0..G9)")

    G: Dict[str, float] = {}
    g0_9 = tokens[:10]; pos = 10
    for idx, val in enumerate(g0_9):
        G[f"G{idx}"] = float(val)

    v_count = int(G.get("G3", 0))
    h_count = int(G.get("G4", 0))
    if v_count <= 0 or h_count <= 0:
        raise ValueError(f"Invalid V/H counts (V={v_count}, H={h_count})")

    need_rest = v_count + h_count + v_count * h_count
    best_ex: Optional[int] = None
    for ex in range(0, 6):
        remain = len(tokens) - (pos + ex)
        if remain == need_rest:
            best_ex = ex; break
    if best_ex is None:
        for ex in range(0, 6):
            remain = len(tokens) - (pos + ex)
            if remain >= need_rest:
                best_ex = ex; break
    if best_ex is None:
        raise ValueError("Insufficient angle/candela values")

    ex = best_ex
    extras = tokens[pos:pos+ex]; pos += ex
    if ex >= 1: G["G10"] = float(extras[0])  # ballast factor
    if ex >= 2: G["G11"] = float(extras[1])  # file generation type (numeric in old files; we’ll mirror flags later)
    if ex >= 3: G["G12"] = float(extras[2])  # input watts
    if ex >= 4: G["G13"] = float(extras[3])  # raw board lumens
    if ex >= 5: G["G14"] = float(extras[4])  # circuit watts
    for k in range(0, 15):
        G.setdefault(f"G{k}", 0.0)

    end_v = pos + v_count
    end_h = end_v + h_count
    end_I = end_h + (v_count * h_count)
    if end_I > len(tokens):
        have = len(tokens) - pos
        raise ValueError(f"Insufficient angle/candela values (need {need_rest}, have {have})")

    V = [float(x) for x in tokens[pos:end_v]]; pos = end_v
    H = [float(x) for x in tokens[pos:end_h]]; pos = end_h
    flat_I  = [float(x) for x in tokens[pos:end_I]]; pos = end_I
    I = np.asarray(flat_I, dtype=float).reshape(h_count, v_count)

    meta = {"header": header, "keywords_order": [{"key": k, "value": v} for k, v in keywords]}
    return {
        "meta": meta,
        "tilt": {"mode": f"TILT={tilt_mode}"},
        "geometry": dict(G),
        "photometry": {"v_angles": V, "h_angles": H, "candela": I.tolist()},
    }


def ordered_metadata_rows(parsed: Dict[str, Any]) -> List[Tuple[str, Any]]:
    meta = parsed.get("meta") or {}
    rows = meta.get("keywords_order") or []
    out: List[Tuple[str, Any]] = []
    if "header" in meta:
        out.append(("IES Header", meta["header"]))
    for r in rows:
        out.append((r.get("key"), r.get("value")))
    tilt = parsed.get("tilt") or {}
    if tilt.get("mode"):
        out.append(("TILT", tilt["mode"]))
    return out


def geometry_rows_ordered(parsed: Dict[str, Any]) -> List[Tuple[str, str, Any]]:
    g = parsed.get("geometry") or {}
    items: List[Tuple[int, str, Any]] = []
    for k, v in g.items():
        if isinstance(k, str) and k.startswith("G"):
            try:
                n = int(k[1:])
                items.append((n, k, v))
            except Exception:
                pass
    items.sort(key=lambda t: t[0])
    return [(f"G{n}", key, g.get(key)) for (n, key, _v) in items]


# ----------------------------- Photometry maths -----------------------------

def luminous_flux_with_note(parsed: Dict[str, Any]) -> Tuple[Optional[float], str]:
    ph = parsed.get("photometry") or {}
    V = np.array(ph.get("v_angles") or [], dtype=float)
    H = np.array(ph.get("h_angles") or [], dtype=float)
    I = np.array(ph.get("candela") or [], dtype=float)
    if I.ndim != 2 or I.shape != (len(H), len(V)):
        return None, ""
    if len(V) < 2 or len(H) < 1:
        return None, ""

    Vrad = np.deg2rad(V)
    Hrad = np.deg2rad(H)
    line = I * np.sin(Vrad)[None, :]
    dPhi_dH = np.trapz(line, Vrad, axis=1)
    Phi = np.trapz(dPhi_dH, Hrad)

    note = ""
    if len(H) > 0 and float(np.max(H)) <= 180.0:
        Phi *= 2.0
        note = "half-azimuth → symmetry ×2"
    return float(Phi), note


def power_watts(parsed: Dict[str, Any]) -> Optional[float]:
    g = parsed.get("geometry") or {}
    for k in ("G12", "G14", "G11"):
        v = g.get(k)
        try:
            fv = float(v)
            if isfinite(fv) and fv > 0:
                return fv
        except Exception:
            continue
    return None


def lumens_per_watt(parsed: Dict[str, Any]) -> Optional[float]:
    lm, _ = luminous_flux_with_note(parsed)
    pw = power_watts(parsed)
    if lm is None or pw in (None, 0):
        return None
    return lm / pw


def _lerp_x(x0: float, x1: float, y0: float, y1: float, y: float) -> float:
    if x1 == x0:
        return x0
    t = (y - y0) / (y1 - y0)
    return x0 + t * (x1 - x0)


def beam_angle_with_note(parsed: Dict[str, Any]) -> Tuple[Optional[float], str]:
    ph = parsed.get("photometry") or {}
    V = np.array(ph.get("v_angles") or [], dtype=float)
    H = np.array(ph.get("h_angles") or [], dtype=float)
    I = np.array(ph.get("candela") or [], dtype=float)
    if I.ndim != 2 or I.shape != (len(H), len(V)):
        return None, ""
    if len(V) < 2:
        return None, ""
    peaks = I.max(axis=1)
    j = int(np.argmax(peaks))
    sweep = I[j, :]
    peak = float(np.max(sweep))
    if not isfinite(peak) or peak <= 0:
        return None, "no peak"

    def first_cross(frac: float) -> Optional[float]:
        tgt = frac * peak
        vmax = 90.0 if V[-1] >= 90.0 else V[-1]
        m = (V >= 0.0) & (V <= vmax)
        vv, ss = V[m], sweep[m]
        if vv.size < 2:
            return None
        dif = ss - tgt
        idx = np.where(np.diff(np.signbit(dif)))[0]
        if idx.size:
            k = int(idx[0])
            return _lerp_x(vv[k], vv[k+1], ss[k], ss[k+1], tgt)
        # fine grid fallback
        vv_f = np.linspace(vv[0], vv[-1], max(3, int((vv[-1]-vv[0])*20)+1))
        ss_f = np.interp(vv_f, vv, ss)
        dif_f = ss_f - tgt
        idx_f = np.where(np.diff(np.signbit(dif_f)))[0]
        if idx_f.size:
            k = int(idx_f[0])
            return _lerp_x(vv_f[k], vv_f[k+1], ss_f[k], ss_f[k+1], tgt)
        return None

    for frac, label in ((0.5, "2×θ@50%"), (0.1, "2×θ@10%"), (exp(-2), "2×θ@1/e²")):
        th = first_cross(frac)
        if th is not None:
            return float(2.0 * th), f"{label}"

    return 0.0, "flat"


# ----------------------------- Flags & geometry -----------------------------

def pack_flags(flags: Tuple[bool, bool, bool, bool, bool]) -> str:
    return ".".join("1" if b else "0" for b in flags)


def apply_geometry(working: Dict[str, Any], g_new: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(working)
    geom = dict(out.get("geometry", {}))
    for k in ("G7","G8","G9","G10","G11","G12","G13","G14","file_generation_flags"):
        if k in g_new:
            geom[k] = g_new[k]
    # mirror flags into G11 as a string like "1.1.1.0.0"
    if "file_generation_flags" in g_new:
        geom["G11"] = g_new["file_generation_flags"]
    out["geometry"] = geom
    return out


# ----------------------------- Interpolation -----------------------------

def source_is_0_to_90(working: Dict[str, Any]) -> bool:
    ph = working.get("photometry") or {}
    V = ph.get("v_angles") or []
    try:
        vmax = float(V[-1]) if V else 0.0
        return vmax <= 90.0 + 1e-9
    except Exception:
        return False


def resample_photometry(
    V: List[float], H: List[float], I: np.ndarray,
    target_v: int, target_h: int,
    force_zero_beyond_90: bool,
    flip_vertical: bool
) -> Tuple[List[float], List[float], np.ndarray]:
    """
    Resample to target counts (linear interp). After resample and optional flip,
    enforce zeros for all V>90° when force_zero_beyond_90 is True.
    """
    V = np.array(V, dtype=float); H = np.array(H, dtype=float)
    I = np.array(I, dtype=float)
    if I.ndim != 2 or I.shape != (len(H), len(V)):
        raise ValueError("I shape mismatch")

    # target grids
    Vmin, Vmax = float(V.min()), float(V.max())
    Hmin, Hmax = float(H.min()), float(H.max())
    Vt = np.linspace(Vmin, Vmax, int(target_v))
    Ht = np.linspace(Hmin, Hmax, int(target_h))

    # interp along V then along H
    Iv = np.vstack([np.interp(Vt, V, I[h, :]) for h in range(len(H))])
    Ivh = np.vstack([np.interp(Ht, H, Iv[:, k]) for k in range(len(Vt))]).T  # shape (Ht, Vt)

    # flip vertically (hemispheres)
    if flip_vertical:
        Ivh = Ivh[:, ::-1]
        Vt = Vt[::-1]

    # enforce zeros beyond 90° AFTER all interpolation/flip is complete
    if force_zero_beyond_90:
        mask = Vt > 90.0 + 1e-9
        if np.any(mask):
            Ivh[:, mask] = 0.0
            k_first = int(np.argmax(mask))
            if 0 <= k_first < Ivh.shape[1]:
                Ivh[:, k_first] = 0.0

    return list(Vt), list(Ht), Ivh


# ----------------------------- Metadata build -----------------------------

def build_metadata_from_photom_layout(
    working: Dict[str, Any],
    db_json: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Returns a list of rows: {FIELD, IES_FUNC, ORIGINAL, EDIT_HERE, INTERPOLATED}
    EDIT_HERE is editable only if IES_FUNC == 'Editable'. Derived are filled from calculations.
    """
    rows: List[Dict[str, Any]] = []
    # baseline: raw ordered keywords
    original_map = {}
    for k, v in ordered_metadata_rows(working):
        original_map[str(k).strip()] = v

    geom = working.get("geometry", {}) or {}
    ph = working.get("photometry", {}) or {}
    V = ph.get("v_angles") or []
    H = ph.get("h_angles") or []
    I = np.array(ph.get("candela") or [], dtype=float)

    # quick calcs
    lm, _note = luminous_flux_with_note(working)
    lm = float(lm or 0.0)
    length_m = float(geom.get("G8", 0.0)) / 1000.0
    circuit_w = float(geom.get("G14", 0.0))
    input_w = float(geom.get("G12", 0.0))
    raw_lm = float(geom.get("G13", 0.0))
    lm_per_w = (lm / input_w) if input_w > 0 else 0.0
    lm_per_m = (lm / length_m) if length_m > 0 else 0.0
    opt_eff = (lm / raw_lm) * 100.0 if raw_lm > 0 else 0.0

    # db_json → PHOTOM_LAYOUT
    table = (db_json or {}).get("PHOTOM_LAYOUT") or []

    def _is_derived(func: str) -> bool:
        return str(func or "").strip().lower().startswith("derived")

    # order by IES_ORDER if present
    def _order_key(rec: Dict[str, Any]) -> int:
        raw = str(rec.get("IES_ORDER", "0")).strip()
        head = raw.split()[0]
        return int(head) if head.isdigit() else 9999

    for rec in sorted(table, key=_order_key):
        field = str(rec.get("FIELD", "")).strip()
        func = str(rec.get("IES_FUNC", "")).strip()
        proposed = str(rec.get("IES_PROPOSED", "")).strip()

        orig_val = original_map.get(field, "")
        interp_val = proposed

        # hardwired derived examples
        if field == "IESNA: LM-63-2019":
            interp_val = "IESNA: LM-63-2019"
        elif field == "[_LUMINOUS_FLUX_REF]":
            interp_val = f"{lm:.4f}"
        elif field == "[_LUMENS_PER_W_REF]":
            interp_val = f"{lm_per_w:.4f}"
        elif field == "[_LUMENS_PER_M_REF]":
            interp_val = f"{lm_per_m:.4f}"
        elif field == "[_OPTICAL_EFFICIENCY_REF]":
            interp_val = f"{opt_eff:.4f}"
        elif field == "[_INPUTWATTS_REF]":
            interp_val = f"{input_w:.4f}"
        elif field == "[_CIRCUIT_WATTS_REF]":
            interp_val = f"{circuit_w:.4f}"
        elif field == "[_LENGTH_M]":
            interp_val = f"{length_m:.4f}"

        rows.append({
            "FIELD": field,
            "IES_FUNC": func,
            "ORIGINAL": orig_val,
            "EDIT_HERE": "" if _is_derived(func) else (orig_val if orig_val else ""),
            "INTERPOLATED": interp_val,
        })
    return rows


# ----------------------------- Export helpers -----------------------------

def export_ies_lm63(working: Dict[str, Any], meta_rows: List[Dict[str, Any]]) -> str:
    """Build a minimal LM-63 text based on current working and meta rows."""
    geom = working.get("geometry", {}) or {}
    ph = working.get("photometry", {}) or {}
    V = ph.get("v_angles") or []
    H = ph.get("h_angles") or []
    I = np.array(ph.get("candela") or [], dtype=float)

    # header & keywords
    lines = ["IESNA:LM-63-2019"]
    # write keywords from edited rows (ORIGINAL + EDIT_HERE/INTERPOLATED)
    for r in meta_rows:
        k = r.get("FIELD", "")
        if not k or k.upper().startswith("IESNA"):
            continue
        v = r.get("EDIT_HERE") or r.get("INTERPOLATED") or r.get("ORIGINAL") or ""
        if not str(k).startswith("["):
            lines.append(f"{k}{v if v else ''}")
        else:
            lines.append(f"{k}{v}")

    lines.append("TILT=NONE")

    # geometry block (G0..G12) — keep numeric
    gnums = [geom.get(f"G{k}", 0.0) for k in range(0, 13)]
    # G11 may be a flags string; LM-63 numeric block expects a number → write 0.0 for that position
    try:
        _ = float(geom.get("G11"))
    except Exception:
        gnums[11] = 0.0
    lines.append(" ".join(str(float(x)) for x in gnums))

    # angles & intensities
    lines.append(" ".join(str(float(x)) for x in V))
    lines.append(" ".join(str(float(x)) for x in H))
    flat = I.reshape(-1)
    lines.append(" ".join(str(float(x)) for x in flat))

    return "\n".join(lines) + "\n"


# ----------------------------- Scaling -----------------------------

def scale_working_length_and_flux(working: Dict[str, Any], target_length_mm: float) -> Dict[str, Any]:
    """
    Linear scaling for linear luminaires:
      k = target_length / current_length
      - G8 (length) *= k
      - G12 (input watts), G14 (circuit watts), G13 (raw lumens) *= k
      - candela matrix *= k
    """
    out = dict(working)
    geom = dict(out.get("geometry", {}) or {})
    ph = dict(out.get("photometry", {}) or {})
    V = list(ph.get("v_angles") or [])
    H = list(ph.get("h_angles") or [])
    I = np.array(ph.get("candela") or [], dtype=float)

    cur_len = float(geom.get("G8", 0.0))
    if cur_len <= 0 or target_length_mm <= 0:
        return out

    k = float(target_length_mm) / float(cur_len)

    # scale geometry figures
    for gk in ("G8",):
        geom[gk] = float(geom.get(gk, 0.0)) * k
    for gk in ("G12", "G14", "G13"):
        geom[gk] = float(geom.get(gk, 0.0)) * k

    # scale intensity matrix
    if I.ndim == 2 and I.shape == (len(H), len(V)):
        I = I * k
        ph["candela"] = I.tolist()

    out["geometry"] = geom
    out["photometry"] = ph
    return out
