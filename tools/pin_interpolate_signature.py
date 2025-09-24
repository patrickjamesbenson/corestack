# tools/pin_interpolate_signature.py
from __future__ import annotations
import importlib, inspect, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy_src" / "ies_norm"
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))

# find engine module
engine = None
for name in ("ies_prep","engine","app","ies_norm","ies_prep_main"):
    try:
        engine = importlib.import_module(name)
        break
    except Exception:
        pass
if engine is None:
    raise SystemExit("Cannot find legacy engine under legacy_src\\ies_norm")

fn = getattr(engine, "interpolate_candela_matrix", None)
if not callable(fn):
    raise SystemExit("Engine has no interpolate_candela_matrix()")

sig = inspect.signature(fn)
params = list(sig.parameters.keys())
# param roles we need to map from UI to engine
roles = {"v":"", "h":"", "clamp":"", "norm":""}

# pick params by simple name cues (no runtime aliasing later; we *pin* the names now)
for p in params:
    lp = p.lower()
    if not roles["v"] and any(k in lp for k in ("v_steps","v_count","v_points","v_target","vlen","vn","v_n")):
        roles["v"] = p
        continue
    if not roles["h"] and any(k in lp for k in ("h_steps","h_count","h_points","h_target","hlen","hn","h_n")):
        roles["h"] = p
        continue
    if not roles["clamp"] and any(k in lp for k in ("one_hemisphere","one_hemi","clamp_dark","fill_above_90","force_zero_above_90","dark_above_90")):
        roles["clamp"] = p
        continue
    if not roles["norm"] and any(k in lp for k in ("normalize_flux","normalize_total_flux","norm_flux")):
        roles["norm"] = p
        continue

# create call building code with *your* names only
lines = []
lines.append("    eng_kwargs = {}")
if roles["v"]:
    lines.append(f"    eng_kwargs['{roles['v']}'] = v_steps")
if roles["h"]:
    lines.append(f"    eng_kwargs['{roles['h']}'] = h_steps")
if roles["clamp"]:
    lines.append(f"    eng_kwargs['{roles['clamp']}'] = clamp_dark_hemi")
if roles["norm"]:
    lines.append(f"    eng_kwargs['{roles['norm']}'] = normalize_flux")

call_block = "\n".join(lines) if lines else "    eng_kwargs = {}"

TARGET = ROOT / "utils" / "photometry_engine_adapter.py"
CONTENT = f'''# utils/photometry_engine_adapter.py (pinned to your engine signature)
from __future__ import annotations
import sys, importlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

APP_ROOT = Path(__file__).resolve().parents[1]
LEGACY_DIR = (APP_ROOT / "legacy_src" / "ies_norm").resolve()
if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))

_engine_mod = importlib.import_module("{engine.__name__}")

def _getfn(name: str):
    fn = getattr(_engine_mod, name, None)
    if not callable(fn):
        raise AttributeError(f"Missing engine function: {{name}}")
    return fn

_fn_parse          = getattr(_engine_mod, "parse_ies_input", None) or _getfn("parse_ies")
_fn_interpolate    = _getfn("interpolate_candela_matrix")
_fn_flip_hemi      = getattr(_engine_mod, "exchange_hemispheres", None) or _getfn("exchange_hemispheres")
_fn_apply_geom     = getattr(_engine_mod, "apply_geometry_overrides", None) or _getfn("apply_geometry_overrides")
_fn_build_ies_text = _getfn("build_ies_text")
_fn_flux           = getattr(_engine_mod, "calculate_luminous_flux", None)

def _photometry(obj: Any) -> Dict[str, Any] | None:
    if isinstance(obj, dict):
        if "photometry" in obj and isinstance(obj["photometry"], dict):
            return obj["photometry"]
        return obj
    return None

def _grids_and_cd(obj: Any):
    ph = _photometry(obj) or {{}}
    v = ph.get("v_grid_deg") or ph.get("v_angles") or []
    h = ph.get("h_grid_deg") or ph.get("h_angles") or []
    cd = ph.get("candela")   or ph.get("cd")       or []
    return v, h, cd

def parse_ies(data: bytes) -> Dict[str, Any]:
    return _fn_parse(data)

def summarize_counts(parsed: Dict[str, Any]):
    v, h, _ = _grids_and_cd(parsed)
    flux = 0.0
    if _fn_flux:
        try:
            flux = float(_fn_flux(parsed))
        except Exception:
            flux = float((_photometry(parsed) or {{}}).get("flux_lm_header") or 0.0)
    else:
        flux = float((_photometry(parsed) or {{}}).get("flux_lm_header") or 0.0)
    return len(v), len(h), flux

def interpolate(parsed: Dict[str, Any], *, v_steps: int, h_steps: int, clamp_dark_hemi: bool, normalize_flux: bool) -> Dict[str, Any]:
{call_block}
    return _fn_interpolate(parsed, **eng_kwargs)

class _ENG:
    exchange_hemispheres = staticmethod(_fn_flip_hemi)

def apply_geometry(ies: Dict[str, Any], geo: Dict[str, float]) -> Dict[str, Any]:
    return _fn_apply_geom(ies, geo)

def apply_metadata(ies: Dict[str, Any], workflow_json_path: Path, *, accredited: bool) -> Dict[str, Any]:
    if isinstance(ies, dict):
        meta = dict(ies.get("meta", {{}}))
        meta["workflow_path"] = str(workflow_json_path)
        meta["accredited_lab"] = bool(accredited)
        ies["meta"] = meta
    return ies

def build_ies_text_from(ies: Dict[str, Any], *, lm63_2019: bool) -> str:
    return _fn_build_ies_text(ies)
'''
TARGET.write_text(CONTENT, encoding="utf-8")
print("[OK] Wrote pinned adapter using your interpolate_candela_matrix signature")
print(f"[INFO] Engine: {engine.__name__}")
print(f"[INFO] interpolate_candela_matrix params: ({', '.join(params)})")
print(f"[INFO] Mapped roles: v='{roles['v']}', h='{roles['h']}', clamp='{roles['clamp']}', norm='{roles['norm']}'")

