import streamlit as st
def _safe_set_page_config(*a, **k):
    try:
        return st.set_page_config(*a, **k)
    except Exception:
        return None

import json, re, math
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
import streamlit as st

_safe_set_page_config(page_title="Configurator + Lumen Solver (JSON + Scenes)", layout="wide")
st.title("Configurator + Lumen Solver")
st.caption("JSON-native • Flux+CCT sliders • DT8 (Tc / xy / RGBWAF) • DALI (linear+IEC) • Scene collector • Endpoint overrides")

# ---------- Endpoint overrides (optional) ----------
OVERRIDES_FILE = Path("solver_endpoints_overrides.json")
OVERRIDES = {"one_sheet_tab": "one_sheet_schema_template", "column_renames": {}}
if OVERRIDES_FILE.exists():
    try:
        with OVERRIDES_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                for k in ("one_sheet_tab","column_renames"):
                    if k in data:
                        OVERRIDES[k] = data[k]
    except Exception:
        pass

# ---------- Utilities ----------
def load_json_payload(auto_path: Optional[Path], uploaded_file):
    if uploaded_file is not None:
        try:
            return json.load(uploaded_file)
        except Exception as e:
            st.error(f"Failed to parse uploaded JSON: {e}")
            st.stop()
    if auto_path and auto_path.exists():
        with auto_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    st.info("Place **novon_workflow.json** next to this app or drag & drop it below.")
    st.stop()

def df_from_one_sheet(payload: Dict) -> pd.DataFrame:
    tab = OVERRIDES.get("one_sheet_tab","one_sheet_schema_template")
    try:
        rows = payload["tables"][tab]
    except KeyError:
        st.error(f"JSON missing `tables -> {tab}`.")
        st.stop()
    if not isinstance(rows, list) or not rows:
        st.error(f"`{tab}` is empty or not a list.")
        st.stop()
    df = pd.DataFrame(rows)
    # Optional column renames so the rest of the app can stay stable
    colmap = OVERRIDES.get("column_renames") or {}
    if isinstance(colmap, dict) and colmap:
        # Try old->new; if keys not present, try inverse mapping
        if not set(colmap.keys()).issubset(df.columns):
            inv = {v:k for k,v in colmap.items()}
            if set(inv.keys()).issubset(df.columns):
                colmap = inv
        df.rename(columns=colmap, inplace=True)
    df.columns = [str(c).strip() for c in df.columns]
    if "section" not in df.columns:
        cols_lower = {c.lower(): c for c in df.columns}
        if "section" in cols_lower:
            df.rename(columns={cols_lower["section"]: "section"}, inplace=True)
        else:
            st.error("Required column `section` not found in JSON sheet.")
            st.stop()
    return df

# ---------- Controller profiles (optional) ----------
# Provide a file "controller_profiles.json" with e.g.:
# [
#   {"name":"zencontrol (Application Controller)","protocols":{"DALI":50,"DMX":25},
#    "docs":[
#       {"label":"DMX integration","url":"https://zencontrol.com/dmx_integration/"},
#       {"label":"DMX wiring guide","url":"https://support.zencontrol.com/hc/en-us/articles/7646572778767-Controller-DMX-Wiring-Guide"}
#    ]}
# ]
PROFILES_FILE = Path("controller_profiles.json")
CONTROLLER_PROFILES = []
if PROFILES_FILE.exists():
    try:
        CONTROLLER_PROFILES = json.loads(PROFILES_FILE.read_text(encoding="utf-8"))
    except Exception:
        CONTROLLER_PROFILES = []

def pick_profile(name: str):
    for p in CONTROLLER_PROFILES:
        if p.get("name","").strip().lower() == name.strip().lower():
            return p
    return None

def pick_first(series):
    for v in series:
        if pd.notna(v):
            return v
    return None

def to_float_or_none(x):
    try:
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)
    except Exception:
        return None

def parse_cri_supported(s: Optional[str]):
    if not s:
        return [80, 90]
    try:
        return [int(x.strip()) for x in str(s).split(",") if x.strip()]
    except Exception:
        return [80, 90]

def ensure_numeric(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def sc_cct_factor_from_anchors(factors_row: pd.Series, cctK: float) -> float:
    anchors = []
    for col in factors_row.index:
        cl = str(col).strip().lower()
        if not cl.endswith("_x"):
            continue
        base = cl[:-2].replace(" ", "")
        m = re.match(r"^(\d+)(k)?$", base)
        if not m:
            continue
        n = float(m.group(1))
        k = 1000.0 * (n if n < 100 else n/10.0) if m.group(2) else n
        try:
            v = float(factors_row[col])
            if not np.isnan(v):
                anchors.append((float(k), v))
        except Exception:
            pass
    if not anchors:
        return 1.0
    anchors.sort(key=lambda t: t[0])
    if cctK <= anchors[0][0]:
        return anchors[0][1]
    if cctK >= anchors[-1][0]:
        return anchors[-1][1]
    for i in range(1, len(anchors)):
        k0, v0 = anchors[i-1]; k1, v1 = anchors[i]
        if cctK <= k1:
            t = (cctK - k0) / (k1 - k0)
            return v0 + t*(v1 - v0)
    return anchors[-1][1]

def mirek(K: float) -> float:
    return 1e6/float(K)

# ---- DALI helpers (both linear & IEC logarithmic) ----
def clip_pct(p, pmin, pmax):
    return float(np.clip(p, pmin, pmax))

def dali_arc_linear(percent: float, pmin: float = 0.1, pmax: float = 100.0) -> int:
    if percent <= 0:
        return 0
    p = clip_pct(percent, pmin, pmax)
    n = 1.0 + 253.0 * (p - pmin) / (pmax - pmin) if pmax > pmin else 254.0
    return int(round(np.clip(n, 1, 254)))

def dali_arc_iec(percent: float, pmin: float = 0.1, pmax: float = 100.0) -> int:
    if percent <= 0:
        return 0
    p = clip_pct(percent, pmin, pmax)
    denom = (math.log10(pmax) - math.log10(pmin)) if (pmax>pmin and pmin>0) else 3.0
    n = 1.0 + 253.0 * (math.log10(p) - math.log10(pmin)) / denom
    return int(round(np.clip(n, 1, 254)))

# ---- Color conversion helpers ----
def hex_to_rgb01(hexstr: str) -> Tuple[float,float,float]:
    h = hexstr.lstrip("#")
    r = int(h[0:2], 16)/255.0
    g = int(h[2:4], 16)/255.0
    b = int(h[4:6], 16)/255.0
    return r,g,b

def srgb_to_linear(c: float) -> float:
    return (c/12.92) if (c <= 0.04045) else (((c+0.055)/1.055)**2.4)

def srgb_hex_to_xy(hexstr: str) -> Tuple[float,float]:
    r,g,b = hex_to_rgb01(hexstr)
    rl, gl, bl = srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b)
    X = 0.4124564*rl + 0.3575761*gl + 0.1804375*bl
    Y = 0.2126729*rl + 0.7151522*gl + 0.0721750*bl
    Z = 0.0193339*rl + 0.1191920*gl + 0.9503041*bl
    denom = (X+Y+Z)
    if denom <= 1e-12:
        return 0.3127, 0.3290  # D65 fallback
    return X/denom, Y/denom

def clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

def rgb01_to_u16(rgb: Tuple[float,float,float]) -> Tuple[int,int,int]:
    r,g,b = [int(round(clamp01(c)*65535)) for c in rgb]
    return r,g,b

# ---------- Load JSON ----------
st.sidebar.header("Data source")
auto_file = Path("novon_workflow.json")
use_auto = st.sidebar.toggle("Auto-detect 'novon_workflow.json' in app folder", value=True)
uploaded = None if use_auto else st.sidebar.file_uploader("Upload novon_workflow.json", type=["json"])

payload = load_json_payload(auto_file if use_auto else None, uploaded)
df = df_from_one_sheet(payload)

# Sections
df["section"] = df["section"].astype(str).str.strip().str.upper()
settings_df = df[df["section"] == "SETTINGS"].copy()
engine_df   = df[df["section"] == "ENGINE"].copy()
factors_df  = df[df["section"] == "FACTORS"].copy()
curve_df    = df[df["section"] == "CURVE"].copy()

# Numeric coercions
ensure_numeric(curve_df, ["temp_C", "mA", "flux_lm_per_m", "power_W_per_m"])
ensure_numeric(engine_df, ["mA_ref","cct_nom_K","cct_min_K","cct_max_K",
                           "tw_warm_K","tw_cool_K",
                           "flux_25C_lm_per_m_ref","flux_65C_lm_per_m_ref",
                           "flux_warm_25C_lm_per_m_ref","flux_warm_65C_lm_per_m_ref",
                           "flux_cool_25C_lm_per_m_ref","flux_cool_65C_lm_per_m_ref"])

def setting_val(k, default):
    srow = settings_df[settings_df["key"].astype(str).str.lower() == str(k).lower()]
    v = pick_first(srow["value"]) if not srow.empty else None
    return v if pd.notna(v) else default

default_temp_C = float(setting_val("default_temp_C", 40))
solver_step    = int(setting_val("default_solver_mA_step", 1))
mA_min_global  = int(setting_val("mA_min_global", 50))
mA_max_global  = int(setting_val("mA_max_global", 600))

# Hard clamps for flux slider (optional)
flux_min_cfg = to_float_or_none(setting_val("flux_min_lm_per_m", None))
flux_max_cfg = to_float_or_none(setting_val("flux_max_lm_per_m", None))

# DALI physical min/max percent (for IEC & linear)
dali_min_pct = to_float_or_none(setting_val("dali_physical_min_pct", 0.1)) or 0.1
dali_max_pct = to_float_or_none(setting_val("dali_physical_max_pct", 100.0)) or 100.0

# Typical controller response time (ms) before fade visibly begins
response_typ_ms = float(to_float_or_none(setting_val("controller_response_typical_ms", 50.0)) or 50.0)

# Supported DT8 colour types (comma-separated, e.g., "TC,XY,RGBWAF")
dt8_supported_default = "TC,XY,RGBWAF"
dt8_supported = str(setting_val("dt8_supported_types", dt8_supported_default)).upper().replace(" ", "")
supported_types = [t for t in dt8_supported.split(",") if t in ("TC","XY","RGBWAF")] or ["TC"]

# For RGBWAF, allow declaring which primaries are present (subset, order matters)
prim_labels = str(setting_val("dt8_primary_labels", "RGBW")).upper().replace(" ", "")
prim_labels = "".join([c for c in prim_labels if c in "RGBWAF"]) or "RGB"

# ---------- UI: Engine / Gen / Mode ----------
if engine_df.empty:
    st.error("No ENGINE rows found in JSON.")
    st.stop()

engines = engine_df["engine_id"].dropna().unique().tolist()
engine_id = st.sidebar.selectbox("Engine", engines, index=0)

engine_rows = engine_df[engine_df["engine_id"] == engine_id]
gens = engine_rows["gen"].dropna().unique().tolist()
gen = st.sidebar.selectbox("Generation", gens, index=0)

e = engine_rows[engine_rows["gen"] == gen]
if e.empty:
    st.error("Selected engine+gen not found.")
    st.stop()

mode = str(pick_first(e["mode"])).strip().upper()
cri_supported = parse_cri_supported(pick_first(e["cri_supported"]))
cri = st.sidebar.selectbox("CRI", cri_supported, index=0)
tempC = st.sidebar.slider("Ambient / Reference Temp (°C)", min_value=25, max_value=65, value=int(default_temp_C), step=1)

st.subheader(f"Selected: {engine_id} • {gen} • {mode} • CRI {cri} • {tempC}°C")

show_myth = st.sidebar.toggle("Show myth buster panel", value=True)
st.sidebar.subheader("Controller profile")
profile_names = ["(none)"] + [p.get("name","") for p in CONTROLLER_PROFILES]
sel_prof = st.sidebar.selectbox("Controller", options=profile_names, index=0)
protocol_choice = st.sidebar.radio("Protocol for response timing", options=["DALI","DMX"], horizontal=True)
sel_prof_obj = pick_profile(sel_prof) if sel_prof != "(none)" else None
if sel_prof_obj:
    # If profile defines a suggestion for this protocol, use it
    prot_map = sel_prof_obj.get("protocols",{})
    if protocol_choice in prot_map:
        response_typ_ms = float(prot_map[protocol_choice])
st.sidebar.metric("Typical response (ms)", f"{int(round(response_typ_ms))}")
if sel_prof_obj and sel_prof_obj.get("docs"):
    with st.sidebar.expander("Controller links"):
        for d in sel_prof_obj["docs"]:
            st.markdown(f"- [{d.get('label','link')}]({d.get('url','#')})")

with st.sidebar.expander("Why so fast?"):
    st.write("DALI is **fast**—not 'supersonic'. A command is ~16 ms and drivers start acting right after the stop bit; what you see is the **fade seconds** you program. DMX refreshes in frames (~23–28 ms full universe). You can tweak the number via `SETTINGS.controller_response_typical_ms`.")


# ---------- SC interpolation helpers ----------
sc_curves = curve_df[(curve_df["engine_id"] == engine_id) &
                     (curve_df["gen"] == gen) &
                     (curve_df["channel"].astype(str).str.upper() == "SC")].copy()

def interp_sc_at(tempC: float, mA: float) -> Tuple[Optional[float], Optional[float]]:
    if sc_curves.empty:
        return (None, None)
    c25 = sc_curves[np.isclose(sc_curves["temp_C"], 25.0, atol=0.51)].sort_values("mA")
    c65 = sc_curves[np.isclose(sc_curves["temp_C"], 65.0, atol=0.51)].sort_values("mA")
    for frame in (c25, c65):
        if frame.empty or frame["mA"].nunique() < 2:
            return (None, None)
    def interp_one(frame, mA):
        x = frame["mA"].values
        fx = frame["flux_lm_per_m"].values
        px = frame["power_W_per_m"].values
        f = float(np.interp(mA, x, fx))
        p = float(np.interp(mA, x, px))
        return f, p
    f25, p25 = interp_one(c25, mA)
    f65, p65 = interp_one(c65, mA)
    t = np.clip((tempC - 25.0) / 40.0, 0.0, 1.0)
    f = (1.0 - t)*f25 + t*f65
    p = (1.0 - t)*p25 + t*p65
    return f, p

factors_row = factors_df[(factors_df["engine_id"] == engine_id) & (factors_df["gen"] == gen)]
factors_row = factors_row.iloc[0] if not factors_row.empty else pd.Series(dtype="object")

def apply_sc_factors(raw_flux: float, cri: int, cct_nom: Optional[float]) -> float:
    g = float(factors_row.get("gen_gain_x", 1.0) or 1.0)
    cri80 = float(factors_row.get("cri80_x", 1.0) or 1.0)
    cri90 = float(factors_row.get("cri90_x", 1.0) or 1.0)
    cri_factor = cri80 if cri == 80 else (cri90 if cri == 90 else 1.0)
    cct_factor = sc_cct_factor_from_anchors(factors_row, cct_nom) if cct_nom else 1.0
    return raw_flux * g * cri_factor * cct_factor

def dali_tc_from_K(targetK: float, cminK: float, cmaxK: float) -> Tuple[int, float]:
    m_min = 1e6 / cmaxK
    m_max = 1e6 / cminK
    m_tgt = 1e6 / targetK
    t = (m_tgt - m_min) / (m_max - m_min) if (m_max - m_min) != 0 else 0.0
    t = float(np.clip(t, 0.0, 1.0))
    return int(round(t * 65535)), float(m_tgt)

# ---------- Mode flows ----------
# Shared: compute DT6 % and arc codes from a running flux estimate
def arc_from_flux(flux_total: float, tempC: float) -> Tuple[float,int,int]:
    # Reference max = SC at mA_max_global if available, otherwise current flux as fallback
    f_sc_max, _ = interp_sc_at(tempC, float(mA_max_global))
    flux_max_for_arc = float(f_sc_max) if f_sc_max else max(flux_total, 1.0)
    pct_out = 100.0 * float(np.clip(flux_total/flux_max_for_arc, 0.0, 1.0))
    arc_linear = dali_arc_linear(pct_out, dali_min_pct, dali_max_pct)
    arc_iec    = dali_arc_iec(pct_out, dali_min_pct, dali_max_pct)
    return pct_out, arc_linear, arc_iec

# UI gradients

def _render_perception_scale(dali_ms: float, dmx_ms: float, max_ms: int = 600):
    # clamp
    dali_ms = max(0.0, min(float(dali_ms), float(max_ms)))
    dmx_ms  = max(0.0, min(float(dmx_ms),  float(max_ms)))
    def _pct(ms): return 100.0 * (ms / max_ms if max_ms>0 else 0.0)
    dali_left = _pct(dali_ms)
    dmx_left  = _pct(dmx_ms)
    html = f"""
    <div style="font-size:13px;margin-top:6px;">
      <div style="position:relative;height:24px;border-radius:8px; background: linear-gradient(90deg, rgba(220,220,220,0.8) 0% 16.6%, rgba(245,245,245,0.9) 16.6% 50%, rgba(255,240,240,0.9) 50% 100%); border:1px solid #ddd;">
        <!-- ticks -->
        <div style="position:absolute;left:0%;top:-18px;">0 ms</div>
        <div style="position:absolute;left:{_pct(100)}%;top:-18px;transform:translateX(-50%);">100 ms</div>
        <div style="position:absolute;left:{_pct(300)}%;top:-18px;transform:translateX(-50%);">300 ms</div>
        <div style="position:absolute;left:100%;top:-18px;transform:translateX(-100%);">{max_ms} ms</div>
        <!-- labels -->
        <div style="position:absolute;left:1%;top:4px;color:#333;">Imperceptible</div>
        <div style="position:absolute;left:36%;top:4px;color:#333;">Barely noticeable</div>
        <div style="position:absolute;left:70%;top:4px;color:#333;">Starting to feel lag</div>
        <!-- markers -->
        <div title="DALI (typical)" style="position:absolute;left:{dali_left}%;top:-6px;transform:translateX(-50%);">
          <div style="width:0;height:0;border-left:6px solid transparent;border-right:6px solid transparent;border-bottom:10px solid #222;margin:0 auto;"></div>
          <div style="font-size:11px;text-align:center;margin-top:2px;">DALI {int(round(dali_ms))} ms</div>
        </div>
        <div title="DMX (typical)" style="position:absolute;left:{dmx_left}%;top:18px;transform:translateX(-50%);">
          <div style="width:0;height:0;border-left:6px solid transparent;border-right:6px solid transparent;border-top:10px solid #222;margin:0 auto;"></div>
          <div style="font-size:11px;text-align:center;margin-top:2px;">DMX {int(round(dmx_ms))} ms</div>
        </div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def gradient_hint(left_hex: str, right_hex: str, height_px: int = 10):
    st.markdown(
        f'<div style="height:{height_px}px;background:linear-gradient(90deg, {left_hex}, {right_hex});border-radius:8px;margin-bottom:6px;"></div>',
        unsafe_allow_html=True
    )

if mode == "SC":
    if sc_curves.empty:
        st.warning("No SC curves found; cannot run SC mode.")
        st.stop()

    mA_domain = np.linspace(max(int(sc_curves["mA"].min()), 50),
                            min(int(sc_curves["mA"].max()), 600), 201)
    flux_blend = []
    for m in mA_domain:
        f_raw, _ = interp_sc_at(tempC, float(m))
        f_raw = 0.0 if f_raw is None or np.isnan(f_raw) else f_raw
        cct_nom = to_float_or_none(pick_first(e["cct_nom_K"]))
        flux_blend.append(apply_sc_factors(f_raw, int(cri), cct_nom))
    flux_blend = np.array(flux_blend, dtype=float)
    valid = np.isfinite(flux_blend)

    st.markdown("**Target Flux (lm/m)**")
    gradient_hint("rgba(200,200,200,0.7)","white",10)

    fmin_nat = int(np.nanmin(flux_blend[valid])) if valid.any() else 0
    fmax_nat = int(np.nanmax(flux_blend[valid])) if valid.any() else 2000
    fmin = int(max(fmin_nat, flux_min_cfg)) if flux_min_cfg is not None else fmin_nat
    fmax = int(min(fmax_nat, flux_max_cfg)) if flux_max_cfg is not None else fmax_nat
    if fmin >= fmax:
        fmin, fmax = fmin_nat, fmax_nat

    target_flux = st.slider(" ", min_value=fmin, max_value=fmax, value=int((fmin+fmax)//2), step=5, key="sc_flux_slider")
    order = np.argsort(flux_blend[valid])
    f_sorted = flux_blend[valid][order]
    m_sorted = mA_domain[valid][order]
    target_mA = float(np.interp(target_flux, f_sorted, m_sorted))

    f_raw_at_mA, p_raw_at_mA = interp_sc_at(tempC, target_mA)
    cct_nom = to_float_or_none(pick_first(e["cct_nom_K"]))
    f_final = apply_sc_factors(f_raw_at_mA or 0.0, int(cri), cct_nom)
    p_final = p_raw_at_mA or np.nan
    lm_per_W = f_final / p_final if p_final and p_final>0 else np.nan

    pct_out, arc_linear, arc_iec = arc_from_flux(f_final, tempC)

    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Drive Current (mA)", f"{int(round(target_mA))}")
    col2.metric("Flux (lm/m)", f"{int(round(f_final))}")
    col3.metric("Power (W/m)", f"{p_final:,.2f}" if p_final==p_final else "—")
    col4.metric("Efficacy (lm/W)", f"{lm_per_W:,.1f}" if lm_per_W==lm_per_W else "—")

    # SC is fixed CCT: we still compute DT8 Tc for reference based on cct_min/max
    tcode, tc_mired = dali_tc_from_K(
        to_float_or_none(pick_first(e["cct_nom_K"])) or 0.0,
        to_float_or_none(pick_first(e["cct_min_K"])) or (to_float_or_none(pick_first(e["tw_warm_K"])) or 2700.0),
        to_float_or_none(pick_first(e["cct_max_K"])) or (to_float_or_none(pick_first(e["tw_cool_K"])) or 6500.0)
    )
    st.markdown(f"**DALI** — DT6 arc: linear `{arc_linear}`, IEC log `{arc_iec}` • DT8 Tc: `{tcode}` • Output vs ref: {pct_out:,.2f}%")

    with st.expander("Scenes"):
        st.info("Add a scene: records target flux & derived mA. (SC mode has fixed nominal CCT).")
        scene_name = st.text_input("Scene name", value="Scene 1")
        fade_mA_s = st.number_input("Fade time for current (seconds)", min_value=0.0, value=0.0, step=0.5)

        if "scenes" not in st.session_state:
            st.session_state["scenes"] = []
        if st.button("Commit scene", use_container_width=True):
            st.session_state["scenes"].append({
                "name": scene_name, "mode": "SC",
                "color_type": "TC",
                "target_cct_K": to_float_or_none(pick_first(e["cct_nom_K"])),
                "target_cct_mired": tc_mired if tc_mired==tc_mired else None,
                "target_flux_lm_per_m": float(target_flux),
                "current_mA": float(target_mA),
                "dim_percent_ref": pct_out,
                "dali_arc_linear_0_254": int(arc_linear),
                "dali_arc_iec_0_254": int(arc_iec),
                "dali_tc_0_65535": int(tcode),
                "iec_min_pct": float(dali_min_pct),
                "iec_max_pct": float(dali_max_pct),
                "fade_cct_s": 0.0,
                "fade_mA_s": float(fade_mA_s),
            })
        if st.session_state.get("scenes"):
            df_s = pd.DataFrame(st.session_state["scenes"])
            df_s = st.data_editor(df_s, num_rows="dynamic", use_container_width=True, key="scenes_editor")
            st.download_button("Export scenes (CSV)", data=df_s.to_csv(index=False), file_name="scenes.csv", mime="text/csv")
            st.download_button("Export scenes (JSON)", data=df_s.to_json(orient="records"), file_name="scenes.json", mime="application/json")

elif mode == "TW":
    warmK = to_float_or_none(pick_first(e["tw_warm_K"])) or 2700.0
    coolK = to_float_or_none(pick_first(e["tw_cool_K"])) or 6500.0
    cmin = to_float_or_none(pick_first(e["cct_min_K"])) or warmK
    cmax = to_float_or_none(pick_first(e["cct_max_K"])) or coolK
    mA_ref = to_float_or_none(pick_first(e["mA_ref"])) or 200.0
    fw25 = to_float_or_none(pick_first(e["flux_warm_25C_lm_per_m_ref"])) or 900.0
    fw65 = to_float_or_none(pick_first(e["flux_warm_65C_lm_per_m_ref"])) or 850.0
    fc25 = to_float_or_none(pick_first(e["flux_cool_25C_lm_per_m_ref"])) or 1050.0
    fc65 = to_float_or_none(pick_first(e["flux_cool_65C_lm_per_m_ref"])) or 1000.0
    t = np.clip((tempC - 25.0) / 40.0, 0.0, 1.0)
    fw_ref = (1.0 - t)*fw25 + t*fw65
    fc_ref = (1.0 - t)*fc25 + t*fc65

    # Choose colour type (from supported set)
    st.markdown("**DT8 Colour Type**")
    ct_choice = st.radio(" ", options=supported_types, horizontal=True, key="dt8_colour_type")

    if ct_choice == "TC":
        st.markdown("**Target CCT (K)**")
        gradient_hint("#ff8f8f","#8fbaff",10)
        targetK = st.slider(" ", min_value=int(cmin), max_value=int(cmax), value=int((cmin+cmax)//2), step=50, key="tw_cct_slider")
        m_warm = 1e6 / warmK
        m_cool = 1e6 / coolK
        m_tgt  = 1e6 / targetK
        warm_share = float(np.clip((m_tgt - m_cool) / (m_warm - m_cool), 0.0, 1.0))
        cool_share = 1.0 - warm_share
        flux_at_ref = fw_ref*warm_share + fc_ref*cool_share

    elif ct_choice == "XY":
        st.markdown("**Pick colour (sRGB)**")
        hexcol = st.color_picker(" ", value="#ffffff", key="xy_color_picker")
        x,y = srgb_hex_to_xy(hexcol)
        st.write(f"xy = ({x:.4f}, {y:.4f})")
        # For reference CCT slider still useful for TW engine mixing, but not required for DT8 xy.
        # We'll compute flux_at_ref as average of warm/cool contributions at a nominal 4000K blend
        m_nom = 1e6 / float((cmin+cmax)/2.0)
        m_warm = 1e6 / warmK; m_cool = 1e6 / coolK
        warm_share = float(np.clip((m_nom - m_cool) / (m_warm - m_cool), 0.0, 1.0))
        cool_share = 1.0 - warm_share
        flux_at_ref = fw_ref*warm_share + fc_ref*cool_share

    else:  # RGBWAF
        st.markdown("**Pick colour (sRGB)**")
        hexcol = st.color_picker(" ", value="#ffffff", key="rgbw_color_picker")
        r01,g01,b01 = hex_to_rgb01(hexcol)
        st.write(f"RGB = ({int(round(r01*255))}, {int(round(g01*255))}, {int(round(b01*255))})")
        # Start from RGB, let user optionally tweak extra primaries
        prim_values = {}
        for P in prim_labels:
            if P == "R": prim_values["R"] = int(round(r01*65535))
            elif P == "G": prim_values["G"] = int(round(g01*65535))
            elif P == "B": prim_values["B"] = int(round(b01*65535))
            else: prim_values[P] = 0
        st.markdown("**Primaries (0..65535)**")
        cols = st.columns(len(prim_labels))
        for i,P in enumerate(prim_labels):
            prim_values[P] = cols[i].number_input(P, min_value=0, max_value=65535, value=int(prim_values[P]), step=256, key=f"prim_{P}")
        # Estimate flux_at_ref similarly to XY path (neutral)
        m_nom = 1e6 / float((cmin+cmax)/2.0)
        m_warm = 1e6 / warmK; m_cool = 1e6 / coolK
        warm_share = float(np.clip((m_nom - m_cool) / (m_warm - m_cool), 0.0, 1.0))
        cool_share = 1.0 - warm_share
        flux_at_ref = fw_ref*warm_share + fc_ref*cool_share

    # Flux slider (shared)
    st.markdown("**Target Flux (lm/m)**")
    gradient_hint("rgba(200,200,200,0.7)","white",10)
    fmin_nat = 0
    fmax_nat = int(round(max(1.2*flux_at_ref, flux_at_ref*2)))
    fmin = int(max(fmin_nat, flux_min_cfg)) if flux_min_cfg is not None else fmin_nat
    fmax = int(min(fmax_nat, flux_max_cfg)) if flux_max_cfg is not None else fmax_nat
    if fmin >= fmax:
        fmin, fmax = fmin_nat, fmax_nat
    target_flux = st.slider("  ", min_value=fmin, max_value=fmax, value=int(round(min(max(flux_at_ref, fmin), fmax))), step=5, key="tw_flux_slider")

    # Scale current from reference
    scale = (target_flux / flux_at_ref) if flux_at_ref>0 else 1.0
    total_mA = float(np.clip(scale * mA_ref, mA_min_global, mA_max_global))
    scale_eff = total_mA / mA_ref if mA_ref>0 else 1.0
    fw = fw_ref * warm_share * scale_eff
    fc = fc_ref * cool_share * scale_eff
    flux_total = fw + fc

    # Power estimate from SC efficacy if available
    f_sc, p_sc = interp_sc_at(tempC, total_mA)
    if f_sc and p_sc and p_sc>0:
        lm_per_W = f_sc / p_sc
        power_total = flux_total / lm_per_W
    else:
        power_total = flux_total / 120.0
        lm_per_W = flux_total / power_total if power_total>0 else np.nan

    pct_out, arc_linear, arc_iec = arc_from_flux(flux_total, tempC)

    # Compute DT8 outputs per colour type
    if ct_choice == "TC":
        tcode, tc_mired = dali_tc_from_K(float(targetK), float(cmin), float(cmax))
        extra = {"dali_tc_0_65535": int(tcode), "target_cct_K": float(targetK), "target_cct_mired": float(tc_mired)}
    elif ct_choice == "XY":
        x,y = srgb_hex_to_xy(hexcol)
        x_code = int(round(x*65535)); y_code = int(round(y*65535))
        extra = {"dt8_x_code_0_65535": x_code, "dt8_y_code_0_65535": y_code, "xy":[x,y], "target_cct_K": None, "target_cct_mired": None}
    else:
        extra = {"dt8_primaries": {P:int(prim_values[P]) for P in prim_labels}, "target_cct_K": None, "target_cct_mired": None, "dt8_primary_labels": prim_labels}

    col1,col2,col3,col4 = st.columns(4)
    if ct_choice == "TC":
        col1.metric("Target CCT (K)", f"{int(extra['target_cct_K'])}")
    elif ct_choice == "XY":
        col1.metric("DT8 x,y codes", f"{extra['dt8_x_code_0_65535']}, {extra['dt8_y_code_0_65535']}")
    else:
        col1.metric("Primaries", prim_labels)
    col2.metric("Flux (lm/m)", f"{int(round(flux_total))}")
    col3.metric("Current (mA) ~", f"{int(round(total_mA))}")
    col4.metric("Power (W/m) ~", f"{power_total:,.2f}")

    st.markdown(f"**DALI** — DT6 arc: linear `{arc_linear}`, IEC log `{arc_iec}` • Output vs ref: {pct_out:,.2f}%")

    with st.expander("Scenes"):
        st.info("Add a scene with fade durations. Stores DT6 arc (linear+IEC) and the DT8 outputs for the selected colour type.")
        scene_name = st.text_input("Scene name", value="Scene 1")
        fade_cct_s = st.number_input("Fade time for CCT/colour (seconds)", min_value=0.0, value=0.0, step=0.5)
        fade_mA_s  = st.number_input("Fade time for current (seconds)", min_value=0.0, value=0.0, step=0.5)

        if "scenes" not in st.session_state:
            st.session_state["scenes"] = []
        if st.button("Commit scene", use_container_width=True):
            base = {
                "name": scene_name, "mode": "TW", "color_type": ct_choice,
                "target_flux_lm_per_m": float(target_flux),
                "current_mA": float(total_mA),
                "dim_percent_ref": pct_out,
                "dali_arc_linear_0_254": int(arc_linear),
                "dali_arc_iec_0_254": int(arc_iec),
                "iec_min_pct": float(dali_min_pct),
                "iec_max_pct": float(dali_max_pct),
                "fade_cct_s": float(fade_cct_s),
                "fade_mA_s": float(fade_mA_s),
                "controller_response_typical_ms": float(response_typ_ms),
                "controller_profile": sel_prof if sel_prof else None,
                "controller_protocol": protocol_choice,
                "est_total_time_flux_s": float(response_typ_ms/1000.0 + fade_mA_s if isinstance(fade_mA_s,float) else response_typ_ms/1000.0 + float(fade_mA_s)),
                "est_total_time_cct_s": float(response_typ_ms/1000.0 + fade_cct_s),
            }
            base.update(extra)
            st.session_state["scenes"].append(base)
        if st.session_state.get("scenes"):
            df_s = pd.DataFrame(st.session_state["scenes"])
            df_s = st.data_editor(df_s, num_rows="dynamic", use_container_width=True, key="scenes_editor")
            st.download_button("Export scenes (CSV)", data=df_s.to_csv(index=False), file_name="scenes.csv", mime="text/csv")
            st.download_button("Export scenes (JSON)", data=df_s.to_json(orient="records"), file_name="scenes.json", mime="application/json")

else:
    st.warning(f"Unknown mode '{mode}'. Please check ENGINE rows.")

with st.expander("Endpoints & Overrides (live)"):
    st.json({
        "one_sheet_tab": OVERRIDES.get("one_sheet_tab"),
        "column_renames": OVERRIDES.get("column_renames"),
        "settings_keys_expected": [
            "default_temp_C","default_solver_mA_step","mA_min_global","mA_max_global",
            "flux_min_lm_per_m","flux_max_lm_per_m","dali_physical_min_pct","dali_physical_max_pct",
            "dt8_supported_types","dt8_primary_labels"
        ],
        "engine_columns_expected": [
            "engine_id","gen","mode","cri_supported","mA_ref",
            "cct_nom_K","cct_min_K","cct_max_K","tw_warm_K","tw_cool_K",
            "flux_warm_25C_lm_per_m_ref","flux_warm_65C_lm_per_m_ref",
            "flux_cool_25C_lm_per_m_ref","flux_cool_65C_lm_per_m_ref"
        ],
        "curve_columns_expected_SC": ["engine_id","gen","channel","temp_C","mA","flux_lm_per_m","power_W_per_m"],
        "factors_columns_expected": ["engine_id","gen","gen_gain_x","cri80_x","cri90_x","<CCT_anchor>_x (e.g., 3000K_x, 35k_x)"]
    })

with st.expander("Notes"):
    st.markdown("""
- **Flux slider** is the primary control. Current (mA) is derived by inverting SC curve or scaling in TW.
- **DT8 colour types**: pick **TC**, **xy**, or **RGBWAF** (primaries). For RGBWAF, declare the order of primaries via `dt8_primary_labels` in **SETTINGS** (e.g., `RGBW` or `RGBWAF`).
- **DALI DT6**: both **linear** and **IEC log** arc codes are shown and stored. IEC bounds from SETTINGS.
- **DT8 outputs** stored per type: `dali_tc_0_65535` for TC; `dt8_x_code_0_65535`/`dt8_y_code_0_65535` for xy; `dt8_primaries` for RGBWAF.
- **Hard clamps**: set `flux_min_lm_per_m` / `flux_max_lm_per_m` in **SETTINGS** to bound the flux slider.
- **Endpoint overrides**: drop `solver_endpoints_overrides.json` next to the app to rename columns or change the one-sheet tab without editing code.
""")


with st.expander("Protocol explainer: DALI vs DMX (latency & fades)"):
    st.markdown("""
- **DALI**: 1200 bps, a forward frame is ~15.8 ms; devices begin action right after the **stop** condition, so **start-of-fade is effectively immediate** and the *fade time* parameter dominates. We expose a configurable “Typical response (ms)” to budget controller/stack overheads.
- **DMX512**: continuous frames at ~**22–28 ms** per full universe (≈ 39–44 frames/sec). Gateways (e.g., zencontrol DMX send) map DALI inputs to DMX levels and let you set fade rate/time on the DMX side.
- Result: For most architectural fades, both are **sub-100 ms to begin**; the human-visible effect is governed by your **fade seconds**.
""")


with st.expander("Helpful links (DALI/DMX + vendors)"):
    st.markdown("""
- **zencontrol**: [DMX integration](https://zencontrol.com/dmx_integration/) • [DMX wiring guide](https://support.zencontrol.com/hc/en-us/articles/7646572778767-Controller-DMX-Wiring-Guide)
- **DMX timing**: [SparkFun intro (23 ms full packet ≈ 44 Hz)](https://learn.sparkfun.com/tutorials/introduction-to-dmx/all) • [DMX512 (Wikipedia)](https://en.wikipedia.org/wiki/DMX512)
- **DALI signalling**: [Microchip DALI protocol brief (1200 bps)](https://ww1.microchip.com/downloads/en/AppNotes/90003200A.pdf) • [Pico Technology note](https://www.picotech.com/library/knowledge-bases/oscilloscopes/dali-serial-protocol-decoding)
""")


with st.expander("Typical research outcome & self‑check prompt"):
    st.markdown("""
**Typical research outcome (neutral summary):**
- For most **architectural** fades, both **DALI** and **DMX** begin acting within **sub‑100 ms**; what people *see* is the **fade seconds** you program.
- **DALI / DT8**: compact commands; drivers act after the stop bit; great for tunable white (**Tc**) and stable colour (**xy**).
- **DMX512**: frame‑based refresh (tens of ms per full universe), excellent for fast effects and pixel mapping; widely known in entertainment.
- Pick based on *workflow and feature set*, not fear: DALI DT8 is usually **more than fast enough** for architectural colour.
""")

    # Ready-to-use prompts
    prompt_core = """Compare DALI (incl. DT8 Tc/xy) vs DMX512 for architectural lighting.
Cover: start-of-fade latency, typical command/frame times, visible response vs programmed fade, gateway behaviours (DALI→DMX), and when DMX is preferable (effects/pixels).
Cite IEC 62386 basics and at least one vendor doc (e.g., zencontrol) plus a neutral DMX timing reference."""

    prompt_brands = """Compare practical response time and fade behaviour for DALI DT8 vs DMX across common vendors (e.g., zencontrol, Signify/Philips Dynalite, Pharos, ETC).
Focus on architectural use (tunable white/xy), controller overheads, and human perception thresholds for ‘instant’ start-of-fade."""

    st.code(prompt_core, language="text")
    st.download_button("Download prompt (core)", data=prompt_core, file_name="dali_vs_dmx_core_prompt.txt", mime="text/plain")
    st.code(prompt_brands, language="text")
    st.download_button("Download prompt (brands)", data=prompt_brands, file_name="dali_vs_dmx_brands_prompt.txt", mime="text/plain")

    # One-click web searches with the prompt embedded
    import urllib.parse as _u
    q1 = _u.quote(prompt_core)
    q2 = _u.quote(prompt_brands)
    st.markdown(f"[Search Google (core prompt)](https://www.google.com/search?q={q1})  |  [Search Bing (core prompt)](https://www.bing.com/search?q={q1})")
    st.markdown(f"[Search Google (brands prompt)](https://www.google.com/search?q={q2})  |  [Search Bing (brands prompt)](https://www.bing.com/search?q={q2})")


if show_myth:
    with st.expander("Myth buster — Architectural vs Stage: DALI is fast (not supersonic)"):
        st.markdown("""
**Architectural lighting (what most projects are):**
- Comfort, ambience, scenes; **no pixel-mapped effects**
- Typical actions: **set/recall scenes, tunable white, modest colour moves**
- **DALI/DT8 fits naturally**: one shared **DALI bus**, addressable gear, Tc/xy colour control
- **Latency**: typically **sub‑100 ms** to start moving; the **fade seconds** you choose dominate
- **Wiring**: two‑wire, topology‑free DALI bus you likely already have

**Stage / entertainment (specialist):**
- High‑speed chases, strobes, **pixel mapping**, media sync
- Frame‑by‑frame updates; **DMX/Art‑Net/sACN** ecosystems excel here
- **Wiring**: dedicated DMX cabling/connectors, universes, patching

**Plain English:** DALI is **fast** for architectural work—users don’t perceive the command delay; they perceive your programmed fades. If you need **supersonic**, frame‑by‑frame effects, that’s a stage problem (DMX et al.).
""")
        # Mini infographic (side-by-side)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            ### Architectural
            - Scenes & ambience
            - Tunable white / stable colour
            - Human comfort priority
            - **Fast** commands, **programmed fades**
            """)
        with c2:
            st.markdown("""
            ### Stage / Entertainment
            - Effects, strobes, pixel mapping
            - Frame-by-frame control
            - Media sync priority
            - **Supersonic** frame updates
            """)
        st.divider()
        st.markdown("### Wiring at a glance")
        w1, w2 = st.columns(2)
        with w1:
            st.markdown("""
            **DALI**
            - 2‑wire, polarity‑agnostic bus
            - Topology‑free (no daisy‑chain required)
            - Power + comms on the same pair
            - Usually **already present** in architectural jobs
            """)
        with w2:
            st.markdown("""
            **DMX512**
            - 120 Ω shielded twisted pair (XLR5/RJ45)
            - Daisy‑chain with terminators
            - Patch universes, address fixtures
            - Extra gateways for DT8 ↔ DMX mapping
            """)
        st.info("Rule of thumb: Choose **DALI/DT8** for architectural; choose **DMX** only when you truly need entertainment‑style effects or pixel mapping.")
        st.caption("Analogy: DALI is **fast**—like smooth camera pans. Stage DMX is **supersonic**—like high‑FPS video for action shots. Use the tool that matches the scene.")


with st.expander("Perception cheat‑sheet (latency vs perception)"):
    st.markdown("Map **controller/stack response** onto what humans perceive. These are guidance bands, not pass/fail lines.")
    # typicals: DALI from current sidebar (includes profile/protocol), DMX from profile default or fallback 25 ms
    dali_typ_ms = float(response_typ_ms)
    dmx_typ_ms = 25.0
    if sel_prof_obj and isinstance(sel_prof_obj.get("protocols",{}), dict):
        dmx_typ_ms = float(sel_prof_obj["protocols"].get("DMX", dmx_typ_ms))
    _render_perception_scale(dali_typ_ms, dmx_typ_ms, max_ms=600)
    st.caption("Cheat‑sheet bands: 0–100 ms ≈ **imperceptible**, 100–300 ms ≈ **barely noticeable**, >300 ms ≈ **people start to feel lag**.")

