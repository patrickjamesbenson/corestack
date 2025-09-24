import streamlit as st
def _safe_set_page_config(*a, **k):
    try:
        return st.set_page_config(*a, **k)
    except Exception:
        return None

import os, json, re
from pathlib import Path
import pandas as pd
import streamlit as st

APP_TITLE = "Picker + IES JSON Finder (Sibling-wired)"
_safe_set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="collapsed")
st.title(APP_TITLE)

# ---------------- JSON-only workbook loader ----------------
# Prefer sibling path: ../novon_db_updater/assets/novon_workflow.json
JSON_CANDIDATES = [
    "../novon_db_updater/assets/novon_workflow.json",  # sibling
    "novon_workflow.json",
    "Select_Output_and_Attribute.json",
]

def find_local_json():
    for name in JSON_CANDIDATES:
        p = Path(name)
        if p.exists():
            return str(p)
    return None

def load_workbook_from_json(path_or_file):
    if isinstance(path_or_file, (str, os.PathLike)):
        with open(path_or_file, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    else:
        payload = json.load(path_or_file)
    tables = payload.get("tables", {})
    wb = {}
    for sheet, rows in tables.items():
        try:
            df = pd.DataFrame(rows)
        except Exception:
            try:
                df = pd.DataFrame(list(rows.values()))
            except Exception:
                df = pd.DataFrame()
        wb[str(sheet)] = df
    return wb

wb = {}
local_json = find_local_json()

# Sidebar uploader (sidebar stays collapsed; we show a hint on main area if missing)
st.sidebar.header("Workbook JSON")
if local_json:
    st.sidebar.success(f"Loaded: {local_json}")
    wb = load_workbook_from_json(local_json)
else:
    up = st.sidebar.file_uploader("Upload workbook JSON (.json)", type=["json"])
    if up is not None:
        wb = load_workbook_from_json(up)
        st.sidebar.success("Loaded uploaded JSON")
    else:
        st.info("No workbook JSON found at ../novon_db_updater/assets/novon_workflow.json.\nOpen the sidebar to upload one.")

# --------------- Tier mapping (from SYS_CODES if present) ---------------
def build_tier_maps_from_wb(wb):
    code_to_name = {"S":"S","I":"I","X":"X","C":"C"}
    name_to_code = {v:k for k,v in code_to_name.items()}
    try:
        if "SYS_CODES" in wb:
            df = wb["SYS_CODES"]
            if isinstance(df, pd.DataFrame) and all(c in df.columns for c in ["category","code","value"]):
                sel = df[df["category"].astype(str).str.strip().str.lower()=="tier"][["code","value"]].dropna()
                if not sel.empty:
                    code_to_name.update(dict(zip(sel["code"].astype(str), sel["value"].astype(str))))
                    name_to_code = {v:k for k,v in code_to_name.items()}
    except Exception:
        pass
    return code_to_name, name_to_code

code_to_name, name_to_code = build_tier_maps_from_wb(wb)

# ---------------- IES JSON Finder utilities ----------------
def safe_read_json(p: Path):
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None

def guess_meta_from_json(payload: dict):
    if not isinstance(payload, dict):
        return {}
    cands = [payload]
    for k in ["meta","metadata","IES_META","header","info","properties"]:
        if isinstance(payload.get(k), dict):
            cands.append(payload[k])
    meta = {}
    # width
    for d in cands:
        for key in ["optic_fam_width","optic_width_mm","body_width_mm","width_mm","optic_width","width"]:
            if key in d:
                meta["width"] = str(d[key]); break
        if "width" in meta: break
    # material
    for d in cands:
        for key in ["optic_material","material","diffuser","lens_material","optic_type"]:
            if key in d:
                meta["material"] = str(d[key]); break
        if "material" in meta: break
    # tier (code or name)
    for d in cands:
        for key in ["tier","tier_code","tier_name","Tier","TierCode","TierName"]:
            if key in d:
                meta["tier_raw"] = str(d[key]); break
        if "tier_raw" in meta: break
    # driver type
    for d in cands:
        for key in ["driver_type","driver","driver_mode","dali_type","dali_profile","type"]:
            if key in d:
                meta["driver_type"] = str(d[key]); break
        if "driver_type" in meta: break
    # CRI
    for d in cands:
        for key in ["cri","CRI","Ra","ra","CRI_Ra"]:
            if key in d:
                meta["cri"] = str(d[key]); break
        if "cri" in meta: break
    # CCT
    for d in cands:
        for key in ["cct","CCT","cct_k","CCT_K","kelvin","CCTK","temp_k"]:
            if key in d:
                meta["cct"] = str(d[key]); break
        if "cct" in meta: break
    # LED generation
    for d in cands:
        for key in ["gen","generation","led_gen","led_generation","chip_gen"]:
            if key in d:
                meta["led_gen"] = str(d[key]); break
        if "led_gen" in meta: break
    return meta

def guess_meta_from_filename(name: str):
    s = name.lower()
    out = {}
    # Deterministic filename parse: LUMCAT__{lm}lm_per_mm__{w}w_per_mm__{thick}mm_{dist}.json
    m = re.search(r"lumcat__([0-9]+(?:\.[0-9]+)?)lm_per_mm__([0-9]+(?:\.[0-9]+)?)w_per_mm__([0-9]+)mm_([bdi])\.json", s)
    if m:
        out["lumens_per_mm"] = m.group(1)
        out["watts_per_mm"] = m.group(2)
        out["thickness_mm"] = m.group(3)
        dist = m.group(4).upper()
        out["distribution"] = {"B":"Bidirectional","D":"Direct","I":"Indirect"}.get(dist, dist)
    # width
    for w in ["45","60","80","90","100","120"]:
        for token in [f"_{w}_", f"-{w}-", f" {w} ", f"_{w}.json", f"-{w}.json"]:
            if token in s:
                out["width"] = w; break
        if "width" in out: break
    # material
    mats = {"Opal":["opal","op-","op_","_op_"], "Microprism":["micro","mpr","prism"], "Asymmetric":["asym","asymmetric","asymm","asym-","-asym"]}
    for label, keys in mats.items():
        if any(k in s for k in keys):
            out["material"] = label; break
    # driver type
    if "dt6" in s or "dt-6" in s: out["driver_type"] = "DT6"
    elif "dt8" in s or "dt-8" in s: out["driver_type"] = "DT8"
    elif "non-dim" in s or "nondim" in s or "non_dim" in s: out["driver_type"] = "non_dim"
    # CRI
    if "cri90" in s or "ra90" in s or "_90_" in s: out["cri"] = "90"
    elif "cri80" in s or "ra80" in s or "_80_" in s: out["cri"] = "80"
    # CCT
    for k,v in [("3000","3000"),("30k","3000"),("3500","3500"),("35k","3500"),("4000","4000"),("40k","4000")]:
        if k in s:
            out["cct"] = v; break
    # tier code or name
    for code in ["s","i","x","c"]:
        if f"_{code}_" in s or f"-{code}-" in s or s.startswith(code+"_"):
            out["tier_raw"] = code.upper(); break
    if "tier_raw" not in out:
        for nm in ["economy","professional","business","first","charter","custom"]:
            if nm in s:
                out["tier_raw"] = nm; break
    # LED generation
    mg = re.search(r'\bgen\s*([0-9]+)\b', s)
    if mg: out["led_gen"] = mg.group(1)
    return out

@st.cache_data(show_spinner=False)
def build_ies_index(search_dirs, code_to_name):
    rows = []
    seen = set()
    for d in search_dirs:
        p = Path(d).expanduser()
        if not p.exists():
            continue
        for fp in p.rglob("*.json"):
            nm = fp.name.lower()
            # Skip workbook JSONs
            if nm in {"novon_workflow.json","select_output_and_attribute.json"}:
                continue
            key = str(fp.resolve())
            if key in seen:
                continue
            seen.add(key)
            meta = {}
            payload = safe_read_json(fp)
            if isinstance(payload, dict):
                meta.update(guess_meta_from_json(payload))
            fn = guess_meta_from_filename(fp.name)
            for k,v in fn.items():
                meta.setdefault(k, v)
            # Normalise tier
            tier_code = ""
            tier_name = ""
            raw = meta.get("tier_raw")
            if isinstance(raw, str):
                raw = raw.strip()
                if raw.upper() in code_to_name:
                    tier_code = raw.upper()
                    tier_name = code_to_name.get(tier_code, tier_code)
                else:
                    inv = {v.lower():k for k,v in code_to_name.items()}
                    tkey = inv.get(raw.lower())
                    if tkey:
                        tier_code = tkey
                        tier_name = code_to_name[tkey]
                    else:
                        tier_name = raw
            rows.append({
                "file": fp.name,
                "path": str(fp.resolve()),
                "width": meta.get("width",""),
                "material": meta.get("material",""),
                "tier_code": tier_code,
                "tier_name": tier_name,
                "driver_type": meta.get("driver_type",""),
                "cri": meta.get("cri",""),
                "cct": meta.get("cct",""),
                "led_gen": meta.get("led_gen",""),
                "lumens_per_mm": meta.get("lumens_per_mm",""),
                "watts_per_mm": meta.get("watts_per_mm",""),
                "thickness_mm": meta.get("thickness_mm",""),
                "distribution": meta.get("distribution",""),
            })
    return pd.DataFrame(rows)

st.subheader("IES JSON Finder")

# Default search dirs: sibling repo path first; keep local ./ies_repo as fallback
DEFAULT_DIRS = [
    "../ies_prep/IES Repo/software_ready",
    "./ies_repo",
]

if "IES_JSON_SEARCH_DIRS" not in st.session_state:
    st.session_state.IES_JSON_SEARCH_DIRS = DEFAULT_DIRS[:]

with st.expander("Search paths (edit as needed)", expanded=False):
    new_dirs = []
    for i, d in enumerate(st.session_state.IES_JSON_SEARCH_DIRS):
        new_dirs.append(st.text_input(f"Path {i+1}", value=d, key=f"iesdir_{i}"))
    add_more = st.text_input("Add path", value="", placeholder="Type a new path and press Enter")
    if add_more:
        new_dirs.append(add_more)
    if new_dirs:
        st.session_state.IES_JSON_SEARCH_DIRS = [p for p in new_dirs if p]

if st.button("Build / refresh index"):
    st.cache_data.clear()

idx_df = build_ies_index(st.session_state.IES_JSON_SEARCH_DIRS, code_to_name)

if idx_df.empty:
    st.info("No IES JSON files found. Update search paths and click refresh.")
else:
    def opts(col):
        vals = sorted({str(v) for v in idx_df[col].dropna().astype(str) if str(v).strip()})
        return ["Any"] + vals

    colA, colB, colC = st.columns(3)
    with colA:
        sel_width = st.selectbox("Width", opts("width"))
        sel_material = st.selectbox("Material", opts("material"))
        sel_tier_name = st.selectbox("Tier (name)", opts("tier_name"))
    with colB:
        sel_driver = st.selectbox("Driver type", opts("driver_type"))
        sel_cri = st.selectbox("CRI", opts("cri"))
        sel_cct = st.selectbox("CCT (K)", opts("cct"))
    with colC:
        sel_led_gen = st.selectbox("LED generation", opts("led_gen"))

    df = idx_df.copy()
    def apply_eq(col, val):
        if val != "Any":
            df.loc[:, col] = df[col].astype(str)
            return df[df[col].str.lower() == str(val).lower()]
        return df
    df = apply_eq("width", sel_width)
    df = apply_eq("material", sel_material)
    df = apply_eq("tier_name", sel_tier_name)
    df = apply_eq("driver_type", sel_driver)
    df = apply_eq("cri", sel_cri)
    df = apply_eq("cct", sel_cct)
    df = apply_eq("led_gen", sel_led_gen)

    df = df.sort_values(["tier_name","width","material","driver_type","cri","cct","led_gen","file"])
    st.write(f"Matches: {len(df)}")
    st.dataframe(df[[
        "file","width","material","tier_name","driver_type","cri","cct","led_gen",
        "lumens_per_mm","watts_per_mm","thickness_mm","distribution","path"
    ]], use_container_width=True)

    pick = st.selectbox("Choose IES JSON", ["(none)"] + df["path"].tolist())
    if pick and pick != "(none)":
        st.session_state["selected_ies_json"] = pick
        st.success(f"Selected IES JSON: {pick}")

