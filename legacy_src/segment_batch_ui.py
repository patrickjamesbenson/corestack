import streamlit as st
def _safe_set_page_config(*a, **k):
    try:
        return st.set_page_config(*a, **k)
    except Exception:
        return None
﻿# segment_batch_ui.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import streamlit as st

APP_TITLE = "Run Segment Solver"  # keep/rename as you like
_safe_set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")


import math, json, os, pandas as pd
# ...import json, math, re, io
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd

# ---- SAFE FLAT JSON LOADER (drop-in) ----
import glob, json, os
# (removed duplicate) import streamlit as st
import re

def _pick_latest_flat_json(search_dir: str = "."):
    cands = sorted(
        glob.glob(os.path.join(search_dir, "flat_*.json")),
        key=lambda p: [int(x) if x.isdigit() else x for part in os.path.basename(p)[5:-5].split(".") for x in [part]],
    )
    return cands[-1] if cands else None

@st.cache_data(show_spinner=False)
def load_flat_json():
    """Try latest flat_*.json; if bad, return None (UI will show uploader)."""
    path = _pick_latest_flat_json(".")
    if not path:
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data, path
    except Exception as e:
        st.warning(f"Found {path} but couldn’t parse it: {e}")
        return None, path

flat_data, flat_path = load_flat_json()

st.sidebar.markdown("### Source")
if flat_data:
    st.sidebar.success(f"Loaded: {os.path.basename(flat_path)}")
else:
    st.sidebar.warning("No valid `flat_*.json` found.")
    up = st.sidebar.file_uploader("…or drop a flat JSON", type=["json"])
    if up:
        try:
            flat_data = json.loads(up.read().decode("utf-8"))
            st.sidebar.success("Loaded from uploaded file")
        except Exception as e:
            st.sidebar.error(f"JSON parse error: {e}")
            flat_data = None

# guard: keep the app alive even if there’s no data
if flat_data is None:
    st.info("Load a `flat_*.json` (or upload one) to begin.")
    st.stop()
# ---- END SAFE LOADER ----

APP_TITLE = "Run Segment Solver"

# ====================== helpers ======================
SEMVER = re.compile(r"^flat_(\d+)\.(\d+)\.(\d+)\.json$", re.I)
def _latest_flat() -> Path | None:
    best = None; ver = None
    for p in Path(".").glob("flat_*.json"):
        m = SEMVER.match(p.name)
        if not m: continue
        v = (int(m[1]), int(m[2]), int(m[3]))
        if ver is None or v > ver:
            ver, best = v, p
    return best

def _read_json_bytes(b: bytes) -> Dict[str, Any]:
    return json.loads(b.decode("utf-8"))

def _runs(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    return obj["runs"] if isinstance(obj.get("runs"), list) else [obj]

def mm1(x: float) -> float: return float(f"{x:.1f}")
def chips(pairs: List[tuple[str,str]]) -> str:
    return "  ".join(f"`{a} {b}`" if b else f"`{a}`" for a,b in pairs)
def abc(n: int) -> List[str]: return [chr(65+i) for i in range(n)]

# ================= segments & placement =================
def greedy_segments(final_len: float, max_seg: float) -> List[float]:
    if max_seg <= 0: return [final_len]
    n_full = int(final_len // max_seg)
    rem = final_len - n_full*max_seg
    segs = [max_seg]*n_full
    if rem > 1e-6: segs.append(rem)
    return segs or [final_len]

def place_short(segs: List[float], mode: str) -> List[float]:
    out = list(segs)
    if len(out) <= 1: return out
    if len(set(round(s,3) for s in out)) == 1: return out
    short = out.pop(-1)
    m = mode.lower()
    if m.startswith("start"):
        out.insert(0, short)
    elif m.startswith("centre") or m.startswith("center"):
        n = len(out)+1
        if n % 2 == 1:  # only place to centre when odd
            out.insert(n//2, short)
        else:
            out.append(short)  # UI will note “centre if odd ignored”
    else:  # End
        out.append(short)
    return out

# ================= ancillaries =================
def normalise_anc_pills(anc_in: List[Dict[str, Any]]) -> List[str]:
    pills: List[str] = []
    for a in anc_in or []:
        code = str(a.get("code","")).upper().strip() or "ANC"
        cover = float(a.get("cover_len_mm", 0) or 0.0)
        if code == "MW" and cover < 1.5:
            pills.append("MW-0")
        else:
            pills.append(f"{code} {int(round(cover))}")
    return pills

def reserve_grid(final_len: float, pitch: float) -> List[tuple[float,float]]:
    slots = []; cur = 0.0
    while cur + pitch <= final_len + 1e-6:
        slots.append((cur, cur+pitch)); cur += pitch
    if cur < final_len - 1e-6: slots.append((cur, final_len))
    return slots

def bundle_anc(anc_in: List[Dict[str, Any]], pitch: float) -> Dict[int, List[Dict[str,Any]]]:
    bundles: Dict[int, List[Dict[str,Any]]] = {}
    for a in anc_in or []:
        x = a.get("x_mm") or a.get("x_mm_run")
        idx = int(round(float(x)/pitch)) if isinstance(x,(int,float)) else 0
        cover = float(a.get("cover_len_mm",0) or 0.0)
        code = str(a.get("code","")).upper().strip() or "ANC"
        if code=="MW" and cover<1.5:
            bundles.setdefault(idx, []).append({"code":"MW-0","cover_len_mm":0.0})
        else:
            bundles.setdefault(idx, []).append({"code":code,"cover_len_mm":cover})
    if not bundles and anc_in:
        bundles[0] = [{"code":( "MW-0" if (str(a.get("code","")).upper()=="MW" and float(a.get("cover_len_mm",0) or 0.0)<1.5) else str(a.get("code","")).upper()),
                       "cover_len_mm": (0.0 if (str(a.get("code","")).upper()=="MW" and float(a.get("cover_len_mm",0) or 0.0)<1.5) else float(a.get("cover_len_mm",0) or 0.0))}
                      for a in anc_in]
    return bundles

def ancillary_policy(anc_in: List[Dict[str, Any]], pitch: float, policy: str):
    """
    Return:
      bundles: [{slot_index, items:[{code,cover_len_mm}], cover_sum_mm, slots_taken, remainder_mm}]
      delta_len: +/- mm on run length (Extend +rem, Shorten -(slots*pitch - cover))
    """
    bundles_in_slots = bundle_anc(anc_in, pitch)
    out = []; delta = 0.0
    for idx, items in sorted(bundles_in_slots.items()):
        cover = sum(it["cover_len_mm"] for it in items)
        full = int(cover // pitch); rem = cover - full*pitch
        pol = policy.lower()
        if pol.startswith("extend"):
            out.append({"slot_index":idx,"items":items,"cover_sum_mm":cover,"slots_taken":full,"remainder_mm":rem})
            delta += rem
        elif pol.startswith("shorten"):
            slots = int(math.ceil(cover/pitch)) if cover>0 else 0
            shorten_by = max(0.0, slots*pitch - cover)
            out.append({"slot_index":idx,"items":items,"cover_sum_mm":cover,"slots_taken":slots,"remainder_mm":0.0})
            delta -= shorten_by
        else:  # Maintain
            slots = int(math.ceil(cover/pitch)) if cover>0 else 0
            out.append({"slot_index":idx,"items":items,"cover_sum_mm":cover,"slots_taken":slots,"remainder_mm":0.0})
    return out, delta

# ================= hungry-fill timeline =================
def timeline_with_covers(final_len: float, pitch: float, bundles: List[Dict[str,Any]], policy: str):
    pol = policy.lower()
    if pol.startswith("shorten"): return None
    slots = reserve_grid(final_len, pitch)
    tokens: List[tuple[str,float,str]] = [("LED", s[1]-s[0], "LED") for s in slots]

    def expand_at(index: int, new_tokens: List[tuple[str,float,str]]):
        tokens[index:index+1] = new_tokens

    for b in sorted(bundles, key=lambda z: z["slot_index"]):
        idx = b["slot_index"]
        if idx >= len(tokens): idx = len(tokens)-1
        cover = b["cover_sum_mm"]; full = b["slots_taken"]; rem = b["remainder_mm"]
        codes = "+".join(sorted({it["code"] for it in b["items"]})) if b["items"] else "ANC"
        if pol.startswith("extend"):
            repl = []
            for _ in range(full): repl.append(("COVER", pitch, codes))
            if rem > 1e-6:
                repl.append(("COVER", rem, codes))
                left = max(0.0, tokens[idx][1] - rem)
                if left > 1e-6: repl.append(("LED", left, "LED"))
            expand_at(idx, repl or [("LED", tokens[idx][1], "LED")])
        else:  # Maintain
            repl = []
            for _ in range(max(1, full) if cover>0 else 0):
                repl.append(("COVER", pitch, codes))
            expand_at(idx, repl or [("LED", tokens[idx][1], "LED")])

    merged: List[tuple[str,float,str]] = []
    for t,ln,lab in tokens:
        if merged and merged[-1][0]==t and merged[-1][2]==lab:
            merged[-1] = (t, merged[-1][1]+ln, lab)
        else:
            merged.append((t,ln,lab))
    return merged

def hungry_fill_led(led_len: float, boards: List[int]) -> List[int]:
    out = []
    rem = int(round(led_len))
    boards = sorted(set(int(b) for b in boards if b>0), reverse=True)
    if not boards: return [rem]
    for b in boards:
        while rem >= b-1e-6:
            out.append(b); rem -= b
    if rem > 0:
        out += [boards[-1]] * int(math.ceil(rem/boards[-1]))
    return out

def timeline_to_df(tl: List[tuple[str,float,str]], boards: List[int]) -> pd.DataFrame:
    rows=[]; cur=0.0; k=0
    for t, ln, lab in tl:
        if t=="LED":
            for b in hungry_fill_led(ln, boards):
                rows.append({"idx":k,"reservation":"LED","type":f"LED {b}","content":"LED","length_mm":mm1(b),"start_mm":mm1(cur),"end_mm":mm1(cur+b)})
                cur += b; k += 1
        else:
            rows.append({"idx":k,"reservation":"COVER","type":lab,"content":"ANC","length_mm":mm1(ln),"start_mm":mm1(cur),"end_mm":mm1(cur+ln)})
            cur += ln; k += 1
    return pd.DataFrame(rows, columns=["idx","reservation","type","content","length_mm","start_mm","end_mm"])

# ================= joins & suspensions =================
def compute_susp(segs: List[float], max_spacing_mm: float, dist_from_join_mm: float) -> tuple[pd.DataFrame,pd.DataFrame]:
    rows=[]; cur=0.0
    for ln in segs:
        start=cur; end=cur+ln
        rows += [{"pos_mm":mm1(start+dist_from_join_mm),"type":"susp"},
                 {"pos_mm":mm1(end-dist_from_join_mm),"type":"susp"}]
        span = ln - 2*dist_from_join_mm
        if max_spacing_mm>0 and span>max_spacing_mm:
            n=int(math.floor(span/max_spacing_mm)); step=span/(n+1)
            for i in range(n):
                rows.append({"pos_mm":mm1(start+dist_from_join_mm+(i+1)*step),"type":"susp"})
        cur=end
    cur=0.0; labs=abc(len(segs)); j=[]
    for i,ln in enumerate(segs):
        j.append({"segment":labs[i],"start_mm":mm1(cur),"end_mm":mm1(cur+ln),"is_join":"join"}); cur+=ln
    return pd.DataFrame(j), pd.DataFrame(rows)

# ================= ancillary breakdown =================
def ancillary_breakdown(anc: List[Dict[str,Any]], unit_costs: Dict[str,float]) -> tuple[pd.DataFrame,float]:
    agg={}
    for a in anc or []:
        code = (str(a.get("code","")).upper().strip() or "ANC")
        cover = float(a.get("cover_len_mm",0) or 0.0)
        key = "MW-0" if (code=="MW" and cover<1.5) else code
        agg.setdefault(key, {"qty":0,"total_cover_mm":0.0,"unit_cost":unit_costs.get(key, unit_costs.get(code,0.0))})
        agg[key]["qty"] += int(a.get("qty",1) or 1)
        agg[key]["total_cover_mm"] += cover*int(a.get("qty",1) or 1)
    rows=[]; total=0.0
    for k,v in sorted(agg.items()):
        ext=v["qty"]*float(v["unit_cost"] or 0.0); total+=ext
        rows.append({"code":k,"qty":v["qty"],"total_cover_mm":int(round(v["total_cover_mm"])),"unit_cost":v["unit_cost"],"ext_cost":ext})
    return pd.DataFrame(rows, columns=["code","qty","total_cover_mm","unit_cost","ext_cost"]), total

def price_break(total_m: float, b1: float, b2: float, b3: float) -> float:
    if total_m>=200: return b3
    if total_m>=100: return b2
    if total_m>=50: return b1
    return 0.0

# ============================= UI =============================
# (removed duplicate) _safe_set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# Source row
c1,c2 = st.columns([1,1])
with c1:
    latest = _latest_flat()
    st.caption("Source")
    st.write(f"Auto config: **{latest.name if latest else '(none)'}**")
with c2:
    up = st.file_uploader("…or drop a runs JSON", type=["json"], label_visibility="collapsed")

cfg={}
if up is not None:
    cfg = _read_json_bytes(up.read())
elif latest:
    cfg = json.loads(latest.read_text(encoding="utf-8"))
else:
    st.warning("No JSON found. Drop a runs JSON, or place a flat_*.json beside this app.")
    st.stop()

runs = _runs(cfg)

# ================= Sidebar =================
st.sidebar.header("Placement overrides")
snap_mode = st.sidebar.selectbox("Run length snap alternatives", ["Closest","Shorter","Longer"], index=0)
short_place = st.sidebar.selectbox("Short piece placement (multi-piece)", ["Start","Centre if odd","End"], index=2)
anc_policy = st.sidebar.selectbox("Ancillary cover policy", ["Maintain","Shorten","Extend"], index=0)

st.sidebar.header("Per-meter rates")
base_pm = float(st.sidebar.number_input("Base $/m", value=150.0, step=1.0))
tech_pm = float(st.sidebar.number_input("Tech add $/m", value=2.0, step=0.5))
finish_pm = float(st.sidebar.number_input("Finish add $/m", value=20.0, step=0.5))

st.sidebar.header("Price breaks (%)")
br50_pct  = int(st.sidebar.number_input("Break 50–100m", value=20, min_value=0, max_value=100, step=1, format="%d"))
br100_pct = int(st.sidebar.number_input("Break 100–200m", value=30, min_value=0, max_value=100, step=1, format="%d"))
br200_pct = int(st.sidebar.number_input("Break 200m+", value=35, min_value=0, max_value=100, step=1, format="%d"))
br50, br100, br200 = br50_pct/100.0, br100_pct/100.0, br200_pct/100.0

st.sidebar.header("Commercial (%)")
markup_pct = int(st.sidebar.number_input("Markup", value=25, min_value=0, max_value=500, step=1, format="%d"))
manual_disc_pct = int(st.sidebar.number_input("Manual discount", value=0, min_value=0, max_value=100, step=1, format="%d"))
markup = markup_pct/100.0
manual_disc = manual_disc_pct/100.0

st.sidebar.header("Ancillary unit costs (optional)")
uc_pir = float(st.sidebar.number_input("PIR unit cost", value=0.0, step=1.0))
uc_spf = float(st.sidebar.number_input("SPF unit cost", value=0.0, step=1.0))
uc_mw0 = float(st.sidebar.number_input("MW-0 unit cost", value=0.0, step=1.0))
unit_costs = {"PIR": uc_pir, "SPF": uc_spf, "MW-0": uc_mw0, "MW": uc_mw0}

st.sidebar.header("Per-run overrides (Admin only)")
st.sidebar.caption("⚠️ Admin only — changing these affects geometry and can ‘break’ a run if misused.")
ovr_final_len = int(st.sidebar.number_input("Override final_run_length_mm (mm)", value=0, step=10, format="%d"))
ovr_board_min = int(st.sidebar.number_input("Override board_family_min_mm (mm)", value=0, step=1, format="%d"))
ovr_end_plate = int(st.sidebar.number_input("Override end_plate_width_mm (mm)", value=0, step=1, format="%d"))

# ====================== Summary ======================
rows=[]
for r in runs:
    rid = r.get("run_id",""); typ=r.get("type",""); qty=int(r.get("qty",1) or 1)
    desired = float(r.get("desired_run_length_mm",0) or 0.0)
    final_len = float(ovr_final_len or r.get("final_run_length_mm", desired) or desired)
    board_min = float(ovr_board_min or r.get("board_family_min_mm", 280) or 280)
    end_plate = float(ovr_end_plate or r.get("end_plate_width_mm",5) or 5)
    max_seg = float(r.get("segment_max_length_mm",3500) or 3500)
    segs = place_short(greedy_segments(final_len, max_seg), short_place)
    seg_pills = chips([(lab, f"{mm1(v)}") for lab,v in zip(abc(len(segs)), segs)])
    anc_pills = "  ".join(f"`{p}`" for p in normalise_anc_pills(r.get("ancillaries_in", [])))
    rows.append({
        "Run ID": rid, "Type": typ, "qty": qty,
        "desired_run_length_mm": mm1(desired),
        "final_run_length_mm": mm1(final_len),
        "board_family_min_mm": mm1(board_min),
        "end_plate_width_mm": mm1(end_plate),
        "segment_qty": len(segs),
        "segment_mm_and_placement": seg_pills,
        "ancillaries": anc_pills,
    })
summary_df = pd.DataFrame(rows, columns=[
    "Run ID","Type","qty","desired_run_length_mm","final_run_length_mm","board_family_min_mm",
    "end_plate_width_mm","segment_qty","segment_mm_and_placement","ancillaries"])
st.subheader("Summary")
st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ====================== Provenance ======================
st.subheader("Provenance (admin overrides vs defaults)")
prov_rows = [
    {"setting":"Run length snap alternatives","value":snap_mode},
    {"setting":"Short piece placement (multi-piece)","value":short_place},
    {"setting":"Ancillary cover policy","value":anc_policy},
    {"setting":"Price break 50–100m","value":f"{br50_pct}%"},
    {"setting":"Price break 100–200m","value":f"{br100_pct}%"},
    {"setting":"Price break 200m+","value":f"{br200_pct}%"},
    {"setting":"Markup","value":f"{markup_pct}%"},
    {"setting":"Manual discount","value":f"{manual_disc_pct}%"},
    {"setting":"Override final_run_length_mm (mm)","value":ovr_final_len},
    {"setting":"Override board_family_min_mm (mm)","value":ovr_board_min},
    {"setting":"Override end_plate_width_mm (mm)","value":ovr_end_plate},
    {"setting":"Base $/m","value":base_pm},
    {"setting":"Tech add $/m","value":tech_pm},
    {"setting":"Finish add $/m","value":finish_pm},
]
prov = pd.DataFrame(prov_rows, columns=["setting","value"])
st.dataframe(prov, use_container_width=True, hide_index=True)

# ====================== Details (one expander per run) ======================
st.subheader("Details")
for r in runs:
    rid=r.get("run_id",""); qty=int(r.get("qty",1) or 1)
    desired=float(r.get("desired_run_length_mm",0) or 0.0)
    final_len=float(ovr_final_len or r.get("final_run_length_mm", desired) or desired)
    board_min=float(ovr_board_min or r.get("board_family_min_mm",280) or 280)
    end_plate=float(ovr_end_plate or r.get("end_plate_width_mm",5) or 5)
    max_seg=float(r.get("segment_max_length_mm",3500) or 3500)
    dist_join=float(r.get("suspension_dist_from_join_mm",15) or 15)
    max_span=float(r.get("suspension_max_spacing_mm",3500) or 3500)
    segs = place_short(greedy_segments(final_len, max_seg), short_place)

    with st.expander(
        f"Run ID {rid} • qty {qty} • Segments {len(segs)}"
        + ("" if not (short_place.lower().startswith('centre') and len(segs)%2==0) else
           " — ‘Centre if odd’ ignored (even segment count)"),
        expanded=False
    ):
        A,B,C = st.columns([2,3,2], gap="large")
        with A:
            st.dataframe(pd.DataFrame({"Segment":abc(len(segs)),"length_mm":[mm1(x) for x in segs]}),
                         use_container_width=True, hide_index=True)

        # Anc policy & timeline/hungry-fill
        anc_bundles, dlt = ancillary_policy(r.get("ancillaries_in", []), board_min, anc_policy)
        vis_final = max(0.0, final_len + dlt)
        available_boards = r.get("available_boards_mm", [])
        tl = timeline_with_covers(vis_final, board_min, anc_bundles, anc_policy)
        if tl and available_boards:
            res_df = timeline_to_df(tl, available_boards)
            res_df.rename(columns={"idx":"slot"}, inplace=True)
        else:
            slots = reserve_grid(vis_final, board_min)
            res_rows=[]
            replaced = {b["slot_index"]:b for b in anc_bundles}
            for i,(s,e) in enumerate(slots):
                if i in replaced and replaced[i]["slots_taken"]>0:
                    lab = "+".join(sorted({it["code"] for it in replaced[i]["items"]})) or "ANC"
                    res_rows.append({"slot":i,"reservation":"COVER","type":lab,"content":"ANC",
                                     "length_mm":mm1(e-s),"start_mm":mm1(s),"end_mm":mm1(e)})
                else:
                    res_rows.append({"slot":i,"reservation":"LED","type":"LED","content":"LED",
                                     "length_mm":mm1(e-s),"start_mm":mm1(s),"end_mm":mm1(e)})
            res_df = pd.DataFrame(res_rows, columns=["slot","reservation","type","content","length_mm","start_mm","end_mm"])

        with B:
            st.dataframe(res_df, use_container_width=True, hide_index=True)

        with C:
            ep = pd.DataFrame([
                {"metric":"total_body_segments_mm","value":mm1(sum(segs))},
                {"metric":"end_plate_width_mm","value":mm1(end_plate)},
                {"metric":"final_run_length_mm","value":mm1(final_len)},
            ])
            st.dataframe(ep, use_container_width=True, hide_index=True)

        # joins & suspensions
        jdf,sdf = compute_susp(segs, max_span, dist_join)
        J,S = st.columns([2,1], gap="large")
        with J:
            st.markdown("**Joins & suspensions**")
            st.dataframe(jdf, use_container_width=True, hide_index=True)
        with S:
            st.dataframe(sdf, use_container_width=True, hide_index=True)

# ====================== Ancillary Breakdown ======================
st.subheader("Ancillary Breakdown")
ablocks=[]; anc_total=0.0
for r in runs:
    df, t = ancillary_breakdown(r.get("ancillaries_in", []), {"PIR":uc_pir,"SPF":uc_spf,"MW-0":uc_mw0,"MW":uc_mw0})
    anc_total += t
    if not df.empty:
        df.insert(0,"Run ID", r.get("run_id",""))
        ablocks.append(df)
anc_table = pd.concat(ablocks, ignore_index=True) if ablocks else pd.DataFrame(columns=["Run ID","code","qty","total_cover_mm","unit_cost","ext_cost"])
st.dataframe(anc_table, use_container_width=True, hide_index=True)

# ---------- BUDGET PRICING (with GM, % formatting) ----------
# (removed duplicate) import pandas as pd

def _fmt_money(x: float) -> str:
    return f"${x:,.2f}"

def _fmt_pct(frac: float) -> str:
    try:
        return f"{float(frac)*100:.1f}%"
    except Exception:
        return "0.0%"

# Your already-computed variables (meters & $ total):
length_m = float(total_final_m)              # e.g., final_run_length_mm/1000 upstream
anc_cost_total = float(ancillary_cost_total) # sum of ancillary $ for the same scope

# Pull config from the flat JSON you loaded earlier
rates = flat_data.get("rates", {})
budget_per_m  = float(rates.get("budget_cost_per_m", 0.0))
tech_per_m    = float(rates.get("tech_add_per_m", 0.0))
finish_per_m  = float(rates.get("finish_add_per_m", 0.0))

commercial = flat_data.get("commercial", {})
markup   = float(commercial.get("markup_recommended", 0.0))   # 0..1
discount = float(commercial.get("discount_pc", 0.0))          # 0..1

pbs = flat_data.get("price_breaks", {})
pb_50_100   = float(pbs.get("50_100m_cost_disc_pc", 0.0))     # 0..1
pb_100_200  = float(pbs.get("100_200m_cost_disc_pc", 0.0))    # 0..1
pb_200_plus = float(pbs.get("200m_plus_cost_disc_pc", 0.0))   # 0..1

def _break_for_length(m: float) -> float:
    if m >= 200.0: return pb_200_plus
    if m >= 100.0: return pb_100_200
    if m >= 50.0:  return pb_50_100
    return 0.0

base_rate_per_m = budget_per_m + tech_per_m + finish_per_m
length_cost = base_rate_per_m * length_m
break_frac = _break_for_length(length_m)
length_cost_after_breaks = length_cost * (1.0 - break_frac)

cost_total = length_cost_after_breaks + anc_cost_total

sell_pre  = cost_total * (1.0 + markup)
sell_post = sell_pre * (1.0 - discount)

gm_pre  = 0.0 if sell_pre  <= 0 else (sell_pre  - cost_total) / sell_pre
gm_post = 0.0 if sell_post <= 0 else (sell_post - cost_total) / sell_post

rows = [
    {"metric": "Total final length (m)",          "value": f"{length_m:,.3f}"},
    {"metric": "Base $/m",                        "value": _fmt_money(budget_per_m)},
    {"metric": "Tech add $/m",                    "value": _fmt_money(tech_per_m)},
    {"metric": "Finish add $/m",                  "value": _fmt_money(finish_per_m)},
    {"metric": "Rate $/m (pre breaks)",           "value": _fmt_money(base_rate_per_m)},
    {"metric": "Length cost (pre breaks)",        "value": _fmt_money(length_cost)},
    {"metric": "Applied price break",             "value": _fmt_pct(break_frac)},
    {"metric": "Length cost (after breaks)",      "value": _fmt_money(length_cost_after_breaks)},
    {"metric": "Ancillary cost (total)",          "value": _fmt_money(anc_cost_total)},
    {"metric": "Cost subtotal",                   "value": _fmt_money(cost_total)},
    {"metric": "Markup",                          "value": _fmt_pct(markup)},
    {"metric": "Sell (pre discount)",             "value": _fmt_money(sell_pre)},
    {"metric": "Discount",                        "value": _fmt_pct(discount)},
    {"metric": "Sell (after discount)",           "value": _fmt_money(sell_post)},
    {"metric": "Gross margin (pre discount)",     "value": _fmt_pct(gm_pre)},
    {"metric": "Gross margin (after discount)",   "value": _fmt_pct(gm_post)},
]

budget_df = pd.DataFrame(rows)
st.dataframe(budget_df, use_container_width=True, hide_index=True)
# ---------- END BUDGET PRICING ----------


# ====================== Downloads (schema/glossary) ======================
schema_csv = io.StringIO()
pd.DataFrame([
    {"key":"run_id","type":"str","purpose":"Run identifier"},
    {"key":"type","type":"str","purpose":"Upstream type label"},
    {"key":"qty","type":"int","purpose":"Quantity"},
    {"key":"desired_run_length_mm","type":"float","purpose":"Design target"},
    {"key":"final_run_length_mm","type":"float","purpose":"Computed/override final"},
    {"key":"board_family_min_mm","type":"float","purpose":"Reservation pitch (min board)"},
    {"key":"end_plate_width_mm","type":"float","purpose":"End plate width (thickness)"},
    {"key":"segment_max_length_mm","type":"float","purpose":"Max segment length"},
    {"key":"suspension_max_spacing_mm","type":"float","purpose":"Max spacing between suspensions"},
    {"key":"suspension_dist_from_join_mm","type":"float","purpose":"Distance of susp from joins"},
    {"key":"available_boards_mm","type":"list[int]","purpose":"Allowed LED board lengths (mm)"},
    {"key":"ancillaries_in","type":"list[dict]","purpose":"[{code, cover_len_mm, qty?, x_mm?}]"},
], columns=["key","type","purpose"]).to_csv(schema_csv, index=False)

glossary_rtf = r"""{\rtf1\ansi
{\b segment_mm_and_placement}: pills for segment labels & lengths (post short-piece rule).\par
{\b Ancillary cover policy}: Maintain (ceil to slots), Shorten (remove slot(s)), Extend (extend by remainder).\par
{\b Reservation fill}: LED hungry-fill (e.g., LED 1120, LED 560) interleaved with COVER (PIR+SPF 90, MW-0).\par
{\b MW-0}: Microwave with cover<1.5mm; occupies no cover slot but has a grid reference.\par
{\b Gross margin}: (Sell - Factory) / Sell.\par
}"""
st.download_button("Download json_schema_dictionary.csv", schema_csv.getvalue(),
                   file_name="json_schema_dictionary.csv", mime="text/csv")
st.download_button("Download segment_solver_glossary.rtf", glossary_rtf.encode("utf-8"),
                   file_name="segment_solver_glossary.rtf", mime="application/rtf")
