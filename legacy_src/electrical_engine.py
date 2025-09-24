from typing import Dict, Any, List, Optional
import math

def _get(d: Dict, *path, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def _count_boards_by_type(reservation_fill: List[Dict]) -> Dict[str, int]:
    counts: Dict[str,int] = {}
    for cell in reservation_fill or []:
        content = cell.get("content","")
        if isinstance(content, str) and content.startswith("LED_"):
            counts[content] = counts.get(content, 0) + 1
    return counts

def _count_boards_by_type_per_segment(segment_reservations: Dict[str, List[Dict]]) -> Dict[str, Dict[str,int]]:
    per_seg: Dict[str, Dict[str,int]] = {}
    for seg_id, rows in (segment_reservations or {}).items():
        counts: Dict[str,int] = {}
        for cell in rows:
            content = cell.get("content","")
            if isinstance(content, str) and content.startswith("LED_"):
                counts[content] = counts.get(content, 0) + 1
        per_seg[seg_id] = counts
    return per_seg

def _snap_current(ma: float, step: int, mode: str) -> int:
    if step <= 0:
        return int(round(ma))
    q = ma / step
    if mode == "down":
        return int(math.floor(q) * step)
    if mode == "nearest":
        return int(round(q) * step)
    return int(math.ceil(q) * step)

def build_run_electrical(contract: Dict[str, Any], physical: Dict[str, Any], catalog: Dict[str, Any]) -> Dict[str, Any]:
    electrical: Dict[str, Any] = {}
    pol = {
        "snap_mode": _get(contract, "policies", "snap_mode", default="up"),
        "allow_cross_wiring": bool(_get(contract, "policies", "allow_cross_wiring", default=False)),
        "max_cross_wire_mm": int(_get(contract, "policies", "max_cross_wire_mm", default=0) or 0),
        "max_capacity_pct": float(_get(contract, "policies", "max_capacity_pct", default=90.0)),
        "vf_margin_v": float(_get(contract, "policies", "vf_margin_v", default=0.0)),
    }

    reservation_fill = physical.get("reservation_fill") or []
    segment_reservations = physical.get("segment_reservations") or {}
    board_counts_run = _count_boards_by_type(reservation_fill)
    board_counts_per_segment = _count_boards_by_type_per_segment(segment_reservations)

    driver_mode = _get(contract, "electrical", "driver_mode", default="unknown")
    target_lm_run = float(_get(contract, "photometry", "lumens_target_run", default=0.0) or 0.0)

    boards_catalog: Dict[str, Any] = catalog.get("boards", {})
    ecg_catalog: Dict[str, Any] = catalog.get("drivers", {})

    if not board_counts_run:
        return {"status": "INSUFFICIENT_DATA", "reason": "No LED boards detected in reservation_fill"}

    lumens_achieved = 0.0
    for board_code, qty in board_counts_run.items():
        lm = float(_get(boards_catalog, board_code, "lumens_at_nominal_ma", default=0) or 0) * qty
        lumens_achieved += lm
    lumens_target = target_lm_run or 0.0
    delta_pct = (lumens_achieved - lumens_target) / lumens_target * 100.0 if lumens_target > 0 else 0.0

    i_step = int(_get(contract, "electrical", "driver_i_step_ma", default=50) or 50)
    i_guess = float(_get(contract, "electrical", "requested_current_ma", default=350.0) or 350.0)
    current_snapped_ma = _snap_current(i_guess, i_step, pol["snap_mode"])

    vf_per_segment: Dict[str, Any] = {}
    some_vf_known = False
    for seg_id, counts in board_counts_per_segment.items():
        strings = 0
        vf_per_string = 0.0
        for board_code, qty in counts.items():
            strings_per_board = int(_get(boards_catalog, board_code, "strings_per_board", default=0) or 0)
            strings += strings_per_board * qty
            off = _get(boards_catalog, board_code, "vf_offset_v", default=None)
            grad = _get(boards_catalog, board_code, "vf_gradient_v_per_ma", default=None)
            if off is not None and grad is not None:
                some_vf_known = True
                vf_per_string = max(vf_per_string, float(off) + float(grad) * current_snapped_ma)
        vf_total = vf_per_string * strings
        model = _get(contract, "electrical", "preferred_ecg_model", default=None)
        vmin = vmax = None
        if model and model in ecg_catalog:
            vmin = float(_get(ecg_catalog, model, "v_min", default=0.0) or 0.0)
            vmax = float(_get(ecg_catalog, model, "v_max", default=0.0) or 0.0)
        window_ok = None
        if vmin is not None and vmax is not None and some_vf_known:
            window_ok = (vf_total + pol["vf_margin_v"]) >= vmin and (vf_total - pol["vf_margin_v"]) <= vmax

        vf_per_segment[seg_id] = {
            "strings_in_series": int(strings),
            "vf_per_string_v": round(vf_per_string, 3),
            "vf_total_v": round(vf_total, 3),
            "window_ok": window_ok,
        }

    ecg_plan: List[Dict[str, Any]] = []
    capacity_ok_all = True
    model = _get(contract, "electrical", "preferred_ecg_model", default=None)
    for seg_id, counts in board_counts_per_segment.items():
        watts = 0.0
        for board_code, qty in counts.items():
            w_nom = float(_get(boards_catalog, board_code, "watts_at_nominal_ma", default=0.0) or 0.0)
            watts += w_nom * qty
        if not model or model not in ecg_catalog:
            ecg_plan.append({
                "segment_id": seg_id, "ecg_model": None, "count": None,
                "placement_mm": [], "cross_wire": False,
                "derated_w": None, "capacity_pct": None, "headroom_pct": None,
            })
            capacity_ok_all = False
            continue

        drv = ecg_catalog[model]
        derated_w = float(drv.get("derated_w") or drv.get("nominal_w") or 0.0)
        if derated_w <= 0:
            ecg_plan.append({
                "segment_id": seg_id, "ecg_model": model, "count": 1,
                "placement_mm": [], "cross_wire": False,
                "derated_w": derated_w, "capacity_pct": None, "headroom_pct": None
            })
            capacity_ok_all = False
            continue

        count = max(1, int(math.ceil((watts * 100.0) / (pol["max_capacity_pct"] * derated_w))))
        capacity_pct = (watts / (count * derated_w)) * 100.0
        headroom_pct = 100.0 - capacity_pct
        if capacity_pct > pol["max_capacity_pct"] + 1e-9:
            capacity_ok_all = False

        ecg_plan.append({
            "segment_id": seg_id, "ecg_model": model, "count": int(count),
            "placement_mm": [], "cross_wire": False,
            "derated_w": round(derated_w, 2), "capacity_pct": round(capacity_pct, 1), "headroom_pct": round(headroom_pct, 1),
        })

    vf_window_ok = all(v.get("window_ok") in (True, None) for v in vf_per_segment.values())
    capacity_ok = capacity_ok_all
    overdrive_ok = current_snapped_ma <= int(_get(contract, "electrical", "max_allowed_ma", default=current_snapped_ma) or current_snapped_ma)
    summary = "PASS" if vf_window_ok and capacity_ok and overdrive_ok else "CHECK"

    status = "OK" if summary == "PASS" else "PENDING"
    electrical.update({
        "status": status,
        "driver_current_ma_snapped": int(current_snapped_ma),
        "driver_mode": driver_mode,
        "vf_per_segment": vf_per_segment,
        "ecg_plan": ecg_plan,
        "warranty_flags": {
            "vf_window_ok": vf_window_ok, "capacity_ok": capacity_ok,
            "overdrive_ok": overdrive_ok, "summary_status": summary
        },
        "power_totals": {
            "watts_run": round(sum((p.get("derated_w") or 0) * (p.get("capacity_pct") or 0) / 100.0 for p in ecg_plan if p.get("capacity_pct") is not None), 2),
            "watts_per_segment": {},
            "lumens_achieved": round(lumens_achieved, 1),
            "lumens_target": round(lumens_target, 1),
            "delta_pct": round(delta_pct, 1),
        }
    })
    return electrical