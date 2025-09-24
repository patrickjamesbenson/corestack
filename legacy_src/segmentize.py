# core_engine/segmentize.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Tuple
import math

# =========================
# Dataclasses / options
# =========================
@dataclass
class Rules:
    # geometry
    std_end_plate_width_mm: float = 5.0
    segment_max_length_mm: float = 3500.0
    segment_min_aesthetic_length_mm: float = 900.0

    # reservations / boards
    reservation_pitch_fallback_mm: float = 280.0

    # suspensions
    suspension_distance_from_join_mm: float = 15.0
    suspension_max_spacing_mm: float = 3500.0

    # pricing (placeholders retained so UI can read them if needed)
    budget_cost_per_m: float = 0.0
    tech_add_per_m: float = 0.0
    finish_add_per_m: float = 0.0
    pb_50_100: float = 0.2
    pb_100_200: float = 0.25
    pb_200_plus: float = 0.3


@dataclass
class SegmentizeOptions:
    # run-length snap
    snap_mode: Literal["Closest","Shorter","Longer"] = "Closest"
    both_prefer: Literal["Shorter","Longer"] = "Longer"  # tie for Closest

    # short-piece placement (UI “Centre if odd” -> pass "Middle")
    short_segment_placement_2to3: Literal["Start","Middle","End"] = "End"
    multi_short_piece_policy_4plus: str = "CenterIfOddElseEnd"

    # overrides
    segment_max_length_override_mm: Optional[float] = None

    # ancillary policy across the run
    ancillary_policy: Literal["Maintain","Shorten","Extend"] = "Maintain"

    # admin overrides
    min_board_mm_override: Optional[float] = None
    endplate_mm_per_end_override: Optional[float] = None


# =========================
# Helpers
# =========================
def _f(x, d=0.0): 
    try: return float(x)
    except: return d

def _i(x, d=0): 
    try: return int(x)
    except: return d

def _normalize_anc(anc_in: Any) -> List[Dict[str, Any]]:
    """
    Normalises ancillary input into a pill-friendly shape:
      [{"code": "PIR", "cover_len_mm": 45, "qty": 1, "x_mm": 6000.0}, ...]
    Special rule:
      - “MW” with cover_len_mm < 1.5 -> zero-space “MW-0” (doesn’t consume slot length)
    Accepts:
      - list of dicts
      - dict {code: {...}} or {code: cover_mm}
    """
    out: List[Dict[str,Any]] = []
    if not anc_in:
        return out

    if isinstance(anc_in, list):
        it = anc_in
    elif isinstance(anc_in, dict):
        it = [{"code":k, **v} if isinstance(v, dict) else {"code":k, "cover_len_mm":v} for k,v in anc_in.items()]
    else:
        return out

    for a in it:
        if not isinstance(a, dict):
            continue
        code = str(a.get("code") or a.get("type") or a.get("name") or "ANC").upper()
        cover = _f(a.get("cover_len_mm") or a.get("cover_mm") or a.get("min_mm") or a.get("length_mm") or 0.0, 0.0)
        qty = _i(a.get("qty") or a.get("quantity") or 1, 1)
        x = a.get("x_mm") or a.get("run_x_mm")

        # MW rule: if effectively zero (<1.5mm), do not occupy length
        if code.startswith("MW") and cover < 1.5:
            code = "MW-0"
            cover = 0.0

        out.append({"code": code, "cover_len_mm": cover, "qty": qty, "x_mm": x})
    return out


def _final_length(desired_mm: float, pitch_mm: float, endplate_mm: float,
                  snap_mode: str, both_prefer: str) -> Tuple[float,int]:
    """
    final = 2*endplate + k*pitch; choose integer k per snap rule.
    Returns (final_length_mm, k)
    """
    base = max(desired_mm - 2*endplate_mm, 0.0)
    if pitch_mm <= 0:
        k = max(1, round(base))
        return 2*endplate_mm + k, k

    k_float = base / pitch_mm
    if snap_mode == "Shorter":
        k = max(1, math.floor(k_float))
    elif snap_mode == "Longer":
        k = max(1, math.ceil(k_float))
    else:  # Closest (with tie-breaker)
        kd = max(1, math.floor(k_float))
        ku = max(1, math.ceil(k_float))
        down_err = abs((2*endplate_mm + kd*pitch_mm) - desired_mm)
        up_err   = abs((2*endplate_mm + ku*pitch_mm) - desired_mm)
        if down_err == up_err:
            k = ku if both_prefer == "Longer" else kd
        else:
            k = kd if down_err < up_err else ku

    return 2*endplate_mm + k*pitch_mm, k


def _split_segments(body_len: float, max_seg: float, placement: str) -> List[float]:
    """
    Split body length into n nearly-equal segments (<= max_seg).
    Keep a single “special” segment (base + remainder) and place it by rule:
      Start / Middle(centre if odd) / End
    """
    if body_len <= 0:
        return []
    n = max(1, math.ceil(body_len / max_seg))
    base = math.floor(body_len / n)
    rem = body_len - base * n

    segs = [base] * n
    special_len = base + rem

    # default “End”; “Middle” only when n is odd (as requested)
    idx = n - 1
    if placement == "Start":
        idx = 0
    elif placement == "Middle" and (n % 2 == 1):
        idx = n // 2

    segs[idx] = special_len
    return segs


def _reservations(body_len: float, pitch: float, endplate_per_end: float) -> List[Dict[str, Any]]:
    """Create LED reservation slots across body."""
    out: List[Dict[str,Any]] = []
    if pitch <= 0 or body_len <= 0:
        return out
    k = int(round(body_len / pitch))
    run_offset = endplate_per_end
    for i in range(k):
        start = run_offset + i*pitch
        end = start + pitch
        centre = (start + end) * 0.5
        out.append({
            "slot": i,
            "start_mm": round(start, 1),
            "end_mm": round(end, 1),
            "centre_mm": round(centre, 1),
            "content": "LED"
        })
    return out


def _find_nearest_slot(x_mm: float, slots: List[Dict[str,Any]]) -> int:
    best = 0
    bestd = 1e18
    for s in slots:
        d = abs(x_mm - s["centre_mm"])
        if d < bestd:
            bestd, best = d, s["slot"]
    return best


def _apply_ancillaries(slots: List[Dict[str,Any]], anc: List[Dict[str,Any]],
                       pitch: float, policy: str) -> Tuple[List[Dict[str,Any]], float, List[Dict[str,Any]]]:
    """
    Apply ancillaries to reservation slots and compute delta length per policy.
    Policies:
      - Maintain: replace slots as needed; run length unchanged
      - Shorten:   replace slots; shorten by (slots*pitch - total_cover)
      - Extend:    replace floor(total_cover/pitch) slots, extend by (total_cover % pitch)
                   (extension <= one pitch)
    MW-0 have cover_len_mm=0 -> never consume length.
    """
    if not slots:
        return slots, 0.0, []

    # default to run centre when no x supplied
    body_start = slots[0]["start_mm"]
    body_end = slots[-1]["end_mm"]
    run_centre = (body_start + body_end) * 0.5

    # explode quantities and map each to nearest slot
    items: List[Tuple[int, float, str]] = []  # (slot_index, cover_len, code)
    anc_xy: List[Dict[str,Any]] = []
    total_cover = 0.0

    for a in anc:
        cover = float(a.get("cover_len_mm", 0.0))
        qty = int(a.get("qty", 1))
        code = a.get("code", "ANC")
        for _ in range(qty):
            cx = a.get("x_mm", None)
            if cx is None:
                cx = run_centre
            slot_idx = _find_nearest_slot(cx, slots)
            items.append((slot_idx, cover, code))
            total_cover += cover

    # group by slot for visual/pill aggregation
    by_slot: Dict[int, List[Tuple[float,str]]] = {}
    for slot_idx, cover, code in items:
        by_slot.setdefault(slot_idx, []).append((cover, code))

    # apply to slots, record ancillary XY
    used_slots = set()
    for slot_idx, lst in by_slot.items():
        coversum = sum(c for c,_ in lst)
        types = "+".join(sorted(c for _,c in lst))

        if coversum <= pitch:
            slots[slot_idx]["content"] = f"ANC:{types}"
            used_slots.add(slot_idx)
        else:
            need = math.ceil(coversum / pitch)
            start = max(0, slot_idx - need//2)
            end = min(len(slots), start + need)
            for s in range(start, end):
                slots[s]["content"] = f"ANC:{types}"
                used_slots.add(s)

        anc_xy.append({
            "type": types,
            "cover_len_mm": round(coversum, 1),
            "run_x_mm": slots[slot_idx]["centre_mm"]
        })

    # policy delta length (aggregate across run)
    covers_effective = total_cover
    if policy == "Maintain":
        delta = 0.0
    elif policy == "Shorten":
        slots_removed = math.ceil(covers_effective / pitch) if covers_effective > 0 else 0
        slack = slots_removed * pitch - covers_effective
        delta = -slack
    else:  # Extend
        slots_removed = math.floor(covers_effective / pitch) if covers_effective > 0 else 0
        overflow = covers_effective - slots_removed * pitch
        # never extend more than one pitch
        delta = overflow if overflow <= pitch else (overflow % pitch)

    return slots, delta, anc_xy


def _map_slots_to_segments(slots: List[Dict[str,Any]], segs: List[float], endplate_per_end: float) -> None:
    # annotate each slot with segment ID (A, B, C...) by its centre position
    pos = endplate_per_end
    boundaries: List[Tuple[float,float,str]] = []
    for i, L in enumerate(segs):
        start = pos
        end = pos + L
        boundaries.append((start, end, chr(ord('A') + i)))
        pos = end

    for s in slots:
        c = s["centre_mm"]
        sid = boundaries[-1][2]
        for a,b,lab in boundaries:
            if a <= c <= b:
                sid = lab
                break
        s["segment_id"] = sid


def _hungry_fill(slots: List[Dict[str,Any]], board_sizes: List[float]) -> List[Dict[str,Any]]:
    """
    Merge consecutive LED slots into the largest available boards.
    Emits rows like {"start_mm":..., "end_mm":..., "content":"LED_1120"} and preserves ANC bands.
    """
    out: List[Dict[str,Any]] = []
    if not slots:
        return out

    sizes = sorted([float(s) for s in board_sizes if float(s) > 0], reverse=True)
    if not sizes:
        return out

    pitch = slots[0]["end_mm"] - slots[0]["start_mm"]

    band_start = None
    band_len = 0.0

    for s in slots + [None]:
        if s is not None and s.get("content") == "LED":
            if band_start is None:
                band_start = s["start_mm"]
            band_len += pitch
            continue

        # close LED band
        if band_start is not None:
            rem = band_len
            cursor = band_start
            while rem > 1e-6:
                taken = None
                for b in sizes:
                    if b <= rem + 1e-6:
                        taken = b
                        break
                if taken is None:
                    taken = sizes[-1]
                out.append({
                    "start_mm": round(cursor, 1),
                    "end_mm": round(cursor + taken, 1),
                    "content": f"LED_{int(round(taken))}"
                })
                cursor += taken
                rem -= taken

            band_start = None
            band_len = 0.0

        # emit ANC slot as-is
        if s is not None and s.get("content","").startswith("ANC"):
            out.append({
                "start_mm": s["start_mm"],
                "end_mm": s["end_mm"],
                "content": s["content"]
            })

    return out


def _segments_table(segs: List[float]) -> List[Dict[str, Any]]:
    rows = []
    for i, L in enumerate(segs):
        rows.append({"Segment": chr(ord('A') + i), "length_mm": round(float(L), 1)})
    return rows


def _joins_and_suspensions(segs: List[float], endplate: float,
                           final_run_len: float,
                           dist_from_join: float,
                           max_span: float) -> List[Dict[str,Any]]:
    # joins per segment, suspensions near joins with mid-spans if gap > max_span
    rows: List[Dict[str,Any]] = []

    # Joins (segment start/end positions)
    pos = endplate
    seg_bounds: List[Tuple[str, float, float]] = []
    for i, L in enumerate(segs):
        sid = chr(ord('A') + i)
        seg_bounds.append((sid, pos, pos + L))
        rows.append({"kind":"segment", "segment": sid, "start_mm": round(pos,1), "end_mm": round(pos+L,1)})
        pos += L

    # Base suspension points near joins
    base_pts: List[float] = []
    for _, a, b in seg_bounds:
        base_pts.append(a + dist_from_join)
        base_pts.append(b - dist_from_join)

    # Add mid points if spans exceed max_span
    body_start = endplate
    body_end = final_run_len - endplate

    base_pts = sorted([p for p in base_pts if body_start <= p <= body_end])
    all_pts: List[float] = []
    last = None
    for p in base_pts:
        if last is None:
            all_pts.append(p)
            last = p
            continue
        gap = p - last
        if gap > max_span:
            need = math.ceil(gap / max_span) - 1
            step = gap / (need + 1)
            for k in range(1, need+1):
                all_pts.append(last + k*step)
        all_pts.append(p)
        last = p

    for x in sorted(all_pts):
        rows.append({"kind":"suspension", "run_x_mm": round(x,1)})

    return rows


# =========================
# Public API
# =========================
def segmentize_from_contract(contract: Dict[str,Any], rules: Rules, options: SegmentizeOptions) -> Dict[str,Any]:
    """
    contract keys expected:
      desired_run_length_mm (float)
      end_plate_width_mm (float)  [optional, uses rules.std_end_plate_width_mm otherwise]
      board_family_min_mm (float) [optional, uses rules.reservation_pitch_fallback_mm otherwise]
      segment_max_length_mm (float) [optional, uses rules.segment_max_length_mm otherwise]
      available_boards_mm (list[float]) e.g. [70,140,280,560,1120,1400]
      ancillaries_in (list/dict) -> normalised by _normalize_anc()
    """
    desired = _f(contract.get("desired_run_length_mm") or contract.get("required_length_mm"))
    endplate = _f(options.endplate_mm_per_end_override or
                  contract.get("end_plate_width_mm") or
                  contract.get("endplate_mm") or
                  rules.std_end_plate_width_mm,
                  rules.std_end_plate_width_mm)

    pitch = _f(options.min_board_mm_override or
               contract.get("board_family_min_mm") or
               rules.reservation_pitch_fallback_mm,
               rules.reservation_pitch_fallback_mm)

    max_seg = _f(options.segment_max_length_override_mm or
                 contract.get("segment_max_length_mm") or
                 rules.segment_max_length_mm,
                 rules.segment_max_length_mm)

    # available boards for hungry fill
    avail = contract.get("available_boards_mm") or \
            contract.get("board_family_availability_mm") or \
            [pitch, pitch*2, pitch*4]
    board_sizes = [float(x) for x in avail if _f(x) > 0]

    # snap to compute k and final length (before ancillary policy)
    final_before_anc, k = _final_length(desired, pitch, endplate, options.snap_mode, options.both_prefer)
    body_len = final_before_anc - 2*endplate

    # split body into segments with short-piece placement
    segs = _split_segments(body_len, max_seg, options.short_segment_placement_2to3)

    # create reservation slots
    slots = _reservations(body_len, pitch, endplate)

    # normalise ancillaries
    anc = _normalize_anc(contract.get("ancillaries_in") or contract.get("ancillaries"))

    # apply ancillaries to slots and compute length delta
    slots2, delta_len, anc_xy = _apply_ancillaries(slots, anc, pitch, options.ancillary_policy)
    final_len = final_before_anc + delta_len

    # map each reservation slot to segment ID
    _map_slots_to_segments(slots2, segs, endplate)

    # build “segment reservation addresses” table
    seg_res_rows = []
    for s in slots2:
        seg_res_rows.append({
            "slot": s["slot"],
            "start_mm": s["start_mm"],
            "end_mm": s["end_mm"],
            "segment_id": s.get("segment_id"),
            "content": s.get("content","LED")
        })

    # hungry fill LED bands into board sizes (keeps ANC spans)
    reservation_fill = _hungry_fill(slots2, board_sizes)

    # joins & suspensions
    joins_suspensions = _joins_and_suspensions(
        segs=segs,
        endplate=endplate,
        final_run_len=final_len,
        dist_from_join=rules.suspension_distance_from_join_mm,
        max_span=rules.suspension_max_spacing_mm
    )

    # end plates & totals record
    endp = {
        "end_plate_width_mm": endplate,
        "desired_run_length_mm": round(desired, 1),
        "final_run_length_mm": round(final_len, 1),
        "segment_qty": len(segs),
        "board_family_min_mm": pitch,
    }

    # ancillary XY annotate segment id for convenience
    for row in anc_xy:
        cx = row["run_x_mm"]
        pos = endplate
        row["segment_id"] = None
        for i, L in enumerate(segs):
            if pos <= cx <= pos + L:
                row["segment_id"] = chr(ord('A') + i)
                break
            pos += L

    return {
        # summary
        "final_run_length_mm": round(final_len, 1),
        "segment_mm_and_placement": [{"id": chr(ord('A')+i), "mm": round(L,1)} for i,L in enumerate(segs)],

        # details
        "segments_table": _segments_table(segs),
        "segment_reservations": seg_res_rows,
        "reservation_fill": reservation_fill,
        "endplates_totals": [endp],
        "joins_suspensions": joins_suspensions,

        # ancillaries
        "ancillary_xy": anc_xy,
        "ancillaries_in": anc,
    }


# Legacy signature kept for older callers
def segmentize(length_mm: float,
               reservation_pitch_mm: float,
               rules: Rules,
               options: SegmentizeOptions,
               ancillaries: Optional[List[Dict[str,Any]]] = None) -> Dict[str,Any]:
    contract = {
        "desired_run_length_mm": length_mm,
        "board_family_min_mm": reservation_pitch_mm,
        "end_plate_width_mm": rules.std_end_plate_width_mm,
        "ancillaries_in": ancillaries or [],
    }
    return segmentize_from_contract(contract, rules, options)

