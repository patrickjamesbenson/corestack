from typing import Literal, List, Dict, Optional

Location = Literal["start", "mid", "end", "mm"]

def compute_power_entry(
    final_run_length_mm: int,
    segments_table: List[Dict],
    location: Location = "start",
    position_mm: Optional[int] = None,
) -> Dict:
    """
    Resolve a power-entry position to an absolute mm and segment-relative mm.
    """
    if location == "mm":
        pos = max(0, min(final_run_length_mm, int(position_mm or 0)))
        loc = "mm"
    elif location == "start":
        pos = 0
        loc = "start"
    elif location == "end":
        pos = int(final_run_length_mm)
        loc = "end"
    else:  # "mid"
        pos = int(round(final_run_length_mm / 2))
        loc = "mid"

    cumulative = 0
    seg_id = None
    seg_offset = 0
    for row in segments_table:
        length = int(row.get("length_mm", 0))
        if cumulative <= pos <= cumulative + length:
            seg_id = row.get("segment_id")
            seg_offset = pos - cumulative
            break
        cumulative += length

    return {
        "power_entry_location": loc,
        "power_entry_position_mm": int(pos),
        "power_entry_segment": seg_id,
        "power_entry_segment_offset_mm": int(seg_offset if seg_id else 0),
    }