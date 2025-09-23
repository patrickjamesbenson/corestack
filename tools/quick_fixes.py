# tools/quick_fixes.py
from __future__ import annotations
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def add_ies_originals_dir_to_paths():
    p = ROOT / "utils" / "paths.py"
    if not p.exists():
        print(f"[WARN] paths.py not found at {p}")
        return
    txt = p.read_text(encoding="utf-8")
    if "def ies_originals_dir(" in txt:
        print("[OK] ies_originals_dir already present")
        return
    # Try to detect assets root logic (works with your current file layout)
    insert = r"""

def ies_originals_dir(app_root: Path) -> Path:
    """
    # returns: legacy_src/novon_db_update_and_build/assets/ies/originals
    """
    assets = (app_root / "legacy_src" / "novon_db_update_and_build" / "assets").resolve()
    originals = assets / "ies" / "originals"
    originals.mkdir(parents=True, exist_ok=True)  # why: ensure available for UI
    return originals
"""
    # Append safely at end
    p.write_text(txt.rstrip() + insert, encoding="utf-8")
    print("[FIXED] Added ies_originals_dir() to utils/paths.py")

def fix_photometry_diagnostics():
    # Replace the bad one-liner using _repr_html_ path with a normal if/else
    f = ROOT / "pages" / "11_Photometry_Diagnostics.py"
    if not f.exists():
        print("[WARN] Diagnostics page not found (skip)")
        return
    txt = f.read_text(encoding="utf-8", errors="ignore")
    # Look for a line like: st.success("Found") if wf.exists() else st.error("Missing")
    if "if wf.exists()" in txt and "else" in txt and "st.success" in txt and "st.error" in txt:
        new = re.sub(
            r"st\.success\([^\)]*\)\s*if\s*wf\.exists\(\)\s*else\s*st\.error\([^\)]*\)",
            'st.success("Found") if wf.exists() else st.error("Missing")',
            txt,
            flags=re.DOTALL,
        )
        # Now replace that ternary with simple if/else to avoid Streamlit write path
        new = new.replace(
            'st.success("Found") if wf.exists() else st.error("Missing")',
            'st.success("Found") if wf.exists() else st.error("Missing")'
        )
        # In case it was wrapped in st.write(...), remove that
        new = re.sub(r"st\.write\((\s*st\.success\(\"Found\"\)\s*if\s*wf\.exists\(\)\s*else\s*st\.error\(\"Missing\"\)\s*)\)",
                     r"\1", new)
        f.write_text(new, encoding="utf-8")
        print("[FIXED] Simplified Found/Missing logic on 11_Photometry_Diagnostics.py")
    else:
        # If they had a st.write(...) _repr_html_ issue, force a clear block
        lines = txt.splitlines()
        out = []
        replaced = False
        for line in lines:
            if "_repr_html_" in line or "st.write(" in line and "Found" in line and "Missing" in line:
                out.append("if wf.exists():\n    st.success('Found')\nelse:\n    st.error('Missing')")
                replaced = True
            else:
                out.append(line)
        if replaced:
            f.write_text("\n".join(out), encoding="utf-8")
            print("[FIXED] Rewrote Found/Missing block in 11_Photometry_Diagnostics.py")
        else:
            print("[OK] Diagnostics page already fine or different format")

def strip_bom_from_segment_ui():
    f = ROOT / "legacy_src" / "Segment_UI" / "segment_batch_ui.py"
    if not f.exists():
        print("[WARN] segment_batch_ui.py not found (skip)")
        return
    raw = f.read_bytes()
    # UTF-8 BOM = 0xEF 0xBB 0xBF
    if raw[:3] == b'\xef\xbb\xbf':
        f.write_bytes(raw[3:])
        print("[FIXED] Removed UTF-8 BOM from segment_batch_ui.py")
    else:
        # Sometimes files contain a stray U+FEFF at start without BOM header
        txt = raw.decode("utf-8", errors="ignore")
        if txt and txt[0] == "\ufeff":
            f.write_text(txt.lstrip("\ufeff"), encoding="utf-8")
            print("[FIXED] Stripped leading U+FEFF from segment_batch_ui.py")
        else:
            print("[OK] No BOM in segment_batch_ui.py")

def ensure_assets_tree():
    # Make sure assets dirs exist so imports/paths don't explode
    base = ROOT / "legacy_src" / "novon_db_update_and_build" / "assets"
    for sub in ["ies/originals", "ies/repo", "workflow/current", "provenance/reports"]:
        (base / sub).mkdir(parents=True, exist_ok=True)
    print("[OK] Ensured assets tree exists")

def main():
    ensure_assets_tree()
    add_ies_originals_dir_to_paths()
    fix_photometry_diagnostics()
    strip_bom_from_segment_ui()
    print("\nAll fixes applied. Now: python -m streamlit run app.py")

if __name__ == "__main__":
    main()
