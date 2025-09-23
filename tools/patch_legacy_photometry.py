# tools/patch_legacy_photometry.py
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy_src"
CANDIDATES = list(LEGACY.rglob("*.py"))

def patch_file(path: Path) -> bool:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    changed = False

    # ---- Fix: FileGenFlags(..., lm63_2019=...) -> set attribute after init
    def repl_filegen(m):
        inner = m.group("inner")
        # Remove lm63_2019 kw if present
        inner2 = re.sub(r"""\s*,\s*lm63_2019\s*=\s*[^,)\s]+""", "", inner, count=1)
        inner2 = re.sub(r"""\s*lm63_2019\s*=\s*[^,)\s]+\s*,\s*""", "", inner2, count=1)
        # Try to capture the value (fallback True)
        vm = re.search(r"""lm63_2019\s*=\s*(?P<val>True|False|[A-Za-z_][A-Za-z0-9_]*|\d+)""", inner)
        val = vm.group("val") if vm else "True"
        return f"(_FLG := FileGenFlags{inner2}); setattr(_FLG, 'lm63_2019', {val}); _FLG"

    new = re.sub(r"""FileGenFlags\s*(?P<inner>\([^\)]*\))""", repl_filegen, txt)
    if new != txt:
        txt = new
        changed = True

    # ---- Guard set_page_config
    if "set_page_config(" in txt and "_safe_set_page_config" not in txt:
        txt = (
            "import streamlit as st\n"
            "def _safe_set_page_config(*a, **k):\n"
            "    try:\n"
            "        return st.set_page_config(*a, **k)\n"
            "    except Exception:\n"
            "        return None\n"
        ) + txt.replace("st.set_page_config(", "_safe_set_page_config(")
        changed = True

    # ---- Neutralize experimental_rerun (avoid tug-of-war)
    if "experimental_rerun(" in txt and "_safe_rerun" not in txt:
        txt = (
            "import streamlit as st\n"
            "def _safe_rerun():\n"
            "    try:\n"
            "        return st.rerun()\n"
            "    except Exception:\n"
            "        return None\n"
        ) + txt.replace("st.experimental_rerun()", "_safe_rerun()")
        changed = True

    if changed:
        path.write_text(txt, encoding="utf-8")
    return changed

def main():
    touched = sum(1 for p in CANDIDATES if patch_file(p))
    print(f"[OK] Legacy photometry patched in {touched} file(s).")

if __name__ == "__main__":
    main()
