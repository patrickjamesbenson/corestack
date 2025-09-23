from pathlib import Path
import os, sys
from utils.manifest import load_manifest, resolve_legacy_path

APP_ROOT = Path(__file__).resolve().parents[1]
mf = load_manifest(APP_ROOT) or {}
legacy_rel = mf.get("select_luminaire_ui") or ""

# Run LEGACY UI FIRST (so it can call st.set_page_config itself)
if legacy_rel:
    p = resolve_legacy_path(APP_ROOT, legacy_rel)
    if p and p.exists():
        sys.path.insert(0, str(APP_ROOT / "legacy_src"))
        sys.path.insert(0, str(p.parent))
        os.chdir(str(p.parent))
        glb = {"__name__":"__main__","__file__": str(p)}
        # seed a few expected globals for safety (non-destructive)
        glb.setdefault("total_final_run_len_mm", 0)
        glb.setdefault("total_final_m", 0.0)
        src = p.read_text(encoding="utf-8-sig")
        exec(compile(src, str(p), "exec"), glb, glb)
    else:
        import streamlit as st
        st.set_page_config(page_title="20 • Select Luminaire Attributes", layout="wide")
        st.error(f"Legacy UI not found: {legacy_rel}")
else:
    import streamlit as st
    st.set_page_config(page_title="20 • Select Luminaire Attributes", layout="wide")
    st.info("Set `select_luminaire_ui` in manifest.yml (path inside legacy_src)")

# Our header banner AFTER legacy UI
from utils.header_banner import header_banner as _hdr
from pathlib import Path as _Path
_hdr(_Path(__file__).resolve().parents[1])
