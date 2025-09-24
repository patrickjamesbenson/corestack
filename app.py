from __future__ import annotations
from pathlib import Path
import streamlit as st

from utils.paths import (
    ensure_assets,
    ASSETS_DIR, DB_DIR, IES_REPO_DIR, BASELINES_DIR, DOCS_DIR, LOG_DIR, KEYS_DIR,
)

st.set_page_config(page_title="Novon CoreStack", layout="wide")
ensure_assets()

st.title("Novon CoreStack")

st.caption(
    "This workspace ties together photometry preparation, data governance, and luminaire build workflows — "
    "parse and upgrade IES files, validate and sync reference data, preview polar/interpolation behaviour, "
    "and export to the Photometry Repo or download LM-63."
)

# ---- AI policy blurb (updated text you provided) ----
with st.expander("🕊️ Transparency by Design — AI Use Policy", expanded=True):
    st.markdown(
        """
**We make no secret of using AI.** We don’t use it deceptively.  
We use it **transparently** — to:

- Collect and structure shared knowledge  
- Build consistency between humans  
- Create traceable, explainable decisions  

If we can’t show we’re **walking the walk**, we **don’t talk the talk**.
        """
    )

st.divider()

# ---- Assets folder is initialized (unchanged layout; just adds DOCS line) ----
st.subheader("Assets folder is initialized")
rows = [
    ("assets",        ASSETS_DIR),
    ("db",            DB_DIR),
    ("ies_repo",      IES_REPO_DIR),
    ("baselines",     BASELINES_DIR),
    ("docs",          DOCS_DIR),     # NEW line
    ("logs",          LOG_DIR),
    ("keys",          KEYS_DIR),
]
for name, p in rows:
    st.success(f"{name}: {p}")

st.divider()

# ---- Reference (HTML) list (uses files you placed under assets/docs) ----
st.subheader("Reference (HTML)")
docs = [
    ("ANSI/IES LM-63-2019 — Quick Reference", DOCS_DIR / "lm63-2019_flat.html",
     "Inline preview below. (Opens inside this page.)"),
    ("UGR & Polar Curves — Mythbusters", DOCS_DIR / "ugr_polar_mythbusters_flat.html",
     "Inline preview below. (Opens inside this page.)"),
    ("LED Luminaire Testing — Diagram", DOCS_DIR / "led_luminaire_testing_diagram.html",
     "Inline preview below. (Opens inside this page.)"),
]
# Lightweight previewer (same UI you had; no structural change)
qs = st.query_params
open_name = qs.get("doc") if "doc" in qs else None

col1, col2, col3 = st.columns([2, 1, 3])
with col1:
    for title, path, hint in docs:
        st.caption(str(path))
        if path.exists():
            if st.button(("Close preview" if open_name == path.name else "Preview"), key=f"btn_{path.name}"):
                st.query_params["doc"] = "" if open_name == path.name else path.name
                st.rerun()
        else:
            st.button("Preview", key=f"btn_{path.name}", disabled=True)
        st.caption(hint)

with col3:
    if open_name:
        target = DOCS_DIR / open_name
        if target.exists():
            st.markdown(f"**Preview: {open_name}**")
            st.components.v1.iframe(src=target.as_uri(), height=600, scrolling=True)
        else:
            st.info("Drop the file into assets/docs to enable preview.")
    else:
        st.caption("Click *Preview* to view an HTML reference inline.")
