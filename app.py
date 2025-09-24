# app.py — Novon CoreStack home
from __future__ import annotations

from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

# --- Paths (native nav; no ui_nav) ---
try:
    from utils.paths import ASSETS_DIR, ensure_assets  # centralized paths
except Exception:
    APP_ROOT = Path(__file__).resolve().parent
    ASSETS_DIR = APP_ROOT / "assets"
    def ensure_assets() -> None:
        for p in (ASSETS_DIR, ASSETS_DIR/"db", ASSETS_DIR/"ies_repo",
                  ASSETS_DIR/"logs", ASSETS_DIR/"keys", ASSETS_DIR/"docs"):
            p.mkdir(parents=True, exist_ok=True)

ensure_assets()
DOCS_DIR = ASSETS_DIR / "docs"

st.set_page_config(page_title="Novon CoreStack", layout="wide")

# -------------------- Header --------------------
st.title("Novon CoreStack")
st.write(
    "This workspace ties together photometry preparation, data governance, "
    "and luminaire build workflows — parse and upgrade IES files, validate and sync reference data, "
    "preview polar/interpolation behavior, and export to the Photometry Repo or download LM-63."
)

# --- AI policy note directly below title (as requested) ---
with st.container():
    st.markdown("### 🕊️ Transparency by Design")
    st.markdown(
        "- We have a live, public-facing **AI Use Policy**.\n"
        "- It outlines **what we do and don’t use AI for**.\n"
        "- It explains **how it helps** and **what humans still do**.\n"
        "- **If we can’t prove we’re walking the walk, we don’t talk the talk.**"
    )

st.divider()

# -------------------- Reference docs --------------------
st.subheader("Reference (HTML)")

# Featured docs (title, filename, tooltip)
DOCS = [
    {
        "title": "ANSI/IES LM-63-2019 — Quick Reference",
        "file": "lm63-2019_flat.html",
        "tip": "Condensed LM-63-2019: required keywords, TILT rules, geometry numbers, and angle/candela layout.",
    },
    {
        "title": "UGR & Polar Curves — Mythbusters",
        "file": "ugr_polar_mythbusters_flat.html",
        "tip": "Glare reality checks, how to read UGR tables, polar-curve shapes, and practical ‘golden rules’.",
    },
    {
        "title": "LED Luminaire Testing — Diagram",
        "file": "led_luminaire_testing_diagram.html",
        "tip": "Annotated diagram of a typical photometry test setup (goniophotometer, planes, angles, zero).",
    },
    # AI Policy card intentionally removed per your instruction.
]

ss = st.session_state
ss.setdefault("preview_doc", None)

def _doc_path(name: str) -> Path:
    return (DOCS_DIR / name).resolve()

# Cards
for i, d in enumerate(DOCS):
    p = _doc_path(d["file"])
    exists = p.exists()
    colA, colB, colC = st.columns([1.6, 0.9, 1.5], gap="small")
    with colA:
        st.markdown(f"**{d['title']}**")
        st.caption(str(p) if exists else f"{p}  (missing)")
    with colB:
        is_current = (ss.get("preview_doc") == d["file"])
        label = "Close preview" if is_current else "Preview"
        if st.button(label, key=f"pv-{i}", help=d["tip"], use_container_width=True, disabled=not exists):
            ss["preview_doc"] = None if is_current else d["file"]
            st.rerun()
    with colC:
        if exists:
            st.caption("Inline preview below. (Opens inside this page.)")
        else:
            st.caption("Drop the file into assets/docs to enable preview.")

st.divider()

# ----------- Preview pane (with safe link handling) -----------
cur = ss.get("preview_doc")
if cur:
    p = _doc_path(cur)
    try:
        raw_html = p.read_text(encoding="utf-8")

        # Inject a tiny helper so clicks behave:
        #  • '#section' anchors -> smooth scroll within preview
        #  • absolute/relative page links -> open in new tab (never hijack the app)
        INJECT = """
        <script>
        (function () {
          document.addEventListener('click', function (e) {
            const a = e.target.closest('a');
            if (!a) return;
            const href = a.getAttribute('href') || '';
            if (href.startsWith('#')) {
              e.preventDefault();
              const id = href.slice(1);
              const el = document.getElementById(id) || document.querySelector('[name="'+id+'"]');
              if (el) el.scrollIntoView({behavior:'smooth', block:'start'});
              return;
            }
            // Any non-hash link: open in new tab to avoid loading app inside iframe
            if (/^(https?:)?\\/\\//.test(href) || href.startsWith('/') || href.startsWith('./') || href.startsWith('../')) {
              e.preventDefault();
              window.open(a.href, '_blank', 'noopener');
            }
          }, {capture:true});
        })();
        </script>
        <style>html { scroll-behavior:smooth; } </style>
        """

        # If the HTML has a head tag, place script right after it; else prepend.
        html = raw_html
        lower = raw_html.lower()
        if "<head" in lower:
            # insert right after the first <head> tag
            idx = lower.index("<head")
            # find end of opening <head ...>
            end = lower.find(">", idx)
            if end != -1:
                html = raw_html[:end+1] + INJECT + raw_html[end+1:]
            else:
                html = INJECT + raw_html
        else:
            html = INJECT + raw_html

        st.markdown(f"##### Preview: {cur}")
        components.html(html, height=900, scrolling=True)
        if st.button("Close preview", key="pv-close-bottom"):
            ss["preview_doc"] = None
            st.rerun()
    except Exception as ex:
        st.error(f"Could not read {p}: {ex}")
else:
    st.caption("Select **Preview** on any reference to view it inline here.")
