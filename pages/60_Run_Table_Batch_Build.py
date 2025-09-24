import streamlit as st
st.set_page_config(page_title="60 • Batch Build (Length Table)", layout="wide")
from utils.header_banner import header_banner as _hdr
from pathlib import Path as _Path
_hdr(_Path(__file__).resolve().parents[1])
st.title("60 • Batch Build (Length Table)")
st.info("Stub page. Hook your exporter here later.")

st.divider()

st.caption("© 2025 LB-Lighting P/L.")