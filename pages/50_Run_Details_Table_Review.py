import streamlit as st
st.set_page_config(page_title="50 • Run Length Table Review", layout="wide")
from utils.header_banner import header_banner as _hdr
from pathlib import Path as _Path
_hdr(_Path(__file__).resolve().parents[1])
st.title("50 • Run Length Table Review")
st.info("Stub page. Your legacy review UI can be wrapped similarly if needed.")

st.divider()

st.caption("© 2025 LB-Lighting P/L.")