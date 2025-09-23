import streamlit as st
st.set_page_config(page_title="WORKFLOW", layout="wide")
from utils.header_banner import header_banner as _hdr
from pathlib import Path as _Path
_hdr(_Path(__file__).resolve().parents[1])
st.markdown("## ")
