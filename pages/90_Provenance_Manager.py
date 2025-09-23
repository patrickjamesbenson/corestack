# pages/90_Provenance_Manager.py
from __future__ import annotations
import streamlit as st
from utils.ui_nav import render_sidebar
from utils.paths import workflow_current_path, ies_repo_current_path

render_sidebar("Admin / Provenance")
st.title("Provenance Files Manager")

col1, col2 = st.columns(2)
with col1:
    st.caption("DB JSON path")
    st.code(str(workflow_current_path()))
with col2:
    st.caption("IES Repo path")
    st.code(str(ies_repo_current_path()))

st.info("Stub page: the manager UI will be completed after Photometry/Metadata are locked in.")
