# app.py
from __future__ import annotations
import streamlit as st
from utils.ui_nav import render_sidebar
from utils.paths import ensure_assets

st.set_page_config(page_title="Novon – Admin Home", layout="wide")
ensure_assets()
render_sidebar("Admin")

st.title("Admin Home")
st.write("Use the left sidebar to navigate.")
st.success("Assets folder is initialized (assets/db, assets/ies_repo, assets/logs, assets/keys).")
