# pages/00__ADMIN__.py
from __future__ import annotations
import streamlit as st
from utils.ui_nav import render_sidebar
from utils.paths import ensure_assets

render_sidebar("Admin")
ensure_assets()

st.title("Admin Home")
st.write("Use the left sidebar to navigate to tools.")
