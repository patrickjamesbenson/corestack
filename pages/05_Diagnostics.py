from __future__ import annotations
import streamlit as st
from utils.ui_nav import render_sidebar

render_sidebar("Admin / Diagnostics")
st.title("Diagnostics")

st.info("Diagnostics have moved to **Admin → DB & Diagnostics**.")
st.page_link("pages/01_DB_Update.py", label="Go to DB & Diagnostics →")
