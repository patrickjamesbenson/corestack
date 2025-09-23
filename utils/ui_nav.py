from __future__ import annotations
import streamlit as st

def render_sidebar(current: str = "") -> None:
    st.sidebar.title("Navigation")

    with st.sidebar.expander("Admin", expanded=False if "Admin" in current else False):
        st.page_link("pages/01_DB_Update.py", label="DB & Diagnostics")
        st.page_link("pages/10_Photometry_Preparation_New.py", label="Photometry Preparation")
        st.page_link("pages/90_Provenance_Manager.py", label="Provenance Files Manager")

    with st.sidebar.expander("Workflow", expanded=True if "Workflow" in current else False):
        st.page_link("pages/19__WORKFLOW__.py", label="Workflow Home")
        st.page_link("pages/20_Select_Luminaire_Attributes.py", label="Select Luminaire Attributes")
        st.page_link("pages/30_Outputs_and_Scenes.py", label="Outputs & Scenes")
        st.page_link("pages/40_Build_Luminaire.py", label="Build Luminaire")
        st.page_link("pages/50_Run_Length_Table_Review.py", label="Run Length Table Review")
        st.page_link("pages/60_Batch_Build_Length_Table.py", label="Batch Build Length Table")

    st.sidebar.caption("© 2025 LB Group")
