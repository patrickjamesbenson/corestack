@echo off
setlocal
set "WS=C:\Users\Patrick\Desktop\CoreStack_workspace"
set "NOVON_WORKFLOW_PATH=%WS%\legacy_src\ies_norm\novon_workflow.json"
cd /d "%WS%"
python -m streamlit run app.py
