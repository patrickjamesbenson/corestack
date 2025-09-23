@echo off
setlocal
set "WS=C:\Users\Patrick\Desktop\CoreStack_workspace"

REM --- ensure folders exist
if not exist "%WS%\legacy_src\novon_db_updater\assets" mkdir "%WS%\legacy_src\novon_db_updater\assets"
if not exist "%WS%\legacy_src\Ouputs_and_Scenes" mkdir "%WS%\legacy_src\Ouputs_and_Scenes"
if not exist "%WS%\legacy_src\Outputs_and_Scenes" mkdir "%WS%\legacy_src\Outputs_and_Scenes"
if not exist "%WS%\legacy_src\Segment_UI" mkdir "%WS%\legacy_src\Segment_UI"
if not exist "%WS%\legacy_src\Select_Luminaire_Attributes" mkdir "%WS%\legacy_src\Select_Luminaire_Attributes"

REM --- canonical source
set "SRC=%WS%\legacy_src\ies_norm\novon_workflow.json"

REM --- copy to every legacy UI that auto-looks nearby
copy "%SRC%" "%WS%\legacy_src\novon_db_updater\assets\novon_workflow.json" /Y >nul
copy "%SRC%" "%WS%\legacy_src\Oupts_and_Scenes\novon_workflow.json" /Y >nul
copy "%SRC%" "%WS%\legacy_src\Outputs_and_Scenes\novon_workflow.json" /Y >nul
copy "%SRC%" "%WS%\legacy_src\Segment_UI\novon_workflow.json" /Y >nul
copy "%SRC%" "%WS%\legacy_src\Select_Luminaire_Attributes\novon_workflow.json" /Y >nul

echo Synced novon_workflow.json to legacy folders.
endlocal
