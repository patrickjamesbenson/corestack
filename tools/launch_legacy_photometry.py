# tools/launch_legacy_photometry.py
from __future__ import annotations
import json, os, sys, subprocess, webbrowser
from pathlib import Path

PORT = int(os.environ.get("PHOTOMETRY_PORT", "8510"))

def main():
    app_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(app_root))
    from utils.manifest import load_manifest, resolve_legacy_path
    from utils.paths import workflow_current_path, ies_repo_current_path

    mf = load_manifest(app_root) or {}
    legacy_rel = mf.get("ies_lookup_or_norm_ui") or "ies_norm/app.py"
    legacy_entry = resolve_legacy_path(app_root, legacy_rel)

    wf = workflow_current_path(app_root)
    repo = ies_repo_current_path(app_root)
    repo.parent.mkdir(parents=True, exist_ok=True)
    if not repo.exists():
        repo.write_text('{"records":[]}', encoding="utf-8")

    if not legacy_entry or not legacy_entry.exists():
        print(json.dumps({"ok": False, "error": f"Legacy entry not found: {legacy_rel}"})); return 2
    if not wf.exists():
        print(json.dumps({"ok": False, "error": "Workflow JSON not found. Build DB first."})); return 2

    env = os.environ.copy()
    env["NOVON_WORKFLOW_PATH"] = str(wf)
    env["NOVON_IES_REPO_PATH"] = str(repo)

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(legacy_entry),
        "--server.port", str(PORT),
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false",
    ]
    CREATE_NEW_CONSOLE = 0x00000010  # Windows
    p = subprocess.Popen(cmd, cwd=str(legacy_entry.parent), env=env, creationflags=CREATE_NEW_CONSOLE)
    url = f"http://localhost:{PORT}"
    try: webbrowser.open(url)
    except Exception: pass
    print(json.dumps({"ok": True, "pid": p.pid, "url": url})); return 0

if __name__ == "__main__":
    raise SystemExit(main())

