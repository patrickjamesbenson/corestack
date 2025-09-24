from __future__ import annotations

import sys, pathlib
# make "utils" importable when running as a script
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pathlib import Path
import argparse
from utils.builder import build_database

def main() -> None:
    ap = argparse.ArgumentParser(description="NOVON DB Builder (JSON/CSV → workflow)")
    ap.add_argument(
        "--source",
        required=True,
        help="Path to source .json or .csv (export Google Sheet as CSV or point to your existing JSON)",
    )
    args = ap.parse_args()

    app_root = Path(__file__).resolve().parents[1]
    current, hist_json, hist_md, build_id = build_database(app_root, args.source)

    print("[OK] DB built")
    print(f"  build_id      : {build_id}")
    print(f"  current JSON  : {current}")
    print(f"  history JSON  : {hist_json}")
    print(f"  changelog MD  : {hist_md}")
    print(f"  latest.txt    : {current.parent / 'latest.txt'}")

if __name__ == "__main__":
    main()

