# tools/fix_dupes.py
from __future__ import annotations
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    *(ROOT / "pages").glob("*.py"),
    *(ROOT / "utils").glob("*.py"),
]

def dedupe_imports(text: str) -> str:
    seen = set()
    out = []
    for line in text.splitlines():
        key = None
        if re.match(r"\s*(import|from)\s+\S+", line):
            key = line.strip()
        if key:
            if key in seen:  # drop exact duplicate import
                continue
            seen.add(key)
        out.append(line)
    return "\n".join(out)

def keep_first_db_init(text: str) -> str:
    # keep the FIRST "db = load_local_db()" and remove the rest
    lines = text.splitlines()
    hits = [i for i,l in enumerate(lines) if re.search(r"\bdb\s*=\s*load_local_db\(\)", l)]
    if len(hits) <= 1:
        return text
    first = hits[0]
    for i in hits[1:]:
        lines[i] = re.sub(r"\bdb\s*=\s*load_local_db\(\).*", "", lines[i])
    # strip accidental blank duplicates
    cleaned = []
    for i,l in enumerate(lines):
        if l.strip()=="" and cleaned and cleaned[-1].strip()=="":
            continue
        cleaned.append(l)
    return "\n".join(cleaned)

def run():
    changed = 0
    for p in FILES:
        src = p.read_text(encoding="utf-8", errors="ignore")
        dst = dedupe_imports(src)
        dst = keep_first_db_init(dst)
        if dst != src:
            p.write_text(dst, encoding="utf-8")
            changed += 1
            print(f"fixed: {p.relative_to(ROOT)}")
    print(f"Done. Files updated: {changed}")

if __name__ == "__main__":
    run()
