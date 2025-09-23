from __future__ import annotations
import os, shutil
from pathlib import Path

def same_inode(a: Path, b: Path) -> bool:
    try:
        sa, sb = a.stat(), b.stat()
        return (sa.st_dev, sa.st_ino) == (sb.st_dev, sb.st_ino)
    except Exception:
        return False

def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return False, f"source missing: {src}"
    if dst.exists() and same_inode(src, dst):
        return True, "already linked"
    try:
        if dst.exists():
            dst.unlink()
    except Exception:
        pass
    try:
        os.link(src, dst)
        return True, "linked"
    except OSError as ex:
        shutil.copyfile(src, dst)
        return False, f"copied (link failed: {ex})"
