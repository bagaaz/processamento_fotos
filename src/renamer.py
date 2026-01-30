from __future__ import annotations
from pathlib import Path
import shutil

def unique_path(out_dir: Path, base_name: str, ext: str) -> Path:
    """
    Evita sobrescrever: se existir, cria sufixo _2, _3...
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate = out_dir / f"{base_name}{ext}"
    if not candidate.exists():
        return candidate

    i = 2
    while True:
        candidate = out_dir / f"{base_name}_{i}{ext}"
        if not candidate.exists():
            return candidate
        i += 1

def copy_with_new_name(src: Path, out_dir: Path, new_base_name: str) -> Path:
    ext = src.suffix.lower()
    dst = unique_path(out_dir, new_base_name, ext)
    shutil.copy2(src, dst)
    return dst
