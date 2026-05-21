"""Filesystem helpers shared by manifest and inference code."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def iter_files(root: str | Path, pattern: str) -> Iterable[Path]:
    base = Path(root).expanduser().resolve()
    yield from sorted(base.rglob(pattern))


def relative_to_root(root: str | Path, file_path: str | Path) -> str:
    root_path = Path(root).expanduser().resolve()
    return str(Path(file_path).expanduser().resolve().relative_to(root_path))
