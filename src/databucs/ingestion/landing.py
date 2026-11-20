"""Write raw responses to the landing volume as immutable, content-addressed files.

Each file name includes a hash of its contents, so:
- re-fetching unchanged data writes nothing (re-runs are idempotent), and
- a corrected re-fetch lands as a new file next to the old one; downstream layers keep
  the latest version.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

HASH_CHARS = 12


@dataclass(frozen=True)
class LandResult:
    path: Path
    written: bool  # False if an identical file already existed


def content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:HASH_CHARS]


def land(content: bytes, directory: Path, stem: str, extension: str) -> LandResult:
    """Write `content` to `directory/<stem>_<hash>.<extension>` unless it already exists.

    Writes to a hidden temporary file first and renames it, so Auto Loader (which skips
    hidden files) never sees a partially written file.
    """
    path = directory / f"{stem}_{content_hash(content)}.{extension}"
    if path.exists():
        return LandResult(path, written=False)
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / f".{path.name}.tmp"
    tmp.write_bytes(content)
    os.replace(tmp, path)
    return LandResult(path, written=True)


def has_landed(directory: Path, stem: str, extension: str) -> bool:
    """True if any version of `stem` has been landed in `directory`."""
    return any(directory.glob(f"{stem}_*.{extension}"))
