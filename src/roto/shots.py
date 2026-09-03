"""Locating the Silhouette project for each shot on disk.

Shot directories are not laid out consistently, so the ``.sfx`` path cannot be hardcoded.
Both conventions in this archive work because the file is found recursively:

    <shot>/scene/<name>.sfx          <shot>/<name>_SFX_script_v02/<name>.sfx
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Shot:
    name: str
    path: Path
    sfx: Path | None


def _find_sfx(directory: Path) -> Path | None:
    """The shot's project file. Largest wins when a directory holds several saves."""
    found = sorted(directory.rglob('*.sfx'), key=lambda p: p.stat().st_size, reverse=True)
    return found[0] if found else None


def find_shot(shot_dir: str | Path) -> Shot:
    d = Path(shot_dir)
    return Shot(d.name, d, _find_sfx(d))


def find_shots(data_root: str | Path) -> list[Shot]:
    """Every immediate subdirectory of ``data_root`` that contains a ``.sfx``."""
    root = Path(data_root)
    shots = [find_shot(d) for d in sorted(root.iterdir()) if d.is_dir()]
    return [s for s in shots if s.sfx is not None]
