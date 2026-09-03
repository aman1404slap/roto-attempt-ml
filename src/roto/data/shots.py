"""Locating the pieces of one archive shot on disk.

A shot directory is not laid out consistently across vendors. Two conventions appear in the
six-shot drop:

    <shot>/scene/<name>.sfx                       <shot>/<name>_SFX_script_v02/<name>.sfx
    <shot>/matte01/<name>.1001.exr                <shot>/<name>_matte_L110_v02/<name>.1001.exr

So neither the .sfx path nor the matte folder names can be hardcoded. We find the .sfx
recursively and treat *any* directory containing .exr files as a matte folder, which is the
one rule that holds for both layouts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

FRAME_RE = re.compile(r'\.(\d{4,})\.exr$', re.IGNORECASE)


@dataclass(slots=True)
class Shot:
    name: str
    path: Path
    sfx: Path | None
    mattes: dict[str, dict[int, Path]] = field(default_factory=dict)
    """``{matte_folder_name: {exr_frame_number: path}}``. Frame numbers are the *file*
    numbers (1001-based), not sfx frames -- convert with ``RotoDoc.start_frame``."""

    @property
    def is_usable(self) -> bool:
        return self.sfx is not None and bool(self.mattes)

    def frames(self, matte: str | None = None) -> list[int]:
        """Sorted file frame numbers for one matte folder (the first, by default)."""
        if not self.mattes:
            return []
        key = matte if matte is not None else next(iter(self.mattes))
        return sorted(self.mattes[key])

    def sample_frames(self, n: int = 3, matte: str | None = None) -> list[int]:
        """``n`` file frames spread evenly across the sequence, first and last included."""
        fr = self.frames(matte)
        if not fr or n <= 0:
            return []
        if len(fr) <= n:
            return fr
        idx = [round(i * (len(fr) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
        return sorted({fr[i] for i in idx})


def find_shot(shot_dir: str | Path) -> Shot:
    d = Path(shot_dir)
    sfx = sorted(d.rglob('*.sfx'))
    mattes: dict[str, dict[int, Path]] = {}
    for sub in sorted(p for p in d.iterdir() if p.is_dir()):
        frames = {int(m.group(1)): f
                  for f in sorted(sub.glob('*.exr'))
                  if (m := FRAME_RE.search(f.name))}
        if frames:
            mattes[sub.name] = frames
    return Shot(name=d.name, path=d, sfx=sfx[0] if sfx else None, mattes=mattes)


def find_shots(data_root: str | Path) -> list[Shot]:
    """Every immediate subdirectory of ``data_root`` that holds a .sfx, in name order."""
    root = Path(data_root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    out = [find_shot(p) for p in sorted(root.iterdir()) if p.is_dir()]
    return [s for s in out if s.sfx is not None]
