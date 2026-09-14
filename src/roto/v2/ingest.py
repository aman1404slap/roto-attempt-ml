"""Locating one shot's final Silhouette project in the 2026-08 delivery.

The layout is not the archive's::

    <shot>/splines/silhouette/*.sfx     the project, plus up to ~20 autosaves
    <shot>/splines/nuke/*.nk            unused
    <shot>/mattes/hard/<name>/*.exr     Silhouette's own render -- referee only
    <shot>/mattes/mb/<name>/*.exr       the same with motion blur -- out of scope (charter S7)
    <shot>/source/mov/*.mov             the plate -- unused in v2 (charter S1, L1)

**Picking the final .sfx cannot use the filesystem.** Every mtime in the delivery is identical
because the copy flattened them, and there is no timestamp inside a .sfx -- the XML carries a
schema ``version`` and nothing else. Filenames do not rank either: ``ts_020028``'s long
descriptive name is a 77 KB early save sitting next to a 747 KB ``project.sfx``.

So the ladder is, in order, and each rung records *why* it fired:

1. ``project.sfx``, if it parses. Exists in 47 of 50 shots and parses in 46.
2. The candidate whose render best explains the shot's own delivered mattes
   (:func:`roto.exr.best_channel`). This is charter S6's referee, and it is what "final" means
   operationally -- the artist rendered from the save they finished on.
3. The parseable candidate with the most shapes.

Rule 3 is last because on its own it is wrong: on ``ts_021262`` the largest file is
``backup.4.sfx`` at 7.7 MB against ``project.sfx`` at 5.1 MB, the artist having deleted shapes
later in the session. Largest-file-wins -- ``roto.shots``' archive rule -- picks a mid-session
autosave in roughly 30 of these 50 shots.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..exr import best_channel
from ..ir import RotoDoc
from ..sfx.read import read_sfx

AUTOSAVE_RE = re.compile(r'^(project|backup|autosave)(\.\d+)?\.sfx$')
"""Silhouette's own rotation. 453 of the delivery's 483 .sfx files match this.

``autosave.sfx`` is a third pattern, seen on ts_020769 and ts_021097 only, and it is the
largest file in both -- so a rule that knew only ``project``/``backup`` would silently treat
it as a named project file.
"""

PREFERRED = 'project.sfx'
"""The artist's own manual save. First rung of the ladder."""

MATTE_KIND = 'hard'
"""``mattes/hard`` is the referee. ``mattes/mb`` is the same roto with motion blur, which
charter S7 puts out of scope until the S3 gates pass."""

FRAME_RE = re.compile(r'[._](\d+)\.exr$')
r"""Frame number in a delivered matte's file name.

**Two separators are in use** and this is the reason v2 discovers mattes itself rather than
calling :func:`roto.exr.matte_sequences`, whose pattern is ``\.(\d+)\.exr$``::

    MON_0050_BG1_v001_Roto_V01.1030.exr          dot     -- 8 of 10 Tier-1 shots
    inn020_split02_v01_Roto_V01_1001.exr         under   -- ts_021555, ts_021182

The dot-only pattern does not fail on the underscore form; it returns **no frames**, the
directory is skipped as empty, and the shot is graded silver for "no delivered mattes" while
its mattes sit on disk. Silent, and it removed the two held-out shots from the referee.
"""


def matte_sequences(directory: str | Path) -> dict[str, dict[int, Path]]:
    """``{matte_dir_name: {sfx_frame: path}}`` under one ``mattes/<kind>`` directory.

    Frame numbers are Silhouette's own (1001-based), so
    ``sfx_frame = doc_internal_frame + doc.start_frame``.
    """
    out: dict[str, dict[int, Path]] = {}
    root = Path(directory)
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        frames = {}
        for f in sorted(d.glob('*.exr')):
            m = FRAME_RE.search(f.name)
            if m:
                frames[int(m.group(1))] = f
        if frames:
            out[d.name] = frames
    return out


@dataclass(slots=True)
class Shot:
    """One delivery shot, with its chosen project file and the reason it was chosen."""
    name: str
    path: Path
    sfx: Path | None
    reason: str
    candidates: int = 0
    error: str | None = None

    @property
    def usable(self) -> bool:
        return self.sfx is not None

    def mattes(self, kind: str = MATTE_KIND) -> dict[str, dict[int, Path]]:
        """``{matte_dir_name: {sfx_frame: path}}`` for this shot's delivered renders.

        Frame numbers are Silhouette's own (1001-based), matching ``exr.matte_sequences``;
        ``sfx_frame = doc_internal_frame + doc.start_frame``.
        """
        return matte_sequences(self.path / 'mattes' / kind)

    def as_dict(self) -> dict:
        return {'shot': self.name, 'sfx': str(self.sfx) if self.sfx else None,
                'reason': self.reason, 'candidates': self.candidates, 'error': self.error}


def sfx_candidates(shot_dir: str | Path) -> list[Path]:
    """Every .sfx in the shot, named projects and autosaves alike, largest first."""
    d = Path(shot_dir) / 'splines' / 'silhouette'
    if not d.is_dir():
        return []
    return sorted(d.glob('*.sfx'), key=lambda p: p.stat().st_size, reverse=True)


def is_autosave(path: str | Path) -> bool:
    return AUTOSAVE_RE.match(Path(path).name) is not None


def _try_read(path: Path) -> RotoDoc | None:
    try:
        return read_sfx(path)
    except Exception:
        return None


def _n_shapes(doc: RotoDoc) -> int:
    return sum(len(list(_walk_shapes(r))) for r in doc.roots)


def _walk_shapes(node) -> Iterable:
    for child in getattr(node, 'children', []):
        if hasattr(child, 'children'):
            yield from _walk_shapes(child)
        else:
            yield child


def pick_sfx(shot_dir: str | Path, referee: bool = True,
             probe_frames: int = 5) -> tuple[Path | None, str, int]:
    """``(chosen, reason, n_candidates)`` for one shot, by the ladder in the module docstring.

    ``referee=False`` skips rung 2, which costs a render per candidate per probe frame. Rung 2
    only ever fires where ``project.sfx`` is missing or unreadable -- 4 of 50 shots -- so the
    default is on and the cost is paid almost nowhere.
    """
    cands = sfx_candidates(shot_dir)
    if not cands:
        return None, 'no .sfx found', 0

    preferred = [p for p in cands if p.name == PREFERRED]
    if preferred and _try_read(preferred[0]) is not None:
        return preferred[0], f'{PREFERRED} (rung 1)', len(cands)

    parsed: list[tuple[Path, RotoDoc]] = []
    for p in cands:
        doc = _try_read(p)
        if doc is not None:
            parsed.append((p, doc))
    if not parsed:
        return None, f'none of {len(cands)} candidates parse', len(cands)

    if referee:
        sequences = matte_sequences(Path(shot_dir) / 'mattes' / MATTE_KIND)
        if sequences:
            scored = []
            for p, doc in parsed:
                frames = _probe_frames(doc, probe_frames)
                _, _, score = best_channel(doc, sequences, frames)
                scored.append((score, p))
            scored.sort(reverse=True)
            if scored and scored[0][0] > 0:
                return scored[0][1], f'best EXR agreement {scored[0][0]:.4f} (rung 2)', len(cands)

    p, doc = max(parsed, key=lambda t: _n_shapes(t[1]))
    return p, f'most shapes, {_n_shapes(doc)} (rung 3)', len(cands)


def _probe_frames(doc: RotoDoc, n: int) -> list[int]:
    """``n`` frames spread across the document, never the first or last."""
    if doc.duration <= 2:
        return list(range(doc.duration))
    step = max(1, (doc.duration - 2) // max(1, n))
    return list(range(1, doc.duration - 1, step))[:n]


def find_shot(shot_dir: str | Path, referee: bool = True) -> Shot:
    d = Path(shot_dir)
    sfx, reason, n = pick_sfx(d, referee)
    return Shot(d.name, d, sfx, reason, n, None if sfx else reason)


def find_shots(data_root: str | Path, names: Iterable[str] | None = None,
               referee: bool = True) -> list[Shot]:
    """Every shot under ``data_root``, or just ``names`` if given, in name order."""
    root = Path(data_root)
    dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if names is not None:
        wanted = set(names)
        dirs = [d for d in dirs if d.name in wanted]
        missing = wanted - {d.name for d in dirs}
        if missing:
            raise FileNotFoundError(f'no such shot under {root}: {sorted(missing)}')
    return [find_shot(d, referee) for d in dirs]
