"""Which of the delivery's 50 shots v2 uses, and why each of the others is out.

Surveyed 2026-09-11: 50 shots, 483 .sfx (30 named, 453 autosaves), 470 parseable, 145
top-level layers, 27,120 shapes, 7,592 frames. Shapes per layer run 1 / 17 / 247 / 9,008
(min / median / p90 / max).

Nothing here is a quality judgement about the roto. ``EXCLUDED`` is four shots that fail
charter S6's ingest QC outright; ``DEFERRED`` is three that are simply too large to start on.
Both lists carry their reason so a later round can reverse them on evidence rather than memory.
"""
from __future__ import annotations

TIER1: tuple[str, ...] = (
    'ts_021658',   # 58f,  1L,  7 shapes
    'ts_020036',   # 104f, 1L,  13
    'ts_021555',   # 71f,  1L,  20            [held out]
    'ts_020355',   # 63f,  2L,  20, 5
    'ts_021150',   # 64f,  1L,  27
    'ts_020876',   # 71f,  2L,  39, 10
    'ts_020028',   # 94f,  5L,  78, 19, 15, 9, 1
    'ts_021243',   # 56f,  1L,  139
    'ts_021182',   # 51f,  3L,  175, 4, 1     [held out]
    'ts_021351',   # 42f,  4L,  247, 27, 16, 3
)
"""10 shots, 21 layers, 875 shapes, 674 frames.

Chosen to span 7 -> 247 shapes per layer so that a failure says *where* it breaks rather than
only that it broke, and to stay short enough that a build-and-ledger cycle is minutes. Every
one has a ``project.sfx`` that parses and ``hard`` mattes matching frame for frame.
"""

HOLDOUT_SHOTS: tuple[str, ...] = ('ts_021555', 'ts_021182')
"""Withheld from training entirely, so there is a held-out-shot number from the first run.

One small (1 layer, 20 shapes) and one dense (3 layers, 175+4+1), mirroring charter S3's
"one small, one dense" rule for held-out layers.
"""

TIER2: tuple[str, ...] = (
    'ts_019698',   # 118f, 3L, 643, 422, 2
    'ts_020530',   # 122f, 1L, 1235
    'ts_020880',   # 49f,  4L, 563, 15, 10, 2
    'ts_020191',   # 140f, 3L, 436, 91, 13
    'ts_021994',   # 212f, 3L, 415, 215, 3
)
"""The mid-range. Added after Step 2 is clean; charter S2 means one re-baseline when it is."""

DEFERRED: dict[str, str] = {
    'ts_021191': '9,077 shapes over 4 layers, 417 frames -- would dominate a first run',
    'ts_021733': 'one layer of 9,008 shapes; the per-(layer, shape) query table is the '
                 'constraint, so this shot re-enters with charter S4 stage S3',
    'ts_020966': '7680x4320 over 393 frames -- a size decision, not a quality one',
}
"""Real data held back for size. Not excluded: each names the condition for its return."""

EXCLUDED: dict[str, str] = {
    'ts_021734': "final .sfx unreadable -- shape 'B-Spline 30' changes point count across "
                 'keyframes, which roto.ir does not model. Only backup.8/9 (the two oldest '
                 'of eleven) parse. Dropped rather than extending a frozen interface '
                 '(charter S2, S7)',
    'ts_021030': 'zero delivered EXRs -- no referee for the .sfx pick and no gold tag',
    'ts_019120': '1 shape in total, no EXRs, no project.sfx',
    'ts_019663': 'no project.sfx, and its EXRs do not decode in this OpenCV build',
}
"""Fails charter S6's "pairing QC gates every ingested shot". Each names its own cause."""


def step1_shots() -> tuple[str, ...]:
    """The shots Step 1 builds. See ``v2-tracker.md``."""
    return TIER1


def trainable(shots: tuple[str, ...] = TIER1) -> tuple[str, ...]:
    return tuple(s for s in shots if s not in HOLDOUT_SHOTS)
