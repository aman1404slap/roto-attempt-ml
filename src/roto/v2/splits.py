"""What a run is allowed to train on, decided when the dataset is built.

Three independent splits, because they answer three different questions. Each is recorded in
``splits.json`` at build time so that two runs quoting a held-out gap are quoting it over the
same frames -- a split that lives in the training loop is a property of the *run*, and two runs
with different integers would produce incomparable numbers with neither report saying so.

**Frame holdout** -- every ``holdout_every``-th frame of every element, withheld from training
and scored apart. Asks whether the network interpolates its memorisation or merely recalls it.
Never the first or last frame of a track, so a held frame always has trained neighbours on both
sides and the temporal window stays well defined.

**Element holdout** -- named elements withheld entirely. This is *not* a generalisation claim:
shape queries are per ``(element, shape)``, so an element that never trains has never had its
query rows updated and cannot reconstruct at all. What it measures is how much of a headline
number lives in the query table rather than in the encoder.

**Shot holdout** -- every element of a named shot withheld. New in v2 and the reason the split
record is v2's own rather than the archive's: with 50 shots a held-out *shot* is available
immediately, and charter S5 says generalisation is claimed only from held-out layers, then
held-out shots. A held-out shot implies its elements are held, so the two compose.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SPLIT_VERSION = 1

DEFAULT_HOLDOUT_EVERY = 7
"""Kept from the archive's split rather than re-chosen, so a v2 frame gap reads directly
against the +0.0005 the archive measured on the same rule."""


def frame_split(n_frames: int, holdout_every: int) -> tuple[np.ndarray, np.ndarray]:
    """``(train_positions, held_positions)`` for a track of ``n_frames``.

    Offset to the middle of each period, first and last frame never held.
    """
    n = int(n_frames)
    if holdout_every <= 1 or n <= 2:
        return np.arange(n), np.zeros(0, np.int64)
    held = np.arange(n)[(np.arange(n) % holdout_every == holdout_every // 2)]
    held = held[(held > 0) & (held < n - 1)]
    return np.setdiff1d(np.arange(n), held), held


def build_splits(elements: Sequence[tuple[str, str, Sequence[int]]],
                 holdout_every: int = DEFAULT_HOLDOUT_EVERY,
                 held_shots: Sequence[str] = (), held_elements: Sequence[str] = (),
                 note: str = '') -> dict[str, Any]:
    """The dataset-level split record. ``elements`` is ``(element_id, shot, frames)`` each."""
    known_shots = {shot for _, shot, _ in elements}
    known_els = {eid for eid, _, _ in elements}
    for label, wanted, known in (('shots', held_shots, known_shots),
                                 ('elements', held_elements, known_els)):
        unknown = sorted(set(wanted) - known)
        if unknown:
            raise ValueError(f'held {label} not in the dataset: {unknown}')

    out: dict[str, Any] = {
        'split_version': SPLIT_VERSION,
        'holdout_every': int(holdout_every),
        'held_shots': sorted(held_shots),
        'held_elements': sorted(held_elements),
        'note': note,
        'elements': {},
    }
    for eid, shot, frames in elements:
        tr, he = frame_split(len(frames), holdout_every)
        held_by = ([f'shot {shot}'] if shot in set(held_shots) else []) + \
                  (['element'] if eid in set(held_elements) else [])
        out['elements'][eid] = {
            'shot': shot,
            'in_train': not held_by,
            'held_because': held_by,
            'n_frames': len(frames),
            'frames_train': [int(frames[i]) for i in tr],
            'frames_held': [int(frames[i]) for i in he],
        }
    out['trained_elements'] = sorted(e for e, v in out['elements'].items() if v['in_train'])
    out['frames_held'] = int(sum(len(v['frames_held']) for v in out['elements'].values()))
    return out


def load_splits(root: str | Path) -> dict[str, Any] | None:
    p = Path(root) / 'splits.json'
    return json.loads(p.read_text()) if p.exists() else None
