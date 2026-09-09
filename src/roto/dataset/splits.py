"""Which frames and which layers a run is allowed to train on, decided when the dataset is
built rather than when a run starts.

v1 and v1.1 chose the frame holdout inside ``train()`` from a single integer, which means the
split was a property of the *run*. That is harmless while every run uses the same integer and
silently wrong the moment two do not: two runs quoting a held-out gap would be quoting gaps
over different frames, and neither report would say so. v1.2 then measured the thing that
makes it matter -- a localised failure can be present in one seed and absent in the next
(``v1.2/OVERVIEW.md``), so a held-out number is only comparable across runs if the held-out
set is identical across runs.

So the split moves into the dataset, is written once by ``dataset.build``, and is read back by
everything downstream. Two independent splits, because they answer different questions:

* **Frame holdout** -- every ``holdout_every``-th frame of every layer is withheld from
  training and scored separately. Asks whether the network interpolates its memorisation or
  merely recalls it. Never the first or last frame of a track, so a held frame always has
  trained neighbours on both sides and the temporal window stays well defined.

* **Layer holdout** -- named layers are withheld from training *entirely*. This is **not** a
  generalisation claim, and saying so is the point of the row: shape queries are per
  ``(layer, shape)`` (``model.net.RotoNet``), so a layer that never trains has never had its
  query rows updated from their initialisation and cannot reconstruct at all. What the split
  measures is exactly that -- **how much of the headline number lives in the query table**
  rather than in the encoder -- which is the number v2's dynamic-query work has to beat. It
  also forces the plumbing (a dataset with untrainable layers, a scorer that reports them
  apart from the rest) that every v2 experiment needs.

The two compose: a held-out layer's frames are all "held" in the sense that none of them
trained, and its own frame split is still recorded so its held frames stay identifiable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SPLIT_VERSION = 1

DEFAULT_HOLDOUT_EVERY = 7
"""Every 7th frame, as v1.1's ``final_holdout`` used and v1.2 re-scored.

Kept rather than re-chosen so the v002 gap reads against v1.1's +0.0005 directly."""


def frame_split(n_frames: int, holdout_every: int) -> tuple[np.ndarray, np.ndarray]:
    """``(train_positions, held_positions)`` for a track of ``n_frames``.

    Identical rule to the one ``train()`` used through v1.2 -- offset to the middle of each
    period, with the first and last frame never held -- so a v002 gap is comparable to v1.1's.
    """
    n = int(n_frames)
    if holdout_every <= 1 or n <= 2:
        return np.arange(n), np.zeros(0, np.int64)
    held = np.arange(n)[(np.arange(n) % holdout_every == holdout_every // 2)]
    held = held[(held > 0) & (held < n - 1)]
    return np.setdiff1d(np.arange(n), held), held


def build_splits(layers: Sequence[tuple[str, Sequence[int]]], holdout_every: int,
                 held_layers: Sequence[str] = (), note: str = '') -> dict[str, Any]:
    """The dataset-level split record. ``layers`` is ``(layer_id, frames)`` per layer."""
    held = list(held_layers)
    unknown = sorted(set(held) - {lid for lid, _ in layers})
    if unknown:
        raise ValueError(f'held layers not in the dataset: {unknown}')
    out: dict[str, Any] = {
        'split_version': SPLIT_VERSION,
        'holdout_every': int(holdout_every),
        'held_layers': held,
        'note': note,
        'layers': {},
    }
    for lid, frames in layers:
        tr, he = frame_split(len(frames), holdout_every)
        out['layers'][lid] = {
            'in_train': lid not in held,
            'n_frames': len(frames),
            'frames_train': [int(frames[i]) for i in tr],
            'frames_held': [int(frames[i]) for i in he],
        }
    out['trained_layers'] = sorted(l for l in out['layers'] if out['layers'][l]['in_train'])
    out['frames_held'] = int(sum(len(v['frames_held']) for v in out['layers'].values()))
    return out


def load_splits(root: str | Path) -> dict[str, Any] | None:
    """The dataset's own split record, or ``None`` for a dataset built before splits existed
    (``datasets/v001``), where the caller falls back to computing one."""
    p = Path(root) / 'splits.json'
    return json.loads(p.read_text()) if p.exists() else None
