"""Recovering which Silhouette layers produced which delivered matte channel.

This mapping is not in the archive. The layers were rendered per-layer into colour-named
folders and a Nuke script packed them into R/G/B; that script was not delivered. So we
recover it by search: score every candidate layer against every (matte folder, channel),
then grow the best single candidate into a union while the score improves.

A channel is frequently the union of several layers -- body + arms + head, or FAM_0060's
``green 2 + r1_w_c + green 1 + r1_t_c`` -- so single-layer matching is not enough.

Candidates include *nested* layers, because some deliveries correspond to a sub-layer
rather than a top-level one. Those are rendered through ``RotoDoc.isolate``, which rebuilds
the ancestor chain so tracked transforms above the candidate still apply. Rendering a
nested layer detached instead produces a plausible-looking image in the wrong place.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..ir import Layer, RotoDoc
from ..matte.exr import load_channels
from ..render.raster import RenderConfig, render
from .metrics import soft_iou


@dataclass(slots=True)
class Candidate:
    label: str
    """Slash-joined path from the top-level layer, e.g. ``green 2/CH1/face``."""
    ancestors: tuple[Layer, ...]
    layer: Layer


@dataclass(slots=True)
class Assignment:
    matte: str
    channel: str
    members: list[Candidate]
    score: float

    @property
    def label(self) -> str:
        return ' + '.join(c.label for c in self.members)


def candidate_layers(doc: RotoDoc, max_depth: int = 2) -> list[Candidate]:
    """Every layer from depth 0 to ``max_depth``, labelled by its hierarchy path."""
    out: list[Candidate] = []

    def rec(layer: Layer, ancestors: tuple[Layer, ...], depth: int, prefix: str) -> None:
        label = f'{prefix}{layer.name}'
        out.append(Candidate(label, ancestors, layer))
        if depth < max_depth:
            for child in layer.children:
                if isinstance(child, Layer):
                    rec(child, ancestors + (layer,), depth + 1, label + '/')

    for root in doc.roots:
        rec(root, (), 0, '')
    return out


def element_doc(doc: RotoDoc, members: Sequence[Candidate]) -> RotoDoc:
    """One document whose roots are the isolated member chains of an element.

    Render it with ``render_union``, never plain ``render``: the members form a union, and
    add-and-clip is not union.
    """
    if not members:
        raise ValueError('an element needs at least one member layer')
    roots = [doc.isolate(c.ancestors, c.layer).roots[0] for c in members]
    return RotoDoc(doc.width, doc.height, doc.duration, roots, doc.frame_rate,
                   doc.start_frame, doc.dialect, doc.source_path, doc.source_label)


class _Renders:
    """Memoised single-candidate renders, keyed by (candidate index, sfx frame)."""

    def __init__(self, doc: RotoDoc, cands: Sequence[Candidate],
                 scale: float, cfg: RenderConfig | None) -> None:
        self._doc, self._cands, self._scale = doc, cands, scale
        self._cfg = cfg or RenderConfig()
        self._cache: dict[tuple[int, int], np.ndarray] = {}

    def one(self, ci: int, frame: int) -> np.ndarray:
        hit = self._cache.get((ci, frame))
        if hit is None:
            c = self._cands[ci]
            hit = render(self._doc.isolate(c.ancestors, c.layer), frame,
                         self._cfg, self._scale)
            self._cache[(ci, frame)] = hit
        return hit

    def union(self, members: Sequence[int], frame: int) -> np.ndarray:
        acc = self.one(members[0], frame)
        for ci in members[1:]:
            acc = np.maximum(acc, self.one(ci, frame))
        return acc


def assign_channels(doc: RotoDoc, mattes: dict[str, dict[int, Path]],
                    scale: float = 0.35, sample_frames: Sequence[int] | None = None,
                    max_depth: int = 2, max_members: int = 8,
                    min_gain: float = 0.01, top_k: int = 24,
                    cfg: RenderConfig | None = None) -> list[Assignment]:
    """Best layer union for each (matte folder, channel), scored on a few sample frames.

    ``sample_frames`` are *file* frame numbers (1001-based); they are converted to sfx
    frames with ``doc.start_frame``. Defaults to first / middle / last.
    """
    cands = candidate_layers(doc, max_depth)
    if not cands:
        return []
    renders = _Renders(doc, cands, scale, cfg)
    out: list[Assignment] = []

    for matte, frames in mattes.items():
        available = sorted(frames)
        if not available:
            continue
        picks = [f for f in (sample_frames or ()) if f in frames]
        if not picks:
            picks = sorted({available[0], available[len(available) // 2], available[-1]})

        # Ground truth per channel, as (sfx_frame, alpha) pairs.
        gt: dict[str, list[tuple[int, np.ndarray]]] = {}
        for f in picks:
            for name, img in load_channels(frames[f], scale).items():
                gt.setdefault(name, []).append((f - doc.start_frame, img))

        for channel, pairs in sorted(gt.items()):
            def score(members: list[int]) -> float:
                vals = []
                for frame, truth in pairs:
                    r = renders.union(members, frame)
                    if r.shape != truth.shape:
                        return -1.0
                    vals.append(soft_iou(r, truth))
                return float(np.mean(vals)) if vals else -1.0

            ranked = sorted(((score([i]), i) for i in range(len(cands))), reverse=True)
            best, members = ranked[0][0], [ranked[0][1]]
            improved = True
            while improved and len(members) < max_members:
                improved = False
                for sc, ci in ranked[1:top_k]:
                    if ci in members or sc <= 0.0:
                        continue
                    grown = score(members + [ci])
                    if grown > best + min_gain:
                        best, members, improved = grown, members + [ci], True
                        break
            out.append(Assignment(matte, channel, [cands[i] for i in members], best))
    return out
