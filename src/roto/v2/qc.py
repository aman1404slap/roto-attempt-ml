"""Pairing QC: does each element correspond to something the artist actually delivered?

Charter S6 requires that "pairing QC (the sh0230 detector) gates every ingested shot". The
failure it is named after is an element that looks fine on its own and is matched to the wrong
delivered matte -- invisible in any per-element number, because both sides render something
plausible.

Each shot ships Silhouette's own render of its mattes, and each delivered EXR packs three
independent mattes into its B, G and R channels (:mod:`roto.exr`). So the question has a
mechanical answer: render each element, score it against every delivered channel, and see
whether the assignment that comes back is a clean bijection.

**This never produces training input.** The delivered EXRs are a referee and a tag; the
training alpha stays our own render of the artist's shapes (charter S3, tracker D1).

**Empty frames are excluded from the score, and that is not a detail.** ``metrics.soft_iou``
returns 1.0 when both sides are empty, which is the right convention for scoring a
reconstruction -- predicting nothing where there is nothing is correct -- and exactly the wrong
one for a referee. An element that is live on a third of the track, compared against a channel
that is blank everywhere, collects a free 1.0 on every frame where neither draws anything.
Measured on ts_020028, that scored ``char`` at 0.8778 against a channel with no content at all.
So a frame counts as evidence only when **both** sides draw something, and a pairing supported
by fewer than :data:`MIN_EVIDENCE` such frames is no pairing.

Three findings, which are the detector:

``unmatched``   no channel clears ``MATCH_MIN`` on enough evidence -- the element is silver
``collision``   two elements claim the same channel -- at most one can be right
``orphan``      a non-empty delivered channel that no element claims -- something is missing

An element is **gold** when it matches a channel above the threshold and holds that channel
alone. Everything else is **silver**: tree-defined, still built, still trained (tracker D4),
but not evidence about render conventions.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..dataset.layers import layer_doc
from ..exr import load_channels
from ..metrics import soft_iou
from ..render.raster import RenderConfig, render_union
from ..sfx.read import read_sfx
from .ingest import Shot
from .manifest import Element, resolve

QC_VERSION = 1

MATCH_MIN = 0.50
"""Below this an element is not considered to explain a channel at all.

Deliberately loose. The referee's job is to identify *which* channel, a coarse question, and a
tight threshold would demote elements whose artist render differs for reasons charter S7 puts
out of scope. Charter S3's exactness claims are made by the ledger against our own render,
never against this number.
"""

AMBIGUOUS_MARGIN = 0.05
"""Two elements claiming one channel within this of each other are genuinely ambiguous.

Collisions are normal and mostly benign: a small element contained inside a large one scores
against the same channel, and the larger wins by a wide margin. That is resolved by score, not
escalated. What the sh0230 failure actually looks like is two elements the referee *cannot
separate* -- and that is the only case worth failing a shot over, because picking either one
would be a coin toss recorded as a fact.
"""

MIN_EVIDENCE = 2
"""Frames on which both sides draw something, below which a pairing is unsupported."""

EMPTY_MAX = 1e-6
"""Mean coverage below which an image is blank. Two of three channels usually are."""

MIN_MATTE_PX = 16
"""Below this a delivered matte is degenerate rather than small.

Fires on ts_021658, whose entire ``hard/matte01`` sequence is **1x1 pixels**. At any reduced
scale it also makes ``cv2.resize`` raise, so it is rejected before being read, not after.
"""

PROBE_FRAMES = 7


@dataclass(slots=True)
class Pairing:
    element_id: str
    matte: str | None
    channel: str | None
    score: float
    evidence: int
    grade: str                      # 'gold' | 'silver'
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {'element_id': self.element_id, 'matte': self.matte, 'channel': self.channel,
                'score': round(self.score, 6), 'evidence': self.evidence,
                'grade': self.grade, 'reasons': self.reasons}


@dataclass(slots=True)
class ShotQC:
    shot: str
    pairings: list[Pairing]
    findings: list[str]
    channels_present: list[str]

    @property
    def passed(self) -> bool:
        """Fails only on an **ambiguous** collision -- the sh0230 failure.

        The other two findings are recorded, not escalated, and the distinction is the whole
        design. An ``orphan`` is a delivered channel no element claims, which in this delivery
        is ordinary: the EXRs carry outputs that are not one .sfx layer -- combined passes,
        elements from another project, utility mattes. An ``unmatched`` element is one whose
        layer contains shapes the delivery does not (ts_021555 carries a full-opacity ``Square``
        that no delivered channel shows). Neither says the *pairing* is wrong, and failing a
        shot for either would drop four of Tier 1's ten over facts about the delivery rather
        than about the roto.
        """
        return not any(f.startswith('ambiguous') for f in self.findings)

    def as_dict(self) -> dict[str, Any]:
        return {'shot': self.shot, 'passed': self.passed, 'findings': self.findings,
                'channels_present': self.channels_present,
                'pairings': [p.as_dict() for p in self.pairings]}


def probe_frames(duration: int, n: int = PROBE_FRAMES) -> list[int]:
    """``n`` frames spread across the track, never the first or last."""
    if duration <= 2:
        return list(range(duration))
    step = max(1, (duration - 2) // max(1, n))
    return list(range(1, duration - 1, step))[:n]


def usable_mattes(sequences: dict[str, dict[int, Path]], width: int, height: int
                  ) -> tuple[dict[str, dict[int, Path]], list[str]]:
    """``(sequences worth scoring, findings)``.

    A delivered matte is a referee only if it depicts the same raster as the document. Two ways
    that fails here, both silent if unchecked:

    ``degenerate``  the sequence is 1x1 px (ts_021658)
    ``resolution``  a different size from the document -- ts_020355 delivers 958x1435 against a
                    2882x2006 document, not even the same aspect, so it is a different format
                    rather than a scaled copy. Cropping both to the smaller shape would return
                    a confident, meaningless number for two misaligned pictures.
    """
    keep, findings = {}, []
    for matte, frames in sequences.items():
        if not frames:
            continue
        mid = sorted(frames)[len(frames) // 2]
        img = cv2.imread(str(frames[mid]), cv2.IMREAD_UNCHANGED)
        if img is None:
            findings.append(f'unreadable: {matte} will not decode in this OpenCV build')
            continue
        h, w = img.shape[:2]
        if min(h, w) < MIN_MATTE_PX:
            findings.append(f'degenerate: {matte} is {w}x{h} px')
        elif (w, h) != (width, height):
            findings.append(f'resolution: {matte} is {w}x{h}, document is {width}x{height}')
        else:
            keep[matte] = frames
    return keep, findings


def _align(pred: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if pred.shape == ref.shape:
        return pred, ref
    h, w = min(pred.shape[0], ref.shape[0]), min(pred.shape[1], ref.shape[1])
    return pred[:h, :w], ref[:h, :w]


def score_element(sub, sequences: dict[str, dict[int, Path]], frames: list[int],
                  start_frame: int, cfg: RenderConfig | None = None, scale: float = 0.4
                  ) -> dict[str, tuple[float, int]]:
    """``{'matte:channel': (mean soft IoU, n evidence frames)}`` for one element.

    Renders each probe frame **once** and scores it against every channel of every matte, which
    is where the cost of this pass lives. Only frames on which both sides draw something count.
    """
    cfg = cfg or RenderConfig(supersample=2)
    totals: dict[str, list[float]] = defaultdict(list)
    for f in frames:
        pred = None
        for matte, seq in sequences.items():
            path = seq.get(int(f) + start_frame)
            if path is None:
                continue
            if pred is None:
                pred = render_union(sub, int(f), cfg, scale)
                if float(pred.mean()) <= EMPTY_MAX:
                    break                       # element not live: no evidence on this frame
            for channel, ref in load_channels(path, scale).items():
                p, r = _align(pred, ref)
                if float(r.mean()) <= EMPTY_MAX:
                    continue                    # channel blank here: no evidence either
                totals[f'{matte}:{channel}'].append(soft_iou(p, r))
    return {k: (float(np.mean(v)), len(v)) for k, v in totals.items()}


def _present_channels(sequences: dict[str, dict[int, Path]], frames: list[int],
                      start_frame: int, scale: float = 0.25) -> list[str]:
    """Delivered channels that carry content on at least one probe frame."""
    present = set()
    for matte, seq in sequences.items():
        for f in frames:
            path = seq.get(int(f) + start_frame)
            if path is None:
                continue
            for channel, img in load_channels(path, scale).items():
                if float(img.mean()) > EMPTY_MAX:
                    present.add(f'{matte}:{channel}')
    return sorted(present)


def check_shot(shot: Shot, els: list[Element], scale: float = 0.4,
               cfg: RenderConfig | None = None) -> ShotQC:
    """Pair every element of one shot against its delivered channels."""
    doc = read_sfx(shot.sfx)
    sequences, findings = usable_mattes(shot.mattes(), doc.width, doc.height)
    frames = probe_frames(doc.duration)
    present = _present_channels(sequences, frames, doc.start_frame) if sequences else []
    if not sequences:
        return ShotQC(shot.name,
                      [Pairing(e.element_id, None, None, 0.0, 0, 'silver',
                               ['no usable delivered matte']) for e in els],
                      findings or ['no delivered mattes for this shot'], present)

    pairings: list[Pairing] = []
    for e in els:
        sub = layer_doc(doc, resolve(doc, e))
        scored = score_element(sub, sequences, frames, doc.start_frame, cfg, scale)
        viable = {k: v for k, v in scored.items() if v[1] >= MIN_EVIDENCE}
        if not viable:
            best = max(scored.values(), default=(0.0, 0))
            pairings.append(Pairing(e.element_id, None, None, best[0], best[1], 'silver',
                                    [f'no channel with >= {MIN_EVIDENCE} evidence frames']))
            continue
        key, (score, n) = max(viable.items(), key=lambda kv: kv[1][0])
        if score < MATCH_MIN:
            pairings.append(Pairing(e.element_id, None, None, score, n, 'silver',
                                    [f'unmatched: best {score:.4f} < {MATCH_MIN}']))
        else:
            matte, channel = key.split(':')
            pairings.append(Pairing(e.element_id, matte, channel, score, n, 'gold', []))

    claimed: dict[str, list[str]] = defaultdict(list)
    for p in pairings:
        if p.matte:
            claimed[f'{p.matte}:{p.channel}'].append(p.element_id)
    by_id = {p.element_id: p for p in pairings}
    for key, owners in sorted(claimed.items()):
        if len(owners) < 2:
            continue
        ranked = sorted(owners, key=lambda i: by_id[i].score, reverse=True)
        winner, rest = ranked[0], ranked[1:]
        margin = by_id[winner].score - by_id[rest[0]].score
        kind = 'ambiguous' if margin < AMBIGUOUS_MARGIN else 'collision'
        findings.append(f'{kind}: {key} claimed by {ranked}, '
                        f'{winner} wins by {margin:.4f}')
        for eid in rest:
            p = by_id[eid]
            p.matte, p.channel, p.grade = None, None, 'silver'
            p.reasons = p.reasons + [f'lost {key} to {winner} by {margin:.4f}']
        if kind == 'ambiguous':
            p = by_id[winner]
            p.grade = 'silver'
            p.reasons = p.reasons + [f'ambiguous claim on {key}']
    for key in present:
        if key not in claimed:
            findings.append(f'orphan: delivered {key} claimed by no element')
    for p in pairings:
        if p.grade == 'silver' and p.reasons and not p.reasons[0].startswith('collision'):
            findings.append(f'unmatched: {p.element_id} best {p.score:.4f} on {p.evidence}f')
    return ShotQC(shot.name, pairings, findings, present)


def check(shots: list[Shot], els: list[Element], scale: float = 0.4) -> dict[str, ShotQC]:
    by_shot: dict[str, list[Element]] = defaultdict(list)
    for e in els:
        by_shot[e.shot].append(e)
    return {s.name: check_shot(s, by_shot.get(s.name, []), scale)
            for s in shots if s.usable}


def grades(qc: dict[str, ShotQC]) -> dict[str, Pairing]:
    return {p.element_id: p for r in qc.values() for p in r.pairings}
