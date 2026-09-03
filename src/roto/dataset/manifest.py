"""Which layers are training elements, and why the delivered EXRs no longer decide.

An element is **one top-level layer of one .sfx**. That is the whole rule, and it is
derived from the file itself -- there is no hand-maintained table any more.

The previous manifest was a *recovered* mapping from layer union to delivered EXR channel,
found by brute-force search and scored against the EXRs. It existed because the EXR was the
training target, so an element had to be something an EXR could certify. Under the clean
alpha strategy the EXR is reference only: the input is rendered from the artist's own
splines, so the layer *is* the element and no channel mapping is needed to select it.

Three blockers dissolve with it, and they are worth naming because they were on the
ask-the-client list: the missing Nuke render-node mapping (which cost 14 of 23
deliverables), the sh0230 project/matte disagreement, and the vendor colour transform on two
shots' alpha. All three are EXR-side problems, and the EXR has left the training loop.

``roto.data.elements`` is kept, unchanged, because ``roto verify`` still scores against the
EXRs -- that check is what gives us confidence in the renderer, and it stays.

EXCLUSIONS
----------
Exclusion is by measurement, not taste, and each rule states the number it fires on.
``keys_per_live_frame`` is the sparsity the model has to learn to reproduce; ``open_frac``
is the share of shapes that are stroked open paths rather than filled regions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ..data.shots import Shot, find_shots
from ..eval.match import Candidate, candidate_layers
from ..ir import Layer, RotoDoc, Shape, opacity_at
from ..sfx.read import read_sfx

MANIFEST_VERSION = 2

MAX_KEYS_PER_LIVE_FRAME = 0.75
"""Above this the element is keyed so densely it teaches key-every-frame.

Measured across the 18 top-level layers: 14 sit at 0.03-0.59, and four sit at 0.95-1.59.
There is no continuum between them -- the gap is a factor of 1.6 -- so the threshold is
reading a real bimodality rather than cutting an arbitrary tail. Over-keying is the
project's measured failure mode (precision 0.21-0.44, ~2.5x too many keys), and these are
the elements that would teach it.
"""

MAX_OPEN_FRACTION = 0.9
"""Above this the element is a paint-stroke pass, not region roto.

sh0230/L110 is 2515 shapes, 2515 of them open strokes; nfl_0080/MB 2 is 508 of 510. Both
are single-key-by-construction hair and motion-blur passes. They are not what the POC
predicts, and mixing them in teaches one key per shape.
"""

MIN_SHAPES = 3
"""Below this there is no set to predict. Fires on FAM_0060/red (2 shapes)."""


@dataclass(slots=True)
class ElementStats:
    shapes: int
    open_strokes: int
    ephemeral: int
    points: int
    keys: int
    live_frame_shapes: int
    groups: int
    max_group: int

    @property
    def keys_per_live_frame(self) -> float:
        return self.keys / self.live_frame_shapes if self.live_frame_shapes else 0.0

    @property
    def open_frac(self) -> float:
        return self.open_strokes / self.shapes if self.shapes else 0.0

    def as_dict(self) -> dict[str, Any]:
        d = {f: getattr(self, f) for f in self.__slots__}
        d['keys_per_live_frame'] = round(self.keys_per_live_frame, 4)
        d['open_frac'] = round(self.open_frac, 4)
        return d


@dataclass(slots=True)
class Element:
    """One top-level layer of one shot: the unit the dataset is built from."""
    element_id: str
    shot: str
    layer: str
    uuid: str | None
    stats: ElementStats
    excluded_by: tuple[str, ...] = ()

    @property
    def is_target(self) -> bool:
        return not self.excluded_by

    def as_dict(self) -> dict[str, Any]:
        return {'element_id': self.element_id, 'shot': self.shot, 'layer': self.layer,
                'uuid': self.uuid, 'excluded_by': list(self.excluded_by),
                'stats': self.stats.as_dict()}


def _shapes_of(node: Layer | Shape) -> Iterator[Shape]:
    if isinstance(node, Shape):
        yield node
    else:
        for child in node.children:
            yield from _shapes_of(child)


def _groups_of(node: Layer | Shape) -> Iterator[int]:
    """Sizes of the leaf layers that directly hold shapes -- the transform-track unit."""
    if isinstance(node, Shape):
        return
    direct = sum(1 for c in node.children if isinstance(c, Shape))
    if direct:
        yield direct
    for child in node.children:
        if isinstance(child, Layer):
            yield from _groups_of(child)


def measure(doc: RotoDoc, root: Layer) -> ElementStats:
    shapes = list(_shapes_of(root))
    groups = list(_groups_of(root)) or [0]
    live = sum(1 for s in shapes for f in range(doc.duration) if opacity_at(s, f) > 0.5)
    return ElementStats(
        shapes=len(shapes),
        open_strokes=sum(1 for s in shapes if not s.closed),
        ephemeral=sum(1 for s in shapes if s.opacity and len(s.opacity) > 1),
        points=sum(s.n_points for s in shapes),
        keys=sum(len(s.path) for s in shapes),
        live_frame_shapes=live,
        groups=len(groups),
        max_group=max(groups),
    )


def exclusions(stats: ElementStats) -> tuple[str, ...]:
    out = []
    if stats.shapes < MIN_SHAPES:
        out.append(f'too_few_shapes({stats.shapes}<{MIN_SHAPES})')
    if stats.open_frac > MAX_OPEN_FRACTION:
        out.append(f'paint_strokes(open_frac={stats.open_frac:.2f})')
    if stats.keys_per_live_frame > MAX_KEYS_PER_LIVE_FRAME:
        out.append(f'over_keyed(k/live={stats.keys_per_live_frame:.2f})')
    return tuple(out)


def element_id(shot: str, layer: str) -> str:
    return f'{shot}__{layer.replace("/", "_").replace(" ", "_")}'


def discover(data_root: str | Path) -> list[Element]:
    """Every top-level layer of every shot, measured and tiered. No EXR involved."""
    out: list[Element] = []
    for shot in find_shots(data_root):
        if shot.sfx is None:
            continue
        doc = read_sfx(shot.sfx, validate=False)
        for root in doc.roots:
            stats = measure(doc, root)
            if stats.shapes == 0:
                continue
            out.append(Element(element_id(shot.name, root.name), shot.name, root.name,
                               root.uuid, stats, exclusions(stats)))
    return out


def resolve(doc: RotoDoc, element: Element) -> list[Candidate]:
    """Locate the element's top-level layer in ``doc``, by uuid with a name cross-check."""
    cands = [c for c in candidate_layers(doc, max_depth=0)]
    if element.uuid:
        for c in cands:
            if c.layer.uuid == element.uuid:
                if c.label != element.layer:
                    raise KeyError(f'{element.element_id}: uuid is {c.label!r}, manifest '
                                   f'says {element.layer!r} -- rediscover the manifest')
                return [c]
    matches = [c for c in cands if c.label == element.layer]
    if len(matches) != 1:
        raise KeyError(f'{element.element_id}: {len(matches)} layers named '
                       f'{element.layer!r} -- rediscover the manifest')
    return matches
