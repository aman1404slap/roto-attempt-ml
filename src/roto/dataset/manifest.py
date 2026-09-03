"""Which Silhouette layers become training samples.

A training sample is **one top-level layer of one ``.sfx``**, because that is the unit that
renders to one matte. The list is derived from the files themselves -- there is no
hand-maintained table.

Exclusion is by measurement, and each rule states the number it fires on. ``keys_per_live_frame``
is how densely the artist keyed the layer; ``open_frac`` is the share of its shapes that are
stroked open paths rather than filled regions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ..shots import Shot, find_shots
from .layers import LayerRef, top_layers
from ..ir import Layer, RotoDoc, Shape, opacity_at
from ..sfx.read import read_sfx

MANIFEST_VERSION = 2

MAX_KEYS_PER_LIVE_FRAME = 0.75
"""Above this the layer is keyed so densely it teaches key-every-frame.

Measured across the 18 top-level layers: 14 sit at 0.03-0.59, and four sit at 0.95-1.59.
There is no continuum between them -- the gap is a factor of 1.6 -- so the threshold is
reading a real bimodality rather than cutting an arbitrary tail. Over-keying is the
project's measured failure mode (precision 0.21-0.44, ~2.5x too many keys), and these are
the layers that would teach it.
"""

MAX_OPEN_FRACTION = 0.9
"""Above this the layer is a paint-stroke pass, not region roto.

Fires on sh0230/L110 (2515 of 2515 shapes open) and nfl_0080/MB 2 (508 of 510). An open
stroke is rendered as a width along a path, not as a filled region, so it is a different
prediction target wearing the same B-spline clothes -- the model would have to learn a
stroke width it is not asked to predict.

The two are otherwise unalike, and it is worth not conflating them. L110 is hair: shapes
live 4.2 frames and carry 4.0 keys each, so it is redrawn rather than animated, and it trips
``over_keyed`` as well. MB 2 is a motion-blur pass that is genuinely animated -- 81.4 live
frames and 25.0 keys per shape, a k/live of 0.31 that sits comfortably inside the kept range.
MB 2 is excluded for what it draws, not for how it is keyed.
"""

MIN_SHAPES = 3
"""Below this there is no set to predict. Fires on FAM_0060/red (2 shapes)."""


@dataclass(slots=True)
class LayerStats:
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
class RotoLayer:
    """One top-level layer of one shot: the unit the dataset is built from."""
    layer_id: str
    shot: str
    name: str
    uuid: str | None
    stats: LayerStats
    excluded_by: tuple[str, ...] = ()

    @property
    def is_target(self) -> bool:
        return not self.excluded_by

    def as_dict(self) -> dict[str, Any]:
        return {'layer_id': self.layer_id, 'shot': self.shot, 'name': self.name,
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


def measure(doc: RotoDoc, root: Layer) -> LayerStats:
    shapes = list(_shapes_of(root))
    groups = list(_groups_of(root)) or [0]
    live = sum(1 for s in shapes for f in range(doc.duration) if opacity_at(s, f) > 0.5)
    return LayerStats(
        shapes=len(shapes),
        open_strokes=sum(1 for s in shapes if not s.closed),
        ephemeral=sum(1 for s in shapes if s.opacity and len(s.opacity) > 1),
        points=sum(s.n_points for s in shapes),
        keys=sum(len(s.path) for s in shapes),
        live_frame_shapes=live,
        groups=len(groups),
        max_group=max(groups),
    )


def exclusions(stats: LayerStats) -> tuple[str, ...]:
    out = []
    if stats.shapes < MIN_SHAPES:
        out.append(f'too_few_shapes({stats.shapes}<{MIN_SHAPES})')
    if stats.open_frac > MAX_OPEN_FRACTION:
        out.append(f'paint_strokes(open_frac={stats.open_frac:.2f})')
    if stats.keys_per_live_frame > MAX_KEYS_PER_LIVE_FRAME:
        out.append(f'over_keyed(k/live={stats.keys_per_live_frame:.2f})')
    return tuple(out)


def layer_id(shot: str, layer: str) -> str:
    return f'{shot}__{layer.replace("/", "_").replace(" ", "_")}'


def discover(data_root: str | Path) -> list[RotoLayer]:
    """Every top-level layer of every shot, measured and screened against the rules above."""
    out: list[RotoLayer] = []
    for shot in find_shots(data_root):
        if shot.sfx is None:
            continue
        doc = read_sfx(shot.sfx, validate=False)
        for root in doc.roots:
            stats = measure(doc, root)
            if stats.shapes == 0:
                continue
            out.append(RotoLayer(layer_id(shot.name, root.name), shot.name, root.name,
                               root.uuid, stats, exclusions(stats)))
    return out


def resolve(doc: RotoDoc, layer: RotoLayer) -> LayerRef:
    """Find this layer in ``doc``, by uuid where there is one, with a name cross-check.

    A uuid/name disagreement is raised rather than resolved silently -- it means the manifest
    was built from a different save of the project than the one being read.
    """
    refs = top_layers(doc)
    if layer.uuid:
        for r in refs:
            if r.layer.uuid == layer.uuid:
                if r.name != layer.name:
                    raise KeyError(f'{layer.layer_id}: uuid is {r.name!r}, manifest says '
                                   f'{layer.name!r} -- rediscover the manifest')
                return r
    matches = [r for r in refs if r.name == layer.name]
    if len(matches) != 1:
        raise KeyError(f'{layer.layer_id}: {len(matches)} layers named {layer.name!r} '
                       f'-- rediscover the manifest')
    return matches[0]
