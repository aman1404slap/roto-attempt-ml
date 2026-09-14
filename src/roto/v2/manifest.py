"""Which Silhouette layers become v2 elements.

An **element** is one top-level layer of one shot's chosen ``.sfx``, because that is the unit
that renders to one matte. The list is derived from the files; there is no hand-maintained
table.

**v2 tags, it does not exclude.** Charter S6 asks for "elements from the .sfx layer tree
(gold = EXR-matched, silver = tree-defined, tagged in meta)", and D4 in ``v2-tracker.md``
records the decision that every layer trains. That is a deliberate break from the archive
manifest, which dropped layers by thresholds (``MAX_KEYS_PER_LIVE_FRAME``, ``MAX_OPEN_FRACTION``,
``MIN_SHAPES``) calibrated on 18 archive layers -- numbers with no standing on a 50-shot
delivery from different artists. The same properties are still *measured*; they ride along as
tags so a later round can slice on them, and so charter S5's per-layer floors can attribute a
failure to a population rather than to a layer id.

The one thing that does remove an element is a shot failing ingest QC, which is a different
question and lives in :mod:`roto.v2.qc`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ..dataset.layers import LayerRef, layer_doc, top_layers
from ..ir import RotoDoc, is_ephemeral, live_frames
from ..sfx.read import read_sfx
from .ingest import Shot

MANIFEST_VERSION = 1
"""v2's own lineage. Unrelated to ``roto.dataset.manifest``'s version 2, which counts the
archive's exclusion rules and is not carried forward."""

_SAFE = re.compile(r'[^A-Za-z0-9_.-]+')


def element_id(shot: str, layer_name: str) -> str:
    """``<shot>__<layer>`` with anything path-hostile collapsed to ``_``."""
    return f'{shot}__{_SAFE.sub("_", layer_name).strip("_") or "unnamed"}'


@dataclass(slots=True)
class Element:
    """One top-level layer, with the measurements that later become tags."""
    element_id: str
    shot: str
    layer: str
    layer_index: int
    uuid: str | None
    sfx: Path
    duration: int
    start_frame: int
    width: int
    height: int
    stats: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {'element_id': self.element_id, 'shot': self.shot, 'layer': self.layer,
                'layer_index': self.layer_index, 'uuid': self.uuid, 'sfx': str(self.sfx),
                'duration': self.duration, 'start_frame': self.start_frame,
                'width': self.width, 'height': self.height,
                'stats': self.stats, 'tags': self.tags}


def _tags(sub: RotoDoc, stats: dict[str, Any]) -> dict[str, Any]:
    """Descriptive tags. Every one of these was an exclusion rule in the archive manifest.

    ``keys_per_live_frame`` is how densely the artist keyed; ``open_fraction`` is the share of
    shapes that are stroked open paths rather than filled regions; ``ephemeral_fraction`` is
    the share that live only a frame or two. Recorded, never enforced.
    """
    shapes = [s for _, s in sub.shapes()]
    live = sum(len(live_frames(s)) for s in shapes)
    n = len(shapes)
    return {
        'shapes': n,
        'open_fraction': round(stats['open_strokes'] / n, 4) if n else 0.0,
        'ephemeral_fraction': round(sum(1 for s in shapes if is_ephemeral(s)) / n, 4) if n else 0.0,
        'keys_per_live_frame': round(stats['keys'] / live, 4) if live else 0.0,
        'points_per_shape': round(stats['points'] / n, 2) if n else 0.0,
        'tracked': bool(stats['tracked_layers']),
        'animated_transform': bool(stats['animated_transforms']),
        # Populations charter S5 will want to attribute a per-layer failure to.
        'trivial': n <= 1,
        'paint_pass': (stats['open_strokes'] / n) > 0.9 if n else False,
        'over_keyed': (stats['keys'] / live) > 0.75 if live else False,
    }


def elements_for(shot: Shot) -> list[Element]:
    """Every top-level layer of one shot's chosen ``.sfx``, in document order."""
    if shot.sfx is None:
        return []
    doc = read_sfx(shot.sfx)
    out = []
    for i, ref in enumerate(top_layers(doc)):
        sub = layer_doc(doc, ref)
        stats = sub.stats()
        out.append(Element(
            element_id=element_id(shot.name, ref.name), shot=shot.name, layer=ref.name,
            layer_index=i, uuid=ref.layer.uuid, sfx=shot.sfx,
            duration=doc.duration, start_frame=doc.start_frame,
            width=doc.width, height=doc.height,
            stats=stats, tags=_tags(sub, stats)))
    return out


def elements(shots: list[Shot]) -> list[Element]:
    """Elements for every usable shot, skipping those ingest could not resolve."""
    out: list[Element] = []
    for s in shots:
        if s.usable:
            out.extend(elements_for(s))
    return out


def resolve(doc: RotoDoc, element: Element) -> LayerRef:
    """The ``LayerRef`` an element names, matched by uuid first and name second.

    uuid first because a layer can be renamed between the manifest being written and a
    dataset being rebuilt, and a silently mismatched layer is the failure this guards.
    """
    refs = top_layers(doc)
    for ref in refs:
        if element.uuid and ref.layer.uuid == element.uuid:
            return ref
    for ref in refs:
        if ref.name == element.layer:
            return ref
    raise KeyError(f'{element.element_id}: no layer with uuid {element.uuid!r} '
                   f'or name {element.layer!r} in {element.sfx}')
