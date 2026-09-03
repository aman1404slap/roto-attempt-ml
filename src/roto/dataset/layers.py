"""Finding the roto layers in a Silhouette document, and isolating one for rendering.

A Silhouette project nests layers; the ones that matter here are the **top-level** layers,
because each of those is what gets rendered to a matte. A layer holds shapes -- B-splines --
and a transform track, usually from a tracker.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..ir import Layer, RotoDoc


@dataclass(slots=True)
class LayerRef:
    """A layer plus the ancestor chain needed to compose its transform."""
    name: str
    ancestors: tuple[Layer, ...]
    layer: Layer


def top_layers(doc: RotoDoc) -> list[LayerRef]:
    """The document's top-level layers, in document order."""
    return [LayerRef(root.name, (), root) for root in doc.roots]


def layer_doc(doc: RotoDoc, ref: LayerRef) -> RotoDoc:
    """A document containing only ``ref``, with its ancestor transforms still composing.

    Render the result with ``render_union``, never plain ``render``: shapes within a layer
    overlap heavily -- measured up to 2.5x the layer's own silhouette area -- and add-and-clip
    is not union. Adding two overlapping soft edges overshoots, and a Subtract shape must not
    punch through a shape it does not belong to.
    """
    root = doc.isolate(ref.ancestors, ref.layer).roots[0]
    return RotoDoc(doc.width, doc.height, doc.duration, [root], doc.frame_rate,
                   doc.start_frame, doc.dialect, doc.source_path, doc.source_label)
