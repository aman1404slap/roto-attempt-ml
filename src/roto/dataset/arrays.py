"""Per-shape arrays derived from an element's document, plus the shape index."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..ir import RotoDoc, opacity_at, shape_class
from ..render.raster import compose, layer_matrix


def _intervals(mask: np.ndarray, frames: Sequence[int]) -> np.ndarray:
    """Contiguous runs of True in ``mask`` as inclusive ``[start_frame, end_frame]`` rows."""
    out, start = [], None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        elif not on and start is not None:
            out.append([frames[start], frames[i - 1]])
            start = None
    if start is not None:
        out.append([frames[start], frames[len(mask) - 1]])
    return np.array(out, dtype=np.int32).reshape(-1, 2)


def derive_tensors(doc: RotoDoc, frames: Sequence[int]
                   ) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Per-shape arrays for the losses, plus an index describing each shape.

    Arrays are named ``<field>/<i>`` where ``i`` is the shape's position in the index, which
    is document order -- the deterministic teacher-forcing sequence the handoff asks for.
    """
    # The transform track is a property of the document, not of which frames we chose to
    # rasterise, so it is sampled densely even when alpha is emitted on a stride. Sampling
    # it on the stride instead leaves it shorter than the key axis it is later indexed by,
    # which is an out-of-range read rather than a wrong number -- silent only until it is
    # not. Dense costs nothing here: composing matrices does no rasterisation.
    dense = list(range(int(min(frames)), int(max(frames)) + 1)) if len(frames) else []
    tensors: dict[str, np.ndarray] = {'frames': np.asarray(frames, dtype=np.int32),
                                      'matrix_frames': np.asarray(dense, dtype=np.int32)}
    index: list[dict[str, Any]] = []

    for i, (ancestors, shape) in enumerate(doc.shapes()):
        key_frames = np.array([k.frame for k in shape.path], dtype=np.int32)
        points = np.stack([np.asarray(k.value, dtype=np.float32) for k in shape.path])
        opacity = np.array([opacity_at(shape, f) for f in dense], dtype=np.float32)
        mats = np.stack([
            compose([m for m in (layer_matrix(l, f) for l in ancestors) if m is not None])
            if any(l.transform or l.trs for l in ancestors) else np.eye(4)
            for f in dense]).astype(np.float32)

        tensors[f'key_frames/{i}'] = key_frames
        tensors[f'points_norm/{i}'] = points          # (n_keys, n_points, k, 2), local
        tensors[f'opacity/{i}'] = opacity             # (Tdense,) in [0,1]
        tensors[f'lifespan/{i}'] = _intervals(opacity > 0.0, dense)
        tensors[f'layer_matrix/{i}'] = mats           # (Tdense, 4, 4), composed

        index.append({
            'i': i,
            'name': shape.name,
            'layer_path': [l.name for l in ancestors],
            'shape_type': shape.shape_type,
            'shape_class': shape_class(shape),
            'closed': bool(shape.closed),
            'stroke_width': float(shape.stroke_width),
            'blend': shape.blend,
            'invert': bool(shape.invert),
            'n_points': int(shape.n_points),
            'n_keys': int(len(shape.path)),
            'coords_per_point': int(points.shape[2]),
            'key_frames': key_frames.tolist(),
            'transform_animated': bool(any(l.transform_is_animated for l in ancestors)),
        })
    tensors['n_shapes'] = np.array(len(index), dtype=np.int32)
    return tensors, index
