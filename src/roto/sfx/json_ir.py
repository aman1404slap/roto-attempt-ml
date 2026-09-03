"""``roto_ir.json`` -- the IR on disk, in a readable form.

Two things a flat schema loses, and how this one keeps them:

* **Interleaved order.** A layer's children are split into separate ``shapes`` and ``children``
  lists, which loses the order they were authored in. Silhouette composites in document order,
  so a Subtract shape that sat between two others must stay between them. ``child_order``
  records the original sequence.
* **Per-key interpolation.** Interpolation is stored per key, never per track, because the
  archive mixes linear and catmullrom within a single shape.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..ir import (ADD, BEZIER, BSPLINE, XSPLINE, Key, Layer, RotoDoc, Shape,
                  shape_class)

_TYPE_OUT = {BSPLINE: 'Bspline', BEZIER: 'Bezier', XSPLINE: 'Xspline'}
_TYPE_IN = {v.lower(): k for k, v in _TYPE_OUT.items()}

SCHEMA_VERSION = 1


# ---- scalar / vector tracks ----------------------------------------------------

def _fmt(value: Any) -> str:
    """Keyed values are strings: scalars bare, vectors parenthesised."""
    v = np.ravel(np.asarray(value, dtype=np.float64))
    if v.size == 1:
        return repr(float(v[0]))
    return '(' + ','.join(repr(float(x)) for x in v) + ')'


def _track_out(track: list[Key] | None, default: Any = None) -> dict[str, Any] | None:
    if not track:
        return None if default is None else {'const': _fmt(default)}
    if len(track) == 1:
        return {'const': _fmt(track[0].value)}
    return {'keys': [[k.frame, k.interp, _fmt(k.value)] for k in track]}


def _nums(text: Any) -> list[float]:
    if isinstance(text, (int, float)):
        return [float(text)]
    if isinstance(text, (list, tuple)):
        return [float(x) for x in text]
    s = str(text).replace('(', '').replace(')', '')
    return [float(p) for p in s.split(',') if p.strip()]


def _track_in(obj: dict[str, Any] | None) -> list[Key] | None:
    if not obj:
        return None
    if 'keys' in obj:
        return [Key(int(f), str(i or 'linear'), np.array(_nums(v), dtype=np.float64))
                for f, i, v in obj['keys']]
    if obj.get('const') is None:
        return None
    return [Key(0, 'hold', np.array(_nums(obj['const']), dtype=np.float64))]


def _matrix_out(track: list[Key] | None) -> dict[str, Any] | None:
    if not track:
        return None
    if len(track) == 1:
        return {'const': [float(x) for x in np.ravel(track[0].value)]}
    return {'keys': [[k.frame, [float(x) for x in np.ravel(k.value)]] for k in track]}


def _matrix_in(obj: dict[str, Any] | None, interp: list[str] | None) -> list[Key] | None:
    if not obj:
        return None
    if 'const' in obj:
        return [Key(0, 'hold', np.asarray(obj['const'], dtype=np.float64).reshape(4, 4))]
    keys = [Key(int(f), 'linear', np.asarray(v, dtype=np.float64).reshape(4, 4))
            for f, v in obj['keys'] if len(v) == 16]
    for k, mode in zip(keys, interp or []):
        k.interp = mode
    return keys or None


# ---- nodes ---------------------------------------------------------------------

def _shape_out(shape: Shape) -> dict[str, Any]:
    return {
        'label': shape.name,
        'uuid': shape.uuid or '',
        'shape_type': _TYPE_OUT.get(shape.shape_type, 'Bspline'),
        'path_keys': [[k.frame, k.interp, shape.closed,
                       np.asarray(k.value, dtype=np.float64).tolist()]
                      for k in shape.path],
        'opacity': _track_out(shape.opacity, 100.0),
        'strokeWidth': {'const': repr(float(shape.stroke_width))},
        'mode': shape.blend,
        'invert': shape.invert,
        # --- structure a flat schema would lose ---
        'feather': _track_out(shape.feather),
        'shape_class': shape_class(shape),
    }


def _shape_in(obj: dict[str, Any]) -> Shape:
    keys = [Key(int(f), str(i or 'linear'), np.asarray(pts, dtype=np.float64))
            for f, i, _closed, pts in obj['path_keys']]
    closed = bool(obj['path_keys'][0][2]) if obj['path_keys'] else True
    sw = obj.get('strokeWidth')
    return Shape(
        name=obj.get('label', ''),
        shape_type=_TYPE_IN.get(str(obj.get('shape_type', 'Bspline')).lower(), BSPLINE),
        closed=closed,
        path=keys,
        opacity=_track_in(obj.get('opacity')),
        feather=_track_in(obj.get('feather')),
        stroke_width=(_nums(sw['const'])[0] if sw and sw.get('const') is not None else 0.0),
        blend=obj.get('mode') or ADD,
        invert=bool(obj.get('invert', False)),
        uuid=obj.get('uuid') or None,
    )


def _layer_out(layer: Layer) -> dict[str, Any]:
    shapes, children, order = [], [], []
    for child in layer.children:
        if isinstance(child, Shape):
            order.append(['shape', len(shapes)])
            shapes.append(_shape_out(child))
        else:
            order.append(['layer', len(children)])
            children.append(_layer_out(child))
    interp = [k.interp for k in layer.transform] if layer.transform else []
    return {
        'label': layer.name,
        'uuid': layer.uuid or '',
        'matrix': _matrix_out(layer.transform),
        'trs': {name: _track_out(track) for name, track in layer.trs.items()},
        'shapes': shapes,
        'children': children,
        # --- structure a flat schema would lose ---
        'order': order,
        'opacity': _track_out(layer.opacity),
        'mode': layer.blend,
        'invert': layer.invert,
        'matrix_interp': interp if any(m != 'linear' for m in interp) else None,
    }


def _layer_in(obj: dict[str, Any]) -> Layer:
    shapes = [_shape_in(s) for s in obj.get('shapes', [])]
    children = [_layer_in(c) for c in obj.get('children', [])]
    order = obj.get('order')
    if order:
        seq: list[Layer | Shape] = [shapes[i] if kind == 'shape' else children[i]
                                    for kind, i in order]
    else:
        seq = [*shapes, *children]      # no child_order recorded: fall back to document order
    return Layer(
        name=obj.get('label', ''),
        transform=_matrix_in(obj.get('matrix'), obj.get('matrix_interp')),
        trs={n: t for n, t in ((n, _track_in(o)) for n, o in (obj.get('trs') or {}).items())
             if t},
        children=seq,
        opacity=_track_in(obj.get('opacity')),
        blend=obj.get('mode') or ADD,
        invert=bool(obj.get('invert', False)),
        uuid=obj.get('uuid') or None,
    )


# ---- document ------------------------------------------------------------------

def to_json_ir(doc: RotoDoc) -> dict[str, Any]:
    return {
        'source_file': Path(doc.source_path).name if doc.source_path else '',
        'session': {
            'width': doc.width,
            'height': doc.height,
            'startFrame': doc.start_frame,
            'duration': doc.duration,
            'frameRate': doc.frame_rate,
        },
        'layers': [_layer_out(r) for r in doc.roots],
        # --- structure a flat schema would lose ---
        'schema_version': SCHEMA_VERSION,
        'dialect': doc.dialect,
        'source_label': doc.source_label,
        'source_path': doc.source_path,
    }


def from_json_ir(obj: dict[str, Any]) -> RotoDoc:
    sess = obj.get('session') or {}
    return RotoDoc(
        width=int(sess.get('width', 0)),
        height=int(sess.get('height', 0)),
        duration=int(sess.get('duration', 0)),
        roots=[_layer_in(l) for l in obj.get('layers', [])],
        frame_rate=float(sess.get('frameRate', 24.0)),
        start_frame=int(sess.get('startFrame', 1001)),
        dialect=obj.get('dialect'),
        source_path=obj.get('source_path'),
        source_label=obj.get('source_label'),
    )


def write_json_ir(doc: RotoDoc, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_json_ir(doc)))
    return p


def read_json_ir(path: str | Path) -> RotoDoc:
    return from_json_ir(json.loads(Path(path).read_text()))
