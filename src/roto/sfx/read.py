"""Read Silhouette .sfx project files into the canonical IR.

Handles both containers seen in the archive:

* plain XML beginning ``<!-- Silhouette Project File -->``
* big-endian uint32 uncompressed-length followed by a zlib stream

and all four dialects measured so far (v2020, v2022.5, v5, v2025.5). The dialects differ
in container, in which optional properties exist, and in property naming -- but *not* in
geometry conventions, which is why one code path suffices.
"""
from __future__ import annotations

import re
import struct
import zlib
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from ..ir import (ADD, BEZIER, BSPLINE, XSPLINE, Key, Layer, RotoDoc, Shape)

_PAIR = re.compile(r'\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)')
_NUM = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')

_SHAPE_TYPES = {'bspline': BSPLINE, 'bezier': BEZIER, 'xspline': XSPLINE}


class SfxParseError(RuntimeError):
    pass


# ---- container -----------------------------------------------------------------

def read_xml_bytes(path: str | Path) -> tuple[bytes, str]:
    """Return ``(xml_bytes, container)`` for a .sfx file."""
    raw = Path(path).read_bytes()
    if raw[:4] == b'<!--':
        return raw, 'plain'
    if len(raw) < 5:
        raise SfxParseError(f'{path}: too short to be a .sfx')
    declared = struct.unpack('>I', raw[:4])[0]
    try:
        xml = zlib.decompress(raw[4:])
    except zlib.error as exc:
        raise SfxParseError(f'{path}: not plain XML and zlib failed: {exc}') from exc
    if len(xml) != declared:
        raise SfxParseError(
            f'{path}: header declares {declared} bytes, decompressed {len(xml)}')
    return xml, 'zlib'


# ---- small helpers -------------------------------------------------------------

def _props(el: ET.Element) -> dict[str, ET.Element]:
    container = el.find('Properties')
    if container is None:
        return {}
    return {p.get('id'): p for p in container.findall('Property') if p.get('id')}


def _value(props: dict[str, ET.Element], name: str) -> str | None:
    p = props.get(name)
    if p is None:
        return None
    v = p.find('Value')
    return None if v is None else (v.text or '')


def _float(props: dict[str, ET.Element], name: str, default: float = 0.0) -> float:
    txt = _value(props, name)
    if not txt:
        return default
    m = _NUM.search(txt)
    return float(m.group()) if m else default


def _bool(props: dict[str, ET.Element], name: str) -> bool:
    return (_value(props, name) or '').strip().lower() == 'true'


def _pairs(text: str | None) -> np.ndarray:
    if not text:
        return np.zeros((0, 2))
    return np.array([[float(a), float(b)] for a, b in _PAIR.findall(text)],
                    dtype=np.float64)


def _scalar_track(prop: ET.Element | None) -> list[Key] | None:
    """A track of single numbers, e.g. opacity."""
    if prop is None:
        return None
    keys = prop.findall('Key')
    if not keys:
        return None
    out = []
    for k in keys:
        m = _NUM.search(k.text or '')
        if m is None:
            continue
        out.append(Key(int(k.get('frame')), k.get('interp') or 'linear',
                       np.array([float(m.group())])))
    out.sort(key=lambda k: k.frame)
    return out or None


_TRS_DEFAULTS = {'position': (0.0, 0.0), 'anchor': (0.0, 0.0),
                 'scale': (1.0, 1.0), 'rotate': (0.0,)}


def _numeric_track(prop: ET.Element | None) -> list[Key] | None:
    """A track of numeric vectors. Falls back to a constant ``<Value>`` as a single key.

    Unlike ``_scalar_track`` this keeps every component, which TRS needs: ``position`` is a
    vector and truncating it to its first number would silently drop the y translation.
    """
    if prop is None:
        return None
    out = []
    for k in prop.findall('Key'):
        nums = [float(m.group()) for m in _NUM.finditer(k.text or '')]
        if not nums:
            continue
        out.append(Key(int(k.get('frame')), k.get('interp') or 'linear',
                       np.array(nums, dtype=np.float64)))
    if out:
        out.sort(key=lambda k: k.frame)
        return out
    txt = prop.find('Value')
    nums = [float(m.group()) for m in _NUM.finditer((txt.text or '') if txt is not None else '')]
    return [Key(0, 'hold', np.array(nums, dtype=np.float64))] if nums else None


def _trs_tracks(props: dict[str, ET.Element]) -> dict[str, list[Key]]:
    """Non-identity ``transform.*`` tracks only -- identity ones are noise in the IR."""
    out: dict[str, list[Key]] = {}
    for name, default in _TRS_DEFAULTS.items():
        track = _numeric_track(props.get(f'transform.{name}'))
        if track is None:
            continue
        if len(track) == 1:
            v = np.ravel(track[0].value)[:len(default)]
            if len(v) == len(default) and np.allclose(v, default):
                continue                      # constant identity: drop it
        out[name] = track
    return out


def _matrix_track(prop: ET.Element | None) -> list[Key] | None:
    """A track of 4x4 matrices. Empty ``<Value/>`` means identity -- represented as None."""
    if prop is None:
        return None
    out = []
    for k in prop.findall('Key'):
        nums = [float(m.group()) for m in _NUM.finditer(k.text or '')]
        if len(nums) != 16:
            continue                      # blank or malformed key: skip, treat as absent
        out.append(Key(int(k.get('frame')), k.get('interp') or 'linear',
                       np.array(nums, dtype=np.float64).reshape(4, 4)))
    out.sort(key=lambda k: k.frame)
    return out or None


def _path_track(prop: ET.Element | None, name: str) -> tuple[list[Key], bool, str | None]:
    """Returns ``(keys, closed, path_type)``. Key values are (n_points, k, 2)."""
    if prop is None:
        return [], True, None
    keys, closed, ptype = [], None, None
    for k in prop.findall('Key'):
        pel = k.find('Path')
        if pel is None:
            continue
        rows = [_pairs(p.text) for p in pel.findall('Point')]
        rows = [r for r in rows if len(r)]
        if not rows:
            continue
        widths = {len(r) for r in rows}
        if len(widths) != 1:
            raise SfxParseError(
                f'shape {name!r} frame {k.get("frame")}: mixed coords-per-point {widths}')
        arr = np.stack(rows)                              # (n_points, k, 2)
        keys.append(Key(int(k.get('frame')), k.get('interp') or 'linear', arr))
        if closed is None:
            closed = pel.get('closed') == 'True'
            ptype = pel.get('type')
    keys.sort(key=lambda k: k.frame)
    return keys, (True if closed is None else closed), ptype


# ---- object tree ---------------------------------------------------------------

def _shape(el: ET.Element) -> Shape | None:
    props = _props(el)
    name = el.get('label') or el.get('id') or '?'
    keys, closed, path_type = _path_track(props.get('path'), name)
    if not keys:
        return None
    declared = (el.get('shape_type') or path_type or 'Bspline').lower()
    shape_type = _SHAPE_TYPES.get(declared)
    if shape_type is None:
        raise SfxParseError(f'shape {name!r}: unknown shape_type {declared!r}')
    return Shape(
        name=name,
        shape_type=shape_type,
        closed=closed,
        path=keys,
        opacity=_scalar_track(props.get('opacity')),
        feather=_scalar_track(props.get('feather')),
        stroke_width=_float(props, 'strokeWidth'),
        blend=(_value(props, 'mode') or ADD).strip() or ADD,
        invert=_bool(props, 'invert'),
        uuid=el.get('uuid'),
    )


def _layer(el: ET.Element) -> Layer:
    props = _props(el)
    layer = Layer(
        name=el.get('label') or el.get('id') or '?',
        transform=_matrix_track(props.get('transform.matrix')),
        trs=_trs_tracks(props),
        opacity=_scalar_track(props.get('opacity')),
        blend=(_value(props, 'mode') or ADD).strip() or ADD,
        invert=_bool(props, 'invert'),
        uuid=el.get('uuid'),
    )
    objects = props.get('objects')
    if objects is not None:
        for child in objects.findall('Object'):
            kind = child.get('type')
            if kind == 'Layer':
                layer.children.append(_layer(child))
            elif kind == 'Shape':
                shape = _shape(child)
                if shape is not None:
                    layer.children.append(shape)
            else:
                raise SfxParseError(f'unknown Object type {kind!r}')
    return layer


# ---- entry point ---------------------------------------------------------------

def read_sfx(path: str | Path, validate: bool = True) -> RotoDoc:
    xml, container = read_xml_bytes(path)
    root = ET.fromstring(xml)

    width = height = None
    source_path = source_label = None
    for src in root.iter('Source'):
        if src.get('width'):
            width, height = int(src.get('width')), int(src.get('height'))
            node = src.find('Path')
            source_path = node.text if node is not None else None
            source_label = src.get('label')
    if width is None:
        raise SfxParseError(f'{path}: no Source element carrying width/height')

    duration, frame_rate, start_frame = 0, 24.0, 1001
    for sess in root.iter('Session'):
        sp = _props(sess)
        duration = int(_float(sp, 'duration', duration))
        frame_rate = _float(sp, 'frameRate', frame_rate)
        start_frame = int(_float(sp, 'startFrame', start_frame))

    roots: list[Layer] = []
    for node in root.iter('Node'):
        if node.get('type') != 'RotoNode':
            continue
        objects = _props(node).get('objects')
        if objects is None:
            continue
        for obj in objects.findall('Object'):
            if obj.get('type') == 'Layer':
                roots.append(_layer(obj))
            elif obj.get('type') == 'Shape':
                shape = _shape(obj)
                if shape is not None:                 # shapes can sit at node level
                    roots.append(Layer(name=f'<bare:{shape.name}>', children=[shape]))

    doc = RotoDoc(width=width, height=height, duration=duration, roots=roots,
                  frame_rate=frame_rate, start_frame=start_frame,
                  dialect=f'{root.get("version")}/{container}',
                  source_path=source_path, source_label=source_label)
    if validate:
        doc.validate()
    return doc
