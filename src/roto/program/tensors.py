"""The model's target: an element's spline program as dense arrays, and back again.

Phase 1 exists to prove this file is right. If the artist program cannot survive a trip
through these arrays, no model trained on them can reproduce it, and every later measurement
would be measuring the representation's ceiling rather than the model. So the gate is:
``decode(encode(program))`` must render bit-comparably to the packet's own alpha.

Four measured facts shape the representation, all verified on the nine Phase 0 packets:

* **Layer transforms are affine, never perspective** -- row 2 is always ``[0,0,1,0]`` and
  column 3 always ``[0,0,0,1]``. So a track is 6 numbers per frame, not 16. Similarity
  (rotation + uniform scale) covers 7 of 9 elements but *not* ``TVC_sh0260`` or
  ``nfl_0080``, which carry shear -- so the head must be full affine, not 4-DOF.
* **Transforms are shared per shape group, not per shape.** 1310 shapes across the nine
  packets resolve to **64 distinct tracks** -- ``nfl_0200`` alone is 75 shapes over 3 tracks.
  Predicting per shape would be a 20x redundant target and would let the model disagree
  with itself about the motion of one rigid group.
* **Opacity is strictly binary.** Not one of the 1310 shapes ever takes a value between 0
  and 100, so a per-frame live mask is a *lossless* encoding of the lifespan, not an
  approximation of a fade.
* **0.8% of path keys sit outside the rendered frame range** (a key at frame -1, say). A
  key mask spanning only the packet's frames would silently drop them, and dropping a key
  changes how every frame up to the next key interpolates. Hence ``key_axis``, which spans
  every key present, while ``frames`` stays the rendered range.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..ir import Key, Layer, RotoDoc, Shape
from ..sfx.json_ir import read_json_ir

AFFINE_DOF = 6
"""``(m00, m01, m10, m11, tx, ty)`` -- the entries of a 4x4 row-vector matrix that vary."""


def affine_from_matrix(m: np.ndarray) -> np.ndarray:
    """(..., 4, 4) -> (..., 6). Discards entries measured to be constant on all packets."""
    m = np.asarray(m)
    return np.stack([m[..., 0, 0], m[..., 0, 1], m[..., 1, 0], m[..., 1, 1],
                     m[..., 3, 0], m[..., 3, 1]], axis=-1)


def matrix_from_affine(a: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 4, 4). Exact inverse of ``affine_from_matrix`` for affine tracks."""
    a = np.asarray(a)
    out = np.zeros(a.shape[:-1] + (4, 4), dtype=np.float64)
    out[..., 2, 2] = 1.0
    out[..., 3, 3] = 1.0
    out[..., 0, 0], out[..., 0, 1] = a[..., 0], a[..., 1]
    out[..., 1, 0], out[..., 1, 1] = a[..., 2], a[..., 3]
    out[..., 3, 0], out[..., 3, 1] = a[..., 4], a[..., 5]
    return out


@dataclass(slots=True)
class ProgramSpec:
    """What the model is *given* in the teacher-forced setting: the breakdown, not the values.

    Everything here comes straight from the artist's IR and is never predicted in Phase 1/2.
    """
    element_id: str
    frames: np.ndarray            # (T,)   rendered frames, matching alpha/
    key_axis: np.ndarray          # (Tk,)  frame axis for key/live masks; spans every key
    n_points: np.ndarray          # (S,)   control points per shape
    coords_per_point: np.ndarray  # (S,)   1 for B-spline/X-spline, 3 for Bezier
    n_keys: np.ndarray            # (S,)   artist key count per shape
    shape_group: np.ndarray       # (S,)   which transform track each shape follows
    n_groups: int

    @property
    def n_shapes(self) -> int:
        return len(self.n_points)

    @property
    def max_keys(self) -> int:
        return int(self.n_keys.max()) if self.n_shapes else 0

    @property
    def max_points(self) -> int:
        return int(self.n_points.max()) if self.n_shapes else 0

    @property
    def max_coords(self) -> int:
        return int(self.coords_per_point.max()) if self.n_shapes else 1

    def padded_floats(self) -> int:
        """Element count of the padded point tensor -- large elements are worth checking."""
        return (self.n_shapes * self.max_keys * self.max_points * self.max_coords * 2)


@dataclass(slots=True)
class ProgramTensors:
    """What the model predicts. Padded and masked; ``spec`` says what is real."""
    points: np.ndarray       # (S, Kmax, Pmax, Cmax, 2) float32, local normalised coords
    key_slot_mask: np.ndarray   # (S, Kmax) bool -- which key slots exist
    key_frames: np.ndarray      # (S, Kmax) int32  -- frame of each slot, -32768 where padded
    key_interp: list[list[str]]  # per shape, per slot
    key_mask: np.ndarray     # (S, Tk) bool -- is this frame a key for this shape
    live_mask: np.ndarray    # (S, Tk) bool -- opacity non-zero (lossless: opacity is binary)
    affine: np.ndarray       # (G, Tk, 6) float32 -- one track per shape group

    def copy(self) -> ProgramTensors:
        return ProgramTensors(self.points.copy(), self.key_slot_mask.copy(),
                              self.key_frames.copy(), [list(r) for r in self.key_interp],
                              self.key_mask.copy(), self.live_mask.copy(),
                              self.affine.copy())


PAD_FRAME = -32768


def _group_transforms(mats: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Collapse per-shape transform tracks to distinct tracks plus an index.

    Exact equality is the right test here, not a tolerance: shapes under one tracked layer
    share the *same* matrices by construction, so distinct tracks are genuinely distinct.
    """
    keys: dict[bytes, int] = {}
    index = np.empty(len(mats), dtype=np.int32)
    tracks: list[np.ndarray] = []
    for i, m in enumerate(mats):
        k = np.ascontiguousarray(m, dtype=np.float64).tobytes()
        if k not in keys:
            keys[k] = len(tracks)
            tracks.append(m)
        index[i] = keys[k]
    return np.stack(tracks), index


def load_program(packet_dir: str | Path) -> tuple[ProgramSpec, ProgramTensors, dict[str, Any]]:
    """Read a Phase 0 packet into ``(spec, tensors, meta)``."""
    d = Path(packet_dir)
    meta = json.loads((d / 'meta.json').read_text())
    t = np.load(d / 'tensors.npz')
    n = int(t['n_shapes'])
    frames = t['frames'].astype(np.int32)

    per_keys = [t[f'key_frames/{i}'].astype(np.int32) for i in range(n)]
    per_points = [t[f'points_norm/{i}'].astype(np.float32) for i in range(n)]
    lo = min([int(frames.min())] + [int(k.min()) for k in per_keys if len(k)])
    hi = max([int(frames.max())] + [int(k.max()) for k in per_keys if len(k)])
    key_axis = np.arange(lo, hi + 1, dtype=np.int32)

    # Transform tracks are sampled densely over `matrix_frames`; extend to the key axis by
    # holding the end values, which is what the renderer's own sampling does outside a
    # keyed range. `matrix_frames` is dense even when alpha was emitted on a stride, so
    # the result lines up with `key_axis` element for element -- asserted, because a short
    # track reads out of range at decode time rather than producing a wrong number.
    mframes = t['matrix_frames'].astype(np.int32) if 'matrix_frames' in t else frames
    mats = []
    for i in range(n):
        m = t[f'layer_matrix/{i}'].astype(np.float64)
        lead = np.repeat(m[:1], max(0, int(mframes.min()) - lo), axis=0)
        tail = np.repeat(m[-1:], max(0, hi - int(mframes.max())), axis=0)
        m = np.concatenate([lead, m, tail]) if len(lead) or len(tail) else m
        if len(m) != len(key_axis):
            raise ValueError(
                f'transform track for shape {i} has {len(m)} frames but the key axis has '
                f'{len(key_axis)} -- the track was not sampled densely')
        mats.append(m)
    tracks, shape_group = _group_transforms(mats)

    index = {s['i']: s for s in meta['shapes']}
    spec = ProgramSpec(
        element_id=meta['element']['element_id'], frames=frames, key_axis=key_axis,
        n_points=np.array([index[i]['n_points'] for i in range(n)], np.int32),
        coords_per_point=np.array([index[i]['coords_per_point'] for i in range(n)], np.int32),
        n_keys=np.array([len(k) for k in per_keys], np.int32),
        shape_group=shape_group, n_groups=len(tracks))

    S, Kx, Px, Cx, Tk = n, spec.max_keys, spec.max_points, spec.max_coords, len(key_axis)
    points = np.zeros((S, Kx, Px, Cx, 2), np.float32)
    slot = np.zeros((S, Kx), bool)
    kf = np.full((S, Kx), PAD_FRAME, np.int32)
    key_mask = np.zeros((S, Tk), bool)
    live_mask = np.zeros((S, Tk), bool)
    interp: list[list[str]] = []

    for i in range(n):
        k, p = per_keys[i], per_points[i]
        points[i, :len(k), :p.shape[1], :p.shape[2]] = p
        slot[i, :len(k)] = True
        kf[i, :len(k)] = k
        key_mask[i, k - lo] = True
        interp.append(_interp_of(d, i))
        # Opacity, like the transform, is sampled densely over `matrix_frames`. Outside
        # that span the key axis is covered by holding the end value -- a shape live at
        # the first rendered frame is live at a key that precedes it.
        live = t[f'opacity/{i}'] > 0.0
        a = int(mframes.min()) - lo
        live_mask[i, a: a + len(live)] = live
        live_mask[i, :a] = live[0] if len(live) else False
        live_mask[i, a + len(live):] = live[-1] if len(live) else False

    tensors = ProgramTensors(points, slot, kf, interp, key_mask, live_mask,
                             affine_from_matrix(tracks).astype(np.float32))
    return spec, tensors, meta


def _interp_of(packet_dir: Path, shape_index: int) -> list[str]:
    """Per-key interpolation modes, read from target_ir.json in document order."""
    cache = _INTERP_CACHE.get(packet_dir)
    if cache is None:
        ir = json.loads((packet_dir / 'target_ir.json').read_text())

        def walk(layer):
            for s in layer['shapes']:
                yield s
            for c in layer['children']:
                yield from walk(c)

        cache = [[k[1] for k in s['path_keys']]
                 for l in ir['layers'] for s in walk(l)]
        _INTERP_CACHE[packet_dir] = cache
    return cache[shape_index]


_INTERP_CACHE: dict[Path, list[list[str]]] = {}


def decode(packet_dir: str | Path, spec: ProgramSpec, pred: ProgramTensors) -> RotoDoc:
    """Rebuild a renderable ``RotoDoc`` from predicted tensors.

    The *structure* comes from the packet's ``target_ir.json`` -- that is what teacher-forcing
    means -- and everything the model predicts is overwritten: each shape's key frames and
    control points, its lifespan, and its group's transform track. Transform tracks are
    written densely, one key per frame, which is lossless because the renderer only ever
    samples integer frames.
    """
    d = Path(packet_dir)
    doc = read_json_ir(d / 'target_ir.json')
    axis = spec.key_axis
    mats = matrix_from_affine(pred.affine)                  # (G, Tk, 4, 4)

    shapes = [s for _, s in doc.shapes()]
    if len(shapes) != spec.n_shapes:
        raise ValueError(f'template has {len(shapes)} shapes, spec has {spec.n_shapes}')

    for i, shape in enumerate(shapes):
        slots = np.nonzero(pred.key_slot_mask[i])[0]
        np_i, nc_i = int(spec.n_points[i]), int(spec.coords_per_point[i])
        modes = pred.key_interp[i]
        keys = []
        for j, s in enumerate(slots):
            frame = int(pred.key_frames[i, s])
            if frame == PAD_FRAME:
                continue
            value = pred.points[i, s, :np_i, :nc_i, :].astype(np.float64)
            keys.append(Key(frame, modes[j] if j < len(modes) else 'linear', value))
        keys.sort(key=lambda k: k.frame)
        shape.path = keys
        shape.opacity = _lifespan_keys(pred.live_mask[i], axis)

    # One dense transform track per group, attached to the layer each shape sits under.
    for i, (ancestors, _) in enumerate(doc.shapes()):
        track = mats[spec.shape_group[i]]
        for layer in ancestors:
            if layer.transform:
                layer.transform = [Key(int(f), 'linear', track[t])
                                   for t, f in enumerate(axis)]
    return doc


def _lifespan_keys(live: np.ndarray, axis: np.ndarray) -> list[Key] | None:
    """Binary live mask -> hold keys at 0/100, the artist's own on/off idiom.

    Returns ``None`` when the shape is live throughout, so an always-on shape carries no
    opacity track at all rather than a redundant constant one.
    """
    if live.all():
        return None
    keys: list[Key] = []
    prev = False
    for t, f in enumerate(axis):
        on = bool(live[t])
        if on != prev:
            keys.append(Key(int(f), 'hold', np.array([100.0 if on else 0.0])))
            prev = on
    if not keys:                                   # never live: a single 0 key
        return [Key(int(axis[0]), 'hold', np.array([0.0]))]
    if keys[0].frame > int(axis[0]):
        keys.insert(0, Key(int(axis[0]), 'hold', np.array([0.0])))
    return keys
