"""A layer's spline program as dense arrays, and back again.

``decode(encode(program))`` must render to the same pixels as the layer's own matte. If the
artist's program cannot survive that trip, no model trained on these arrays could reproduce
it, and every later measurement would be reporting the representation's ceiling rather than
the model's skill. That round trip is an automated test.

Four measured facts shape the representation:

* **Layer transforms are projective, not affine** -- row 2 and column 2 are always
  ``[0,0,1,0]``, but column 3 is *not* always ``[0,0,0,1]``. So a track is 8 numbers per frame
  (a normalised 2D homography), not 6 and not 16. Similarity covers most layers but not all:
  two carry shear and two carry perspective. See :func:`proj_from_matrix`.

  ``ProgramTensors.transform`` carries those 8 numbers as of v1.2. It carried
  :func:`affine_from_matrix`'s 6 through v1 and v1.1, which meant the round trip at the top of
  this docstring *did not hold* on the two perspective layers -- by 350 crop px on one of
  them. ``affine_from_matrix`` is kept, and kept documented as lossy, only so that v1's
  transform-head numbers still reproduce.
* **Transforms are shared per shape group, not per shape.** The 2753 shapes in this archive
  resolve to 208 distinct tracks. Predicting per shape would be a redundant target and would
  let the model disagree with itself about the motion of one rigid group.
* **Shape opacity is strictly binary.** No shape ever takes a value between 0 and 100, so a
  per-frame live mask is a *lossless* encoding of when a shape exists, not an approximation of
  a fade.
* **A few keyframes sit outside the rendered frame range** (a key at frame -1, say). A key mask
  spanning only the rendered frames would silently drop them, and dropping a key changes how
  every frame up to the next one interpolates. Hence ``key_axis``, which spans every key
  present, while ``frames`` stays the rendered range.
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
"""``(m00, m01, m10, m11, tx, ty)`` -- v1's transform target. **Measured lossy; see below.**"""

PROJ_DOF = 8
"""``(m00, m01, m02w, m10, m11, m12w, tx, ty)`` -- a normalised 2D homography.

The entries of a 4x4 row-vector layer matrix that actually vary, after dividing the whole
matrix through by ``m33``. Row 2 and column 2 are exactly ``[0,0,1,0]`` on every sample in
the archive, so nothing else is carried.
"""

PROJ_ENTRIES = ((0, 0), (0, 1), (0, 3), (1, 0), (1, 1), (1, 3), (3, 0), (3, 1))
"""Which entries :func:`proj_from_matrix` keeps, in order. ``m33`` is the normaliser."""


def affine_from_matrix(m: np.ndarray) -> np.ndarray:
    """(..., 4, 4) -> (..., 6). **Lossy on two of the thirteen layers -- prefer
    :func:`proj_from_matrix`.**

    This drops the perspective column ``(m03, m13, m33)`` on the assumption, stated in this
    module's own header until v1.1 measured it, that layer transforms are affine. Eleven
    layers satisfy that. Two do not: ``TVC_sh0260 Layer_52`` runs ``m33`` from 0.672 to 1.801
    and ``FAM red_1`` carries ``m03`` up to 0.028, so ``matrix_from_affine`` is not the
    inverse of this function on them.

    The cost is not a rounding error. Round-tripping the *artist's own* transform track
    through these 6 numbers and re-rendering -- the case that must come out at 1.000 -- moves
    control points by 0.45 crop px mean and 6.2 px worst on ``red_1``, which renders at
    0.9100 soft IoU, and by 23.8 px worst on ``Layer_52``. A transform head trained against
    this target therefore cannot be scored above that ceiling no matter how well it predicts,
    which is most of why v1's head read 0.756 and looked unusable.

    Kept, unchanged, because every v1 number was measured through it.
    """
    m = np.asarray(m)
    return np.stack([m[..., 0, 0], m[..., 0, 1], m[..., 1, 0], m[..., 1, 1],
                     m[..., 3, 0], m[..., 3, 1]], axis=-1)


def proj_from_matrix(m: np.ndarray) -> np.ndarray:
    """(..., 4, 4) -> (..., 8). Lossless for every transform in this archive.

    A projective map is defined only up to overall scale, so the matrix is first divided
    through by ``m33`` to fix the gauge. That is safe here and not in general: ``m33`` is
    measured to stay in [0.672, 1.801] across the archive, never near zero. Fixing the gauge
    rather than predicting nine free numbers is what stops a scale-invariant target from
    drifting toward the degenerate ``m33 -> 0``.

    Strictly contains :func:`affine_from_matrix`: an affine track has ``m03 = m13 = 0`` and
    ``m33 = 1``, so three of these eight entries are then constant.
    """
    m = np.asarray(m, np.float64)
    w = m[..., 3, 3]
    assert np.all(np.abs(w) > 1e-9), 'm33 at zero: the gauge normalisation is undefined'
    m = m / w[..., None, None]
    return np.stack([m[..., r, c] for r, c in PROJ_ENTRIES], axis=-1)


def matrix_from_proj(a: np.ndarray) -> np.ndarray:
    """(..., 8) -> (..., 4, 4). Exact inverse of :func:`proj_from_matrix` as a map on the
    plane -- measured at 7e-16 normalised units across every matrix in the archive, against
    0.98 for the 6-number affine form.

    "As a map on the plane" is the necessary qualifier and not a hedge. Dividing through by
    ``m33`` leaves ``m22 = 1/m33``, so the reconstructed matrix is not entry-for-entry equal
    to the original. It does not need to be: ``render.raster.apply_transform`` lifts points as
    ``[x, y, 0, 1]``, so row 2 is multiplied by zero and column 2 is never read. ``m22`` is
    written as 1 here because that is the value the rest of the pipeline expects to see.
    """
    a = np.asarray(a)
    out = np.zeros(a.shape[:-1] + (4, 4), dtype=np.float64)
    out[..., 2, 2] = 1.0
    out[..., 3, 3] = 1.0
    for i, (r, c) in enumerate(PROJ_ENTRIES):
        out[..., r, c] = a[..., i]
    return out


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

    Everything here comes straight from the artist's IR and is never predicted.
    """
    layer_id: str
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
        """RotoLayer count of the padded point tensor -- large layers are worth checking."""
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
    transform: np.ndarray    # (G, Tk, 8) float32 -- one projective track per shape group
    """The layer transform track, as ``PROJ_DOF`` numbers per frame.

    **This was 6 numbers -- ``affine_from_matrix`` -- through v1 and v1.1, and it was wrong.**
    The module header has said since v1.1 that these transforms are projective and that the
    6-number form is lossy on two layers; the header said it while this field still carried
    it, so ``decode(encode(program))`` did not round trip on those two layers. Measured on the
    archive's own matrices, the 6-number form displaces a planar point by up to 0.086
    normalised units on ``FAM red_1`` and **1.564 on ``Layer_52``** -- 350 crop px, a shape
    rendered off its own crop -- while the 8-number form is exact to 7e-16.

    It survived because the round-trip test covers ``nfl_0200 blue``, whose track *is* affine,
    so the case that fails was the case not sampled. ``scripts/ledger.py`` walks the
    perspective layers instead, which is how it was found."""

    def copy(self) -> ProgramTensors:
        return ProgramTensors(self.points.copy(), self.key_slot_mask.copy(),
                              self.key_frames.copy(), [list(r) for r in self.key_interp],
                              self.key_mask.copy(), self.live_mask.copy(),
                              self.transform.copy())


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


def load_program(layer_dir: str | Path) -> tuple[ProgramSpec, ProgramTensors, dict[str, Any]]:
    """Read one layer sample into ``(spec, tensors, meta)``."""
    d = Path(layer_dir)
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
    # the result lines up with `key_axis` layer for layer -- asserted, because a short
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
        layer_id=meta['layer']['layer_id'], frames=frames, key_axis=key_axis,
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
                             proj_from_matrix(tracks).astype(np.float32))
    return spec, tensors, meta


def _interp_of(layer_dir: Path, shape_index: int) -> list[str]:
    """Per-key interpolation modes, read from target_ir.json in document order."""
    cache = _INTERP_CACHE.get(layer_dir)
    if cache is None:
        ir = json.loads((layer_dir / 'target_ir.json').read_text())

        def walk(layer):
            for s in layer['shapes']:
                yield s
            for c in layer['children']:
                yield from walk(c)

        cache = [[k[1] for k in s['path_keys']]
                 for l in ir['layers'] for s in walk(l)]
        _INTERP_CACHE[layer_dir] = cache
    return cache[shape_index]


_INTERP_CACHE: dict[Path, list[list[str]]] = {}


def decode(layer_dir: str | Path, spec: ProgramSpec, pred: ProgramTensors) -> RotoDoc:
    """Rebuild a renderable ``RotoDoc`` from predicted tensors.

    The *structure* comes from the sample's ``target_ir.json`` -- that is what teacher-forcing
    means -- and everything the model predicts is overwritten: each shape's key frames and
    control points, its lifespan, and its group's transform track. Transform tracks are
    written densely, one key per frame, which is lossless because the renderer only ever
    samples integer frames.
    """
    d = Path(layer_dir)
    doc = read_json_ir(d / 'target_ir.json')
    axis = spec.key_axis
    mats = matrix_from_proj(pred.transform)                 # (G, Tk, 4, 4)

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
