"""RotoLayer -> tensors the network trains on.

Targets are the *dense per-frame* control points, not the artist's key values. The network
predicts geometry frame by frame; choosing which of those frames become keys is the DP's job
(``roto.keys``), and keeping the two separate is what lets each be measured on its own.

Dense tracks are built with ``roto.ir.sample`` rather than by interpolating the key arrays
directly, so the target is exactly what the renderer would draw -- per-key interp modes
included. The archive is ~75% linear and ~25% catmullrom, varying by shot, so re-deriving the
interpolation here would silently disagree with the picture on a quarter of the segments.

Targets are stored in **crop space**, in [0,1] across the alpha the network is given, not in
the IR's local normalised coordinates. See ``roto.model.geometry`` -- predicting local
coordinates directly was measured to stall at ~260 px, because they are absolute document
positions whose range is many times the crop the network can see. ``local`` is kept alongside
so the conversion can be checked, and the local points are what the IR is rebuilt from.

The **transform** target now follows the same rule, and did not before. ``affine`` is the
document-space 6-number form v1 predicted; ``proj_crop`` is the same track expressed as the
local-to-crop map of each frame, in the 8-number projective form that is actually lossless.
Both are carried so the two can be compared on one checkpoint, but only ``proj_crop`` states
the target in the space the picture depicts. See :func:`~roto.model.geometry.crop_matrix` and
:func:`~roto.program.proj_from_matrix`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..dataset import load_alpha, load_splits
from ..ir import opacity_at, sample
from .geometry import crop_matrix, local_to_crop
from ..program import affine_from_matrix, proj_from_matrix
from ..sfx.json_ir import from_json_ir


def shift_alpha(a: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate one alpha by ``(dx, dy)`` crop pixels, bilinear, zero outside.

    Forward mapping: content at ``p`` in the input lands at ``p + (dx, dy)`` in the output,
    which is the direction :meth:`LayerData.window` needs -- a point of the plane sits at
    ``(source - offset) * scale`` in each frame's own window, so moving a neighbour into the
    anchor's window is a shift by ``(offset_neighbour - offset_anchor) * scale``.
    """
    m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    return cv2.warpAffine(a, m, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def group_probes(local: np.ndarray, live: np.ndarray, group_of: np.ndarray,
                 point_mask: np.ndarray, n_groups: int) -> np.ndarray:
    """``(G, 5, 2)`` probe points per transform group, in local normalised coordinates.

    A transform target needs a loss in pixels, the same unit as the point target, or its
    weight is a free parameter nobody can set (v1 used 20, chosen so the term was not simply
    noise). The way to get one is to stop measuring the matrix and measure what the matrix
    *does*: push a few fixed points through the predicted transform and the true one, and
    take the distance between where they land.

    Five points -- the corners and centre of the bounding box of every control point the
    group's shapes ever hold -- are enough to pin all eight degrees of freedom while staying
    free: the loss costs ``G x 5`` transformed points per frame, against ``S x Pmax``
    for the point term. Corners rather than a random sample because the extremes are what
    make rotation, shear and perspective visible; a probe set clustered at the centroid would
    price a rotation error at nearly nothing.
    """
    probes = np.zeros((n_groups, 5, 2), np.float32)
    for g in range(n_groups):
        shapes = np.where(group_of == g)[0]
        pts = [local[:, si][live[:, si]][:, point_mask[si]] for si in shapes]
        pts = [q.reshape(-1, 2) for q in pts if q.size]
        if not pts:
            continue
        q = np.concatenate(pts)
        lo, hi = q.min(0), q.max(0)
        probes[g] = [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]],
                     [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2]]
    return probes


@dataclass
class LayerData:
    """One layer, fully materialised. Dense tracks dominate memory; ~200 MB for all 13."""
    layer_id: str
    directory: Path
    frames: np.ndarray          # (F,)
    points: np.ndarray          # (F, S, Pmax, Cmax, 2) float32, [0,1] across the crop
    local: np.ndarray           # (F, S, Pmax, Cmax, 2) float32, IR-native local normalised
    live: np.ndarray            # (F, S) bool
    affine: np.ndarray          # (F, G, 6) float32, document space -- v1's target
    group_of: np.ndarray        # (S,) int32
    desc: np.ndarray            # (S, 3) float32 -- n_points, closed, coords_per_point
    point_mask: np.ndarray      # (S, Pmax, Cmax) bool
    px_per_norm: float
    matrices: np.ndarray        # (F, S, 4, 4) composed ancestor transform per shape per frame
    crop: dict                  # width/height/scale/out_px/offsets, for the inverse mapping
    proj_crop: np.ndarray       # (F, G, 8) float32, local->crop projective -- v1.1's target
    crop_matrices: np.ndarray   # (F, 4, 4) doc-normalised -> crop, per frame
    probe_local: np.ndarray     # (G, K, 2) float32, per-group probe points in local coords
    group_live: np.ndarray      # (F, G) bool -- has this group any live shape at this frame
    out_px: int = 256
    render: dict = field(default_factory=dict)
    """The ``meta['render']`` block: the conventions these alphas were **drawn** with.

    Carried on the layer rather than looked up per call site because the v002 rebuild showed
    what happens when it is not: every scorer built its own ``RenderConfig`` from the
    supersample alone and silently reasserted the class defaults for the other three
    conventions, which was invisible while the dataset agreed with those defaults and wrong
    the moment it did not. See ``render.raster.config_from_meta``."""
    key_mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), bool))
    """``(F, S)`` bool -- is this rendered frame an *artist* keyframe for this shape.

    The key-timing head's target, and the one thing in this project that has never been
    supervised. Keys outside the rendered frame range are dropped here on purpose: this array
    is aligned to ``frames`` because it is a per-frame prediction target, and the full key
    axis (which spans keys at frame -1 and beyond the last rendered frame) lives in
    ``program.ProgramTensors.key_mask`` where nothing is asked to predict it."""
    split_train: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int64))
    split_held: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int64))
    in_train: bool = True
    """The dataset's own split, read from ``splits.json`` rather than recomputed per run.

    ``in_train=False`` is a layer withheld from training *entirely*. It is not a
    generalisation claim -- shape queries are per ``(layer, shape)``, so such a layer has
    never had its query rows updated -- it is how much of the headline number lives in the
    query table. See ``roto.dataset.splits``."""

    @property
    def n_shapes(self) -> int:
        return self.points.shape[1]

    @property
    def n_groups(self) -> int:
        return self.affine.shape[1]

    # ``desc`` stores these normalised by the layer's own Pmax/Cmax, which is right for
    # conditioning and wrong for anything that needs the real count -- the polyline maps
    # need the true point count to pick the right basis matrix. Read them off the mask,
    # which is exact, rather than multiplying the normalised value back up.

    @property
    def n_points_per_shape(self) -> np.ndarray:
        return self.point_mask[:, :, 0].sum(1).astype(np.int32)

    @property
    def coords_per_shape(self) -> np.ndarray:
        return self.point_mask[:, 0, :].sum(1).astype(np.int32)

    @property
    def closed_per_shape(self) -> np.ndarray:
        return self.desc[:, 1].astype(bool)

    def alpha(self, i: int) -> np.ndarray:
        return load_alpha(self.directory, int(self.frames[i]))

    _alphas: np.ndarray | None = None

    @property
    def alphas(self) -> np.ndarray:
        """(F, 256, 256) float32, materialised on first use and kept.

        All 13 layers at once is ~475 MB, which fits, but the read is slow enough that the
        training loop must not pay it per step.
        """
        if self._alphas is None:
            self._alphas = np.stack(
                [self.alpha(i) for i in range(len(self.frames))]).astype(np.float32)
        return self._alphas

    _offsets_crop_px: np.ndarray | None = None

    @property
    def offsets_crop_px(self) -> np.ndarray:
        """``(F, 2)`` the crop window's own top-left, in crop pixels, per frame.

        The offsets are stored in *source* pixels; multiplying by the crop scale puts the
        window twitch in the units the network sees, which is the only scale at which it can
        be compared to the model's own jitter.
        """
        if self._offsets_crop_px is None:
            off = np.array([self.crop['offsets'][int(f)] for f in self.frames], np.float64)
            self._offsets_crop_px = off * self.crop['scale']
        return self._offsets_crop_px

    def window(self, idx: np.ndarray, in_frames: int, align: bool = False) -> np.ndarray:
        """``(len(idx), in_frames, H, W)`` -- each frame stacked with its neighbours.

        The window is centred and **clamped at the ends of the track**, so the first frame
        sees itself twice on the left. Wrapping or zero-padding would both hand the network a
        discontinuity that does not exist in the shot; holding the edge is what the rest of
        the pipeline already does with tracks that run out.

        ``frames`` is contiguous for every layer in this archive (the dataset is built at
        stride 1), so index adjacency *is* frame adjacency. Asserted rather than assumed,
        because a strided build would silently make this a window over the wrong frames.

        ``align`` is the correction v1.1 needed before the window could be judged at all.
        Each frame's alpha is rendered in *its own* crop window, and that window twitches --
        1.09 crop px per step on average and 4.1 px at worst on FAM blue, measured in
        ``v1.1/results/offset_jitter.json``. Stacking three neighbours raw therefore hands the
        network three copies of the shape displaced by *more* than the 0.77 px of per-frame
        jitter the window exists to remove, so "the window did not help" was never a test of
        the idea. With ``align=True`` each neighbour is shifted by
        ``(offset_neighbour - offset_anchor) * scale`` into the anchor's window, and all three
        channels then describe the same picture.

        The shift is subpixel and bilinear rather than rounded: an alpha is soft by
        construction (the dataset renders at supersample 4), so resampling it costs a little
        edge blur, while rounding to whole pixels would leave up to half a pixel of the exact
        misalignment being corrected. Content shifted in from beyond the neighbour's window is
        zero, which is what the renderer would have produced there anyway.
        """
        if in_frames <= 1:
            return self.alphas[idx][:, None]
        assert np.all(np.diff(self.frames) == 1), \
            f'{self.layer_id}: frames are not contiguous, a temporal window would span gaps'
        half = in_frames // 2
        offs = np.arange(-half, in_frames - half)
        pos = np.clip(idx[:, None] + offs[None], 0, len(self.frames) - 1)
        stack = self.alphas[pos]
        if not align:
            return stack
        stack = stack.copy()
        off = self.offsets_crop_px
        for i, anchor in enumerate(idx):
            for j in range(in_frames):
                if pos[i, j] == anchor:
                    continue
                d = off[pos[i, j]] - off[int(anchor)]
                if abs(d[0]) < 1e-3 and abs(d[1]) < 1e-3:
                    continue
                stack[i, j] = shift_alpha(stack[i, j], float(d[0]), float(d[1]))
        return stack

    def split(self, holdout_every: int) -> tuple[np.ndarray, np.ndarray]:
        """``(train_idx, held_idx)`` frame positions. ``holdout_every <= 0`` holds nothing.

        **A dataset that records its own split wins.** ``datasets/v002`` writes
        ``splits.json`` at build time (``roto.dataset.splits``), and when it is present this
        returns it verbatim and ignores ``holdout_every`` -- because the split is then a
        property of the data, and two runs quoting a held-out gap over different frames is
        the failure the record exists to prevent. ``datasets/v001`` has no record, so the
        rule below still computes one and every v1/v1.1/v1.2 number reproduces.

        Every Nth frame is withheld from training and scored separately. v1 trained and
        measured on the same frames, which answers "can we regenerate roto we have been
        shown" but cannot distinguish a network that interpolates its memorisation from one
        that only recalls it. The gap between the two columns is that answer.

        The held frames are never the first or last of the track, so a held frame always has
        trained neighbours on both sides and the temporal window stays well defined.
        """
        if len(self.split_train):
            return self.split_train, self.split_held
        n = len(self.frames)
        if holdout_every <= 1:
            return np.arange(n), np.zeros(0, np.int64)
        held = np.arange(n)[(np.arange(n) % holdout_every == holdout_every // 2)]
        held = held[(held > 0) & (held < n - 1)]
        train = np.setdiff1d(np.arange(n), held)
        return train, held


def _group_index(mats: np.ndarray) -> tuple[np.ndarray, int]:
    """Collapse identical per-shape transform tracks to a group index. Exact equality: shapes
    under one tracked layer share the same matrices by construction."""
    keys: dict[bytes, int] = {}
    idx = np.empty(len(mats), np.int32)
    for i, m in enumerate(mats):
        k = np.ascontiguousarray(m, np.float64).tobytes()
        idx[i] = keys.setdefault(k, len(keys))
    return idx, len(keys)


def load_element(directory: str | Path, with_local: bool = True) -> LayerData:
    """Materialise one layer.

    ``with_local=False`` skips keeping the IR-native copy of the control points. It is the
    same size as the crop-space copy -- 306 MB for the 1036-shape layer alone -- and only
    the reconstruction path reads it, so training holds roughly half as much memory. On a
    16 GB host that is the difference between two concurrent runs and none.
    """
    d = Path(directory)
    meta = json.loads((d / 'meta.json').read_text())
    doc = from_json_ir(json.loads((d / 'target_ir.json').read_text()))
    t = np.load(d / 'tensors.npz')
    frames = np.asarray(meta['frames']['index'], np.int32)
    n = int(t['n_shapes'])

    mats = [t[f'layer_matrix/{i}'].astype(np.float64) for i in range(n)]
    mframes = t['matrix_frames'].astype(np.int32)
    group_of, n_groups = _group_index(mats)

    shapes = [s for _, s in doc.shapes()]
    assert len(shapes) == n, f'{d.name}: IR has {len(shapes)} shapes, tensors say {n}'
    Pmax = max(int(t[f'points_norm/{i}'].shape[1]) for i in range(n))
    Cmax = max(int(t[f'points_norm/{i}'].shape[2]) for i in range(n))

    source = meta['source']
    crop = meta['crop']
    out_px = int(crop['out_px'][0])
    offsets = {int(k): tuple(v) for k, v in crop['offsets'].items()}
    pos = {int(f): k for k, f in enumerate(mframes)}

    # The artist's own keyframes, as a per-frame mask on the rendered frame axis. This is
    # the key-timing head's target; see LayerData.key_mask for why keys outside the rendered
    # range are dropped rather than clamped.
    at_frame = {int(f): i for i, f in enumerate(frames)}
    key_mask = np.zeros((len(frames), n), bool)
    for i in range(n):
        for kf in t[f'key_frames/{i}'].tolist():
            j = at_frame.get(int(kf))
            if j is not None:
                key_mask[j, i] = True

    local = np.zeros((len(frames), n, Pmax, Cmax, 2), np.float32)
    points = np.zeros((len(frames), n, Pmax, Cmax, 2), np.float32)
    matrices = np.zeros((len(frames), n, 4, 4), np.float64)
    live = np.zeros((len(frames), n), bool)
    pmask = np.zeros((n, Pmax, Cmax), bool)
    desc = np.zeros((n, 3), np.float32)
    for i, shape in enumerate(shapes):
        P, C = shape.n_points, int(t[f'points_norm/{i}'].shape[2])
        pmask[i, :P, :C] = True
        desc[i] = (P / Pmax, float(shape.closed), C / max(1, Cmax))
        for fi, f in enumerate(frames):
            v = np.asarray(sample(shape.path, int(f)), np.float32)
            local[fi, i, :P, :C] = v
            m = mats[i][pos[int(f)]]
            matrices[fi, i] = m
            points[fi, i, :P, :C] = local_to_crop(
                local[fi, i, :P, :C], m, width=source['width'], height=source['height'],
                offset=offsets[int(f)], scale=crop['scale'], out_px=out_px)
            live[fi, i] = opacity_at(shape, int(f)) > 0.5

    # Transform tracks are dense over `matrix_frames`; index them at the rendered frames.
    pos = {int(f): k for k, f in enumerate(mframes)}
    # The window's own map, per frame. Composing it onto the layer matrix is what moves the
    # transform target out of document space; see model.geometry.crop_matrix.
    crop_mats = np.stack([crop_matrix(width=source['width'], height=source['height'],
                                      offset=offsets[int(f)], scale=crop['scale'],
                                      out_px=out_px) for f in frames])
    affine = np.zeros((len(frames), n_groups, 6), np.float32)
    proj_crop = np.zeros((len(frames), n_groups, 8), np.float32)
    for g in range(n_groups):
        src = mats[int(np.where(group_of == g)[0][0])][[pos[int(f)] for f in frames]]
        affine[:, g] = affine_from_matrix(src).astype(np.float32)
        proj_crop[:, g] = proj_from_matrix(src @ crop_mats).astype(np.float32)

    # Built before `local` is released: the probes are local-space points and there is no
    # way back to them once the array is dropped.
    probes = group_probes(local, live, group_of, pmask, n_groups)

    # Whether a group has anything on screen at all, per frame. 34% of (frame, group) cells
    # across the archive have no live shape -- 59% on Layer_52, 70% on FAM blue_2 -- and a
    # transform loss that scores those is asking where an invisible group is. See
    # ``train.affine_probe_term``.
    group_live = np.zeros((len(frames), n_groups), bool)
    for g in range(n_groups):
        group_live[:, g] = live[:, group_of == g].any(1)

    if not with_local:
        local = np.zeros((0,), np.float32)
    # The dataset's own split, when it has one. Read here rather than in train() so that
    # every consumer -- training, scoring, the gate script -- sees the same one.
    rec = (load_splits(d.parent) or {}).get('layers', {}).get(meta['layer']['layer_id'], {})
    pos = {int(f): i for i, f in enumerate(frames)}
    tr = np.array([pos[f] for f in rec.get('frames_train', []) if f in pos], np.int64)
    he = np.array([pos[f] for f in rec.get('frames_held', []) if f in pos], np.int64)
    return LayerData(meta['layer']['layer_id'], d, frames, points, local, live, affine,
                       group_of, desc, pmask, float(crop['px_per_norm']), matrices,
                       {'width': source['width'], 'height': source['height'],
                        'scale': crop['scale'], 'out_px': out_px, 'offsets': offsets,
                        'supersample': int(meta.get('render', {}).get('supersample', 2)),
                        # Also in the crop bag, not only in ``render``, because the figure
                        # path is handed this dict alone and draws outlines through the
                        # renderer's own polyline evaluator -- which reads ``open_end_rule``.
                        # A figure drawn under the wrong rule disagrees with the IoU printed
                        # beside it at every open-stroke end.
                        'render': dict(meta.get('render', {}))},
                       proj_crop, crop_mats, probes, group_live, out_px,
                       render=dict(meta.get('render', {})), key_mask=key_mask,
                       split_train=tr, split_held=he,
                       in_train=bool(rec.get('in_train', True)))


def load_dataset(root: str | Path, with_local: bool = True,
                 trained_only: bool = False) -> list[LayerData]:
    """Every layer of a built dataset, in directory order.

    ``trained_only`` drops the layers the dataset's own split withholds from training. It is
    the *training* loader's flag and never the scorer's: a held-out layer still has to be
    scored, and reported apart from the rest, or the split measures nothing.
    """
    els = [load_element(p, with_local) for p in sorted(Path(root).iterdir())
           if (p / 'meta.json').exists()]
    return [e for e in els if e.in_train] if trained_only else els
