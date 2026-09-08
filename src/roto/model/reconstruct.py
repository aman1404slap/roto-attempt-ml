"""Model output -> a real spline program -> pixels, scored against the layer's own alpha.

This is the step that decides whether the model worked, and it is deliberately the strictest
reading available. The network's per-frame geometry is converted back to local coordinates,
the keyframe selector picks which frames become keys, a ``RotoDoc`` is rebuilt from those
keys alone, and that document is *rendered* and compared to the clean alpha the network was
given.

Scoring the rendered result rather than the control points is the whole point. A low point
error can still draw the wrong picture -- a handful of points on a small shape can be badly
wrong while the mean stays flattering -- and a keyframe set that looks sparse can interpolate
into something an artist would reject. Rendering collapses geometry, keys, and lifespan into
the single question that matters: does it look like the matte.

The render must be at the **dataset's own supersample**. Scoring a prediction rendered at
supersample 2 against a target written at 4 disagrees on every anti-aliased boundary pixel,
which is precisely what soft IoU is built to notice: re-rendering the *artist's own shapes*
that way scores 0.953-0.995 instead of 1.000. Every v1 number carried that handicap.

What is teacher-forced, and therefore not claimed: the shape breakdown (how many shapes,
their point counts, their groups), each shape's lifespan, and the layer transform track. The
network supplies the geometry; ``roto.keys`` supplies the timing. ``RebuildConfig.
predicted_affine`` puts the transform head's own output on the table instead -- see
``affine_doc``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..dataset import load_alpha
from ..metrics import iou, soft_iou
from ..ir import Key, RotoDoc
from ..keys import f1 as key_f1
from ..keys import select
from ..keys.refit import refit_key_values
from ..program import PROJ_DOF, matrix_from_affine, matrix_from_proj
from ..render.raster import RenderConfig, render_union
from ..sfx.json_ir import from_json_ir
from .data import LayerData, load_element
from .geometry import crop_to_local
from .net import RotoNet
from .smoothing import BOXCAR, smooth_track

DEFAULT_TOL_PX = 1.0
"""Keyframe tolerance, in crop pixels.

On *ground-truth* tracks the best tolerance is 0.1 px. On *predicted* tracks that value is
badly wrong, and the reason couples two components that look independent: the keyframe search
cannot be tuned below the geometry's own noise floor. A predicted track carries a per-frame
jitter of roughly the model's point error, so asking the search to reproduce it within 0.1 px
forces a key on almost every frame -- measured at 16.5x the artist's key count. The tolerance
has to sit above the model's noise, not above the artist's precision.

So the operating point is chosen against predicted tracks, and it is re-chosen whenever the
model changes. Measured per model in ``v1.1/results/operating_point_<run>.json``
(``scripts/sweep_operating_point.py``), and the answer is not stable across models: v1's own
model peaks at a 13-frame window and 0.5 px, while the converged v1.1 model peaks at a
13-frame window, 1.0 px, savgol and a key-value refit -- worth +0.0035 soft IoU and +0.027 key
F1 over these defaults, for no retraining.

The defaults below stay at v1's values deliberately, so that re-running any v1 number
reproduces it. The better operating point is a flag, not a silent change.
"""

SMOOTH_WINDOW = 9
"""Frames of temporal smoothing applied to a predicted track before knots are chosen.

The artist's true track is piecewise linear between sparse keys; the model's error is
approximately independent frame to frame. A short centred filter therefore removes a large
part of the noise while leaving genuine motion almost untouched, which lets the search see
the structure it is meant to find. Set to 1 to disable.

Measured in v1, at 1.0 px tolerance: turning smoothing on raised rendered soft-IoU on every
layer tried (0.9820 -> 0.9846, 0.9605 -> 0.9653, 0.7651 -> 0.7763) *and* cut the key count
from 3.01x the artist's to 0.99x. Both improve together because the keys it removes were
spent tracking noise, not motion. Which *filter* does the smoothing is a separate question --
see ``roto.model.smoothing``.
"""


@dataclass(slots=True)
class RebuildConfig:
    """Everything about turning a predicted track into keyframed splines, in one place.

    These are the axes of the operating-point sweep, so they belong together rather than as
    a growing tail of keyword arguments.
    """
    tol_px: float = DEFAULT_TOL_PX
    smooth: int = SMOOTH_WINDOW
    smooth_kind: str = BOXCAR
    """``'boxcar'`` reproduces v1; ``'savgol'`` preserves the motion peaks artists key."""
    refit_values: bool = False
    """Fit key values to the raw track once the key frames are chosen. See ``keys.refit``.

    Off by default so v1's behaviour is reproducible; the sweep decides whether it earns its
    place, and it is a strictly cheaper change than swapping the filter."""
    predicted_affine: bool = False
    """Score the *transform head's* output instead of the artist's track. See ``affine_doc``."""
    supersample: int | None = None
    """Render supersample; ``None`` means the dataset's own, which is the only like-for-like
    choice. An explicit value exists so the anti-aliasing axis can be measured on purpose."""


@dataclass(slots=True)
class Reconstruction:
    layer_id: str
    doc: RotoDoc
    frames: np.ndarray
    soft_iou: np.ndarray
    iou: np.ndarray
    point_err_px: float
    keys_predicted: int
    keys_artist: int
    key_precision: float
    key_recall: float
    key_f1: float
    jitter_px: float = 0.0
    """Mean frame-to-frame change in the *error* of the predicted track, in crop pixels.

    The number the review identified as the system's master constraint: it is what forces
    smoothing, which forces the key tolerance, which caps key F1. Reported so the chain can
    be seen moving rather than inferred."""

    def summary(self) -> dict[str, Any]:
        return {
            'layer_id': self.layer_id,
            'frames': int(len(self.frames)),
            'mean_soft_iou': float(self.soft_iou.mean()),
            'min_soft_iou': float(self.soft_iou.min()),
            'mean_iou': float(self.iou.mean()),
            'point_err_px': self.point_err_px,
            'jitter_px': self.jitter_px,
            'keys_predicted': self.keys_predicted,
            'keys_artist': self.keys_artist,
            'key_ratio': self.keys_predicted / max(1, self.keys_artist),
            'key_precision': self.key_precision,
            'key_recall': self.key_recall,
            'key_f1': self.key_f1,
            'worst_frame': int(self.frames[int(np.argmin(self.soft_iou))]),
        }


def load_model(checkpoint: str | Path,
               device: str | torch.device | None = None) -> tuple[RotoNet, dict[str, Any]]:
    """Load a checkpoint onto ``device`` (GPU when one exists).

    ``arch`` carries ``in_frames`` and ``self_attn`` from v1.1 on; both default off in
    ``RotoNet``, so a v1 checkpoint written before they existed still loads unchanged.
    """
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    dev = torch.device(device) if device is not None else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = RotoNet(**ck['arch'])
    net.load_state_dict(ck['state_dict'])
    net.eval()
    return net.to(dev), ck


@torch.no_grad()
def predict(net: RotoNet, el: LayerData, shape_base: int = 0, group_base: int = 0,
            batch: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """``(points (F,S,Pmax,Cmax,2) in crop space, affine (F,G,6))``.

    ``shape_base``/``group_base`` are the layer's offsets into the query tables and must
    match training exactly; they come from the checkpoint.

    The temporal window is read from the network, not passed in: a model trained on 3
    stacked frames must be *given* 3 stacked frames, and getting that wrong would show up as
    a mysterious quality loss at inference rather than as an error. Window *alignment* travels
    the same way and for the same reason -- see ``net.RotoNet``.

    The second return is ``(F, G, affine_dim)``, 6 wide for a v1-style document-space head and
    8 for v1.1's crop-space projective one. ``affine_doc`` reads the width to know which it is
    holding.
    """
    out = np.zeros_like(el.points)
    dof = getattr(net, 'affine_dim', 6)
    aff = np.zeros(el.affine.shape[:2] + (dof,), np.float32)
    P, C = el.points.shape[2], el.points.shape[3]
    dev = next(net.parameters()).device
    shape_ids = (torch.arange(el.n_shapes) + shape_base)[None].to(dev)
    group_ids = (torch.arange(el.n_groups) + group_base)[None].to(dev)
    desc = torch.from_numpy(el.desc)[None].to(dev)
    in_frames = getattr(net, 'in_frames', 1)
    align = bool(getattr(net, 'align_window', False))
    for i in range(0, len(el.frames), batch):
        idx = np.arange(i, min(i + batch, len(el.frames)))
        a = torch.from_numpy(el.window(idx, in_frames, align)).to(dev)
        n = a.shape[0]
        pts, af = net(a, shape_ids.expand(n, -1), group_ids.expand(n, -1),
                      desc.expand(n, -1, -1))
        out[idx] = pts[:, :, :P, :C].cpu().numpy()
        aff[idx] = af[:, :el.n_groups].cpu().numpy()
    return out, aff


def predict_crop_points(net: RotoNet, el: LayerData, shape_base: int = 0,
                        group_base: int = 0, batch: int = 8) -> np.ndarray:
    """Points only -- the common case. See :func:`predict`."""
    return predict(net, el, shape_base, group_base, batch)[0]


def to_local(el: LayerData, crop_pts: np.ndarray) -> np.ndarray:
    """Crop-space predictions -> IR-native local normalised coordinates."""
    out = np.zeros_like(crop_pts)
    c = el.crop
    for fi, f in enumerate(el.frames):
        off = c['offsets'][int(f)]
        for si in range(el.n_shapes):
            out[fi, si] = crop_to_local(
                crop_pts[fi, si], el.matrices[fi, si], width=c['width'], height=c['height'],
                offset=off, scale=c['scale'], out_px=c['out_px'])
    return out


def track_jitter(crop_pts: np.ndarray, target: np.ndarray, live: np.ndarray,
                 point_mask: np.ndarray, out_px: float) -> float:
    """Frame-to-frame change in the prediction's *error*, in crop pixels.

    Differencing the error rather than the position is what separates jitter from motion: a
    shape travelling smoothly across the frame has a large position delta and no jitter.
    """
    if len(crop_pts) < 2:
        return 0.0
    err = crop_pts - target
    d = np.linalg.norm(err[1:] - err[:-1], axis=-1) * out_px
    m = point_mask[None] & (live[1:] & live[:-1])[..., None, None]
    return float(d[m].mean()) if m.any() else 0.0


def transform_matrices(el: LayerData, pred_affine: np.ndarray) -> np.ndarray:
    """``(F, G, 4, 4)`` document-space layer matrices from whatever the head predicted.

    The width says which representation is in hand, so one call site serves both and a
    v1 checkpoint keeps scoring exactly as it did:

    * **6 wide** -- v1's document-space affine. Filled straight in, ``m33`` forced to 1.
      That last part is the lossy step; see ``program.affine_from_matrix``.
    * **8 wide** -- v1.1's crop-space projective map. The prediction is the composed
      *local-to-crop* transform, so the window's own map has to be divided back out:
      ``M_doc = M_crop @ inv(C_f)``, with ``C_f`` from ``geometry.crop_matrix``. Exact, and
      per frame, because the crop window translates.
    """
    if pred_affine.shape[-1] == PROJ_DOF:
        crop_mats = matrix_from_proj(pred_affine)               # (F, G, 4, 4), local->crop
        inv = np.linalg.inv(el.crop_matrices)                   # (F, 4, 4)
        return np.einsum('fgij,fjk->fgik', crop_mats, inv)
    return matrix_from_affine(pred_affine)


def affine_doc(el: LayerData, pred_affine: np.ndarray) -> RotoDoc:
    """The artist's own shapes, moved by the *predicted* transform track.

    This is how the transform head gets a number on the table. It is trained but v1 never
    consumed it, which spends gradient on an output nobody reads.

    Isolating it this way is deliberate, and the alternative is a measurement that cannot
    say anything. Feeding predicted points *and* the predicted transform is self-cancelling:
    the points are regressed in crop space and converted to local by inverting the same
    matrix the renderer then re-applies, so any transform error divides out and the picture
    is identical to the geometry-only one. Holding the artist's control points fixed and
    varying only the transform is what makes the head's error visible in pixels.

    **Feed it the artist's own track and this must return 1.000.** It does for v1.1's
    representation and does not for v1's -- 0.9100 on ``FAM red_1`` -- which is how the 6-number
    target was found to be lossy rather than merely badly framed. ``scripts/exp_affine_target.py``
    is that check.
    """
    doc = from_json_ir(json.loads((el.directory / 'target_ir.json').read_text()))
    mats = transform_matrices(el, pred_affine)                  # (F, G, 4, 4)
    for si, (ancestors, _) in enumerate(doc.shapes()):
        g = int(el.group_of[si])
        for layer in ancestors:
            if layer.transform:
                layer.transform = [Key(int(f), 'linear', mats[fi, g])
                                   for fi, f in enumerate(el.frames)]
    return doc


def rebuild(el: LayerData, local_pts: np.ndarray,
            cfg: RebuildConfig | None = None) -> tuple[RotoDoc, dict[str, Any]]:
    """Choose keys per shape and write a document carrying only those keys.

    The keyframe search runs on the point positions only, not on Bezier handles. Handles are
    carried at whatever the chosen keys hold: they describe the curve *between* points, so
    letting them drive key timing would key on tangent wobble the picture barely shows. Two
    of thirteen layers have handles at all.

    Key *timing* is chosen on the smoothed track, because that is a question about structure
    and noise would corrupt it. Key *values* come from the smoothed track too unless
    ``refit_values`` is set, in which case they are fitted to the raw track -- see
    ``keys.refit`` for why storing the filter's own output biases every motion extreme.

    Key scoring reads the artist's keys from a second, untouched copy of the IR, so the
    comparison is never the rebuilt document against itself.
    """
    cfg = cfg or RebuildConfig()
    doc = from_json_ir(json.loads((el.directory / 'target_ir.json').read_text()))
    truth_doc = from_json_ir(json.loads((el.directory / 'target_ir.json').read_text()))
    shapes = [s for _, s in doc.shapes()]
    truth_shapes = [s for _, s in truth_doc.shapes()]
    at = {int(f): i for i, f in enumerate(el.frames)}

    n_pred = n_true = 0
    precs, recs, f1s, weights = [], [], [], []
    for si, (shape, tshape) in enumerate(zip(shapes, truth_shapes)):
        live = np.array([int(f) for f in el.frames if el.live[at[int(f)], si]], np.int32)
        P = shape.n_points
        C = int(np.asarray(shape.path[0].value).shape[1])
        interp = {k.frame: k.interp for k in tshape.path}

        def value(frame: int) -> np.ndarray:
            return local_pts[at[int(frame)], si, :P, :C].astype(np.float64)

        if len(live) < 2:
            # Nothing to interpolate: one key at the frame the shape exists on.
            f = int(live[0]) if len(live) else int(el.frames[0])
            shape.path = [Key(f, interp.get(f, 'linear'), value(f))]
            n_pred += 1
            n_true += len({k.frame for k in tshape.path})
            continue

        raw = np.stack([value(f) for f in live])
        clean = smooth_track(raw, cfg.smooth, cfg.smooth_kind)
        sel = select(clean[:, :, 0, :] * el.px_per_norm, live, cfg.tol_px)
        modes = [interp.get(int(f), 'linear') for f in sel.frames]
        if cfg.refit_values:
            vals = refit_key_values(raw, sel.frames, modes, live)
        else:
            pick = {int(f): i for i, f in enumerate(live)}
            vals = np.stack([clean[pick[int(f)]] for f in sel.frames])
        shape.path = [Key(int(f), modes[i], vals[i]) for i, f in enumerate(sel.frames)]

        truth = np.array(sorted({int(np.clip(k.frame, live[0], live[-1]))
                                 for k in tshape.path}), np.int32)
        p, r, s = key_f1(sel.frames, truth, tolerance=1)
        precs.append(p); recs.append(r); f1s.append(s); weights.append(len(truth))
        n_pred += len(sel.frames)
        n_true += len(truth)

    w = np.array(weights, float)
    w = w / w.sum() if w.sum() else w
    stats = {
        'keys_predicted': int(n_pred), 'keys_artist': int(n_true),
        'key_precision': float(np.dot(w, precs)) if len(w) else 0.0,
        'key_recall': float(np.dot(w, recs)) if len(w) else 0.0,
        'key_f1': float(np.dot(w, f1s)) if len(w) else 0.0,
    }
    return doc, stats


def score_doc(el: LayerData, doc: RotoDoc, want: Sequence[int],
              supersample: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Render ``doc`` on ``want`` and compare to the stored alpha. Returns (soft, hard)."""
    c = el.crop
    cfg = RenderConfig(supersample=int(supersample or c.get('supersample', 2)))
    softs, hards = [], []
    for f in want:
        x0, y0 = c['offsets'][int(f)]
        box = (x0, y0, c['out_px'] / c['scale'], c['out_px'] / c['scale'])
        pred = render_union(doc, int(f), cfg, c['scale'], box)
        truth = load_alpha(el.directory, int(f))
        if pred.shape != truth.shape:
            pred = pred[:truth.shape[0], :truth.shape[1]]
        softs.append(soft_iou(pred, truth))
        hards.append(iou(pred, truth))
    return np.asarray(softs), np.asarray(hards)


def reconstruct(layer_dir: str | Path, net: RotoNet, cfg: RebuildConfig | None = None,
                frames: Sequence[int] | None = None, shape_base: int = 0,
                group_base: int = 0) -> Reconstruction:
    cfg = cfg or RebuildConfig()
    el = load_element(layer_dir)
    crop_pts, pred_aff = predict(net, el, shape_base, group_base)

    mask = el.point_mask[None] & el.live[..., None, None]
    # Euclidean, in crop pixels. This was ``(|dx| + |dy|) / 2 * out_px``, which is neither
    # L1 nor L2 and reads low against the distance an artist would measure.
    err = np.linalg.norm(crop_pts - el.points, axis=-1)[mask] * el.out_px
    point_err = float(err.mean())
    jitter = track_jitter(crop_pts, el.points, el.live, el.point_mask, el.out_px)

    local = to_local(el, crop_pts)
    doc, kstats = rebuild(el, local, cfg)
    if cfg.predicted_affine:
        doc = affine_doc(el, pred_aff)

    want = list(frames) if frames is not None else [int(f) for f in el.frames]
    softs, hards = score_doc(el, doc, want, cfg.supersample)
    return Reconstruction(el.layer_id, doc, np.asarray(want), softs, hards,
                          point_err, jitter_px=jitter, **kstats)
