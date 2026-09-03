"""Model output -> a real spline program -> pixels, scored against the element's own alpha.

This is the step that decides whether v1 worked, and it is deliberately the strictest reading
available. The network's per-frame geometry is converted back to local coordinates, the
keyframe selector picks which frames become keys, a ``RotoDoc`` is rebuilt from those keys
alone, and that document is *rendered* and compared to the clean alpha the network was given.

Scoring the rendered result rather than the control points is the whole point. A low point
error can still draw the wrong picture -- a handful of points on a small shape can be badly
wrong while the mean stays flattering -- and a keyframe set that looks sparse can interpolate
into something an artist would reject. Rendering collapses geometry, keys, and lifespan into
the single question that matters: does it look like the matte.

What v1 teacher-forces, and therefore does not claim: the shape breakdown (how many shapes,
their point counts, their groups), each shape's lifespan, and the layer transform track. The
network supplies the geometry; ``roto.keys`` supplies the timing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..dataset import load_alpha
from ..eval.metrics import iou, soft_iou
from ..ir import Key, RotoDoc, Shape
from ..keys import f1 as key_f1
from ..keys import select
from ..render.raster import RenderConfig, render_union
from ..sfx.json_ir import from_json_ir
from .data import ElementData, load_element
from .geometry import crop_to_local
from .net import RotoNet

DEFAULT_TOL_PX = 0.1
"""Keyframe tolerance, in packet pixels. Measured sweep on ground-truth tracks: F1 peaks at
0.1 (0.766) while 0.2 lands the key *count* within 4% of the artist's. 0.1 is the default
because recall costs more than a few extra keys when the result is going to be rendered."""


@dataclass(slots=True)
class Reconstruction:
    element_id: str
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

    def summary(self) -> dict[str, Any]:
        return {
            'element_id': self.element_id,
            'frames': int(len(self.frames)),
            'mean_soft_iou': float(self.soft_iou.mean()),
            'min_soft_iou': float(self.soft_iou.min()),
            'mean_iou': float(self.iou.mean()),
            'point_err_px': self.point_err_px,
            'keys_predicted': self.keys_predicted,
            'keys_artist': self.keys_artist,
            'key_ratio': self.keys_predicted / max(1, self.keys_artist),
            'key_precision': self.key_precision,
            'key_recall': self.key_recall,
            'key_f1': self.key_f1,
            'worst_frame': int(self.frames[int(np.argmin(self.soft_iou))]),
        }


def load_model(checkpoint: str | Path) -> tuple[RotoNet, dict[str, Any]]:
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    net = RotoNet(**ck['arch'])
    net.load_state_dict(ck['state_dict'])
    net.eval()
    return net, ck


@torch.no_grad()
def predict_crop_points(net: RotoNet, el: ElementData, shape_base: int = 0,
                        group_base: int = 0, batch: int = 8) -> np.ndarray:
    """(F, S, Pmax, Cmax, 2) predicted control points, in crop space.

    ``shape_base``/``group_base`` are the element's offsets into the query tables and must
    match training exactly; they come from the checkpoint.
    """
    out = np.zeros_like(el.points)
    P, C = el.points.shape[2], el.points.shape[3]
    shape_ids = (torch.arange(el.n_shapes) + shape_base)[None]
    group_ids = (torch.arange(el.n_groups) + group_base)[None]
    desc = torch.from_numpy(el.desc)[None]
    for i in range(0, len(el.frames), batch):
        sl = slice(i, min(i + batch, len(el.frames)))
        a = torch.from_numpy(el.alphas[sl]).unsqueeze(1)
        n = a.shape[0]
        pts, _ = net(a, shape_ids.expand(n, -1), group_ids.expand(n, -1), desc.expand(n, -1, -1))
        out[sl] = pts[:, :, :P, :C].numpy()
    return out


def to_local(el: ElementData, crop_pts: np.ndarray) -> np.ndarray:
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


def rebuild(el: ElementData, local_pts: np.ndarray, tol_px: float
            ) -> tuple[RotoDoc, dict[str, Any]]:
    """Choose keys per shape and write a document carrying only those keys.

    The keyframe search runs on the point positions only, not on Bezier handles. Handles are
    carried at whatever the chosen keys hold: they describe the curve *between* points, so
    letting them drive key timing would key on tangent wobble the picture barely shows. Two of
    thirteen elements have handles at all.

    Key scoring reads the artist's keys from a second, untouched copy of the IR, so the
    comparison is never the rebuilt document against itself.
    """
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

        track = np.stack([value(f)[:, 0, :] for f in live]) * el.px_per_norm
        sel = select(track, live, tol_px)
        shape.path = [Key(int(f), interp.get(int(f), 'linear'), value(f)) for f in sel.frames]

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


def reconstruct(element_dir: str | Path, net: RotoNet, tol_px: float = DEFAULT_TOL_PX,
                frames: Sequence[int] | None = None, shape_base: int = 0,
                group_base: int = 0) -> Reconstruction:
    el = load_element(element_dir)
    crop_pts = predict_crop_points(net, el, shape_base, group_base)

    mask = el.point_mask[None] & el.live[..., None, None]
    err = np.abs(crop_pts - el.points).sum(-1)[mask] * el.out_px / 2.0
    point_err = float(err.mean())

    local = to_local(el, crop_pts)
    doc, kstats = rebuild(el, local, tol_px)

    c = el.crop
    cfg = RenderConfig(supersample=2)
    want = list(frames) if frames is not None else [int(f) for f in el.frames]
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

    return Reconstruction(el.element_id, doc, np.asarray(want), np.asarray(softs),
                          np.asarray(hards), point_err, **kstats)
