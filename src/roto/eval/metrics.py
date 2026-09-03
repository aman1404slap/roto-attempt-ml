"""Metrics for scoring a predicted roto program against ground truth.

Two families, and both matter:

* **raster** -- does it render to the right pixels? Cheap, automatable, and the thing a
  naive vectoriser can already win. Must be measured at *non-key* frames too: a program
  can match every key and still drift between them.
* **structural** -- is it shaped like artist work? Point counts, key counts, lifespans,
  and whether gross motion sits in layer transforms rather than being baked into every
  control point. This is what separates usable from unusable at equal raster fidelity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..ir import Key, RotoDoc, sample
from ..render.raster import apply_transform, compose


def iou(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
    p, g = pred > threshold, gt > threshold
    union = int((p | g).sum())
    return 1.0 if union == 0 else float((p & g).sum()) / union


def soft_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """IoU on fractional coverage: ``sum(min) / sum(max)``, no threshold.

    This is the headline number once the renderer keeps anti-aliased edges, because a
    thresholded IoU scores a correct soft edge as wrong on every boundary pixel.
    """
    denom = float(np.maximum(pred, gt).sum())
    return 1.0 if denom <= 0.0 else float(np.minimum(pred, gt).sum()) / denom


def dice(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
    """Overlap weighted toward small regions -- more forgiving than IoU on thin shapes."""
    p, g = pred > threshold, gt > threshold
    total = int(p.sum()) + int(g.sum())
    return 1.0 if total == 0 else 2.0 * float((p & g).sum()) / total


def soft_l1(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean absolute coverage error -- sensitive to soft edges, unlike thresholded IoU."""
    return float(np.abs(pred.astype(np.float64) - gt.astype(np.float64)).mean())


@dataclass(slots=True)
class ErrorProfile:
    """Where disagreement lives. Near-boundary error is anti-aliasing/feather; error far
    from the boundary means a whole shape is wrong, which is a different class of bug."""
    iou: float
    error_px: int
    frac_within_1px: float
    frac_within_3px: float
    frac_within_10px: float
    largest_blob_px: int
    largest_blob_frac: float
    n_blobs: int

    @property
    def is_edge_only(self) -> bool:
        return self.frac_within_3px > 0.85


def error_profile(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> ErrorProfile:
    p, g = pred > threshold, gt > threshold
    err = p ^ g
    n_err = int(err.sum())
    if n_err == 0:
        return ErrorProfile(1.0, 0, 1.0, 1.0, 1.0, 0, 0.0, 0)

    edge = cv2.morphologyEx(g.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    dist = cv2.distanceTransform((1 - edge).astype(np.uint8), cv2.DIST_L2, 3)
    d = dist[err]
    n_blobs, _, stats, _ = cv2.connectedComponentsWithStats(err.astype(np.uint8), 8)
    areas = stats[1:, 4] if n_blobs > 1 else np.array([0])
    gt_area = max(int(g.sum()), 1)
    return ErrorProfile(
        iou=iou(pred, gt, threshold), error_px=n_err,
        frac_within_1px=float((d <= 1).mean()),
        frac_within_3px=float((d <= 3).mean()),
        frac_within_10px=float((d <= 10).mean()),
        largest_blob_px=int(areas.max()), largest_blob_frac=float(areas.max() / gt_area),
        n_blobs=int(n_blobs - 1))


# ---- structural -----------------------------------------------------------------

@dataclass(slots=True)
class Economy:
    shapes: int = 0
    points: int = 0
    keys: int = 0
    points_per_shape: float = 0.0
    keys_per_shape: float = 0.0
    keys_per_frame: float = 0.0
    with_lifespan: int = 0
    tracked_layers: int = 0


def economy(doc: RotoDoc) -> Economy:
    shapes = [s for _, s in doc.shapes()]
    n = max(len(shapes), 1)
    keys = sum(len(s.path) for s in shapes)
    return Economy(
        shapes=len(shapes),
        points=sum(s.n_points for s in shapes),
        keys=keys,
        points_per_shape=sum(s.n_points for s in shapes) / n,
        keys_per_shape=keys / n,
        keys_per_frame=keys / max(doc.duration, 1),
        with_lifespan=sum(1 for s in shapes if s.opacity and len(s.opacity) > 1),
        tracked_layers=sum(1 for _, l in doc.layers() if l.transform))


def key_frames(doc: RotoDoc) -> set[int]:
    return {k.frame for _, s in doc.shapes() for k in s.path}


def non_key_frames(doc: RotoDoc) -> list[int]:
    """Frames with no artist key at all -- where interpolation drift shows up."""
    keyed = key_frames(doc)
    return [f for f in range(doc.duration) if f not in keyed]


def motion_split(doc: RotoDoc, samples: int = 12) -> dict[str, float]:
    """How much shape motion is carried by layer transforms vs control points.

    Returns mean per-frame centroid displacement in pixels, in screen space and in the
    shape's own local frame. A high ``carried_by_transform`` means the artist let the
    tracker do the work -- the factorisation the model should reproduce.
    """
    screen, local = [], []
    for ancestors, shape in doc.shapes():
        mats = [l.transform for l in ancestors if l.transform]
        if not mats or len(shape.path) < 2:
            continue
        f0, f1 = max(shape.path[0].frame, 0), shape.path[-1].frame
        if f1 - f0 < 6:
            continue
        frames = np.unique(np.linspace(f0, f1, samples).astype(int))
        if len(frames) < 3:
            continue
        L, S = [], []
        for fr in frames:
            pts = sample(shape.path, fr)[:, 0, :]
            L.append(pts.mean(0) * doc.height)
            S.append(apply_transform(compose([sample(m, fr) for m in mats]),
                                     pts).mean(0) * doc.height)
        df = np.diff(frames)[:, None].astype(float)
        local.append(float(np.abs(np.diff(np.array(L), axis=0) / df).sum(1).mean()))
        screen.append(float(np.abs(np.diff(np.array(S), axis=0) / df).sum(1).mean()))
    if not screen:
        return {'shapes': 0, 'screen_px_per_frame': 0.0, 'local_px_per_frame': 0.0,
                'carried_by_transform': 0.0}
    sm, lm = float(np.mean(screen)), float(np.mean(local))
    return {'shapes': len(screen), 'screen_px_per_frame': sm, 'local_px_per_frame': lm,
            'carried_by_transform': 1.0 - lm / max(sm, 1e-9)}
