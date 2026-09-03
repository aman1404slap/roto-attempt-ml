"""Agreement between a rendered matte and a reference matte.

``soft_iou`` is the one that matters here. Thresholded IoU throws away the anti-aliased edge,
which on a roto shape is precisely where the interesting error lives: a control point a pixel
out moves only the soft boundary, and a hard threshold scores that as perfect.
"""
from __future__ import annotations

import numpy as np


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
