"""Between the IR's local normalised coordinates and the crop the network actually sees.

The first version of this model predicted control points directly in the IR's local
normalised space and stalled at ~260 px mean error. The reason is a mismatch of ranges rather
than a shortage of capacity: local coordinates are absolute positions in the *document*, so
they span roughly +/-0.5, while a tracking crop of a small element covers as little as 0.059
of that -- ``px_per_norm`` reaches 4365 on ``nfl_0200/green``. The network was being asked to
regress an absolute document coordinate eight times larger than anything visible in its input,
from a picture that cannot disambiguate it.

Predicting in crop space fixes the framing: the target is where the point sits *in the
picture*, in [0,1] across the crop, which is exactly what the alpha shows. The mapping back to
local coordinates is exact and closed-form, so nothing is lost -- the IR still stores native
local points under the layer transform, and the factorisation the renderer depends on
survives.

The layer transform is applied, not predicted, on this path. v1 teacher-forces motion; the
affine head is trained and reported separately so its error is visible rather than folded into
the geometry number.
"""
from __future__ import annotations

import numpy as np


def local_to_crop(points: np.ndarray, matrix: np.ndarray, *, width: int, height: int,
                  offset: tuple[float, float], scale: float, out_px: int) -> np.ndarray:
    """``(..., 2)`` local normalised -> ``(..., 2)`` in [0,1] across the crop.

    ``matrix`` is the composed 4x4 ancestor transform for the same frame, row-vector
    convention, matching ``render.raster.apply_transform``.
    """
    flat = points.reshape(-1, 2)
    h = np.concatenate([flat, np.zeros((len(flat), 1)), np.ones((len(flat), 1))], axis=1)
    r = h @ matrix
    w = r[:, 3:4]
    n = r[:, :2] / np.where(np.abs(w) < 1e-12, 1.0, w)
    x0, y0 = offset
    px = (n[:, 0] * height + width / 2.0 - x0) * scale
    py = (n[:, 1] * height + height / 2.0 - y0) * scale
    return (np.stack([px, py], axis=-1) / out_px).reshape(points.shape)


def crop_to_local(crop_pts: np.ndarray, matrix: np.ndarray, *, width: int, height: int,
                  offset: tuple[float, float], scale: float, out_px: int) -> np.ndarray:
    """Exact inverse of :func:`local_to_crop`."""
    flat = crop_pts.reshape(-1, 2) * out_px
    x0, y0 = offset
    nx = (flat[:, 0] / scale + x0 - width / 2.0) / height
    ny = (flat[:, 1] / scale + y0 - height / 2.0) / height
    n = np.stack([nx, ny], axis=-1)
    h = np.concatenate([n, np.zeros((len(n), 1)), np.ones((len(n), 1))], axis=1)
    r = h @ np.linalg.inv(matrix)
    w = r[:, 3:4]
    out = r[:, :2] / np.where(np.abs(w) < 1e-12, 1.0, w)
    return out.reshape(crop_pts.shape)
