"""A loss on the curve the model draws, not just on the points it moves.

The point loss is L1 on control points, but a B-spline's control polygon is not unique in
any useful sense: visibly identical curves can have different polygons, and on a sparse
polygon a small point error can move the drawn curve a long way. The render is the judge at
evaluation time, so the loss should look more like the render.

The economy that makes this nearly free: ``bspline_to_bezier`` and Bezier sampling are both
**fixed linear maps** of the control points. For a given (point count, closed) pair the
polyline is ``M @ P`` for a constant matrix ``M``, so "differentiable rendering" collapses to
one matmul -- no rasteriser in the loop, no approximation.

``M`` is built by pushing basis vectors through ``render.curves`` itself rather than by
re-deriving the basis here. That is deliberate: a hand-written copy of the B-spline basis
could drift from the renderer's and the loss would then optimise a curve nobody draws, which
is the exact failure this module is meant to avoid.

**Bezier shapes are excluded.** Their drawn curve depends on the in/out handles as well as
the point, so this point-only map would describe the wrong geometry. 26 shapes archive-wide
are Bezier (24 in nfl_0200 blue, 2 in nfl_0080 mb 1); they keep the point term and sit out
this one. X-splines are included because the renderer already draws them as B-splines.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from ..render.curves import eval_bspline

SAMPLES_PER_SEG = 4
"""Polyline samples per curve segment for the loss.

The renderer uses 12. Four is enough here and cheaper: the loss needs the curve's *shape*
to be represented densely enough that a wrong bulge costs something, not the sub-pixel
fidelity a rasteriser needs. Measured cost at S=1036, P=93: one 372-row matmul per group.
"""


@lru_cache(maxsize=512)
def polyline_matrix(n_points: int, closed: bool,
                    samples_per_seg: int = SAMPLES_PER_SEG) -> np.ndarray:
    """``(n_samples, n_points)`` linear map from control points to polyline samples.

    Built column by column from ``eval_bspline`` on basis vectors, so it is exactly the
    renderer's own curve. Cached: there are only a few dozen distinct (n_points, closed)
    pairs in the archive.
    """
    cols = []
    for j in range(n_points):
        basis = np.zeros((n_points, 2))
        basis[j, 0] = 1.0
        cols.append(eval_bspline(basis, closed, samples_per_seg)[:, 0])
    return np.stack(cols, axis=1)


class PolylineMaps:
    """Per-layer polyline maps, grouped by the shapes that share one.

    Shapes are padded to a common ``Pmax`` in the tensors but each has its own real point
    count, and the map depends on it, so shapes are bucketed by ``(n_points, closed)`` and
    one matmul is issued per bucket.
    """

    def __init__(self, n_points: np.ndarray, closed: np.ndarray, coords: np.ndarray,
                 device: torch.device | str = 'cpu',
                 samples_per_seg: int = SAMPLES_PER_SEG) -> None:
        self.groups: list[tuple[torch.Tensor, int, torch.Tensor]] = []
        buckets: dict[tuple[int, bool], list[int]] = {}
        for i, (p, c, k) in enumerate(zip(n_points, closed, coords)):
            if int(k) != 1 or int(p) < 3:      # Bezier handles, or too few points to curve
                continue
            buckets.setdefault((int(p), bool(c)), []).append(i)
        for (p, c), idx in sorted(buckets.items()):
            m = torch.from_numpy(polyline_matrix(p, c, samples_per_seg)).float().to(device)
            self.groups.append((torch.as_tensor(idx, device=device), p, m))

    @property
    def n_shapes_covered(self) -> int:
        return sum(len(idx) for idx, _, _ in self.groups)


def curve_loss(pred: torch.Tensor, target: torch.Tensor, live: torch.Tensor,
               maps: PolylineMaps, out_px: float) -> torch.Tensor:
    """Mean L1 distance between predicted and target polylines, in crop pixels.

    ``pred``/``target`` are ``(B, S, Pmax, Cmax, 2)`` in crop space; ``live`` is ``(B, S)``.
    Correspondence is index-wise, which is exact here because topology is fixed and the
    shape breakdown is teacher-forced. When v2 makes queries dynamic and correspondence
    weakens, a chamfer distance over these same polylines is the natural replacement.
    """
    total = pred.new_zeros(())
    count = pred.new_zeros(())
    for idx, p, m in maps.groups:
        w = live[:, idx].to(pred.dtype)[..., None, None]          # (B, N, 1, 1)
        if float(w.sum()) == 0.0:
            continue
        pp = torch.einsum('mp,bnpc->bnmc', m, pred[:, idx, :p, 0, :])
        tt = torch.einsum('mp,bnpc->bnmc', m, target[:, idx, :p, 0, :])
        total = total + ((pp - tt).abs() * w).sum()
        count = count + w.sum() * pp.shape[2] * pp.shape[3]
    if float(count) == 0.0:
        return total
    return total / count * out_px
