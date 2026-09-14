"""Curve-space loss for v2 (charter S4 loss column).

Ported from ``roto.model.curveloss`` at v2 Step 2. Kept as v2's own because charter S4 moves
the curve/render term from a secondary term at S0 to the **primary** one at S2, with dot-L1
demoted to 0.25x -- this file is where that reweighting lands.
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
