"""Keyframe selection as curve simplification, not per-frame classification.

The measured failure of the earlier approach was precision 0.21-0.44 at recall 0.75-1.00 --
the keys were found, and then roughly 2.5x too many frames were keyed besides. That is not a
tuning problem, it is the wrong problem statement. A per-frame binary head decides "is this a
key" independently for each frame, but an artist's keys are a *jointly optimal sparse set*:
key 30 is only worth spending because keys 10 and 50 leave frame 30 badly interpolated. No
independent per-frame decision can express that, so the head fires wherever the picture moves
and precision collapses.

Restated as curve simplification the problem is exact and needs no training. Given a shape's
control-point track over time, choose the fewest knots such that interpolating between them
reproduces the track within a tolerance. That is the same decision the artist made at the
desk, and it is solvable optimally.

``select`` runs a minimax dynamic program: ``dp[k][b]`` is the smallest achievable worst-case
segment error for a k-knot path ending at frame ``b``, composed with ``max`` rather than ``+``
because a keyframe set is judged by its worst frame, not its average one. The result is the
provably smallest knot set meeting the tolerance -- there is no threshold to tune except the
tolerance itself, which is in pixels and therefore means something.

The interpolation must match the renderer's, which interpolates *linearly between adjacent
path keys* in local normalised coordinates. Anything else would select knots that are optimal
for a curve nobody draws.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DP_MAX_FRAMES = 64
"""Above this live span the exact DP is replaced by furthest-reach.

The DP needs a full segment-cost matrix: O(T^2) segments each scored over O(T P) samples, so
it is cubic in the live span and quartic-ish once every shape in an element pays it. Measured
on this archive that is minutes per element at T=91, which is not a tradeoff worth making for
a tolerance-driven objective. Below 64 frames it is cheap and exact, so it runs.
"""

REACH_PATIENCE = 8
"""Infeasible segments to look past before giving up on extending a greedy segment.

Interpolation error is not monotone in segment length -- extending the endpoint moves the
whole chord, so a segment can be infeasible at b and feasible again at b+1. Stopping at the
first failure would over-key exactly where the shape reverses direction. Scanning to the end
of the track instead would restore the cubic cost this exists to avoid.
"""


@dataclass(slots=True)
class Selection:
    frames: np.ndarray
    """Chosen knot frames, ascending, always including the first and last live frame."""
    max_error: float
    """Worst per-point interpolation error over the whole track, in the track's own units."""
    exact: bool
    """True only when the exact DP ran. Furthest-reach is near-optimal, not optimal, and the
    distinction is recorded rather than assumed so a number is never quietly downgraded."""
    method: str = 'dp'

    def __len__(self) -> int:
        return len(self.frames)


def segment_errors(track: np.ndarray, a: int, b: int) -> float:
    """Worst per-point error from interpolating ``track`` linearly between knots ``a`` and ``b``.

    ``track`` is ``(T, P, 2)``. Endpoints are exact by construction, so a segment of length 1
    or 2 costs nothing -- which is what makes a dense key set trivially achievable and the
    tolerance, not the DP, the thing that decides sparsity.
    """
    if b - a < 2:
        return 0.0
    w = np.linspace(0.0, 1.0, b - a + 1, dtype=np.float64)[1:-1, None, None]
    approx = track[a][None] * (1.0 - w) + track[b][None] * w
    return float(np.abs(approx - track[a + 1:b]).sum(axis=-1).max())


def _cost_matrix(track: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """``C[i, j]`` = worst error of a segment from ``cand[i]`` to ``cand[j]``, ``inf`` if j<=i."""
    n = len(cand)
    C = np.full((n, n), np.inf)
    for i in range(n):
        C[i, i] = 0.0
        for j in range(i + 1, n):
            C[i, j] = segment_errors(track, cand[i], cand[j])
    return C


def _feasible(track: np.ndarray, a: int, b: int, tol: float) -> bool:
    return segment_errors(track, a, b) <= tol


def _greedy(track: np.ndarray, tol: float) -> list[int]:
    """Furthest-reach: from each knot, take the furthest endpoint that still fits.

    Optimal when feasibility is interval-like (if (a,b) fits then so does every (a,b') with
    b' < b). Interpolation error is not quite that, which is what ``REACH_PATIENCE`` covers,
    so the result is near-optimal rather than provably minimal.
    """
    T = len(track)
    knots, a = [0], 0
    while a < T - 1:
        best, misses, b = a + 1, 0, a + 1
        while b < T and misses < REACH_PATIENCE:
            if _feasible(track, a, b, tol):
                best, misses = b, 0
            else:
                misses += 1
            b += 1
        knots.append(best)
        a = best
    return knots


def _dp(track: np.ndarray, tol: float) -> list[int]:
    """Exact minimax DP. ``dp[k][b]`` = smallest achievable worst segment error to ``b``."""
    T = len(track)
    C = np.full((T, T), np.inf)
    for i in range(T):
        C[i, i] = 0.0
        for j in range(i + 1, T):
            C[i, j] = segment_errors(track, i, j)

    dp, back = C[0].copy(), [np.zeros(T, np.int32)]
    best_k = 1 if dp[T - 1] <= tol else None
    for k in range(1, T):
        cost = np.maximum(dp[:, None], C)      # a path is only as good as its worst segment
        arg = np.argmin(cost, axis=0)
        dp = cost[arg, np.arange(T)]
        back.append(arg.astype(np.int32))
        if best_k is None and dp[T - 1] <= tol:
            best_k = k + 1
            break
    if best_k is None:
        best_k = len(back)

    idx, b = [T - 1], T - 1
    for k in range(best_k - 1, 0, -1):
        b = int(back[k][b])
        idx.append(b)
    idx.append(0)
    return sorted(set(idx))


def select(track: np.ndarray, frames: np.ndarray, tol: float) -> Selection:
    """Fewest knots whose linear interpolation stays within ``tol`` of ``track``.

    ``track`` is ``(T, P, 2)`` in whatever units ``tol`` is stated in -- packet pixels, in
    this project, so the tolerance is a distance an artist could be shown.
    """
    T = len(frames)
    if T <= 2:
        return Selection(frames.copy(), 0.0, True, 'trivial')

    if T <= DP_MAX_FRAMES:
        idx, method, exact = _dp(track, tol), 'dp', True
    else:
        idx, method, exact = _greedy(track, tol), 'greedy', False

    chosen = np.asarray(idx, dtype=np.int32)
    worst = max((segment_errors(track, int(chosen[i]), int(chosen[i + 1]))
                 for i in range(len(chosen) - 1)), default=0.0)
    return Selection(frames[chosen], float(worst), exact, method)


def f1(predicted: np.ndarray, truth: np.ndarray, tolerance: int = 0) -> tuple[float, float, float]:
    """Precision, recall, F1 of a predicted key set against the artist's.

    ``tolerance`` allows a predicted key to count if it lands within N frames of a real one.
    Matching is greedy nearest-first and one-to-one, so two predictions cannot both claim the
    same artist key -- without that, over-keying would inflate recall for free, which is the
    exact failure this module exists to avoid measuring badly.
    """
    if len(truth) == 0:
        return (0.0, 1.0, 0.0) if len(predicted) else (1.0, 1.0, 1.0)
    if len(predicted) == 0:
        return 1.0, 0.0, 0.0
    unused = set(range(len(truth)))
    hits = 0
    for p in predicted:
        near = [(abs(p - truth[t]), t) for t in unused if abs(p - truth[t]) <= tolerance]
        if near:
            unused.discard(min(near)[1])
            hits += 1
    prec = hits / len(predicted)
    rec = hits / len(truth)
    return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)
