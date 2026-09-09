"""Choosing keyframes by curve simplification.

A shape's control points move over time. The artist stores that motion as a handful of
**keyframes** and lets Silhouette interpolate between them. To write a usable ``.sfx`` we have
to make the same choice: given the motion, which frames should carry a key?

Keying every frame is technically correct and practically useless -- an artist opening a file
with a key on every frame of every shape is worse off than starting from nothing. So the
question is the fewest keyframes whose interpolation still reproduces the motion within a
tolerance.

Stated that way it is curve simplification, and it has an exact answer. There is nothing to
train and one setting to choose: a tolerance, in pixels.

``select`` runs a minimax dynamic program below ``DP_MAX_FRAMES``. ``dp[k][b]`` is the smallest
achievable worst-case segment error for a k-knot path ending at frame ``b``, composed with
``max`` rather than ``+`` because a keyframe set is judged by its worst frame, not its average
one. Above that span the cost matrix is cubic in the number of frames, so furthest-reach takes
over and ``Selection.method`` records which ran.

The interpolation used here must match the renderer's, which interpolates **linearly between
adjacent keys** in local normalised coordinates. Anything else would choose keys that are
optimal for a curve nobody draws.

**The tolerance can now vary along the track**, and that is the only way the key-timing head
reaches this module. Given a per-frame key probability, ``select`` tightens the tolerance
where the head expects a key and leaves it alone where it does not, so a segment spanning a
predicted key frame has to approximate that frame well -- which the DP satisfies by putting a
knot at or beside it. Nothing here lets the head *emit* a key: the objective is still "fewest
knots within tolerance", the DP still decides, and the head only changes what "within
tolerance" means locally. That asymmetry is deliberate. Over-keying is this project's measured
signature failure -- precision 0.21-0.44 at ~2.5x the artist's key count when a head was
allowed to fire directly -- and a bias on a minimising objective cannot produce it, because
every extra knot still costs.

Mechanically a per-frame tolerance turns the feasibility test from "error <= tol" into
"error/tol(t) <= 1", so the cost minimised becomes a *ratio* rather than a distance. The
uniform case is left on the old code path bit for bit rather than routed through the ratio,
because every v1, v1.1 and v1.2 number was measured through it and a float division is not
worth re-baselining a ladder over.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DP_MAX_FRAMES = 64
"""Above this live span the exact DP is replaced by furthest-reach.

The DP needs a full segment-cost matrix: O(T^2) segments each scored over O(T P) samples, so
it is cubic in the live span and quartic-ish once every shape in an layer pays it. Measured
on this archive that is minutes per layer at T=91, which is not a tradeoff worth making for
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
    tol_min: float = 0.0
    tol_max: float = 0.0
    """The range of the tolerance actually used along this track.

    Equal to each other, and to the tolerance asked for, unless a key-timing bias was in
    play. Recorded because a key set is only interpretable against the tolerance that
    produced it, and with a local bias that is no longer a single number in the report."""

    def __len__(self) -> int:
        return len(self.frames)


def segment_errors(track: np.ndarray, a: int, b: int) -> float:
    """Worst per-point error from interpolating ``track`` linearly between knots ``a`` and ``b``.

    ``track`` is ``(T, P, 2)``. Endpoints are exact by construction, so a segment of length 1
    or 2 costs nothing -- which is what makes a dense key set trivially achievable and the
    tolerance, not the DP, the thing that decides sparsity.

    The error is the **Euclidean** distance a point sits from where interpolation would put
    it. This used to be ``|dx| + |dy|``, which reads a purely diagonal miss as sqrt(2) times
    its real size, so a tolerance stated in pixels did not mean the distance an artist would
    measure. Because L1 >= L2, the same tolerance now admits slightly longer segments.
    """
    if b - a < 2:
        return 0.0
    w = np.linspace(0.0, 1.0, b - a + 1, dtype=np.float64)[1:-1, None, None]
    approx = track[a][None] * (1.0 - w) + track[b][None] * w
    return float(np.linalg.norm(approx - track[a + 1:b], axis=-1).max())


def segment_ratio(track: np.ndarray, a: int, b: int, inv_tol: np.ndarray) -> float:
    """Worst ``error / tolerance`` over the interior of segment ``(a, b)``.

    The local-tolerance form of :func:`segment_errors`: ``inv_tol[t]`` is ``1 / tol(t)``, so a
    frame the key head expects a key on carries a larger multiplier and a segment spanning it
    is charged more for the same absolute error. Feasibility is then simply ``<= 1``, which
    keeps the minimax DP below unchanged -- it composes an arbitrary scalar cost with ``max``
    and never looks at the units.
    """
    if b - a < 2:
        return 0.0
    w = np.linspace(0.0, 1.0, b - a + 1, dtype=np.float64)[1:-1, None, None]
    approx = track[a][None] * (1.0 - w) + track[b][None] * w
    d = np.linalg.norm(approx - track[a + 1:b], axis=-1)          # (b-a-1, P)
    return float((d * inv_tol[a + 1:b, None]).max())


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


def _greedy(track: np.ndarray, tol: float, cost=segment_errors) -> list[int]:
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
            if cost(track, a, b) <= tol:
                best, misses = b, 0
            else:
                misses += 1
            b += 1
        knots.append(best)
        a = best
    return knots


def _dp(track: np.ndarray, tol: float, cost=segment_errors) -> list[int]:
    """Exact minimax DP. ``dp[k][b]`` = smallest achievable worst segment cost to ``b``.

    ``cost`` is the per-segment scalar being minimaxed: the interpolation error in the
    track's own units for a uniform tolerance, or ``error / tol(t)`` when the tolerance
    varies along the track. The recursion never looks at the units, only at ``max``.
    """
    T = len(track)
    C = np.full((T, T), np.inf)
    for i in range(T):
        C[i, i] = 0.0
        for j in range(i + 1, T):
            C[i, j] = cost(track, i, j)

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


MIN_TOL_SCALE = 0.2
"""Floor on the local tolerance, as a fraction of the tolerance asked for.

Without one, a key probability of 1.0 at full bias would drive the tolerance to zero and force
a knot on that frame unconditionally -- which is the head emitting a key, and the one authority
it is deliberately not given. At 0.2 the tightest the head can make any frame is 5x, which is
enough to move a knot onto it when the geometry agrees and not enough to override the DP when
it does not.
"""


MAX_TOL_SCALE = 4.0
"""Ceiling on the local tolerance, as a multiple of the tolerance asked for.

The counterpart of ``MIN_TOL_SCALE``. Without one, a confident "no key here" could raise the
tolerance far enough to span a whole track in one segment and lose real motion the head simply
failed to flag -- the head's mistakes should cost keys, not shapes.
"""


def local_tolerances(tol: float, key_prob: np.ndarray | None, bias: float,
                     slack: float = 0.0) -> np.ndarray | None:
    """Per-frame tolerance from a per-frame key probability, or ``None`` for the uniform case.

    ``tol(t) = tol * (1 - bias * p(t) + slack * (1 - p(t)))``, clipped to
    ``[MIN_TOL_SCALE, MAX_TOL_SCALE] * tol``. Two knobs, and they push in opposite directions
    for different reasons:

    * ``bias`` **tightens** where the head expects a key, so the DP puts a knot there. This is
      the obvious half, and on its own it can only *add* keys.
    * ``slack`` **loosens** where the head expects none, so the DP spans further and places
      fewer. This is the half that attacks the actual failure mode. Over-keying is this
      project's signature failure -- precision 0.21-0.44 against recall 0.75-1.00 -- so the
      binding constraint on key F1 is precision, and only ``slack`` can improve it. It is also
      what lets the pair move keys *without* moving the key count, which matters because the
      operating point has to stay inside an editable key economy.

    ``slack = 0`` is the one-sided form, kept as the default and as a measurable rung so the
    two can be compared rather than assumed.

    Linear in the probability rather than thresholded, because the head's output is a
    probability and a threshold would throw away what distinguishes "probably a key" from
    "certainly one" -- and because a hard threshold reintroduces emit-a-key behaviour through
    the back door.

    With neither knob set this returns ``None`` rather than an array of ``tol``, so the caller
    stays on the original code path and every earlier number reproduces bit for bit.
    """
    if key_prob is None or not (bias or slack):
        return None
    p = np.clip(np.asarray(key_prob, np.float64), 0.0, 1.0)
    scale = 1.0 - bias * p + slack * (1.0 - p)
    return tol * np.clip(scale, MIN_TOL_SCALE, MAX_TOL_SCALE)


def select(track: np.ndarray, frames: np.ndarray, tol: float,
           key_prob: np.ndarray | None = None, bias: float = 0.0,
           slack: float = 0.0) -> Selection:
    """Fewest knots whose linear interpolation stays within tolerance of ``track``.

    ``track`` is ``(T, P, 2)`` in whatever units ``tol`` is stated in -- crop pixels, in
    this project, so the tolerance is a distance an artist could be shown.

    ``key_prob`` is the key-timing head's per-frame probability on the same ``T`` axis;
    ``bias`` tightens the tolerance where a key is expected and ``slack`` loosens it where one
    is not (see :func:`local_tolerances`). With the probability or both knobs absent the
    behaviour is v1's exactly, on the same code path.
    """
    T = len(frames)
    if T <= 2:
        return Selection(frames.copy(), 0.0, True, 'trivial', tol, tol)

    tol_track = local_tolerances(tol, key_prob, bias, slack)
    if tol_track is None:
        cost, budget = segment_errors, tol
        lo = hi = tol
    else:
        inv = 1.0 / tol_track
        cost, budget = (lambda tr, a, b: segment_ratio(tr, a, b, inv)), 1.0
        lo, hi = float(tol_track.min()), float(tol_track.max())

    if T <= DP_MAX_FRAMES:
        idx, method, exact = _dp(track, budget, cost), 'dp', True
    else:
        idx, method, exact = _greedy(track, budget, cost), 'greedy', False

    chosen = np.asarray(idx, dtype=np.int32)
    # Reported in the track's own units either way, so ``max_error`` stays a distance an
    # artist could be shown rather than becoming a ratio when the bias is on.
    worst = max((segment_errors(track, int(chosen[i]), int(chosen[i + 1]))
                 for i in range(len(chosen) - 1)), default=0.0)
    return Selection(frames[chosen], float(worst), exact, method, lo, hi)


def f1(predicted: np.ndarray, truth: np.ndarray, tolerance: int = 0) -> tuple[float, float, float]:
    """Precision, recall, F1 of a predicted key set against the artist's.

    ``tolerance`` allows a predicted key to count if it lands within N frames of a real one.
    Matching is one-to-one, so two predictions cannot both claim the same artist key --
    without that, over-keying would inflate recall for free, which is the exact failure this
    module exists to avoid measuring badly.

    Candidate pairs are sorted by distance **globally** before being assigned. Matching them
    in input order instead let an early, more distant prediction take a truth key that a
    later, closer one needed, so the score depended on the order predictions arrived in. The
    hit *count* can only improve or stay equal under global sorting; what it removes is the
    order dependence.
    """
    if len(truth) == 0:
        return (0.0, 1.0, 0.0) if len(predicted) else (1.0, 1.0, 1.0)
    if len(predicted) == 0:
        return 1.0, 0.0, 0.0
    pairs = sorted((abs(int(p) - int(t)), pi, ti)
                   for pi, p in enumerate(predicted)
                   for ti, t in enumerate(truth)
                   if abs(int(p) - int(t)) <= tolerance)
    taken_p: set[int] = set()
    taken_t: set[int] = set()
    for _, pi, ti in pairs:
        if pi not in taken_p and ti not in taken_t:
            taken_p.add(pi)
            taken_t.add(ti)
    hits = len(taken_p)
    prec = hits / len(predicted)
    rec = hits / len(truth)
    return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)
