"""Curve evaluation and exact B-spline -> cubic-Bezier conversion.

The archive is authored in uniform cubic B-splines. We keep that as the native
representation (see ir.py) and convert only at a render boundary -- for our own
rasteriser, which consumes cubic Bezier paths.

The conversion is exact, not an approximation: a uniform cubic B-spline segment *is* a
cubic Bezier segment, just written in a different basis.
"""
from __future__ import annotations

import numpy as np

# Uniform cubic B-spline basis, segment parameter t in [0,1], control points P0..P3:
#   S(t) = [ (1-t)^3 P0 + (3t^3 - 6t^2 + 4) P1 + (-3t^3 + 3t^2 + 3t + 1) P2 + t^3 P3 ] / 6


def _wrap_closed(P: np.ndarray) -> np.ndarray:
    """Control polygon extended so every segment i uses P[i-1..i+2], periodic."""
    n = len(P)
    return np.vstack([P[(i - 1) % n] for i in range(n + 3)])


TRIPLICATE, DUPLICATE, REFLECT, INTERIOR = 'triplicate', 'duplicate', 'reflect', 'interior'
OPEN_END_RULES = (TRIPLICATE, DUPLICATE, REFLECT, INTERIOR)

OPEN_END_RULE = TRIPLICATE
"""How an open B-spline's control polygon is extended past its ends.

**Measured against Silhouette's own delivered EXRs, not assumed** -- see
``v1.1/results/render_conventions.json`` and ``scripts/exp_conventions.py``. The four rules
are genuinely different curves, not variations in taste:

* ``triplicate`` -- repeat each end point three times, the standard clamped cubic B-spline.
  The curve passes exactly through the first and last control point.
* ``duplicate`` -- repeat each end point twice. The curve stops short of its end points.
* ``reflect`` -- extend with ``2*P0 - P1``, the "natural" extension.
* ``interior`` -- no extension; the curve spans only the interior segments, which is what a
  plain uniform B-spline basis gives and leaves the ends undrawn.

Open shapes are 51% of the archive, so this rule decides the geometry of half of it.

**Measured result: the choice among the first three is nearly immaterial here.** Against the
delivered EXRs at full resolution, on the two layers whose channel match is strong enough to
read (FAM blue 1 at 0.969, nfl_0080 MB 2 at 0.898), triplicate/duplicate/reflect score
0.9737/0.9743/0.9737 and 0.8992/0.9070/0.8992. ``duplicate`` is best by 0.0006 and 0.0078,
which is not enough to move off the standard rule.

``interior`` is **not a candidate** and is excluded from that comparison despite scoring
highest, because it is not a rival convention -- it is less curve. These strokes are short
(median 5 control points), so dropping the four end segments discards ~65% of the drawn
length on average, and on the 21-30% of shapes with <=4 points it degenerates to drawing the
control polygon itself. It wins by drawing less, which is the same confound the stroke-width
sweep shows directly. It is kept only because refereeing it is what identified that confound.
"""


def _clamp_open(P: np.ndarray, rule: str = OPEN_END_RULE) -> np.ndarray:
    """Extend an open control polygon past its ends according to ``rule``."""
    if rule == TRIPLICATE:
        return np.vstack([P[0], P[0], *P, P[-1], P[-1]])
    if rule == DUPLICATE:
        return np.vstack([P[0], *P, P[-1]])
    if rule == REFLECT:
        if len(P) < 2:
            return np.vstack([P[0], *P, P[-1]])
        return np.vstack([2 * P[0] - P[1], *P, 2 * P[-1] - P[-2]])
    if rule == INTERIOR:
        return np.asarray(P)
    raise ValueError(f'unknown open end rule {rule!r}, want one of {OPEN_END_RULES}')


def bspline_segments(P: np.ndarray, closed: bool, rule: str = OPEN_END_RULE) -> np.ndarray:
    """Split a control polygon into per-segment (4, 2) windows."""
    Q = _wrap_closed(P) if closed else _clamp_open(P, rule)
    n_seg = len(Q) - 3
    if n_seg < 1:
        return np.stack([Q[:4]]) if len(Q) >= 4 else np.zeros((0, 4, Q.shape[-1]))
    return np.stack([Q[i:i + 4] for i in range(n_seg)])


def bspline_to_bezier(P: np.ndarray, closed: bool, rule: str = OPEN_END_RULE) -> np.ndarray:
    """Exact conversion. Returns (n_seg, 4, 2) cubic Bezier control points.

    For one uniform cubic B-spline segment with control points P0..P3:
        B0 = (P0 + 4 P1 + P2) / 6
        B1 = (2 P1 + P2) / 3
        B2 = (P1 + 2 P2) / 3
        B3 = (P1 + 4 P2 + P3) / 6

    Endpoints and tangents match by construction:
        S(0) = B0, S(1) = B3, S'(0) = (P2 - P0)/2 = 3(B1 - B0).
    """
    seg = bspline_segments(np.asarray(P, dtype=np.float64), closed, rule)
    p0, p1, p2, p3 = seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 3]
    return np.stack([(p0 + 4 * p1 + p2) / 6.0,
                     (2 * p1 + p2) / 3.0,
                     (p1 + 2 * p2) / 3.0,
                     (p1 + 4 * p2 + p3) / 6.0], axis=1)


def eval_bezier(B: np.ndarray, samples_per_seg: int = 12,
                include_end: bool = False) -> np.ndarray:
    """Polyline through cubic Bezier segments B of shape (n_seg, 4, 2)."""
    t = np.linspace(0.0, 1.0, samples_per_seg, endpoint=False)[:, None]
    out = []
    for b0, b1, b2, b3 in B:
        out.append(((1 - t) ** 3) * b0 + 3 * ((1 - t) ** 2) * t * b1
                   + 3 * (1 - t) * t ** 2 * b2 + t ** 3 * b3)
    poly = np.vstack(out)
    if include_end:
        poly = np.vstack([poly, B[-1, 3][None, :]])
    return poly


def eval_bspline(P: np.ndarray, closed: bool, samples_per_seg: int = 12,
                 rule: str = OPEN_END_RULE) -> np.ndarray:
    """Polyline through a uniform cubic B-spline control polygon."""
    B = bspline_to_bezier(P, closed, rule)
    if not len(B):
        return np.asarray(P, dtype=np.float64)
    return eval_bezier(B, samples_per_seg, include_end=not closed)


def eval_silhouette_bezier(P: np.ndarray, closed: bool,
                           samples_per_seg: int = 12) -> np.ndarray:
    """Polyline for Silhouette's own Bezier form: P is (n, 3, 2) as
    (point, in-handle, out-handle) per control point."""
    P = np.asarray(P, dtype=np.float64)
    n = len(P)
    n_seg = n if closed else n - 1
    segs = []
    for i in range(n_seg):
        j = (i + 1) % n
        segs.append(np.stack([P[i, 0], P[i, 2], P[j, 1], P[j, 0]]))
    return eval_bezier(np.stack(segs), samples_per_seg, include_end=not closed)
