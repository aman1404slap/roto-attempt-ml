"""The B-spline -> Bezier conversion must be exact, not approximate.

If it drifts, every raster loss and every DiffVG gradient is quietly wrong.
"""
import numpy as np
import pytest

from roto.render.curves import bspline_to_bezier, eval_bezier, eval_bspline


def bspline_direct(P, closed, spp=64):
    """Evaluate the uniform cubic B-spline basis directly, as a reference."""
    n = len(P)
    Q = (np.vstack([P[(i - 1) % n] for i in range(n + 3)]) if closed
         else np.vstack([P[0], P[0], *P, P[-1], P[-1]]))
    t = np.linspace(0, 1, spp, endpoint=False)[:, None]
    b0 = (1 - t) ** 3 / 6
    b1 = (3 * t ** 3 - 6 * t ** 2 + 4) / 6
    b2 = (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6
    b3 = t ** 3 / 6
    return np.vstack([b0 * Q[i] + b1 * Q[i + 1] + b2 * Q[i + 2] + b3 * Q[i + 3]
                      for i in range(len(Q) - 3)])


@pytest.mark.parametrize('closed', [True, False])
@pytest.mark.parametrize('n', [4, 7, 13, 40])
def test_conversion_is_exact(closed, n):
    P = np.random.default_rng(n).normal(size=(n, 2))
    direct = bspline_direct(P, closed)
    via_bezier = eval_bezier(bspline_to_bezier(P, closed), 64)
    assert np.abs(direct - via_bezier).max() < 1e-12


def test_segment_count():
    P = np.zeros((10, 2))
    assert len(bspline_to_bezier(P, closed=True)) == 10      # one per control point
    assert len(bspline_to_bezier(P, closed=False)) == 11     # clamped extension


def test_bezier_endpoints_and_tangents_match():
    """S(0)=B0, S(1)=B3, S'(0)=(P2-P0)/2=3(B1-B0) for one segment."""
    P = np.random.default_rng(0).normal(size=(4, 2))
    B = bspline_to_bezier(P, closed=False)[2]               # the interior segment
    assert np.allclose(B[0], (P[0] + 4 * P[1] + P[2]) / 6)
    assert np.allclose(B[3], (P[1] + 4 * P[2] + P[3]) / 6)
    assert np.allclose(3 * (B[1] - B[0]), (P[2] - P[0]) / 2)


def test_closed_curve_is_periodic():
    P = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    poly = eval_bspline(P, closed=True, samples_per_seg=8)
    assert len(poly) == 32
    assert np.allclose(poly[0], eval_bspline(np.roll(P, 0, axis=0), True, 8)[0])
