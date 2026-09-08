"""Keyframe selection by curve simplification.

The planted-knot tests are the ones that matter: a track built by interpolating between known
frames has an unambiguous correct answer, so the selector is measured against ground truth
rather than against its own output.
"""
import numpy as np
import pytest

from roto.keys import f1, select
from roto.keys.dp import DP_MAX_FRAMES, segment_errors


def planted(knots, T, P=12, seed=0):
    """A track that is exactly piecewise-linear through ``knots``."""
    rng = np.random.default_rng(seed)
    ctrl = rng.normal(scale=30.0, size=(len(knots), P, 2))
    track = np.zeros((T, P, 2))
    for i in range(len(knots) - 1):
        a, b = knots[i], knots[i + 1]
        w = np.linspace(0.0, 1.0, b - a + 1)[:, None, None]
        track[a:b + 1] = ctrl[i][None] * (1 - w) + ctrl[i + 1][None] * w
    return track


@pytest.mark.parametrize('knots,T', [
    ([0, 17, 40, 41, 88, 119], 120),      # includes an adjacent pair
    ([0, 30, 60], 61),                    # short enough for the exact DP
    ([0, 5, 6, 7, 50], 51),
])
def test_recovers_planted_knots_exactly(knots, T):
    sel = select(planted(knots, T), np.arange(T), tol=0.5)
    assert sel.frames.tolist() == knots
    assert sel.max_error <= 0.5


def test_exact_dp_below_the_threshold_and_greedy_above():
    assert select(planted([0, 20, 40], 41), np.arange(41), 0.5).method == 'dp'
    n = DP_MAX_FRAMES + 40
    assert select(planted([0, 30, n - 1], n), np.arange(n), 0.5).method == 'greedy'


def test_a_static_track_needs_only_its_endpoints():
    track = np.tile(np.arange(20, dtype=float).reshape(10, 2)[None], (50, 1, 1))
    sel = select(track, np.arange(50), tol=0.5)
    assert sel.frames.tolist() == [0, 49]


def test_tolerance_trades_keys_for_error_monotonically():
    track = planted([0, 11, 23, 40, 55, 70, 99], 100)
    track += np.random.default_rng(1).normal(scale=0.4, size=track.shape)
    counts = [len(select(track, np.arange(100), t)) for t in (0.5, 1.0, 2.0, 4.0)]
    assert counts == sorted(counts, reverse=True), counts


def test_f1_matching_is_one_to_one():
    # Two predictions crowding one artist key must not both score as hits, or over-keying
    # would inflate recall for free -- the exact failure this module exists to avoid.
    p, r, s = f1(np.array([10, 11]), np.array([10]), tolerance=1)
    assert r == 1.0 and p == 0.5


def test_f1_assignment_does_not_depend_on_prediction_order():
    """Globally-sorted assignment: an early distant prediction cannot deny a closer later one.

    Predictions 9 and 10 against truth 10 with tolerance 1. Matching in input order gives 9
    the key and leaves 10 -- an exact hit -- unmatched. Sorted globally, the exact pair wins,
    and the score is the same whichever order the predictions arrive in.
    """
    assert f1(np.array([9, 10]), np.array([10]), tolerance=1) == \
           f1(np.array([10, 9]), np.array([10]), tolerance=1)
    # Two truth keys, two predictions, one crossing pair: both must match.
    p, r, s = f1(np.array([11, 20]), np.array([10, 20]), tolerance=1)
    assert (p, r) == (1.0, 1.0)


def test_segment_error_is_zero_on_its_endpoints():
    track = planted([0, 25], 26)
    assert segment_errors(track, 0, 25) == pytest.approx(0.0, abs=1e-9)


def test_segment_error_is_euclidean_not_manhattan():
    """A purely diagonal miss must read as its true distance, not sqrt(2) times it.

    The track runs straight from 0 to (10, 10) over two frames with the middle frame pushed
    (1, 1) off the chord. Euclidean says sqrt(2); ``|dx| + |dy|`` said 2.0, which is what
    made a '1.0 px tolerance' mean something an artist could not measure.
    """
    track = np.zeros((3, 1, 2))
    track[1] = [[5.0 + 1.0, 5.0 + 1.0]]
    track[2] = [[10.0, 10.0]]
    assert segment_errors(track, 0, 2) == pytest.approx(np.sqrt(2.0))
