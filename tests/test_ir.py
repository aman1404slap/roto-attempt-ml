"""Track sampling: hold/linear/catmullrom, and the boundary behaviour the archive relies on."""
import numpy as np
import pytest

from roto.ir import CATMULLROM, HOLD, LINEAR, Key, Shape, opacity_at, sample


def _track(mode):
    return [Key(0, mode, np.array([0.0])), Key(10, mode, np.array([10.0])),
            Key(20, mode, np.array([0.0])), Key(30, mode, np.array([10.0]))]


def test_holds_outside_keyed_range():
    t = _track(LINEAR)
    assert sample(t, -50)[0] == 0.0
    assert sample(t, 999)[0] == 10.0


def test_linear_midpoint():
    assert sample(_track(LINEAR), 5)[0] == pytest.approx(5.0)


def test_hold_does_not_interpolate():
    assert sample(_track(HOLD), 5)[0] == 0.0
    assert sample(_track(HOLD), 9.99)[0] == 0.0
    assert sample(_track(HOLD), 10)[0] == 10.0


def test_catmullrom_passes_through_keys_and_curves_between():
    """Interpolates the keys exactly, but takes a curved path between them.

    Note: with symmetric key values the midpoint coincides with linear, so this uses an
    asymmetric track -- a symmetric one silently passes a broken implementation.
    """
    vals = [0.0, 10.0, 12.0, 40.0]
    cr = [Key(f, CATMULLROM, np.array([v])) for f, v in zip(range(0, 40, 10), vals)]
    lin = [Key(f, LINEAR, np.array([v])) for f, v in zip(range(0, 40, 10), vals)]
    for k in cr:
        assert sample(cr, k.frame)[0] == pytest.approx(k.value[0], abs=1e-9)
    assert sample(lin, 15)[0] == pytest.approx(11.0)
    assert sample(cr, 15)[0] == pytest.approx(9.875)      # hand-computed from the basis


def test_catmullrom_clamps_at_track_ends_rather_than_dropping_to_linear():
    """The first and last segments lack an outer neighbour, so the endpoint stands in for it.

    Hand-computed: on the first segment p0 is clamped to k0, giving
    ``0.5 * (2*0 + 10*0.5 + 40*0.25 - 30*0.125) = 5.625`` at the midpoint -- curved, where
    the earlier code silently returned the linear 5.0. Both ends stay exact on their keys.
    """
    t = _track(CATMULLROM)
    assert sample(t, 5)[0] == pytest.approx(5.625)
    assert sample(t, 25)[0] == pytest.approx(4.375)         # tail segment, mirror image
    for k in t:
        assert sample(t, k.frame)[0] == pytest.approx(k.value[0], abs=1e-9)


def test_interp_is_per_key_not_global():
    """A track may mix modes; the segment uses the mode of the key that starts it."""
    t = [Key(0, HOLD, np.array([0.0])), Key(10, LINEAR, np.array([10.0])),
         Key(20, LINEAR, np.array([20.0]))]
    assert sample(t, 5)[0] == 0.0                      # hold segment
    assert sample(t, 15)[0] == pytest.approx(15.0)     # linear segment


def test_unknown_interp_rejected():
    with pytest.raises(ValueError):
        Key(0, 'bogus', None)


def test_opacity_gating():
    """Silhouette stores 0-100; lifespans are hold keys 0 -> 100 -> 0."""
    s = Shape('x', path=[Key(0, LINEAR, np.zeros((4, 1, 2)))],
              opacity=[Key(-1, HOLD, np.array([0.0])), Key(5, HOLD, np.array([100.0])),
                       Key(6, HOLD, np.array([0.0]))])
    assert opacity_at(s, 0) == 0.0
    assert opacity_at(s, 5) == 1.0
    assert opacity_at(s, 20) == 0.0
    assert opacity_at(Shape('y', path=[Key(0, LINEAR, np.zeros((4, 1, 2)))]), 3) == 1.0


def test_fixed_topology_invariant_enforced():
    s = Shape('x', path=[Key(0, LINEAR, np.zeros((4, 1, 2))),
                         Key(9, LINEAR, np.zeros((5, 1, 2)))])
    with pytest.raises(ValueError, match='changes point count'):
        s.validate()
