"""What v1.1 added: temporal window, curve loss, peak-preserving smoothing, key refit.

Each test here pins a claim the v1 review asked to be made true, and the ones that matter
most are the ones measuring a *difference* rather than a value -- that savgol keeps a peak a
boxcar flattens, that a refit recovers a value sampling the filter's output cannot, that the
polyline map is the renderer's own curve and not a second implementation of it.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from roto.ir import CATMULLROM, LINEAR, Key, Layer, RotoDoc, Shape, sample
from roto.keys.refit import basis_matrix, refit_key_values
from roto.model.curveloss import PolylineMaps, curve_loss, polyline_matrix
from roto.model.data import group_probes, shift_alpha
from roto.geometry import crop_matrix, local_to_crop
from roto.model.net import RotoNet
from roto.model.smoothing import smooth_track
from roto.model.train import (TrainConfig, affine_probe_term, affine_temporal_term,
                              apply_proj, temporal_term)
from roto.program import (affine_from_matrix, matrix_from_affine, matrix_from_proj,
                          proj_from_matrix)
from roto.render.curves import OPEN_END_RULES, eval_bspline
from roto.render.raster import RenderConfig, apply_transform, render


# ---- smoothing (review 2a) -----------------------------------------------------

def test_savgol_keeps_a_peak_that_boxcar_flattens():
    """The whole reason to change filter: artists key the extremes a boxcar shaves off."""
    x = np.arange(41.0)
    peak = (-(x - 20) ** 2 / 20.0)[:, None, None]      # apex exactly 0 at frame 20
    assert smooth_track(peak, 9, 'savgol')[20, 0, 0] == pytest.approx(0.0, abs=1e-9)
    assert smooth_track(peak, 9, 'boxcar')[20, 0, 0] == pytest.approx(-1.0 / 3.0, abs=1e-9)


def test_savgol_reproduces_a_quadratic_exactly_everywhere_including_the_ends():
    """Order-2 fit on quadratic data is the data, and the edge fit must not flatten it."""
    x = np.arange(30.0)
    quad = (0.5 * x ** 2 - 3 * x + 7)[:, None, None]
    got = smooth_track(quad, 9, 'savgol')
    assert np.abs(got - quad).max() < 1e-8


def test_smoothing_is_a_no_op_below_a_usable_window():
    t = np.random.default_rng(0).normal(size=(2, 3, 2))
    assert smooth_track(t, 9, 'savgol') is t
    assert smooth_track(t, 1, 'boxcar') is t


def test_unknown_smoothing_kind_is_rejected():
    with pytest.raises(ValueError, match='unknown smoothing kind'):
        smooth_track(np.zeros((10, 2, 2)), 5, 'gaussian')


# ---- key value refit (review 2b) -----------------------------------------------

@pytest.mark.parametrize('mode', [LINEAR, CATMULLROM])
def test_basis_matrix_is_the_renderers_own_interpolation(mode):
    """Built from ir.sample, so it cannot drift from what gets drawn."""
    frames, kf = np.arange(31), np.array([0, 10, 20, 30])
    modes = [mode] * 4
    vals = np.random.default_rng(0).normal(size=(4, 3, 1, 2))
    keys = [Key(int(f), mode, vals[i]) for i, f in enumerate(kf)]
    want = np.stack([np.asarray(sample(keys, int(f)), float) for f in frames])
    got = np.einsum('tk,k...->t...', basis_matrix(kf, modes, frames), vals)
    assert np.abs(got - want).max() < 1e-12
    # Partition of unity: a constant track must be reproduced by constant key values.
    assert np.allclose(basis_matrix(kf, modes, frames).sum(1), 1.0)


def test_refit_recovers_a_peak_that_sampling_the_smoothed_track_loses():
    """v1 stored key values from the smoothed track, so every motion extreme undershot.

    A track with a sharp mid-track peak, smoothed, then keyed at the peak frame: reading the
    smoothed value understates the apex, while fitting to the raw track recovers it.
    """
    frames = np.arange(41)
    raw = np.zeros((41, 1, 1, 2))
    # A full-width triangle: exactly representable by keys at 0/20/40, so the refit's
    # least-squares optimum *is* the true apex and any shortfall is the filter's bias.
    raw[:, 0, 0, 0] = 10.0 * (1.0 - np.abs(frames - 20.0) / 20.0)
    kf = np.array([0, 20, 40])
    modes = [LINEAR] * 3
    smoothed = smooth_track(raw, 9, 'boxcar')
    from_filter = smoothed[20, 0, 0, 0]
    fitted = refit_key_values(raw, kf, modes, frames)[1, 0, 0, 0]
    assert from_filter < 9.0                      # the filter has flattened the apex (8.89)
    assert fitted == pytest.approx(10.0, abs=1e-6)   # the refit recovers it exactly
    # And the fit is the least-squares optimum, so it beats the filtered value on the track.
    err = lambda v: np.abs(np.einsum('tk,k->t', basis_matrix(kf, modes, frames),
                                     np.array([0.0, v, 0.0])) - raw[:, 0, 0, 0])
    assert (err(fitted) ** 2).sum() < (err(from_filter) ** 2).sum()


def test_refit_is_exact_when_every_frame_is_a_key():
    frames = np.arange(12)
    raw = np.random.default_rng(1).normal(size=(12, 4, 1, 2))
    got = refit_key_values(raw, frames, [LINEAR] * 12, frames)
    assert np.abs(got - raw).max() < 1e-9


# ---- curve loss (review 3) -----------------------------------------------------

@pytest.mark.parametrize('P,closed', [(4, True), (7, False), (93, True)])
def test_polyline_map_is_exactly_the_renderers_curve(P, closed):
    pts = np.random.default_rng(0).normal(size=(P, 2))
    assert np.abs(polyline_matrix(P, closed, 4) @ pts
                  - eval_bspline(pts, closed, 4)).max() < 1e-12


def test_curve_loss_is_zero_only_when_the_curves_agree():
    n_points = np.array([6, 6]); closed = np.array([True, True]); coords = np.array([1, 1])
    maps = PolylineMaps(n_points, closed, coords)
    pred = torch.rand(2, 2, 6, 1, 2)
    live = torch.ones(2, 2, dtype=torch.bool)
    assert float(curve_loss(pred, pred.clone(), live, maps, 256.0)) == pytest.approx(0.0)
    assert float(curve_loss(pred, pred + 0.01, live, maps, 256.0)) > 0.0


def test_curve_loss_excludes_bezier_shapes():
    """Their drawn curve depends on handles, so a point-only map is the wrong geometry."""
    maps = PolylineMaps(np.array([6, 6, 6]), np.array([True] * 3), np.array([1, 3, 1]))
    assert maps.n_shapes_covered == 2


def test_curve_loss_sees_a_control_polygon_error_the_point_loss_underrates():
    """Two polygons, equal mean point error, different curves -- the point term cannot tell.

    Displacing one interior point of a B-spline moves the drawn curve; displacing a point
    along the curve's own direction barely does. The curve term separates them.
    """
    maps = PolylineMaps(np.array([8]), np.array([True]), np.array([1]))
    base = torch.tensor(np.array([[np.cos(t), np.sin(t)] for t in
                                  np.linspace(0, 2 * np.pi, 8, endpoint=False)]),
                        dtype=torch.float32).reshape(1, 1, 8, 1, 2)
    live = torch.ones(1, 1, dtype=torch.bool)
    bulge = base.clone(); bulge[0, 0, 3, 0, :] += 0.10                  # normal to the curve
    slide = base.clone(); slide[0, 0, :, 0, :] += 0.10 / np.sqrt(8)     # a rigid shift
    l_point = lambda p: float((p - base).abs().mean())
    assert l_point(bulge) < l_point(slide)          # point loss prefers the bulge
    assert float(curve_loss(bulge, base, live, maps, 1.0)) > 0.0


# ---- temporal window and consistency (review 1a, 1b) ---------------------------

def test_temporal_term_is_zero_for_a_perfectly_tracked_motion():
    """It penalises jitter, not motion: matching the target's deltas costs nothing."""
    b = {'points': torch.zeros(3, 2, 4, 1, 2), 'live': torch.ones(3, 2, dtype=torch.bool),
         'point_mask': torch.ones(1, 2, 4, 1, dtype=torch.bool),
         'frames': torch.arange(3)}
    b['points'][:, :, :, :, 0] = torch.arange(3.0)[:, None, None, None]   # steady travel
    assert float(temporal_term(b['points'].clone(), b, 256.0)) == pytest.approx(0.0)
    jittered = b['points'].clone()
    jittered[1] += 0.01
    assert float(temporal_term(jittered, b, 256.0)) > 0.0


def test_temporal_term_ignores_pairs_that_are_not_adjacent_frames():
    """A held-out frame leaves a gap in the run; a gap is not a frame-to-frame delta."""
    b = {'points': torch.zeros(2, 1, 4, 1, 2), 'live': torch.ones(2, 1, dtype=torch.bool),
         'point_mask': torch.ones(1, 1, 4, 1, dtype=torch.bool),
         'frames': torch.tensor([0, 2])}
    pred = b['points'].clone(); pred[1] += 1.0
    assert float(temporal_term(pred, b, 256.0)) == pytest.approx(0.0)


def test_network_accepts_a_temporal_window_and_v1_arch_still_builds():
    for in_frames, self_attn in [(1, False), (3, False), (3, True)]:
        net = RotoNet(10, 2, 6, 1, dim=32, depth=1, in_frames=in_frames,
                      self_attn=self_attn)
        pts, aff, _ = net(torch.rand(2, in_frames, 256, 256), torch.arange(4)[None].expand(2, -1),
                       torch.arange(2)[None].expand(2, -1), torch.rand(2, 4, 3))
        assert pts.shape == (2, 4, 6, 1, 2) and aff.shape == (2, 2, 6)


def test_self_attention_lets_one_query_change_another():
    """The point of restoring it: without it, queries cannot negotiate at all."""
    torch.manual_seed(0)
    net = RotoNet(10, 2, 6, 1, dim=32, depth=1, self_attn=True).eval()
    # ``point_head`` is deliberately zero-initialised so training starts at the crop centre;
    # left that way it emits 0.5 for every query and would hide the wiring under test.
    torch.nn.init.normal_(net.point_head[-1].weight, std=0.02)
    alpha, gid, desc = torch.rand(1, 1, 256, 256), torch.arange(2)[None], torch.rand(1, 3, 3)
    a, _, _ = net(alpha, torch.tensor([[0, 1, 2]]), gid, desc)
    b, _, _ = net(alpha, torch.tensor([[0, 1, 5]]), gid, desc)  # change the third query only
    assert not torch.allclose(a[0, 0], b[0, 0], atol=1e-6), \
        'query 0 did not react to a change in query 2 -- self-attention is not wired in'

    flat = RotoNet(10, 2, 6, 1, dim=32, depth=1, self_attn=False).eval()
    torch.nn.init.normal_(flat.point_head[-1].weight, std=0.02)
    c, _, _ = flat(alpha, torch.tensor([[0, 1, 2]]), gid, desc)
    d, _, _ = flat(alpha, torch.tensor([[0, 1, 5]]), gid, desc)
    assert torch.allclose(c[0, 0], d[0, 0], atol=1e-6), \
        'without self-attention a query must be unable to see any other -- that was v1'


# ---- render conventions (review 5, 8) ------------------------------------------

def test_the_four_open_end_rules_are_four_different_curves():
    P = np.array([[0., 0.], [1., 1.], [2., 0.], [3., 1.], [4., 0.]])
    got = {r: eval_bspline(P, False, 6, r) for r in OPEN_END_RULES}
    # triplicate and reflect both interpolate the end points; duplicate and interior do not.
    assert np.allclose(got['triplicate'][0], P[0])
    assert np.allclose(got['reflect'][0], P[0])
    assert not np.allclose(got['duplicate'][0], P[0])
    lens = {r: len(v) for r, v in got.items()}
    assert len(set(lens.values())) > 1, lens


def test_clip_ordering_only_matters_where_a_subtract_follows_overlapping_adds():
    """Two overlapping Adds then a Subtract: clipping first discards the overshoot."""
    def box(x0, x1, blend):
        pts = np.array([[[x0, -0.2]], [[x1, -0.2]], [[x1, 0.2]], [[x0, 0.2]]])
        return Shape('s', closed=True, blend=blend, path=[Key(0, LINEAR, pts)])

    overlap = RotoDoc(64, 64, 1, [Layer('l', children=[
        box(-0.2, 0.2, 'Add'), box(-0.2, 0.2, 'Add'), box(-0.2, 0.2, 'Subtract')])])
    single = RotoDoc(64, 64, 1, [Layer('l', children=[
        box(-0.2, 0.2, 'Add'), box(-0.2, 0.2, 'Subtract')])])
    on = render(overlap, 0, RenderConfig(supersample=1, clip_per_shape=True))
    off = render(overlap, 0, RenderConfig(supersample=1, clip_per_shape=False))
    # Clipping first throws away the overshoot, so the Subtract cuts through to nothing;
    # compositing unclipped leaves one Add's worth behind. A real difference, not a nuance.
    assert on.sum() == 0.0 and off.sum() > 0.0
    # With no overlap to overshoot, the two rules agree exactly.
    a = render(single, 0, RenderConfig(supersample=1, clip_per_shape=True))
    b = render(single, 0, RenderConfig(supersample=1, clip_per_shape=False))
    assert np.abs(a - b).max() == 0.0


# ---- frame window and holdout (review 1a, 4) -----------------------------------

class _FakeLayer:
    """The array behaviours ``LayerData`` adds, without a 200 MB fixture."""
    from roto.model.data import LayerData
    window = LayerData.window
    split = LayerData.split
    offsets_crop_px = LayerData.offsets_crop_px

    def __init__(self, n):
        self.layer_id = 'fake'
        self.frames = np.arange(n)
        # Empty as of v1.3: ``split`` prefers a split the *dataset* recorded and falls back
        # to the rule when there is none, which is the path this fixture is testing.
        self.split_train = np.zeros(0, np.int64)
        self.split_held = np.zeros(0, np.int64)
        self._a = np.arange(n, dtype=np.float32)[:, None, None] * np.ones((1, 4, 4), np.float32)

    @property
    def alphas(self):
        return self._a


def test_window_is_centred_and_clamped_at_the_track_ends():
    el = _FakeLayer(10)
    w = el.window(np.array([0, 5, 9]), 3)
    assert w.shape == (3, 3, 4, 4)
    assert [w[0, i, 0, 0] for i in range(3)] == [0, 0, 1]      # first frame sees itself twice
    assert [w[1, i, 0, 0] for i in range(3)] == [4, 5, 6]
    assert [w[2, i, 0, 0] for i in range(3)] == [8, 9, 9]
    assert el.window(np.array([5]), 1).shape == (1, 1, 4, 4)


def test_holdout_split_is_disjoint_and_never_takes_a_track_end():
    el = _FakeLayer(40)
    train, held = el.split(7)
    assert len(held) > 0
    assert not set(train) & set(held)
    assert sorted(set(train) | set(held)) == list(range(40))
    assert 0 not in held and 39 not in held, 'a held end frame would have no trained neighbour'
    assert el.split(0)[1].size == 0


# ---- the transform representation (v1.1 review item 2) -------------------------

def _projective_matrix(m33=1.4, m03=0.03, m13=-0.02):
    """A row-vector 4x4 of the shape the archive actually contains: affine 2x2 plus a real
    perspective column. ``TVC_sh0260 Layer_52`` runs m33 from 0.672 to 1.801."""
    m = np.zeros((4, 4))
    m[0, 0], m[0, 1] = 1.3, 0.4
    m[1, 0], m[1, 1] = -0.2, 0.9
    m[2, 2] = 1.0
    m[3, 0], m[3, 1] = 0.11, -0.07
    m[0, 3], m[1, 3], m[3, 3] = m03, m13, m33
    return m


def test_the_six_number_affine_target_is_lossy_on_a_perspective_transform():
    """Why the transform head could not be judged: two layers carry perspective, and v1's
    target throws it away, so a *perfect* prediction still draws the wrong picture."""
    m = _projective_matrix()
    pts = np.array([[0.2, -0.3], [-0.4, 0.1], [0.0, 0.0]])
    want = apply_transform(m, pts)
    got = apply_transform(matrix_from_affine(affine_from_matrix(m)), pts)
    assert np.abs(got - want).max() > 0.1, \
        'the affine round trip looks lossless -- the fixture has no perspective in it'


def test_the_eight_number_projective_target_is_exact_on_the_plane():
    m = _projective_matrix()
    pts = np.array([[0.2, -0.3], [-0.4, 0.1], [0.0, 0.0], [0.5, 0.5]])
    got = apply_transform(matrix_from_proj(proj_from_matrix(m)), pts)
    assert np.abs(got - apply_transform(m, pts)).max() < 1e-12
    # Strictly a superset: an affine track leaves three of the eight entries at their
    # constants, so nothing about the affine case changes.
    a = _projective_matrix(m33=1.0, m03=0.0, m13=0.0)
    p = proj_from_matrix(a)
    assert p[2] == 0.0 and p[5] == 0.0
    assert np.abs(apply_transform(matrix_from_proj(p), pts)
                  - apply_transform(a, pts)).max() < 1e-12


def test_the_gauge_normalisation_makes_scaled_copies_identical():
    """A projective map is defined up to overall scale, so 2M and M must encode the same."""
    m = _projective_matrix()
    assert np.abs(proj_from_matrix(m) - proj_from_matrix(2.0 * m)).max() < 1e-12
    with pytest.raises(AssertionError, match='m33 at zero'):
        proj_from_matrix(_projective_matrix(m33=0.0))


def test_crop_matrix_composes_with_the_layer_matrix_to_give_local_to_crop():
    """The identity the crop-space transform target rests on: ``local_to_crop`` is
    ``p @ (M @ C)``, so a document-space matrix becomes a crop-space one by one multiply."""
    m = _projective_matrix()
    kw = dict(width=2880, height=1620, offset=(410.0, 233.0), scale=0.25, out_px=256)
    pts = np.array([[0.2, -0.3], [-0.4, 0.1]])
    want = local_to_crop(pts, m, **kw)
    got = apply_transform(m @ crop_matrix(**kw), pts)
    assert np.abs(got - want).max() < 1e-12
    # And the perspective column survives the change of space untouched, which is what makes
    # the inverse a plain ``@ inv(C)``.
    assert np.abs((m @ crop_matrix(**kw))[:, 3] - m[:, 3]).max() < 1e-12


def test_apply_proj_matches_the_renderers_own_transform():
    """The loss's fused expression and the rasteriser must agree, or the term optimises a
    motion nobody draws -- the same trap ``curveloss`` avoids by construction."""
    m = _projective_matrix()
    probe = torch.tensor([[[0.2, -0.3], [-0.4, 0.1], [0.0, 0.0]]], dtype=torch.float64)
    params = torch.tensor(proj_from_matrix(m), dtype=torch.float64)[None]
    got = apply_proj(params, probe).numpy()[0]
    want = apply_transform(matrix_from_proj(proj_from_matrix(m)), probe.numpy()[0])
    assert np.abs(got - want).max() < 1e-12


def test_affine_probe_term_is_zero_only_when_the_transforms_agree():
    probe = torch.rand(2, 5, 2, dtype=torch.float64) - 0.5
    # Two *affine* groups here, so the pixel figure asserted below is closed-form; the
    # perspective path is covered by ``test_apply_proj_matches_the_renderers_own_transform``.
    t = torch.tensor(np.stack([
        proj_from_matrix(_projective_matrix(m33=1.0, m03=0.0, m13=0.0)),
        proj_from_matrix(2.0 * _projective_matrix(m33=1.0, m03=0.0, m13=0.0)),
    ]))[None]
    assert float(affine_probe_term(t, t.clone(), probe, 256.0)) == pytest.approx(0.0)
    off = t.clone(); off[0, 0, 6] += 0.01          # 0.01 crop units of translation, group 0
    # In crop pixels, so one weight means the same thing here as for the point term: the
    # error lands on the x coordinate of 5 probes out of the 20 numbers averaged.
    assert float(affine_probe_term(off, t, probe, 256.0)) == pytest.approx(0.01 * 256 / 4)


def test_affine_probe_term_prices_a_rotation_by_how_far_it_moves_the_layer():
    """The reason to score what the matrix does rather than its entries: identical radians
    on a large group and a small one are not the same error in pixels."""
    small = torch.tensor([[[0.01, 0.0], [0.0, 0.01]]], dtype=torch.float64)
    large = small * 50.0
    ident = torch.tensor(proj_from_matrix(np.eye(4)))[None, None]
    rot = np.eye(4); c, s_ = np.cos(0.05), np.sin(0.05)
    rot[0, 0], rot[0, 1], rot[1, 0], rot[1, 1] = c, s_, -s_, c
    turned = torch.tensor(proj_from_matrix(rot))[None, None]
    a = float(affine_probe_term(turned, ident, small, 256.0))
    b = float(affine_probe_term(turned, ident, large, 256.0))
    assert b > 40 * a, (a, b)


def test_affine_temporal_term_ignores_non_adjacent_frames_like_the_point_one():
    probe = torch.rand(1, 5, 2, dtype=torch.float64) - 0.5
    t = torch.tensor(proj_from_matrix(np.eye(4)))[None, None].repeat(2, 1, 1)
    pred = t.clone(); pred[1, 0, 6] += 0.05
    assert float(affine_temporal_term(pred, t, probe, torch.tensor([0, 2]), 256.0)) \
        == pytest.approx(0.0)
    assert float(affine_temporal_term(pred, t, probe, torch.tensor([0, 1]), 256.0)) > 0.0


def test_group_probes_span_the_groups_own_points_and_ignore_dead_frames():
    local = np.zeros((3, 2, 4, 1, 2), np.float32)
    local[:, 0, :, 0] = [[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]]
    local[2, 1, :, 0] = 99.0                        # a shape only live on a frame we exclude
    live = np.array([[True, False], [True, False], [True, False]])
    pmask = np.ones((2, 4, 1), bool)
    probes = group_probes(local, live, np.array([0, 0]), pmask, 1)
    assert probes.shape == (1, 5, 2)
    assert np.allclose(probes[0, :4], [[0, 0], [1, 0], [1, 2], [0, 2]])
    assert np.allclose(probes[0, 4], [0.5, 1.0])    # centre


def test_transform_head_width_follows_the_target_and_v1_checkpoints_still_load():
    for dof in (6, 8):
        net = RotoNet(10, 3, 6, 1, dim=32, depth=1, affine_dim=dof)
        _, aff, _ = net(torch.rand(2, 1, 256, 256), torch.arange(4)[None].expand(2, -1),
                     torch.arange(3)[None].expand(2, -1), torch.rand(2, 4, 3))
        assert aff.shape == (2, 3, dof)
    assert RotoNet(10, 3, 6, 1, dim=32, depth=1).affine_dim == 6, 'v1 default must stay 6'
    assert RotoNet(10, 3, 6, 1, dim=32, depth=1).align_window is False


def test_unknown_affine_space_is_rejected():
    from roto.model.train import train
    with pytest.raises(ValueError, match='unknown affine_space'):
        train('datasets/v001', '/tmp/never', TrainConfig(steps=1, affine_space='world'))


# ---- window alignment (v1.1 review item 1) -------------------------------------

def test_shift_alpha_moves_content_forward_and_zeroes_what_comes_in():
    a = np.zeros((8, 8), np.float32)
    a[4, 4] = 1.0
    assert shift_alpha(a, 2.0, -1.0)[3, 6] == pytest.approx(1.0)
    assert shift_alpha(a, 0.0, 0.0)[4, 4] == pytest.approx(1.0)
    # Subpixel, and it conserves mass rather than rounding the shift away.
    half = shift_alpha(a, 0.5, 0.0)
    assert half[4, 4] == pytest.approx(0.5) and half[4, 5] == pytest.approx(0.5)
    edge = np.ones((8, 8), np.float32)
    assert shift_alpha(edge, 3.0, 0.0)[:, :3].max() == 0.0


class _CropLayer(_FakeLayer):
    """A layer whose content sits still in the *world* while its crop window travels, which
    is the situation window alignment exists for."""

    def __init__(self, n=7, step=3.0):
        super().__init__(n)
        self.crop = {'offsets': {i: (i * step, 0.0) for i in range(n)}, 'scale': 1.0}
        self._offsets_crop_px = None
        a = np.zeros((n, 1, 32), np.float32)
        for i in range(n):
            a[i, 0, 16 - int(i * step)] = 1.0       # fixed in the world, moving in the crop
        self._a = np.repeat(a, 32, axis=1)


def test_aligned_window_puts_a_world_fixed_feature_in_one_place():
    el = _CropLayer()
    raw = el.window(np.array([3]), 3)
    aligned = el.window(np.array([3]), 3, align=True)
    col = lambda w, j: int(np.argmax(w[0, j, 0]))
    assert [col(raw, j) for j in range(3)] == [10, 7, 4], \
        'the fixture must show the feature moving in the raw stack'
    assert [col(aligned, j) for j in range(3)] == [7, 7, 7], \
        'after alignment all three channels must agree on where the shape is'
    assert np.abs(aligned[0, 1] - raw[0, 1]).max() == 0.0, 'the anchor must not be touched'


def test_alignment_is_off_by_default_and_a_still_window_is_unchanged():
    el = _CropLayer(step=0.0)
    idx = np.array([2, 4])
    assert np.abs(el.window(idx, 3, align=True) - el.window(idx, 3)).max() == 0.0
    assert np.abs(_CropLayer().window(idx, 3)
                  - _CropLayer().window(idx, 3, align=False)).max() == 0.0


# ---- render conventions as one switch (v1.1 review item 4) ---------------------

def test_measured_conventions_carry_all_four_refereed_answers():
    """The v2 rebuild has to flip three conventions together or the dataset is a mixture.
    Pinned here so a partial flip fails a test rather than shipping."""
    from roto.render.curves import DUPLICATE, TRIPLICATE
    from roto.render.raster import RenderConfig, measured_conventions
    m, v1 = measured_conventions(), RenderConfig()
    assert (m.fill_open_zero_width, m.open_end_rule) == (False, DUPLICATE)
    assert m.stroke_px(0.030864, 1620) == 1.0, 'stroke must sit on the 1 px floor'
    # ...and v1's must not move, or every published number stops reproducing.
    assert (v1.fill_open_zero_width, v1.open_end_rule) == (True, TRIPLICATE)
    assert v1.stroke_px(0.030864, 1620) == pytest.approx(0.030864 * 1620 * 0.0625)


def test_the_dataset_build_reads_its_conventions_off_the_crop_config():
    from roto.dataset import CropConfig
    from roto.dataset.build import render_config_for
    assert CropConfig().conventions == 'v1', 'the default must keep building v001'
    assert render_config_for(CropConfig()).fill_open_zero_width is True
    assert render_config_for(CropConfig(conventions='measured')).fill_open_zero_width is False
    with pytest.raises(ValueError, match='unknown conventions'):
        render_config_for(CropConfig(conventions='v2'))


def test_affine_probe_term_ignores_groups_with_nothing_on_screen():
    """34% of (frame, group) cells in the archive have no live shape. Unmasked, a third of
    this term asks where an invisible group is -- unlearnable from the alpha, and unscorable
    because ``affine_doc`` only ever moves live shapes."""
    probe = torch.rand(2, 5, 2, dtype=torch.float64) - 0.5
    t = torch.tensor(proj_from_matrix(np.eye(4)))[None, None].repeat(1, 2, 1)
    pred = t.clone()
    pred[0, 1, 6] += 1.0                            # a huge error, on group 1 only
    live = torch.tensor([[True, False]])            # ...which is dead at this frame
    assert float(affine_probe_term(pred, t, probe, 256.0, live)) == pytest.approx(0.0)
    assert float(affine_probe_term(pred, t, probe, 256.0)) > 0.0, \
        'without the mask the dead group must contribute -- else this test proves nothing'
    both = torch.tensor([[True, True]])
    assert float(affine_probe_term(pred, t, probe, 256.0, both)) > 0.0


def test_affine_temporal_term_skips_a_group_blinking_on():
    """A group appearing between two frames carries a step change, not jitter."""
    probe = torch.rand(1, 5, 2, dtype=torch.float64) - 0.5
    t = torch.tensor(proj_from_matrix(np.eye(4)))[None, None].repeat(2, 1, 1)
    pred = t.clone(); pred[1, 0, 6] += 0.05
    frames = torch.tensor([0, 1])
    blink = torch.tensor([[False], [True]])
    assert float(affine_temporal_term(pred, t, probe, frames, 256.0, blink)) \
        == pytest.approx(0.0)
    on = torch.tensor([[True], [True]])
    assert float(affine_temporal_term(pred, t, probe, frames, 256.0, on)) > 0.0
