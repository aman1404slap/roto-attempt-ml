"""What v1.2 added: the two missing exactness rows, worst-case metrics, predicted motion.

The handover splits every pixel of disagreement into pipeline error, which must be provably
zero, and model error, which is minimised and measured worst-case. This file is the first
half. Two rows of that ledger had no test -- window alignment and the transform target's
probe geometry -- and both are checked here against the *real* archive rather than a fixture,
because both are claims about recorded dataset metadata and a synthetic offset cannot be
wrong in the way a recorded one can.

The rest pins the paths v1.2 adds, each by the habit that found three of v1.1's four bugs:
feed the pipeline the artist's own answer and anything short of exact is the measurement's
fault. `test_predicted_motion_reproduces_the_teacher_forced_run...` is that check for the
de-teacher-forcing path, and it is the reason that path can be trusted at all.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from roto.ir import sample
from roto.model.data import load_element
from roto.model.geometry import crop_matrix, local_to_crop
from roto.model.net import RotoNet
from roto.model.reconstruct import (ARTIST, PREDICTED, RebuildConfig, assemble,
                                    predicted_shape_matrices, to_local, with_transforms)
from roto.model.report import run_totals, spread
from roto.model.train import apply_proj
from roto.program import proj_from_matrix
from roto.sfx.json_ir import from_json_ir

DATASET = Path('datasets/v001')

PERSPECTIVE = 'FAM_0060_L1_A0003C007_v001__red_1'
"""89 shapes in 7 transform groups, and the layer whose track is *not* affine: |m03| reaches
0.028 and m33 leaves 1.0. Any test of the transform path that only uses affine layers is
testing the easy half."""

EXTREME_SCALE = 'nfl_0200_bg01_v001_compplate_roto_v001__green'
"""px_per_norm 4365 -- its crop covers 0.059 of the document, so a coordinate mapping that is
slightly wrong in normalised units is very wrong in crop pixels here. The cheapest layer to
load, and the one where the arithmetic has the least margin."""

SMALL = 'FAM_0060_L1_A0003C007_v001__r1_t_c'
"""4 shapes: for tests that render."""


def element(name: str):
    d = DATASET / name
    if not (d / 'meta.json').exists():
        pytest.skip('dataset not built')
    return load_element(d)


# ---- ledger row: window alignment (plan S2) -----------------------------------

@pytest.mark.parametrize('name', [PERSPECTIVE, EXTREME_SCALE])
def test_window_alignment_uses_the_recorded_offsets_exactly(name):
    """The delta that shifts a neighbour into the anchor's window must be the delta the
    *targets* were built with -- to 1e-9 crop pixels, on real recorded offsets.

    ``LayerData.window(align=True)`` warps neighbour alphas by
    ``(offset_neighbour - offset_anchor) * scale``. That is only the correct warp if the same
    delta, applied to the neighbour's target control points, lands them exactly where
    ``local_to_crop`` puts them in the *anchor's* window. If the two disagree, the network is
    handed three channels that agree with each other and disagree with the target it is being
    trained against -- a bias no loss can see and no aggregate would reveal.

    The alpha warp itself is bilinear and softens the edge, which is fine and is not what is
    being checked. Coordinates are what must be exact.

    Both sides are computed in float64 from the same stored local points, because the claim
    is about the recorded *offsets*, not about storage. What the float32 target arrays cost on
    top of that is measured separately below, so neither hides inside the other's tolerance.
    """
    el = element(name)
    c = el.crop
    kw = dict(width=c['width'], height=c['height'], scale=c['scale'], out_px=c['out_px'])
    off = el.offsets_crop_px                                   # (F, 2), recorded * scale
    n = len(el.frames)
    worst = 0.0
    for anchor in (1, n // 2, n - 2):                           # every window has both sides
        for j in (anchor - 1, anchor + 1):
            delta = off[j] - off[anchor]                        # what `window` shifts by
            for si in range(el.n_shapes):
                if not el.live[j, si]:
                    continue
                P, C = int(el.point_mask[si, :, 0].sum()), int(el.point_mask[si, 0].sum())
                local = el.local[j, si, :P, :C].astype(np.float64)
                at = lambda fi: local_to_crop(local, el.matrices[j, si],
                                              offset=c['offsets'][int(el.frames[fi])],
                                              **kw) * el.out_px
                worst = max(worst, float(np.abs(at(j) + delta - at(anchor)).max()))
    assert worst <= 1e-9, f'{name}: alignment delta is off by {worst:.3e} crop px'


@pytest.mark.parametrize('name', [PERSPECTIVE, EXTREME_SCALE])
def test_the_float32_point_targets_cost_far_less_than_a_pixel(name):
    """And what the stored targets add to it. ``points`` is float32 across a [0,1] crop, so
    reading a target back and shifting it by the recorded delta agrees with the anchor-window
    mapping to ~1e-5 crop px -- five orders of magnitude under the model's own error, and on
    the ledger as a measured number rather than an assumed zero."""
    el = element(name)
    c = el.crop
    kw = dict(width=c['width'], height=c['height'], scale=c['scale'], out_px=c['out_px'])
    off, n = el.offsets_crop_px, len(el.frames)
    anchor, j = n // 2, n // 2 + 1
    worst = 0.0
    for si in range(el.n_shapes):
        if not el.live[j, si]:
            continue
        P, C = int(el.point_mask[si, :, 0].sum()), int(el.point_mask[si, 0].sum())
        want = local_to_crop(el.local[j, si, :P, :C].astype(np.float64), el.matrices[j, si],
                             offset=c['offsets'][int(el.frames[anchor])], **kw) * el.out_px
        got = el.points[j, si, :P, :C] * el.out_px + (off[j] - off[anchor])
        worst = max(worst, float(np.abs(got - want).max()))
    assert worst < 1e-3, f'{name}: float32 point targets cost {worst:.2e} crop px'


def test_window_alignment_is_not_vacuous_on_this_archive():
    """The check above passes trivially if the window never moves. It moves."""
    el = element(PERSPECTIVE)
    step = np.abs(np.diff(el.offsets_crop_px, axis=0)).max(1)
    assert step.max() > 0.5, 'fixture must be a layer whose crop window actually twitches'


# ---- ledger row: the probe-point transform loss (plan S2) ---------------------

@pytest.mark.parametrize('name', [PERSPECTIVE, EXTREME_SCALE])
def test_probe_targets_are_the_renderers_own_local_to_crop(name):
    """The transform loss scores where five probe points land. Those landings must be the
    same points ``local_to_crop`` produces -- to 1e-9 crop pixels, through the real track.

    Two compositions have to agree: the loss pushes probes through the 8-number target
    (``apply_proj`` of ``proj_from_matrix(M @ crop_matrix(f))``), and the geometry path pushes
    the same probes through ``local_to_crop`` with the same matrix and the same recorded
    offset. If they disagree, the transform term and the point term are pulling the encoder
    toward two different pictures, in the same unit, with nothing to reveal it.

    The comparison is in float64 because it is a claim about the algebra. What the dataset
    *stores* is float32, and that cost is pinned separately below rather than hidden inside
    a looser tolerance.
    """
    el = element(name)
    c = el.crop
    kw = dict(width=c['width'], height=c['height'], scale=c['scale'], out_px=c['out_px'])
    first_shape = [int(np.where(el.group_of == g)[0][0]) for g in range(el.n_groups)]
    worst = 0.0
    for fi in (0, len(el.frames) // 2, len(el.frames) - 1):
        f = int(el.frames[fi])
        cm = crop_matrix(offset=c['offsets'][f], **kw)
        for g in range(el.n_groups):
            m = el.matrices[fi, first_shape[g]]
            target = proj_from_matrix(m @ cm)                   # float64, as built
            got = apply_proj(torch.from_numpy(target)[None],
                             torch.from_numpy(el.probe_local[g].astype(np.float64))[None]
                             ).numpy()[0] * el.out_px
            want = local_to_crop(el.probe_local[g].astype(np.float64), m,
                                 offset=c['offsets'][f], **kw) * el.out_px
            worst = max(worst, float(np.abs(got - want).max()))
    assert worst <= 1e-9, f'{name}: probe target is off by {worst:.3e} crop px'


def test_the_float32_transform_target_costs_less_than_a_thousandth_of_a_pixel():
    """What the algebra above is exact to, and what the stored target actually carries.

    ``proj_crop`` is written float32, so the number the network is trained against is not the
    float64 target the test above pins. That is a real quantisation and it belongs on the
    ledger as a measured value rather than as an assumption -- it is four orders of magnitude
    under the model's own error, which is why float32 is the right call and why saying so
    requires having measured it.
    """
    el = element(PERSPECTIVE)
    c = el.crop
    kw = dict(width=c['width'], height=c['height'], scale=c['scale'], out_px=c['out_px'])
    first_shape = [int(np.where(el.group_of == g)[0][0]) for g in range(el.n_groups)]
    worst = 0.0
    for fi in range(0, len(el.frames), 17):
        f = int(el.frames[fi])
        cm = crop_matrix(offset=c['offsets'][f], **kw)
        for g in range(el.n_groups):
            exact = proj_from_matrix(el.matrices[fi, first_shape[g]] @ cm)
            probe = torch.from_numpy(el.probe_local[g].astype(np.float64))[None]
            a = apply_proj(torch.from_numpy(exact)[None], probe).numpy()
            b = apply_proj(torch.from_numpy(el.proj_crop[fi, g].astype(np.float64))[None],
                           probe).numpy()
            worst = max(worst, float(np.abs(a - b).max()) * el.out_px)
    assert worst < 1e-3, f'float32 target costs {worst:.2e} crop px'


# ---- the transform substitution, and predicted motion ------------------------

def test_no_shape_has_two_transformed_ancestors():
    """``with_transforms`` *replaces* the carrier's track with the group matrix.

    With two transformed ancestors both would be replaced and the renderer would compose the
    group matrix with itself, drawing a plausible picture in the wrong place -- silent, and
    invisible to every aggregate. Checked on every layer because it is a property of the
    archive, not of the code, and the next drop could break it.
    """
    for d in sorted(p for p in DATASET.iterdir() if (p / 'meta.json').exists()):
        doc = from_json_ir(json.loads((d / 'target_ir.json').read_text()))
        counts = {sum(1 for l in ancestors if l.transform) for ancestors, _ in doc.shapes()}
        assert counts <= {0, 1}, f'{d.name}: shapes with {sorted(counts)} transformed ancestors'


def test_the_fallback_transform_carrier_is_never_shared_across_groups():
    """513 of the archive's 2,753 shapes have *no* transformed ancestor, so v1.2 writes the
    predicted matrix to their innermost ancestor instead of skipping them -- see
    ``with_transforms``. That is only safe while no such ancestor is shared with a shape in a
    different group; if one were, one group's motion would silently move another's shapes.

    This is the assumption the fix rests on, and it is a fact about the archive rather than
    about the code, which is exactly the kind that has to be re-checked per drop."""
    from roto.model.data import load_element as _load
    for d in sorted(p for p in DATASET.iterdir() if (p / 'meta.json').exists()):
        doc = from_json_ir(json.loads((d / 'target_ir.json').read_text()))
        el = _load(d, with_local=False)
        ancs = [a for a, _ in doc.shapes()]
        seen: dict[int, set[int]] = {}
        for si, a in enumerate(ancs):
            for layer in a:
                seen.setdefault(id(layer), set()).add(int(el.group_of[si]))
        for si, a in enumerate(ancs):
            if any(l.transform for l in a):
                continue
            assert len(seen[id(a[-1])]) == 1, \
                f'{d.name}: shape {si} would write its group matrix onto a layer shared ' \
                f'with groups {sorted(seen[id(a[-1])])}'


def test_predicted_motion_reproduces_the_teacher_forced_run_when_the_track_is_the_artists():
    """Hand the de-teacher-forcing path the artist's own transform track and it must return
    the teacher-forced reconstruction exactly.

    This is the check that makes the end-to-end number trustworthy. ``motion='predicted'``
    inverts the predicted matrix on the way to local coordinates and re-applies it in the
    render; any asymmetry between those two halves -- a dropped perspective column, a
    per-frame crop map applied at the wrong frame -- would show up as model error and would be
    quoted as the cost of de-teacher-forcing.
    """
    el = element(SMALL)
    c = el.crop
    kw = dict(width=c['width'], height=c['height'], scale=c['scale'], out_px=c['out_px'])
    # The artist's own track, in the head's representation, at full precision.
    first_shape = [int(np.where(el.group_of == g)[0][0]) for g in range(el.n_groups)]
    artist = np.stack([
        np.stack([proj_from_matrix(el.matrices[fi, first_shape[g]]
                                   @ crop_matrix(offset=c['offsets'][int(f)], **kw))
                  for g in range(el.n_groups)])
        for fi, f in enumerate(el.frames)])

    # The local track the keys are chosen on must come back unchanged...
    mats = predicted_shape_matrices(el, artist)
    assert np.abs(mats - el.matrices).max() < 1e-9
    pts64 = el.points.astype(np.float64)       # float64: the claim is about the algebra
    assert np.abs(to_local(el, pts64, mats) - to_local(el, pts64)).max() < 1e-9

    # ...and so must the rendered number, which is what actually gets quoted.
    frame = [int(el.frames[len(el.frames) // 2])]
    forced = assemble(el, el.points, artist, RebuildConfig(), frame)
    dropped = assemble(el, el.points, artist, RebuildConfig(motion=PREDICTED), frame)
    assert float(dropped.soft_iou[0]) == pytest.approx(float(forced.soft_iou[0]), abs=1e-9)
    assert dropped.keys_predicted == forced.keys_predicted


def test_the_two_transform_measurements_cannot_be_asked_for_at_once():
    """They answer different questions -- the head alone, and the whole system without a
    teacher -- and averaging them by accident would be the worst of both."""
    el = element(SMALL)
    with pytest.raises(ValueError, match='two different'):
        assemble(el, el.points, el.proj_crop,
                 RebuildConfig(motion=PREDICTED, predicted_affine=True), [int(el.frames[0])])
    with pytest.raises(ValueError, match='unknown motion'):
        assemble(el, el.points, el.proj_crop, RebuildConfig(motion='camera'),
                 [int(el.frames[0])])


# ---- worst-case metrics (plan S3) --------------------------------------------

def test_point_error_p95_is_the_tail_and_is_in_crop_pixels():
    """A bimodal error: 5 px on a tenth of the points, none anywhere else. The mean reads
    0.5 px and the p95 reads 5 -- which is the whole reason the handover asks for both."""
    el = element(SMALL)
    live = el.live[..., None, None] & el.point_mask[None]
    idx = np.argwhere(live)
    hurt = idx[::10]                                             # a tenth of live points
    pred = el.points.copy()
    pred[hurt[:, 0], hurt[:, 1], hurt[:, 2], hurt[:, 3], 0] += 5.0 / el.out_px
    rec = assemble(el, pred, el.proj_crop, RebuildConfig(), [int(el.frames[0])])
    assert rec.point_err_px == pytest.approx(0.5, abs=0.05)
    assert rec.point_err_p95_px == pytest.approx(5.0, abs=1e-3)


def test_run_totals_reports_the_worst_layer_and_frame_rather_than_averaging_them():
    rows = [{'layer_id': 'easy', 'frames': 300, 'mean_soft_iou': 0.99, 'min_soft_iou': 0.97,
             'mean_iou': 0.99, 'point_err_px': 0.4, 'p95_point_err_px': 1.0,
             'jitter_px': 0.2, 'keys_predicted': 90, 'keys_artist': 100, 'key_f1': 0.4,
             'worst_frame': 12, 'frames_below_0.95': 0, 'frames_below_0.90': 0},
            {'layer_id': 'hard', 'frames': 100, 'mean_soft_iou': 0.90, 'min_soft_iou': 0.51,
             'mean_iou': 0.89, 'point_err_px': 2.0, 'p95_point_err_px': 9.0,
             'jitter_px': 0.6, 'keys_predicted': 50, 'keys_artist': 100, 'key_f1': 0.3,
             'worst_frame': 77, 'frames_below_0.95': 40, 'frames_below_0.90': 25}]
    t = run_totals(rows)
    assert t['mean_soft_iou'] == pytest.approx(0.9675)           # frame-weighted, not 0.945
    assert (t['worst_layer'], t['worst_layer_soft_iou']) == ('hard', 0.90)
    assert (t['worst_frame_layer'], t['worst_frame'], t['worst_frame_soft_iou']) \
        == ('hard', 77, 0.51)
    assert t['p95_point_err_worst_layer_px'] == 9.0
    assert t['frames_below_0.95'] == 40 and t['frames_below_0.95_pct'] == pytest.approx(10.0)
    # Key numbers are weighted by the artist's keys, not by frames: 100 each here.
    assert t['key_f1'] == pytest.approx(0.35)
    assert t['key_ratio'] == pytest.approx(0.7)


def test_run_totals_refuses_to_summarise_nothing():
    with pytest.raises(ValueError, match='no layers'):
        run_totals([])


def test_spread_reports_a_range_and_withholds_sd_from_two_runs():
    """Two seeds give a range and no standard deviation. Quoting an sd from n=2 would dress
    one difference up as a distribution."""
    s = spread([{'mean_soft_iou': 0.9234}, {'mean_soft_iou': 0.9210}], ['mean_soft_iou'])
    assert s['mean_soft_iou']['range'] == pytest.approx(0.0024)
    assert s['mean_soft_iou']['sd'] is None
    s3 = spread([{'x': 1.0}, {'x': 2.0}, {'x': 3.0}], ['x'])
    assert s3['x']['sd'] == pytest.approx(1.0)
    assert spread([{'x': 1.0}], ['y']) == {'n_runs': 1}


# ---- the transform head's own decoder depth ---------------------------------

def test_affine_depth_is_independent_and_v1_arch_still_builds():
    """``affine_depth`` deepens only the transform decoder, and a checkpoint written before
    the option existed must still load into the network it was trained as."""
    v1_arch = dict(max_shapes=8, max_groups=2, max_points=6, max_coords=1, dim=64, depth=3)
    assert len(RotoNet(**v1_arch).gblocks) == 3                  # follows `depth`
    deep = RotoNet(**v1_arch, affine_depth=6)
    assert len(deep.gblocks) == 6 and len(deep.blocks) == 3      # only the transform side
    assert RotoNet(**v1_arch, affine_depth=None).affine_depth == 3
    alpha = torch.zeros(2, 1, 256, 256)
    ids = torch.arange(8)[None].expand(2, -1)
    gids = torch.arange(2)[None].expand(2, -1)
    pts, aff = deep(alpha, ids, gids, torch.zeros(2, 8, 3))
    assert pts.shape == (2, 8, 6, 1, 2) and aff.shape == (2, 2, 6)


# ---- the program round trip, on the layers that were never sampled -----------

@pytest.mark.parametrize('name,px_per_norm', [
    (PERSPECTIVE, 124.0),
    ('TVC_SHOTS_sh0260_BG01_v003_roto_v02__Layer_52', 224.0),
])
def test_the_program_transform_track_round_trips_on_a_perspective_layer(name, px_per_norm):
    """``decode(encode(program))`` must reproduce the artist's transform track exactly.

    It did not until v1.2. ``ProgramTensors`` carried ``affine_from_matrix``'s 6 numbers,
    which cannot express these two layers -- and the round-trip test that was supposed to
    catch it samples ``nfl_0200 blue``, whose track *is* affine. Measured on the archive's own
    matrices, the 6-number form displaces a planar point by up to 0.086 normalised units on
    ``red_1`` and 1.564 on ``Layer_52``; at that layer's 224 px per normalised unit, 1.564 is
    350 crop pixels -- the shape rendered clean off its own crop.

    Checked on the *planar* action of the matrix rather than on its entries, because that is
    all the renderer ever uses and the 8-number form is a gauge-fixed homography: two matrices
    that differ by a scale factor draw the same picture.
    """
    from roto.program import decode, load_program
    from roto.render.raster import apply_transform
    d = DATASET / name
    if not (d / 'meta.json').exists():
        pytest.skip('dataset not built')
    spec, tensors, _ = load_program(d)
    doc = decode(d, spec, tensors)
    el = load_element(d, with_local=False)
    probe = np.array([[0.3, -0.2], [-0.4, 0.35], [0.0, 0.0]])

    worst = 0.0
    for si, (ancestors, _) in enumerate(doc.shapes()):
        carrier = [l for l in ancestors if l.transform]
        if not carrier:
            continue
        for fi, f in enumerate(el.frames[::11]):
            want = el.matrices[fi * 11, si]
            got = np.asarray(sample(carrier[0].transform, int(f)), float)
            worst = max(worst, float(np.abs(apply_transform(got, probe)
                                            - apply_transform(want, probe)).max()))
    assert worst * px_per_norm < 1e-2, \
        f'{name}: transform round trip moves points {worst * px_per_norm:.3e} crop px'
