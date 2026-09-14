"""Charter S4 stages **S2 and S3**: structure predicted rather than given.

S2 moves lifespans and point counts from the input to the output; S3 replaces the per
``(element, shape)`` query table with one bank of slots shared across every element. They share
a file because they share machinery -- the lifespan head *is* S3's existence signal, which is
why S3 needs no separate no-object class.

Two families. The pure ones -- hysteresis, the balanced weights, the descriptor slice -- run
anywhere. The must-come-out-perfect ones need a built ``datasets/v003`` and skip without it.

**The load-bearing tests here are the two identity checks.** S2 adds a second way for the
same document to be built: the artist's lifespans arrive as an opacity track already on the
IR, and a predicted lifespan is written onto it as a new one. If those two paths are not
*exactly* equivalent when handed the same answer, then every S2 number is measuring the
rewriting rather than the head -- and it would look like model error, which is the failure v1.1
found three times and the reason ``assemble`` can be driven by arrays at all.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from roto.v2.net import RotoNetV2
from roto.v2.reconstruct import (ARTIST, PREDICTED, RebuildConfig, alive_mask, assemble,
                                 lifespan_stats, rebuild)
from roto.v2.traindata import load_element, query_desc
from roto.v2.train import (TrainConfig, count_term, live_rate, live_rates, positive_weights)

DATASET = Path('datasets/v003')
needs_built = pytest.mark.skipif(not (DATASET / 'manifest.json').exists(),
                                 reason='datasets/v003 not built')
SMALL = 'ts_020355__Green'
"""5 shapes, 63 frames, live 0.733 -- small enough to render in a test and one of the six
elements where lifespans actually carry cost."""


def element(name: str = SMALL):
    return load_element(DATASET / name)


# ---- the hysteresis ----------------------------------------------------------

def test_hysteresis_holds_a_shape_on_through_a_dip():
    """The whole reason for two thresholds. One frame of doubt mid-span is not a lifespan
    boundary, and treating it as one costs twice: a false negative at 3.2x price, and two
    spurious transitions in the economy column."""
    prob = np.array([[0.9], [0.9], [0.3], [0.9], [0.9]])
    assert alive_mask(prob, on=0.5, off=0.2)[:, 0].tolist() == [True] * 5
    # ...and a plain threshold, which is what on == off means, does blink.
    assert alive_mask(prob, on=0.5, off=0.5)[:, 0].tolist() == [1, 1, 0, 1, 1]


def test_hysteresis_needs_a_real_drop_to_switch_off_but_a_clear_rise_to_switch_on():
    prob = np.array([[0.1], [0.6], [0.3], [0.1], [0.3]])
    #                 dead   on      hold   off    stays dead (0.3 < on)
    assert alive_mask(prob, on=0.5, off=0.2)[:, 0].tolist() == [0, 1, 1, 0, 0]


def test_off_above_on_is_refused_rather_than_silently_inverted():
    """Getting these the wrong way round makes a shape *harder* to switch on than off, which
    is the opposite of what the measured 3.2x asymmetry asks for -- and it would still run."""
    with pytest.raises(ValueError, match='must be >='):
        alive_mask(np.zeros((3, 1)), on=0.2, off=0.5)


def test_min_gap_fills_interior_dropouts_and_leaves_the_ends_alone():
    """A dead run touching either end is the shape not having started or having finished.
    Filling those would extend every shape to the whole track."""
    prob = np.array([[0.1], [0.9], [0.1], [0.9], [0.1]])
    assert alive_mask(prob, 0.5, 0.5, min_gap=2)[:, 0].tolist() == [0, 1, 1, 1, 0]


def test_there_is_no_min_run_to_match_min_gap():
    """Deliberate, not missing. Closing a gap is a cheap false positive; deleting a short live
    run is the expensive false negative, so the knob is offered on one side only."""
    assert 'min_run' not in RebuildConfig.__dataclass_fields__


# ---- the balanced weights ----------------------------------------------------

def test_an_always_live_element_is_pinned_rather_than_deleted_from_the_term():
    """``(1-r)/r`` is 0 at r = 1, which would zero the positive class -- and four of v003's
    seventeen trained elements are always live, so this is a real row, not a hypothetical."""
    w = positive_weights({'always': 1.0, 'never': 0.0, 'half': 0.5, 'sparse': 0.2})
    assert w['always'] == 1.0 and w['never'] == 1.0
    assert w['half'] == pytest.approx(1.0)
    assert w['sparse'] == pytest.approx(4.0)


# ---- the point-count term ----------------------------------------------------

def test_the_style_statistic_is_scale_free_so_one_point_costs_more_on_a_small_shape():
    """Deviation 4.3. Plan section 5's unweighted ``|P̂ - P|`` prices one point the same at
    P = 68 and at P = 4; the render does not, and dividing by P is what encodes that."""
    C = 80
    logits = torch.full((1, 2, C), -20.0)
    logits[0, 0, 5] = 20.0                       # a 4-point shape predicted at 5: out by 1
    logits[0, 1, 69] = 20.0                      # a 68-point shape predicted at 69: out by 1
    _, style = count_term(logits, torch.tensor([4, 68]), style_weight=1.0)
    # Same absolute error, and the mean of 1/4 and 1/68 is dominated by the small shape.
    assert float(style) == pytest.approx((1 / 4 + 1 / 68) / 2, abs=2e-3)


def test_the_count_cross_entropy_reads_the_class_index_as_the_count():
    """No vocabulary travels with the checkpoint, so a decode is ``argmax`` and nothing else.
    A confident, correct head must score near zero."""
    logits = torch.full((1, 1, 30), -20.0)
    logits[0, 0, 17] = 20.0
    ce, _ = count_term(logits, torch.tensor([17]))
    assert float(ce) < 1e-6


# ---- the training wheel ------------------------------------------------------

@needs_built
def test_dropping_the_point_count_from_the_query_leaves_type_and_closure():
    """Charter S4's S2 gives shape count and identity and nothing else, so ``n_points`` goes.
    ``closed`` and ``coords_per_point`` stay: they are shape *type*, and plan section 5 asks
    for the count "over allowed counts per shape type", which presumes the type is known."""
    el = element()
    assert query_desc(el, True).shape[1] == 3
    assert query_desc(el, False).shape[1] == 2
    assert np.array_equal(query_desc(el, False), el.desc[:, 1:])


def test_the_network_records_its_descriptor_width_so_a_checkpoint_cannot_be_misfed():
    """A model trained without ``n_points`` must not be handed one at inference. The width is
    on the checkpoint for the same reason ``in_frames`` is."""
    net = RotoNetV2(4, 2, 20, 1, dim=32, depth=1, desc_dim=2, alive_head=True)
    assert net.desc_dim == 2 and net.desc.in_features == 2
    out = net(torch.rand(2, 1, 256, 256), torch.arange(4)[None].expand(2, -1),
              torch.arange(2)[None].expand(2, -1), torch.rand(2, 4, 2))
    assert out.alive.shape == (2, 4)
    assert out.key is None and out.count is None      # heads not asked for stay None


# ---- lifespan bookkeeping ----------------------------------------------------

def test_lifespan_stats_keeps_the_two_error_kinds_apart():
    """They cost 1 : 3.2 at the render, so an accuracy that adds them is the one number that
    cannot answer the question S2's gate asks."""
    truth = np.array([[True], [True], [False], [False]])
    pred = np.array([[True], [False], [True], [False]])       # one FN, one FP
    st = lifespan_stats(pred, truth)
    assert st['lifespan_fp_rate'] == pytest.approx(0.25)
    assert st['lifespan_fn_rate'] == pytest.approx(0.25)
    assert st['lifespan_accuracy'] == pytest.approx(0.5)
    assert st['lifespan_transitions_artist'] == 1 and st['lifespan_transitions_pred'] == 3


# ---- the identity checks -----------------------------------------------------

@needs_built
def test_a_perfect_lifespan_head_reproduces_the_teacher_forced_render_exactly():
    """**The must-come-out-perfect check for S2's new decode path.**

    Hand the lifespan head's answer the artist's own mask and the rebuilt document must render
    identically to the teacher-forced one. If it does not, the difference is the opacity
    rewriting -- and every S2 number would carry it while looking like model error.
    """
    el = element()
    want = [int(f) for f in el.frames[::7]]
    forced = assemble(el, el.points, el.proj_crop, RebuildConfig(), want)
    # A probability that is exactly the artist's mask, so the hysteresis has nothing to do.
    perfect = el.live.astype(np.float32)
    predicted = assemble(el, el.points, el.proj_crop,
                         RebuildConfig(lifespan=PREDICTED), want, alive_prob=perfect)
    assert np.allclose(predicted.soft_iou, forced.soft_iou, atol=1e-9)
    assert predicted.keys_predicted == forced.keys_predicted
    assert predicted.key_f1 == pytest.approx(forced.key_f1, abs=1e-12)
    assert predicted.lifespan_accuracy == 1.0
    assert predicted.lifespan_fp_rate == 0.0 and predicted.lifespan_fn_rate == 0.0


@needs_built
def test_a_perfect_point_count_head_reproduces_the_teacher_forced_render_exactly():
    """The same check for the other half. With the artist's own counts, the decode must be a
    no-op -- otherwise the count path is charging for itself before the head is even wrong."""
    el = element()
    want = [int(f) for f in el.frames[::7]]
    forced = assemble(el, el.points, el.proj_crop, RebuildConfig(), want)
    counts = el.n_points_per_shape.astype(np.int32)
    predicted = assemble(el, el.points, el.proj_crop,
                         RebuildConfig(point_count=PREDICTED), want, point_count=counts)
    assert np.allclose(predicted.soft_iou, forced.soft_iou, atol=1e-9)
    assert predicted.point_count_exact == 1.0
    assert predicted.point_count_over == 0 and predicted.point_count_under == 0


@needs_built
def test_the_two_lifespan_paths_agree_on_every_frame_not_only_a_sample():
    """The strongest form of the identity check: the same comparison as above, run over the
    whole track rather than every seventh frame, and asserted **frame by frame** rather than
    on the mean.

    Not "renders at 1.0" -- it does not, and should not. ``assemble`` reduces the dense track
    to sparse keys and smooths it first, so even the artist's own control points come back as
    a keyframe approximation of themselves (0.988 at worst here). The ledger's exact
    ``1.000000`` is the raw IR with no keyframe stage in the way. What has to be exact is that
    routing an identical answer through the new path changes *nothing*.
    """
    el = element()
    forced = assemble(el, el.points, el.proj_crop, RebuildConfig())
    predicted = assemble(el, el.points, el.proj_crop, RebuildConfig(lifespan=PREDICTED),
                         alive_prob=el.live.astype(np.float32))
    assert np.abs(predicted.soft_iou - forced.soft_iou).max() < 1e-12
    assert np.abs(predicted.iou - forced.iou).max() < 1e-12


@needs_built
def test_a_wrong_lifespan_actually_changes_the_picture():
    """The complement of the identity checks, and the reason they are not vacuous: if the
    opacity track were being ignored, every test above would pass and mean nothing. Always-alive
    is the trivial answer the design note prices at 0.0971 below the truth."""
    el = element()
    want = [int(f) for f in el.frames]
    forced = assemble(el, el.points, el.proj_crop, RebuildConfig(), want)
    always = assemble(el, el.points, el.proj_crop, RebuildConfig(lifespan=PREDICTED), want,
                      alive_prob=np.ones_like(el.live, np.float32))
    assert always.soft_iou.mean() < forced.soft_iou.mean() - 0.005
    assert always.lifespan_fn_rate == 0.0            # never omits
    assert always.lifespan_fp_rate > 0.0             # only intrudes


# ---- refusals ----------------------------------------------------------------

@needs_built
def test_asking_for_a_predicted_lifespan_without_a_head_is_an_error_not_a_default():
    """Falling back to the artist's track would report a teacher-forced number as a
    de-teacher-forced one -- and a uniform lifespan is "always alive", a trivial baseline that
    would read as a model result."""
    el = element()
    with pytest.raises(ValueError, match='no lifespan head'):
        rebuild(el, el.local, RebuildConfig(lifespan=PREDICTED))
    with pytest.raises(ValueError, match='no'):
        rebuild(el, el.local, RebuildConfig(point_count=PREDICTED))
    with pytest.raises(ValueError, match='unknown lifespan'):
        rebuild(el, el.local, RebuildConfig(lifespan='sometimes'))


# ---- the measured rates the run records --------------------------------------

@needs_built
def test_the_live_rate_is_measured_over_every_cell_not_only_the_live_ones():
    """A lifespan loss masked by the lifespan would only ever see frames where the answer is
    already yes. The denominator being every cell is what makes the head's task the real one."""
    el = element()
    assert live_rate([el]) == pytest.approx(float(el.live.mean()))
    assert live_rates([el])[el.element_id] == pytest.approx(float(el.live.mean()))


def test_s2_config_defaults_leave_every_new_head_off():
    """S0 and S1 must reproduce bit for bit after S2 lands. A default that switched a head on
    would silently re-baseline every earlier number."""
    cfg = TrainConfig()
    assert cfg.alive_weight == 0.0 and cfg.count_weight == 0.0
    assert cfg.point_weight == 1.0 and cfg.curve_weight == 0.5
    assert cfg.give_point_count is True
    assert RebuildConfig().lifespan == ARTIST and RebuildConfig().point_count == ARTIST


# ---- S3: shared slot queries (charter S4 stage S3) ---------------------------

def test_slot_queries_share_one_bank_across_every_element():
    """The substitution S3 *is*. With a table, a row means "shape 7 of this layer" and the
    bank grows with the dataset; with slots, one bank serves every element and nothing in it
    can name a layer."""
    from roto.v2.net import SLOTS, TABLE
    slots = RotoNetV2(256, 4, 20, 1, dim=64, depth=1, query_mode=SLOTS, alive_head=True)
    assert slots.query_mode == SLOTS and slots.token_pool is not None
    table = RotoNetV2(256, 4, 20, 1, dim=64, depth=1, query_mode=TABLE)
    assert table.token_pool is None


def test_slot_queries_ignore_the_shape_descriptor():
    """``closed`` and ``coords_per_point`` are per-(element, shape) facts. Feeding them to a
    shared slot would hand back the identity the stage exists to remove, so the desc term is
    dropped entirely -- and two different descs must therefore give the same answer."""
    from roto.v2.net import SLOTS
    torch.manual_seed(0)
    net = RotoNetV2(8, 2, 20, 1, dim=32, depth=1, query_mode=SLOTS).eval()
    a = torch.rand(1, 1, 256, 256)
    ids, gids = torch.arange(8)[None], torch.arange(2)[None]
    with torch.no_grad():
        x = net(a, ids, gids, torch.zeros(1, 8, 3))
        y = net(a, ids, gids, torch.ones(1, 8, 3))
    assert torch.allclose(x.points, y.points)


def test_an_unknown_query_mode_is_refused():
    with pytest.raises(ValueError, match='unknown query_mode'):
        RotoNetV2(8, 2, 20, 1, dim=32, depth=1, query_mode='dynamic')


@needs_built
def test_canonical_order_is_a_permutation_and_is_derivable_from_the_picture():
    """Slots need an ordering the model could work out from a cutout. Document order is not
    one -- "the fifth shape in her file" is a fact about how the file was built. Group then
    centroid is: the groups move differently and a centroid is a position in the picture."""
    from roto.v2.traindata import canonical_order
    el = element()
    perm = canonical_order(el)
    assert sorted(perm.tolist()) == list(range(el.n_shapes))
    # Deterministic: the same element must always land in the same slots, or a checkpoint
    # means something different on every load.
    assert np.array_equal(perm, canonical_order(el))
    # Ordered by group first, so a group's shapes stay contiguous.
    groups = el.group_of[perm]
    assert np.array_equal(groups, np.sort(groups))


@needs_built
def test_slot_padding_leaves_geometry_targets_alone_and_pads_only_existence():
    """Padding the geometry targets to 256 slots would cost 392 MB on a 94-frame element for
    rows every geometry term masks out anyway. Only the alive target is padded -- it is the
    one head with something to say about an empty slot."""
    from roto.v2.train import _batch
    from roto.v2.traindata import canonical_order
    el = element()
    idx = np.arange(3)
    b = _batch(el, idx, n_slots=256, perm=canonical_order(el))
    assert b['points'].shape[1] == el.n_shapes          # not padded
    assert b['alive_target'].shape == (3, 256)          # padded
    assert bool(b['alive_target'][:, el.n_shapes:].any()) is False
    assert b['shape_ids'].shape == (3, 256)


@needs_built
def test_the_live_rate_counts_empty_slots_because_the_head_must_learn_them():
    """With shared slots the denominator is every slot, not every shape. On v003 that moves
    the rate from 0.627 to ~0.04 and flips the head's degenerate answer from "always alive"
    back to "never" -- which is why the positive weight is re-measured rather than carried."""
    el = element()
    assert live_rate([el]) == pytest.approx(float(el.live.mean()))
    slotted = live_rate([el], n_slots=256)
    assert slotted < float(el.live.mean()) / 10
    assert slotted == pytest.approx(int(el.live.sum()) / (len(el.frames) * 256))


def test_a_slot_bank_too_small_for_the_widest_element_is_refused():
    """Silently dropping the tail of a 247-shape element would read as model error."""
    from dataclasses import replace as dc_replace
    from roto.v2.train import train
    with pytest.raises(ValueError, match='cannot hold the widest'):
        train('datasets/v003', '/tmp/nope',
              dc_replace(TrainConfig(), n_slots=4, alive_weight=1.0, steps=1))


def test_an_untrained_element_stays_out_of_the_headline_even_when_properly_scored():
    """Regression. Through S2, "withheld from training" and "scored with fake query rows"
    were the same fact, so the table split on either. S3's shared slot bank separates them: a
    held-out shot is now scored with the real model *and* was still never trained on.

    Splitting on the wrong one put four untrained elements into the headline block and read
    as a model collapse -- 0.9355 on-screen reported as 0.7998, 1.26 px reported as 16.3.
    It was caught only because the training log disagreed with the scorer.
    """
    from roto.v2.report import by_training_status
    def row(eid, iou, trained):
        return {'element_id': eid, 'frames': 10, 'mean_soft_iou': iou, 'min_soft_iou': iou,
                'mean_iou': iou, 'point_err_px': 1.0, 'p95_point_err_px': 1.0,
                'max_point_err_px': 1.0, 'jitter_px': 0.1, 'worst_frame': 0,
                'frames_below_0.95': 0, 'frames_below_0.90': 0, 'keys_predicted': 5,
                'keys_artist': 5, 'key_f1': 0.5, 'key_precision': 0.5, 'key_recall': 0.5,
                'in_train': trained, 'scored_with_real_queries': True}
    rows = [row('trained_a', 0.96, True), row('trained_b', 0.94, True),
            row('unseen', 0.21, False)]
    out = by_training_status(rows, [r['element_id'] for r in rows if not r['in_train']])
    # The headline must be the trained pair only; the unseen element has its own block.
    assert out['mean_soft_iou'] == pytest.approx(0.95)
    assert out['held_elements']['mean_soft_iou'] == pytest.approx(0.21)
    # ...and splitting on "was it scored with fake queries" would have merged them.
    wrong = by_training_status(rows, [])
    assert wrong['mean_soft_iou'] < 0.95 and wrong['held_elements'] is None
