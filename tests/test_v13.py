"""v1.3: the rebuilt dataset, the .sfx writer, the build-time split, the key-timing head.

Four of these tests exist because of a bug the round found, and in each case the bug survived
earlier rounds for the same reason: a convention or an axis lived in a *default* that two code
paths happened to agree on. So the tests here mostly assert that two paths agree on the real
archive rather than on a fixture -- a synthetic case cannot be wrong in the way a recorded one
can, which is exactly how the transform round trip survived two rounds of review in v1.2.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from roto.dataset import build_splits, frame_split, load_splits
from roto.dataset.build import load_alpha
from roto.keys import (MAX_TOL_SCALE, MIN_TOL_SCALE, local_tolerances,
                       segment_errors, segment_ratio, select)
from roto.metrics import soft_iou
from roto.model.data import load_element
from roto.model.net import RotoNet
from roto.model.reconstruct import RebuildConfig, assemble, rebuild, untrained_queries
from roto.model.report import by_training_status, run_totals
from roto.model.train import key_positive_rate, key_term
from roto.render.raster import (RenderConfig, config_from_meta, conventions,
                                measured_conventions, render_union)
from roto.sfx.json_ir import from_json_ir
from roto.sfx.read import read_sfx
from roto.sfx.write import DIALECT_CONTAINER, num, rewrite_sfx, split_dialect, write_sfx
from roto.shots import find_shot, find_shots

from common import DATA

V002 = Path('datasets/v002')
OPEN_STROKE_LAYER = 'FAM_0060_L1_A0003C007_v001__blue_1'    # 279 of 1036 shapes are open
PLAIN_SHOT = 'MAT_0130_L1_C002_260809_v001'                 # dialect 5, plain container
ZLIB_SHOT = 'nfl_0200_bg01_v001_compplate_roto_v001'        # dialect 2020, zlib container

pytestmark = pytest.mark.skipif(not V002.exists(),
                                reason='datasets/v002 not built; run roto dataset')


# ---- 1. the conventions have to travel with the data ----------------------------

def test_config_from_meta_reproduces_the_conventions_the_alphas_were_drawn_with():
    """Every scorer through v1.2 rebuilt three of four conventions from class defaults.

    It could not produce a wrong number on ``datasets/v001``, whose conventions *are* the
    defaults, and it produces one on ``v002``. This is that difference, measured on the layer
    where it is largest -- 279 open strokes -- against the artist's own program, which must
    come out perfect.
    """
    d = V002 / OPEN_STROKE_LAYER
    el = load_element(d, with_local=False)
    artist = from_json_ir(json.loads((d / 'target_ir.json').read_text()))
    c = el.crop
    f = int(el.frames[len(el.frames) // 2])
    x0, y0 = c['offsets'][f]
    box = (x0, y0, c['out_px'] / c['scale'], c['out_px'] / c['scale'])
    truth = load_alpha(d, f)

    def score(cfg):
        pred = render_union(artist, f, cfg, c['scale'], box)[:truth.shape[0], :truth.shape[1]]
        return soft_iou(pred, truth)

    recorded = score(config_from_meta(el.render))
    defaults = score(RenderConfig(supersample=el.render['supersample']))
    assert recorded > 0.99999, \
        f'the artist\'s own program must render back to its own alpha; got {recorded:.6f}'
    assert defaults < 0.999, \
        (f'the class defaults score {defaults:.6f} here, so this layer no longer '
         'demonstrates the bug and the test has stopped testing anything')


def test_config_from_meta_raises_when_the_record_disagrees_with_the_code():
    """A dataset whose meta.json contradicts a named convention set is an error, not a
    re-baseline nobody notices."""
    good = dict(conventions='measured', supersample=4, samples_per_seg=12,
                fill_open_zero_width=False, open_end_rule='duplicate', clip_per_shape=True)
    assert config_from_meta(good).fill_open_zero_width is False
    with pytest.raises(ValueError, match='disagree'):
        config_from_meta({**good, 'fill_open_zero_width': True})
    with pytest.raises(ValueError, match='unknown conventions'):
        config_from_meta({**good, 'conventions': 'v3'})
    # supersample is the one axis a caller may legitimately override.
    assert config_from_meta(good, supersample=2).supersample == 2


def test_every_v002_layer_records_the_measured_conventions():
    for d in sorted(p for p in V002.iterdir() if (p / 'meta.json').exists()):
        rec = json.loads((d / 'meta.json').read_text())['render']
        assert rec['conventions'] == 'measured', d.name
        want = measured_conventions(int(rec['supersample']))
        assert rec['fill_open_zero_width'] == want.fill_open_zero_width
        assert rec['open_end_rule'] == want.open_end_rule
        assert config_from_meta(rec).open_end_rule == 'duplicate'
    assert conventions('v1', 2).fill_open_zero_width is True, \
        "v1's conventions must stay reproducible: datasets/v001 was drawn with them"


# ---- 2. the .sfx writer ---------------------------------------------------------

def _diff(a, b, path=''):
    """First disagreement between two IR values, as a string, or ''. Exact, not approximate."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        x, y = np.asarray(a, np.float64), np.asarray(b, np.float64)
        if x.shape != y.shape:
            return f'{path}: shape {x.shape} != {y.shape}'
        return '' if not x.size or np.array_equal(x, y) else f'{path}: values differ'
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return f'{path}: {len(a)} != {len(b)} items'
        return next((m for i, (x, y) in enumerate(zip(a, b))
                     if (m := _diff(x, y, f'{path}[{i}]'))), '')
    if isinstance(a, dict):
        if sorted(a) != sorted(b):
            return f'{path}: keys differ'
        return next((m for k in a if (m := _diff(a[k], b[k], f'{path}.{k}'))), '')
    if hasattr(a, '__slots__'):
        return next((m for f in a.__slots__
                     if (m := _diff(getattr(a, f), getattr(b, f), f'{path}.{f}'))), '')
    return '' if a == b else f'{path}: {a!r} != {b!r}'


@pytest.mark.parametrize('shot_name', [ZLIB_SHOT, PLAIN_SHOT])
def test_read_write_read_is_bit_exact(shot_name, tmp_path):
    """``read(write(IR))`` returns the same IR bit for bit, in both containers.

    Bit-exact rather than exact-to-a-tolerance is a choice in the writer -- numbers go out at
    shortest-round-trip precision instead of Silhouette's fixed 9 decimals -- and it is the
    choice that makes this row of the ledger a proof rather than a measurement.
    """
    shot = find_shot(Path(DATA) / shot_name)
    doc = read_sfx(shot.sfx)
    back = read_sfx(write_sfx(doc, tmp_path / 'out.sfx'))
    for f in ('width', 'height', 'duration', 'frame_rate', 'start_frame', 'dialect',
              'source_path', 'source_label'):
        assert getattr(doc, f) == getattr(back, f), f
    assert not _diff(doc.roots, back.roots, shot_name)


def test_the_container_is_the_one_the_dialect_uses():
    assert split_dialect('2020/zlib') == ('2020', 'zlib')
    assert split_dialect('5') == ('5', 'plain')          # bare version picks its own
    assert split_dialect(None)[1] == 'zlib'
    for version, container in DIALECT_CONTAINER.items():
        assert split_dialect(version) == (version, container)
    with pytest.raises(ValueError, match='unknown container'):
        split_dialect('2020/gzip')
    # A float must survive the text: shortest-round-trip, not fixed decimals.
    for x in (0.1, 1 / 3, -1e-7, 65535.0, 1.0000000000000002):
        assert float(num(x)) == x, x


def test_rewrite_keeps_every_untouched_layer_and_renders_it_identically(tmp_path):
    """The form to open in a seat: the artist's own project, one layer substituted.

    If this file renders differently from the artist's, the difference is our shapes --
    which is only true if everything else is genuinely untouched. So the *other* root has to
    come back rendering pixel-identical, and the substituted one has to actually change.
    """
    shot = find_shot(Path(DATA) / ZLIB_SHOT)
    doc = read_sfx(shot.sfx)
    names = [r.name for r in doc.roots]
    assert len(names) >= 2, 'this test needs a shot with an untouched second root'
    target, untouched = names[0], names[1]

    # A visibly different version of the target layer: every point pushed 0.02 normalised.
    moved = read_sfx(shot.sfx).layer(target)
    for _, shape in moved.shapes():
        for k in shape.path:
            k.value = np.asarray(k.value, np.float64) + 0.02
    out = rewrite_sfx(shot.sfx, moved, tmp_path / 'rw.sfx', layers=[target])
    back = read_sfx(out)

    assert [r.name for r in back.roots] == names, 'a root went missing or moved'
    assert next(r for r in doc.roots if r.name == target).uuid == \
        next(r for r in back.roots if r.name == target).uuid, 'the layer uuid must survive'

    cfg = RenderConfig(supersample=2)
    for f in (0, doc.duration // 2):
        same = render_union(doc.layer(untouched), f, cfg, 0.25)
        also = render_union(back.layer(untouched), f, cfg, 0.25)
        assert np.array_equal(same, also), f'untouched layer changed at frame {f}'
        before = render_union(doc.layer(target), f, cfg, 0.25)
        after = render_union(back.layer(target), f, cfg, 0.25)
        assert not np.array_equal(before, after), \
            f'the substituted layer is unchanged at frame {f} -- the rewrite did nothing'

    with pytest.raises(KeyError):
        rewrite_sfx(shot.sfx, moved, tmp_path / 'x.sfx', layers=['no such layer'])


def test_every_archive_shot_can_be_written(tmp_path):
    """All four dialects, both containers -- because the container is the half of a format
    most likely to be got wrong silently: a zlib stream that decompresses to *almost* the
    right bytes still parses."""
    seen = set()
    for shot in find_shots(DATA):
        if shot.sfx is None:
            continue
        doc = read_sfx(shot.sfx)
        seen.add(doc.dialect)
        read_sfx(write_sfx(doc, tmp_path / f'{shot.name}.sfx'))
    assert len(seen) >= 4, f'expected all four dialects, saw {sorted(seen)}'


# ---- 3. the split is the dataset's ----------------------------------------------

def test_frame_split_reproduces_the_rule_v12_used():
    """The recorded split has to be the same rule, or a v002 held-frame gap is not comparable
    to v1.1's +0.0005."""
    for n in (10, 40, 77, 133, 191):
        tr, he = frame_split(n, 7)
        assert set(tr) | set(he) == set(range(n))
        assert not set(tr) & set(he)
        assert 0 not in he and n - 1 not in he, \
            'a held frame must have trained neighbours on both sides'
        assert list(he) == [i for i in range(1, n - 1) if i % 7 == 3]
    assert len(frame_split(40, 0)[1]) == 0 and len(frame_split(40, 1)[1]) == 0


def test_recorded_split_matches_the_rule_and_names_its_held_layers():
    rec = load_splits(V002)
    assert rec is not None and rec['holdout_every'] == 7
    assert len(rec['held_layers']) == 2 and rec['note'], 'held layers need a recorded reason'
    assert set(rec['held_layers']).isdisjoint(rec['trained_layers'])
    for d in sorted(p for p in V002.iterdir() if (p / 'meta.json').exists()):
        meta = json.loads((d / 'meta.json').read_text())
        frames = meta['frames']['index']
        got = rec['layers'][meta['layer']['layer_id']]
        tr, he = frame_split(len(frames), 7)
        assert got['frames_train'] == [frames[i] for i in tr]
        assert got['frames_held'] == [frames[i] for i in he]
    # And the record is what build_splits would write again from the same inputs.
    again = build_splits([(k, v['frames_train'] + v['frames_held'])
                          for k, v in rec['layers'].items()], 7, rec['held_layers'])
    assert sorted(again['trained_layers']) == sorted(rec['trained_layers'])


def test_layer_data_prefers_the_recorded_split_over_the_rule():
    """A dataset that records its own split wins; one that does not falls back, so every
    v1/v1.1/v1.2 number still reproduces."""
    d = V002 / OPEN_STROKE_LAYER
    el = load_element(d, with_local=False)
    rec = load_splits(V002)['layers'][el.layer_id]
    tr, he = el.split(0)                       # holdout_every=0 must NOT hold nothing back
    assert len(he) == len(rec['frames_held']) > 0, \
        'the recorded split must override whatever integer a run passes'
    assert [int(el.frames[i]) for i in he] == rec['frames_held']
    assert el.in_train is True

    v001 = Path('datasets/v001') / OPEN_STROKE_LAYER
    if v001.exists():
        old = load_element(v001, with_local=False)
        assert len(old.split_train) == 0, 'v001 must have no record'
        assert len(old.split(0)[1]) == 0 and len(old.split(7)[1]) > 0


def test_held_out_layers_are_marked_and_the_report_keeps_them_apart():
    """A layer withheld from training is scored and reported *apart*, never diluted in."""
    rec = load_splits(V002)
    for lid in rec['held_layers']:
        assert load_element(V002 / lid, with_local=False).in_train is False

    rows = [{'layer_id': 'trained', 'frames': 100, 'mean_soft_iou': 0.97, 'min_soft_iou': 0.9,
             'p05_soft_iou': 0.93, 'frames_below_0.95': 1, 'frames_below_0.90': 0,
             'mean_iou': 0.95, 'point_err_px': 1.0, 'p95_point_err_px': 2.0,
             'max_point_err_px': 9.0, 'jitter_px': 0.3, 'keys_predicted': 80,
             'keys_artist': 100, 'key_precision': 0.4, 'key_recall': 0.4, 'key_f1': 0.4,
             'worst_frame': 5},
            {**{'layer_id': 'frozen', 'frames': 100, 'mean_soft_iou': 0.30,
                'min_soft_iou': 0.05, 'p05_soft_iou': 0.1, 'frames_below_0.95': 100,
                'frames_below_0.90': 100, 'mean_iou': 0.2, 'point_err_px': 40.0,
                'p95_point_err_px': 90.0, 'max_point_err_px': 200.0, 'jitter_px': 5.0,
                'keys_predicted': 300, 'keys_artist': 100, 'key_precision': 0.1,
                'key_recall': 0.1, 'key_f1': 0.1, 'worst_frame': 7}}]
    tot = by_training_status(rows, ['frozen'])
    assert tot['mean_soft_iou'] == pytest.approx(0.97), \
        'the headline must be the layers the run trained on'
    assert tot['held_layers']['mean_soft_iou'] == pytest.approx(0.30)
    assert tot['all_layers']['mean_soft_iou'] == pytest.approx(0.635)
    assert tot['withheld_layers'] == ['frozen']
    # With nothing withheld the blocks collapse, so a pre-split run reads unchanged.
    assert by_training_status(rows, [])['held_layers'] is None


def test_worst_layer_percentage_is_per_layer_not_per_run():
    """"frames below 0.90 <= 1% per layer" is a per-layer statement: 1% of 1,810 frames is 18,
    which one 77-frame layer could supply entirely while the run-level figure still passes."""
    base = {'min_soft_iou': 0.5, 'p05_soft_iou': 0.6, 'frames_below_0.95': 0,
            'mean_iou': 0.9, 'point_err_px': 1.0, 'p95_point_err_px': 2.0,
            'max_point_err_px': 5.0, 'jitter_px': 0.2, 'keys_predicted': 10,
            'keys_artist': 10, 'key_precision': 0.4, 'key_recall': 0.4, 'key_f1': 0.4,
            'worst_frame': 1}
    rows = [{**base, 'layer_id': 'big', 'frames': 1733, 'mean_soft_iou': 0.98,
             'frames_below_0.90': 0},
            {**base, 'layer_id': 'small', 'frames': 77, 'mean_soft_iou': 0.91,
             'frames_below_0.90': 15}]
    tot = run_totals(rows)
    assert tot['frames_below_0.90_pct'] < 1.0, 'the run-level figure passes'
    assert tot['worst_layer_frames_below_0.90_pct'] > 19.0, 'the per-layer figure must not'
    assert tot['worst_layer_frames_below_0.90'] == 'small'


# ---- 4. the key-timing head ----------------------------------------------------

def test_key_target_is_the_artists_own_keys_on_the_rendered_axis():
    """Keys outside the rendered frame range are dropped, not clamped. Clamping would invent
    a key on frame 0 -- the frame artists key most often -- and inflate recall for free."""
    dropped = 0
    for d in sorted(p for p in V002.iterdir() if (p / 'meta.json').exists()):
        el = load_element(d, with_local=False)
        t = np.load(d / 'tensors.npz')
        frames = {int(f) for f in el.frames}
        for i in range(el.n_shapes):
            kf = [int(k) for k in t[f'key_frames/{i}'].tolist()]
            inside = sorted(k for k in kf if k in frames)
            dropped += len(kf) - len(inside)
            got = sorted(int(el.frames[j]) for j in np.where(el.key_mask[:, i])[0])
            assert got == inside, f'{d.name} shape {i}'
    assert dropped > 0, \
        'no key falls outside the rendered range, so this test is no longer checking the rule'


def test_key_positive_rate_is_measured_over_live_cells():
    els = [load_element(p, with_local=False)
           for p in sorted(V002.iterdir()) if (p / 'meta.json').exists()]
    rate = key_positive_rate(els)
    assert 0.05 < rate < 0.20, rate
    # The denominator is live cells only, matching how the term is masked and how keys are
    # chosen. Using all cells instead would understate the rate and over-weight the positives.
    all_cells = sum(e.live.size for e in els)
    assert sum(int(e.live.sum()) for e in els) < all_cells


def test_the_key_head_varies_with_the_frame():
    """The load-bearing wiring check. The head reads the shape query *after* cross-attention,
    which is the only thing in the forward pass that depends on the frame -- a head on the raw
    embedding could only ever predict a constant, and would still train, and would still
    report a falling loss."""
    torch.manual_seed(0)
    net = RotoNet(8, 2, 6, 1, dim=32, depth=1, key_head=True).eval()
    ids, gids, desc = torch.arange(3)[None], torch.arange(2)[None], torch.zeros(1, 3, 3)
    a = net(torch.rand(1, 1, 256, 256), ids, gids, desc)[2]
    b = net(torch.rand(1, 1, 256, 256), ids, gids, desc)[2]
    assert a.shape == (1, 3), a.shape
    assert not torch.allclose(a, b, atol=1e-6), \
        'the key logit did not move when the picture did -- it is not reading the image'
    # And it is off by default, so every checkpoint before v1.3 still loads and scores.
    assert RotoNet(8, 2, 6, 1, dim=32, depth=1).key_head is False
    assert net(torch.rand(1, 1, 256, 256), ids, gids, desc)[2] is not None
    plain = RotoNet(8, 2, 6, 1, dim=32, depth=1).eval()
    assert plain(torch.rand(1, 1, 256, 256), ids, gids, desc)[2] is None


def test_key_term_is_class_balanced_so_never_firing_is_not_the_minimum():
    """At a 9.6% positive rate an unweighted BCE is minimised by a head that never fires --
    which scores 90.4% accuracy and F1 zero. This is what the positive weight is for."""
    torch.manual_seed(0)
    target = torch.zeros(1, 100, dtype=torch.bool)
    target[0, ::10] = True                                 # 10% positive
    live = torch.ones(1, 100, dtype=torch.bool)
    never = torch.full((1, 100), -8.0)
    perfect = torch.where(target, 8.0, -8.0)
    w = (1 - 0.1) / 0.1
    assert key_term(perfect, target, live, w) < key_term(never, target, live, w)
    # Unweighted, "never fire" is already cheap; weighted, it is nine times as expensive.
    assert (key_term(never, target, live, w)
            > 5 * key_term(never, target, live, 1.0)), 'the positive weight is not biting'
    # Dead cells contribute nothing at all.
    assert float(key_term(never, target, torch.zeros_like(live), w)) == 0.0


# ---- 5. how the head reaches the output ----------------------------------------

def test_local_tolerance_reduces_to_the_uniform_case_exactly():
    """``bias=0`` must stay on v1's own code path, bit for bit: every v1, v1.1 and v1.2 number
    was measured through it and a float division is not worth re-baselining a ladder over."""
    rng = np.random.default_rng(0)
    track = np.cumsum(rng.normal(size=(30, 4, 2)), axis=0)
    frames = np.arange(30, dtype=np.int32)
    p = rng.random(30)
    a = select(track, frames, 1.5)
    b = select(track, frames, 1.5, key_prob=p, bias=0.0)
    assert np.array_equal(a.frames, b.frames) and a.max_error == b.max_error
    assert (a.tol_min, a.tol_max) == (1.5, 1.5)
    assert local_tolerances(1.5, p, 0.0) is None and local_tolerances(1.5, None, 0.5) is None
    # And the ratio cost is exactly error/tol for a uniform tolerance.
    inv = np.full(30, 1 / 1.5)
    for i, j in ((0, 10), (3, 29), (5, 7)):
        assert segment_ratio(track, i, j, inv) == pytest.approx(
            segment_errors(track, i, j) / 1.5, rel=1e-12)


def test_the_bias_moves_a_knot_onto_a_predicted_key_without_adding_keys():
    """What the head is for. A gentle ramp with a small bump: at a loose tolerance the DP skips
    the bump entirely, and a head that localises it should pull a knot there -- for the same
    number of knots, which is the part that matters. Over-keying is the failure this design is
    shaped to avoid."""
    t = np.arange(41)
    track = np.zeros((41, 3, 2))
    track[:, :, 0] = (t * 0.2 + 1.5 * np.exp(-((t - 20) ** 2) / 8.0))[:, None]
    frames = t.astype(np.int32)
    p = np.zeros(41)
    p[19:22] = [0.6, 0.95, 0.6]

    plain = select(track, frames, 2.0)
    biased = select(track, frames, 2.0, key_prob=p, bias=0.5)
    assert not any(19 <= f <= 21 for f in plain.frames), \
        'the unbiased DP already keys the bump, so this test proves nothing'
    assert any(19 <= f <= 21 for f in biased.frames), 'the bias did not reach the DP'
    assert len(biased) <= len(plain) + 1, \
        f'the bias added {len(biased) - len(plain)} knots -- it is emitting keys, not biasing'
    assert biased.tol_min < biased.tol_max == 2.0


def test_slack_moves_and_removes_keys_where_the_head_expects_none():
    """The other half of the bias, and the half that can raise precision.

    Over-keying is this project's measured failure -- precision 0.21-0.44 against recall
    0.75-1.00 -- so key F1 is precision-bound, and tightening alone can only *add* keys.
    Loosening where the head is confident there is no key does two things a tightening cannot,
    and both are asserted here on the same track: it removes keys, and at a fixed key count it
    *moves* one to where the head says it belongs.

    The fixture has two identical bumps and a head that knows about the first one only.
    """
    t = np.arange(41)
    track = np.zeros((41, 3, 2))
    track[:, :, 0] = (t * 0.1 + 1.2 * np.exp(-((t - 10) ** 2) / 6.0)
                      + 1.2 * np.exp(-((t - 30) ** 2) / 6.0))[:, None]
    frames = t.astype(np.int32)
    p = np.zeros(41)
    p[9:12] = 0.95
    near = lambda sel, c: any(abs(int(f) - c) <= 2 for f in sel.frames)

    # 1. It removes keys. At a tolerance where the DP keys both bumps, slack keeps the flagged
    #    one and stops paying for the frames around the other.
    plain = select(track, frames, 0.3)
    slacked = select(track, frames, 0.3, key_prob=p, bias=0.0, slack=1.0)
    assert near(plain, 10) and near(plain, 30), 'the unbiased DP must key both bumps here'
    assert len(slacked) < len(plain), \
        f'slack removed no keys ({len(plain)} -> {len(slacked)}); it is not reaching the DP'
    assert near(slacked, 10), 'slack dropped the flagged bump, which is the wrong one'
    assert slacked.tol_min > 0.3 and slacked.tol_max > slacked.tol_min, \
        'slack must only ever raise the tolerance above the one asked for'

    # 2. It moves keys. At a looser tolerance the unbiased DP can afford one interior knot and
    #    puts it at the *unflagged* bump; slack alone moves it, for the same key count.
    plain = select(track, frames, 1.0)
    slacked = select(track, frames, 1.0, key_prob=p, bias=0.0, slack=1.0)
    assert len(slacked) == len(plain) == 3
    assert near(plain, 30) and not near(plain, 10), \
        'the unbiased DP no longer prefers the unflagged bump; the fixture has stopped biting'
    assert near(slacked, 10) and not near(slacked, 30), \
        'slack did not move the key to the frames the head flagged'

    # 3. Both knobs together push in opposite directions, as documented.
    both = select(track, frames, 0.3, key_prob=p, bias=0.5, slack=0.5)
    assert both.tol_min < 0.3 < both.tol_max
    # And neither knob set is still v1's own code path, bit for bit.
    assert np.array_equal(select(track, frames, 0.3, key_prob=p, bias=0.0, slack=0.0).frames,
                          select(track, frames, 0.3).frames)


def test_the_tolerance_floor_stops_the_head_forcing_a_key():
    """The head is deliberately not allowed to emit a key. The floor is what enforces it: at
    probability 1 and full bias the tolerance is still ``MIN_TOL_SCALE`` of what was asked."""
    tol = np.array(local_tolerances(2.0, np.array([0.0, 0.5, 1.0]), 1.0))
    assert tol[0] == 2.0
    assert tol[2] == pytest.approx(2.0 * MIN_TOL_SCALE)
    assert (tol > 0).all(), 'a zero tolerance would force a knot unconditionally'
    assert np.all(np.asarray(local_tolerances(2.0, np.array([2.0, -1.0]), 1.0)) > 0), \
        'a probability outside [0, 1] must be clipped, not trusted'
    # And the ceiling: a confident "no key here" must not be able to span a whole track.
    loose = np.asarray(local_tolerances(2.0, np.array([0.0, 1.0]), 0.0, slack=99.0))
    assert loose[0] == pytest.approx(2.0 * MAX_TOL_SCALE)
    assert loose[1] == pytest.approx(2.0)


def test_rebuild_with_no_key_head_is_unchanged():
    """A checkpoint with no key head must reconstruct exactly as it did in v1.2, including
    when a caller passes a bias -- there is no probability to bias with."""
    d = V002 / 'FAM_0060_L1_A0003C007_v001__r1_t_c'
    el = load_element(d)
    a, sa = rebuild(el, el.local, RebuildConfig(), None)
    b, sb = rebuild(el, el.local, RebuildConfig(key_bias=0.7), None)
    assert sa['keys_predicted'] == sb['keys_predicted'] and sa['key_f1'] == sb['key_f1']
    assert 'head_key_f1' not in sa
    for (_, x), (_, y) in zip(a.shapes(), b.shapes()):
        assert [k.frame for k in x.path] == [k.frame for k in y.path]


def test_the_artists_own_keys_as_a_perfect_head_score_the_head_at_one():
    """The must-come-out-perfect check for the new head: hand the reporting path the artist's
    own key mask as the "prediction" and the head's own F1 must be 1.000.

    Three of the four bugs v1.1 found came from scoring the case that has to be perfect, and
    a head's own metric is exactly the kind of thing that can be silently misaligned -- by a
    frame, by a shape index, or by the live mask.
    """
    d = V002 / 'FAM_0060_L1_A0003C007_v001__r1_t_c'
    el = load_element(d)
    perfect = el.key_mask.astype(np.float32)
    _, st = rebuild(el, el.local, RebuildConfig(key_bias=0.0, key_thresh=0.5), perfect)
    assert st['head_key_f1'] == pytest.approx(1.0, abs=1e-9), st['head_key_f1']
    assert st['head_key_precision'] == pytest.approx(1.0, abs=1e-9)
    # And an all-zero head scores zero rather than erroring or scoring vacuously well.
    _, st0 = rebuild(el, el.local, RebuildConfig(), np.zeros_like(perfect))
    assert st0['head_key_f1'] == 0.0


def test_key_metrics_carry_their_trivial_baselines():
    """The metric that caught a gate. Both key F1 figures are reported with baselines because
    neither is readable without one.

    The head's own F1 is nearly saturated by firing constantly on a densely-keyed layer, and
    the pipeline's is nearly saturated by keying often -- key density ranges 13x across these
    layers and the two densest carry 40% of the archive's keys. So a perfect head must beat
    the baselines, a trivial head must not, and a head that *is* the baseline must tie it.
    """
    d = V002 / 'FAM_0060_L1_A0003C007_v001__blue'
    el = load_element(d)
    cfg = RebuildConfig(key_thresh=0.5)

    perfect = el.key_mask.astype(np.float32)
    _, st = rebuild(el, el.local, cfg, perfect)
    # Not exactly 1.0, and the reason is a real discrepancy rather than a rounding error: the
    # head's *target* (`LayerData.key_mask`) drops artist keys that fall outside the rendered
    # axis, while the key-F1 *truth* clips keys outside a shape's live range onto its boundary.
    # 7.91% of this archive's keys are outside their shape's live range and `FAM blue` has one
    # of them, which is the whole of this 0.007. A layer with none scores exactly 1.0 --
    # `test_the_artists_own_keys_as_a_perfect_head_score_the_head_at_one` covers that on
    # `r1_t_c`. Both definitions are defensible for their own purpose and both are on the
    # record; what is not acceptable is either of them being wrong by more than this.
    assert st['head_key_f1'] > 0.99, st['head_key_f1']
    assert st['baseline_key_f1_all_live'] < 0.5, \
        'the all-live baseline is near 1 here, so this layer cannot discriminate'
    assert st['head_key_f1_over_best_baseline'] > 0.5

    # A head that fires everywhere must score exactly the all-live baseline -- which is the
    # identity that makes the baseline meaningful rather than decorative.
    _, allst = rebuild(el, el.local, cfg, np.ones_like(perfect))
    assert allst['head_key_f1'] == pytest.approx(allst['baseline_key_f1_all_live'], abs=1e-9)
    assert allst['head_key_f1_over_best_baseline'] <= 0.0

    # And the pipeline's own control: the DP must beat the same number of keys placed at
    # random. On a sparsely-keyed layer that margin is large; the point of quoting it is the
    # layers where it is not.
    _, pst = rebuild(el, el.local, RebuildConfig())
    assert 'baseline_pipeline_key_f1_random' in pst
    assert pst['key_f1_over_random'] > 0.1, \
        f'the DP beat random placement by only {pst["key_f1_over_random"]:+.3f} on a sparsely-' \
        'keyed layer, where it should have a wide margin'
    assert pst['key_f1'] == pytest.approx(
        pst['baseline_pipeline_key_f1_random'] + pst['key_f1_over_random'], abs=1e-12)

    # And the strict figure, which removes the free credit a clipped key gets from the knot
    # `rebuild` forces at the first and last live frame. It can only be lower.
    assert pst['key_f1_strict'] <= pst['key_f1'] + 1e-12
    assert pst['key_f1_strict'] > 0.0


def test_assemble_still_comes_out_perfect_with_a_key_head_present():
    """The artist's own points, the artist's own transform, and a key probability in hand: the
    reconstruction must still be perfect. Adding an output must not perturb the path."""
    d = V002 / 'FAM_0060_L1_A0003C007_v001__r1_t_c'
    el = load_element(d)
    want = [int(f) for f in el.frames[::17]]
    cfg = RebuildConfig(tol_px=0.01, smooth=1)
    rec = assemble(el, el.points, el.proj_crop, cfg, want, el.key_mask.astype(np.float32))
    assert rec.point_err_px == pytest.approx(0.0, abs=1e-4)
    assert rec.soft_iou.min() > 0.999, rec.soft_iou.min()
    assert rec.head_key_f1 == pytest.approx(1.0, abs=1e-9)


# ---- 6. scoring a layer the model never saw ------------------------------------

def test_untrained_queries_are_fresh_rows_not_another_layers():
    """A layer withheld from training has no query rows, and the two available answers are not
    equivalent: another layer's rows measure nothing, fresh rows measure the encoder alone."""
    torch.manual_seed(0)
    net = RotoNet(10, 3, 6, 1, dim=32, depth=1)
    before = net.shape_bank.weight.detach().clone()
    el = load_element(V002 / 'FAM_0060_L1_A0003C007_v001__r1_t_c', with_local=False)

    sb, gb = untrained_queries(net, el, seed=3)
    assert (sb, gb) == (10, 3), 'the new rows must start past the trained ones'
    assert net.shape_bank.num_embeddings == 10 + el.n_shapes
    assert torch.equal(net.shape_bank.weight[:10], before), \
        'growing the table must not disturb a single trained row'
    fresh = net.shape_bank.weight[10:]
    assert not torch.allclose(fresh, torch.zeros_like(fresh)), 'rows were left at zero'

    # Seeded, so the number a report quotes is reproducible.
    torch.manual_seed(0)
    other = RotoNet(10, 3, 6, 1, dim=32, depth=1)
    untrained_queries(other, el, seed=3)
    assert torch.equal(other.shape_bank.weight, net.shape_bank.weight)
    torch.manual_seed(0)
    third = RotoNet(10, 3, 6, 1, dim=32, depth=1)
    untrained_queries(third, el, seed=4)
    assert not torch.equal(third.shape_bank.weight[10:], net.shape_bank.weight[10:])


# ---- 7. the rebuild cleared the debt it was owed -------------------------------

def test_the_v002_ceiling_is_exact_where_v001_was_not():
    """The RED row v1.2 left open, closed.

    ``datasets/v001`` capped every score in the project at 0.999529 mean and 0.992632 on its
    worst frame, for two reasons found a round apart: v1.1's Catmull-Rom endpoint fix landed
    after those alphas were rendered, and every scorer rebuilt three render conventions from
    class defaults. Both are gone. What is left is 16-bit PNG quantisation and nothing else,
    which is checked exactly rather than to six decimals -- put the render through the same
    uint16 round trip the dataset writes and it must be *identical*.
    """
    worst_raw, worst_px = 1.0, 0.0
    for d in sorted(p for p in V002.iterdir() if (p / 'meta.json').exists()):
        el = load_element(d, with_local=False)
        cfg = config_from_meta(el.render)
        artist = from_json_ir(json.loads((d / 'target_ir.json').read_text()))
        c = el.crop
        for f in (int(el.frames[0]), int(el.frames[len(el.frames) // 2])):
            x0, y0 = c['offsets'][f]
            box = (x0, y0, c['out_px'] / c['scale'], c['out_px'] / c['scale'])
            truth = load_alpha(d, f)
            pred = render_union(artist, f, cfg, c['scale'],
                                box)[:truth.shape[0], :truth.shape[1]]
            q = (np.round(np.clip(pred, 0, 1) * 65535).astype(np.uint16).astype(np.float32)
                 / 65535.0)
            assert soft_iou(q, truth) == 1.0, f'{d.name} @{f}: {soft_iou(q, truth):.9f}'
            worst_raw = min(worst_raw, soft_iou(pred, truth))
            worst_px = max(worst_px, float(np.abs(pred - truth).max()))
    assert worst_px <= 1.0 / 131070 * 1.0001, \
        f'worst pixel {worst_px:.3e} exceeds half a uint16 quantum -- not quantisation'
    assert worst_raw > 0.9999, worst_raw
