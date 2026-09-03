"""Core pipeline invariants: shot discovery, IR round-trip, transforms, soft edges.

These are the safety net under everything else. If the IR cannot survive a round trip, or a
transform composes in the wrong order, every number downstream is wrong in a way that looks
like a model problem.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from roto.shots import find_shot, find_shots
from roto.dataset.layers import top_layers
from roto.metrics import iou, soft_iou
from roto.ir import Key, Layer, RotoDoc, Shape, is_ephemeral, shape_class
from roto.render.raster import RenderConfig, layer_matrix, render, trs_matrix
from roto.sfx.json_ir import from_json_ir, to_json_ir
from roto.sfx.read import read_sfx

from common import DATA


# ---- shot discovery ------------------------------------------------------------

def test_finds_all_six_shots_across_both_layouts():
    shots = find_shots(DATA)
    assert len(shots) == 6
    assert all(s.sfx is not None for s in shots)
    # Two layouts: scene/*.sfx + matte01/, and <name>_SFX_script_v02/ + <name>_matte_*/
    by_name = {s.name: s for s in shots}
    assert by_name['nfl_0200_bg01_v001_compplate_roto_v001'].sfx.parent.name == 'scene'
    tvc = by_name['TVC_SHOTS_sh0260_BG01_v003_roto_v02']
    assert tvc.sfx.suffix == '.sfx'


@pytest.mark.parametrize('shot', [s.name for s in find_shots(DATA)])
def test_json_ir_round_trips_and_renders_identically(shot):
    doc = read_sfx(find_shot(f'{DATA}/{shot}').sfx)
    back = from_json_ir(json.loads(json.dumps(to_json_ir(doc))))
    assert back.stats() == doc.stats()
    frame = doc.duration // 2
    a = render(doc, frame, scale=0.15)
    b = render(back, frame, scale=0.15)
    np.testing.assert_array_equal(a, b)


def test_json_ir_keeps_its_documented_keys():
    doc = read_sfx(find_shot(f'{DATA}/nfl_0200_bg01_v001_compplate_roto_v001').sfx)
    obj = to_json_ir(doc)
    assert set(obj['session']) == {'width', 'height', 'startFrame', 'duration', 'frameRate'}
    layer = obj['layers'][0]
    for key in ('label', 'uuid', 'matrix', 'trs', 'shapes', 'children'):
        assert key in layer, key

    def first_shape(node):
        if node['shapes']:
            return node['shapes'][0]
        for child in node['children']:
            found = first_shape(child)
            if found is not None:
                return found
        return None

    shape = first_shape(layer)
    assert shape is not None
    for key in ('label', 'uuid', 'shape_type', 'path_keys', 'opacity',
                'strokeWidth', 'mode', 'invert'):
        assert key in shape, key
    frame, interp, closed, points = shape['path_keys'][0]
    assert isinstance(frame, int) and isinstance(closed, bool)
    assert interp in ('linear', 'hold', 'catmullrom')
    assert np.asarray(points).ndim == 3       # (n_points, coords_per_point, 2)


def test_json_ir_preserves_interleaved_child_order():
    """The IR schema splits shapes and layers into separate lists, losing their
    relative order. Order is the model's teacher-forcing sequence, so we keep it."""
    doc = RotoDoc(width=100, height=100, duration=2, roots=[Layer(name='root', children=[
        Shape(name='s0', path=[Key(0, 'linear', np.zeros((3, 1, 2)))]),
        Layer(name='l0'),
        Shape(name='s1', path=[Key(0, 'linear', np.zeros((3, 1, 2)))]),
    ])])
    back = from_json_ir(to_json_ir(doc))
    assert [c.name for c in back.roots[0].children] == ['s0', 'l0', 's1']


# ---- TRS transforms ------------------------------------------------------------

def test_trs_identity_is_none_and_translation_moves_points():
    assert trs_matrix({}, 0) is None
    assert trs_matrix({'position': [Key(0, 'hold', np.array([0.0, 0.0]))]}, 0) is None

    m = trs_matrix({'position': [Key(0, 'hold', np.array([0.25, -0.5]))]}, 0)
    p = np.array([[1.0, 2.0, 0.0, 1.0]]) @ m
    np.testing.assert_allclose(p[0, :2], [1.25, 1.5])


def test_trs_rotation_composes_before_the_baked_matrix():
    rot = trs_matrix({'rotate': [Key(0, 'hold', np.array([90.0]))]}, 0)
    np.testing.assert_allclose(np.array([[1.0, 0.0, 0.0, 1.0]]) @ rot,
                               [[0.0, 1.0, 0.0, 1.0]], atol=1e-12)
    baked = np.eye(4); baked[3, 0] = 10.0
    layer = Layer(name='l', transform=[Key(0, 'hold', baked)],
                  trs={'rotate': [Key(0, 'hold', np.array([90.0]))]})
    # Row vectors: TRS applies first, then the tracker matrix translates.
    np.testing.assert_allclose(np.array([[1.0, 0.0, 0.0, 1.0]]) @ layer_matrix(layer, 0),
                               [[10.0, 1.0, 0.0, 1.0]], atol=1e-12)


def test_trs_is_identity_on_every_measured_shot():
    """No shot in the archive relies on TRS, so it
    cannot explain any IoU gap. The support exists so a future shot cannot fail silently."""
    for shot in find_shots(DATA):
        doc = read_sfx(shot.sfx, validate=False)
        for _, layer in doc.layers():
            assert trs_matrix(layer.trs, 0) is None, (shot.name, layer.name)


# ---- soft edges and soft IoU ---------------------------------------------------

def test_supersampling_is_on_by_default_and_produces_soft_edges():
    assert RenderConfig().supersample == 2
    doc = read_sfx(find_shot(f'{DATA}/MAT_0130_L1_C002_260809_v001').sfx).layer(['Red Matte'])
    soft = render(doc, 10, RenderConfig(supersample=2), scale=0.3)
    hard = render(doc, 10, RenderConfig(supersample=1), scale=0.3)
    fractional = ((soft > 0.01) & (soft < 0.99)).sum()
    assert fractional > 100, 'anti-aliased render has no partial-coverage pixels'
    assert ((hard > 0.01) & (hard < 0.99)).sum() < fractional


def test_soft_iou_sees_the_edge_error_that_thresholded_iou_hides():
    """Silhouette's edge reads 0.00 -> 0.88 -> 1.00
    across a row; a hard render gives 0.00 -> 1.00 -> 1.00. Both agree once thresholded at
    0.5, so thresholded IoU calls the hard render perfect. soft-IoU does not."""
    truth = np.array([[0.0, 0.88, 1.0]])
    soft = np.array([[0.0, 0.88, 1.0]])
    hard = np.array([[0.0, 1.00, 1.0]])

    assert iou(hard, truth) == iou(soft, truth) == 1.0      # threshold hides it
    assert soft_iou(soft, truth) == 1.0                     # soft-IoU does not
    assert soft_iou(hard, truth) == pytest.approx(1.88 / 2.0)


def test_soft_iou_edge_cases():
    z = np.zeros((4, 4))
    assert soft_iou(z, z) == 1.0
    assert soft_iou(np.ones((4, 4)), z) == 0.0


# ---- layer -> channel matching -------------------------------------------------

def test_top_layers_are_the_documents_roots():
    doc = read_sfx(find_shot(f'{DATA}/nfl_0200_bg01_v001_compplate_roto_v001').sfx)
    refs = top_layers(doc)
    assert [r.name for r in refs] == [root.name for root in doc.roots] == ['blue', 'green']
    assert all(r.ancestors == () for r in refs)


def test_isolate_rebuilds_the_ancestor_chain_so_transforms_still_compose():
    """A nested candidate rendered detached loses every transform above it, which looks
    plausible and is wrong. isolate() rebuilds the ancestor chain as childless stubs."""
    shift = np.eye(4); shift[3, 0] = 0.25
    square = np.array([[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]])[:, None, :]
    leaf = Layer(name='leaf', children=[Shape(name='s', path=[Key(0, 'hold', square)])])
    root = Layer(name='root', transform=[Key(0, 'hold', shift)], children=[leaf])
    doc = RotoDoc(width=200, height=200, duration=1, roots=[root])

    with_chain = render(doc.isolate((root,), leaf), 0)
    detached = render(doc.isolate((), leaf), 0)
    assert with_chain.sum() > 0 and detached.sum() > 0
    assert not np.array_equal(with_chain, detached)
    # The transform is a pure +0.25 x-shift, i.e. +50px at height 200.
    cx = lambda a: float((np.nonzero(a.sum(0))[0]).mean())
    assert cx(with_chain) - cx(detached) == pytest.approx(50, abs=2)


def test_tracked_layers_are_always_leaves():
    """Measured invariant across all six shots: a layer carrying a tracker matrix owns
    shapes and never other layers. Named group layers (core/face/body/CH1) are never
    tracked; tracked ones are auto-named 'Layer N' leaves.

    Two consequences. (a) No candidate can have a tracked ancestor, so detached rendering
    happens to be safe on this drop -- isolate() is insurance, not a fix. (b) The artist's
    motion factorization is one tracker per shape group, not a hierarchy of nested
    transforms, which is the shape the model's transform head should predict.
    """
    for shot in find_shots(DATA):
        doc = read_sfx(shot.sfx, validate=False)
        for _, layer in doc.layers():
            if layer.transform:
                nested = [c for c in layer.children if isinstance(c, Layer)]
                assert not nested, (shot.name, layer.name, [n.name for n in nested])


def test_shape_class_splits_the_four_training_populations():
    def sh(**kw):
        return Shape(name='s', path=[Key(0, 'linear', np.zeros((4, 1, 2)))], **kw)

    assert shape_class(sh()) == 'persistent_fill'
    assert shape_class(sh(closed=False)) == 'persistent_stroke'
    gated = [Key(10, 'hold', np.array([0.0])), Key(11, 'hold', np.array([100.0])),
             Key(12, 'hold', np.array([0.0]))]
    assert shape_class(sh(closed=False, opacity=gated)) == 'ephemeral_stroke'
    assert is_ephemeral(sh(opacity=gated))
    # A shape with a constant opacity key is not gated, however low the value.
    assert not is_ephemeral(sh(opacity=[Key(0, 'hold', np.array([50.0]))]))


def test_ephemeral_counts_match_the_measured_archive():
    """The 51%-of-shapes-are-single-frame finding, as a regression test: it is the reason
    ephemeral shapes exist and are counted."""
    counts = {s.name: read_sfx(s.sfx, validate=False).stats() for s in find_shots(DATA)}
    total = sum(v['shapes'] for v in counts.values())
    ephemeral = sum(v['ephemeral'] for v in counts.values())
    assert total == 6969
    assert 0.45 < ephemeral / total < 0.55
    assert counts['TVC_SHOTS_sh0230_BG01_v003_roto_v02']['ephemeral'] > 3000
    assert counts['nfl_0200_bg01_v001_compplate_roto_v001']['ephemeral'] == 0
