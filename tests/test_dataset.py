"""The clean-alpha dataset path: manifest -> build -> program -> render back.

The gate that matters is the last one. If the artist's program cannot survive a trip through
the tensors, no model trained on them could reproduce it, and every later measurement would
be reporting the representation's ceiling rather than the model's skill. So the assertion is
decode(encode(program)) rendering back to the element's own alpha.

``ROUND_TRIP_TOL`` is 1e-4 rather than 0 for one reason, and it is not slack in the
representation. The stored alpha is quantised to 16-bit PNG, so a pixel carries at most
1/65535 = 1.5e-5 of error before anything is compared; the re-render is float. The tolerance
is that quantisation, and nothing else -- the tensors themselves round-trip bit-for-bit. If
it ever needs raising, the representation lost something and that is the bug.

The stride cases are here because of a specific bug. Opacity and the transform track are
document properties, but they used to be sampled on whichever frames were rasterised, then
indexed by the dense key axis. At stride 1 the two axes coincide and everything passes; above
stride 1 the track runs off its end and the live mask collapses, so shapes render dead. It is
invisible in exactly the configuration a smoke run uses.
"""
import numpy as np
import pytest

from fixtures import DATA
from roto.dataset import CropConfig, build, discover, load_alpha, load_meta
from roto.eval.metrics import soft_iou
from roto.program import decode, load_program
from roto.render.raster import RenderConfig, render_union

ELEMENT = 'nfl_0200_bg01_v001_compplate_roto_v001__blue'
ROUND_TRIP_TOL = 1e-4


def test_manifest_is_top_level_layers_and_uses_no_exr():
    els = discover(DATA)
    assert len(els) == 18, 'six shots carry 18 top-level layers with shapes'
    ids = {e.element_id for e in els}
    assert ELEMENT in ids
    # Selection must not depend on a delivered matte existing for the layer.
    assert all(not hasattr(e, 'channel') for e in els)


def test_exclusions_fire_on_the_measured_cases():
    by_id = {e.element_id: e for e in discover(DATA)}
    hair = by_id['TVC_SHOTS_sh0230_BG01_v003_roto_v02__L110']
    assert not hair.is_target and any('paint_strokes' in r for r in hair.excluded_by)
    dense = by_id['MAT_0130_L1_C002_260809_v001__Red_Matte']
    assert not dense.is_target and any('over_keyed' in r for r in dense.excluded_by)
    assert sum(e.is_target for e in by_id.values()) == 13


@pytest.mark.parametrize('stride', [1, 7, 20])
def test_program_round_trips_to_the_element_alpha(tmp_path, stride):
    element = next(e for e in discover(DATA) if e.element_id == ELEMENT)
    built = build(element, DATA, tmp_path, CropConfig(size=256, supersample=2, stride=stride))
    assert built.meta['render']['exr_used'] is False

    spec, tensors, meta = load_program(built.directory)
    assert spec.n_shapes == 75 and spec.n_groups == 3
    doc = decode(built.directory, spec, tensors)

    cfg = RenderConfig(supersample=meta['render']['supersample'])
    size, scale = meta['crop']['size_src_px'], meta['crop']['scale']
    for f in meta['frames']['index']:
        x0, y0 = meta['crop']['offsets'][str(f)]
        pred = render_union(doc, f, cfg, scale, (x0, y0, size, size))
        assert soft_iou(pred, load_alpha(built.directory, f)) == pytest.approx(1.0, abs=ROUND_TRIP_TOL)


def test_transform_track_is_dense_regardless_of_stride(tmp_path):
    element = next(e for e in discover(DATA) if e.element_id == ELEMENT)
    built = build(element, DATA, tmp_path, CropConfig(size=128, supersample=1, stride=25))
    t = np.load(built.directory / 'tensors.npz')
    frames, dense = t['frames'], t['matrix_frames']
    assert len(frames) < len(dense), 'a stride must not thin the transform track'
    assert dense.tolist() == list(range(int(frames.min()), int(frames.max()) + 1))
    assert len(t['layer_matrix/0']) == len(dense)
    assert len(t['opacity/0']) == len(dense)
