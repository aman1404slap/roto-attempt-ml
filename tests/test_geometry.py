"""The local <-> crop coordinate mapping the model predicts through.

This is the mapping that, when it was absent, made the network stall at ~260 px on what looked
like a capacity problem. It has to be exactly invertible: the network predicts in crop space,
and every prediction is converted back to local normalised coordinates before it becomes a
spline. A lossy mapping here would put a floor under every geometry number in the project and
would look like the model being bad.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from roto.geometry import crop_to_local, local_to_crop
from roto.render.raster import RenderConfig, apply_transform, shape_polyline
from roto.sfx.json_ir import from_json_ir

KW = dict(width=2880, height=1978, offset=(812.0, 433.0), scale=0.788, out_px=256)


def random_affine(rng):
    m = np.eye(4)
    m[0, 0], m[1, 1] = 1.0 + rng.normal(scale=0.1), 1.0 + rng.normal(scale=0.1)
    m[0, 1], m[1, 0] = rng.normal(scale=0.05), rng.normal(scale=0.05)
    m[3, 0], m[3, 1] = rng.normal(scale=0.3), rng.normal(scale=0.3)
    return m


@pytest.mark.parametrize('seed', range(5))
def test_round_trip_is_exact(seed):
    rng = np.random.default_rng(seed)
    m = random_affine(rng)
    pts = rng.normal(scale=0.3, size=(40, 3, 2))
    back = crop_to_local(local_to_crop(pts, m, **KW), m, **KW)
    assert np.abs(back - pts).max() < 1e-12


def test_identity_transform_places_the_frame_centre_predictably():
    """With no layer transform, local (0,0) is the centre of the *source frame*."""
    c = local_to_crop(np.zeros((1, 2)), np.eye(4), **KW)
    expect_x = (KW['width'] / 2 - KW['offset'][0]) * KW['scale'] / KW['out_px']
    expect_y = (KW['height'] / 2 - KW['offset'][1]) * KW['scale'] / KW['out_px']
    assert c[0, 0] == pytest.approx(expect_x)
    assert c[0, 1] == pytest.approx(expect_y)


def test_matches_the_renderer_on_a_real_shape():
    """The mapping must agree with how the rasteriser places pixels, not merely be invertible.

    Re-deriving the normalised-to-pixel formula is the obvious shortcut and would let the
    model train against coordinates the renderer does not draw at.
    """
    d = Path('datasets/v001/nfl_0200_bg01_v001_compplate_roto_v001__blue')
    if not (d / 'meta.json').exists():
        pytest.skip('dataset not built')
    meta = json.loads((d / 'meta.json').read_text())
    doc = from_json_ir(json.loads((d / 'target_ir.json').read_text()))
    crop, src = meta['crop'], meta['source']
    frame = int(meta['frames']['index'][len(meta['frames']['index']) // 2])
    x0, y0 = crop['offsets'][str(frame)]

    ancestors, shape = next(iter(doc.shapes()))
    from roto.ir import sample
    from roto.render.raster import compose, layer_matrix
    mats = [m for m in (layer_matrix(l, frame) for l in ancestors) if m is not None]
    matrix = compose(mats) if mats else np.eye(4)

    local = np.asarray(sample(shape.path, frame), float)[:, 0, :]
    ours = local_to_crop(local, matrix, width=src['width'], height=src['height'],
                         offset=(x0, y0), scale=crop['scale'],
                         out_px=crop['out_px'][0]) * crop['out_px'][0]

    # The renderer's own path: transform to normalised screen, then its pixel formula.
    n = apply_transform(matrix, local)
    hn = src['height'] * crop['scale']
    theirs = np.stack([n[:, 0] * hn + (src['width'] / 2 - x0) * crop['scale'],
                       n[:, 1] * hn + (src['height'] / 2 - y0) * crop['scale']], axis=-1)
    assert np.abs(ours - theirs).max() < 1e-6
