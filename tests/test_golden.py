"""Golden regression tests against real delivered pixels.

These are the safety net for every future refactor. Each row asserts that parsing a real
.sfx and rendering it reproduces the matte the vendor actually shipped, on frames sampled
across the shot. Thresholds are set ~1 point below measured values so genuine regressions
fail while noise does not.

Recorded 2026-08-27. If a threshold needs *raising*, that is an improvement -- update it.
If one needs lowering, something broke.
"""
import glob

import numpy as np
import pytest

from fixtures import DATA, PAIRS
from roto.eval.metrics import error_profile
from roto.matte.exr import load_matte, sequence
from roto.render.raster import render
from roto.sfx.read import read_sfx

# (shot, sfx_glob, layers, matte_folder, channel, frames, min_iou, edge_only_expected)
GOLDEN = [
    ('MAT_0130_L1_C002_260809_v001', 'scene/*.sfx', ['Red Matte'], 'matte01', 'R',
     [11, 22, 27, 33, 44], 0.985, True),
    ('nfl_0200_bg01_v001_compplate_roto_v001', 'scene/*.sfx', ['blue'], 'matte01', 'R',
     [38, 76, 95, 114, 152], 0.974, True),
    ('nfl_0200_bg01_v001_compplate_roto_v001', 'scene/*.sfx', ['green'], 'matte01', 'G',
     [38, 76, 95, 114, 152], 0.958, True),
    ('FAM_0060_L1_A0003C007_v001', 'scene/*.sfx', ['green'], 'matte01', 'G',
     [26, 53, 66, 79, 106], 0.971, True),
    ('FAM_0060_L1_A0003C007_v001', 'scene/*.sfx', ['blue'], 'matte02', 'B',
     [26, 53, 66, 79, 106], 0.983, True),
    ('FAM_0060_L1_A0003C007_v001', 'scene/*.sfx', ['red', 'red 1'], 'matte02', 'R',
     [26, 53, 66, 79, 106], 0.968, False),
    # 52% of this element is open shapes with zero stroke width, rendered as fills by a
    # heuristic (RenderConfig.fill_open_zero_width). Lowest score in the set, and the
    # clearest candidate for improvement once we can diff against Silhouette.
    ('nfl_0080_bg02_v001_compplate_roto_v001', 'scene/*.sfx', ['MB 2'], 'matte01', 'R',
     [30, 61, 77, 92, 123], 0.873, False),
]

IDS = [f'{s.split("_")[0]}_{"+".join(l)}' for s, _, l, _, _, _, _, _ in GOLDEN]


def _sfx(shot, pattern):
    hits = glob.glob(f'{DATA}/{shot}/{pattern}')
    if not hits:
        pytest.skip(f'test data not extracted: {DATA}/{shot}')
    return hits[0]


@pytest.mark.parametrize(
    'shot,pattern,layers,folder,channel,frames,min_iou,edge_only', GOLDEN, ids=IDS)
def test_render_matches_delivered_matte(shot, pattern, layers, folder, channel, frames,
                                        min_iou, edge_only):
    doc = read_sfx(_sfx(shot, pattern)).element(layers)
    files = sequence(f'{DATA}/{shot}/{folder}')
    profiles = []
    for frame in frames:
        gt, _ = load_matte(files[frame], channel)
        profiles.append(error_profile(render(doc, frame), gt))

    worst = min(p.iou for p in profiles)
    assert worst >= min_iou, (
        f'{shot}/{layers} regressed: worst IoU {worst:.4f} < {min_iou}; '
        f'all = {[round(p.iou, 4) for p in profiles]}')

    if edge_only:
        # Error confined near the boundary means anti-aliasing/feather, not wrong geometry.
        assert all(p.is_edge_only for p in profiles), (
            f'{shot}/{layers}: disagreement moved away from the boundary, which means a '
            f'structural bug, not an edge-rendering gap. '
            f'within_3px = {[round(p.frac_within_3px, 3) for p in profiles]}')


def test_all_six_shots_parse_and_validate():
    files = sorted(glob.glob(f'{DATA}/*/*/*.sfx'))
    if not files:
        pytest.skip('test data not extracted')
    assert len(files) == 6
    for f in files:
        doc = read_sfx(f, validate=True)
        assert doc.width > 0 and doc.height > 0 and doc.duration > 0
        assert doc.roots, f'{f}: no top-level layers'


def test_srgb_encoding_detected_per_shot():
    """Two of six shots wrongly store sRGB-encoded alpha. Detection must be automatic."""
    expect = {'FAM_0060_L1_A0003C007_v001': 'srgb',
              'MAT_0130_L1_C002_260809_v001': 'srgb',
              'nfl_0200_bg01_v001_compplate_roto_v001': 'linear',
              'nfl_0080_bg02_v001_compplate_roto_v001': 'linear'}
    for shot, want in expect.items():
        folder = f'{DATA}/{shot}/matte01'
        if not glob.glob(f'{folder}/*.exr'):
            pytest.skip('test data not extracted')
        files = sequence(folder)
        _, enc = load_matte(files[len(files) // 2], 'R')
        assert enc == want, f'{shot}: detected {enc}, expected {want}'
