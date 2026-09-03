"""Recovered layer -> delivered-matte-channel mapping.

This mapping is NOT in the archive. The Silhouette layers were rendered per-layer into
colour-named folders and a Nuke script packed them into RGB channels; that script was not
delivered. The pairings below were recovered by brute-force IoU search over every
(top-level layer union) x (matte file, channel) combination.

Treat as measured fact for these six shots, and as a warning: any future delivery should
use named EXR channels so this step is unnecessary. See FINDINGS.md.
"""
DATA = 'data/extracted/test_data'

# (shot, sfx_glob, element_layers, matte_folder, channel)
PAIRS = [
    ('MAT_0130_L1_C002_260809_v001', 'scene/*.sfx', ['Red Matte'], 'matte01', 'R'),
    ('nfl_0200_bg01_v001_compplate_roto_v001', 'scene/*.sfx', ['blue'], 'matte01', 'R'),
    ('nfl_0200_bg01_v001_compplate_roto_v001', 'scene/*.sfx', ['green'], 'matte01', 'G'),
    ('FAM_0060_L1_A0003C007_v001', 'scene/*.sfx', ['green'], 'matte01', 'G'),
    ('FAM_0060_L1_A0003C007_v001', 'scene/*.sfx', ['blue'], 'matte02', 'B'),
    ('FAM_0060_L1_A0003C007_v001', 'scene/*.sfx', ['red', 'red 1'], 'matte02', 'R'),
    ('nfl_0080_bg02_v001_compplate_roto_v001', 'scene/*.sfx', ['MB 2'], 'matte01', 'R'),
]
