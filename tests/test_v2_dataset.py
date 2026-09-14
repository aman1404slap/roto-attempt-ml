"""v2 ingest, manifest, QC, splits and ledger, against the real delivery.

Everything here reads ``data/spline_dataset_08_25_26`` and ``datasets/v003`` and skips when
they are absent, so a checkout without the data still runs the rest of the suite.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from roto.v2 import subset
from roto.v2.ingest import AUTOSAVE_RE, FRAME_RE, find_shot, is_autosave, sfx_candidates
from roto.v2.manifest import element_id, elements_for
from roto.v2.qc import AMBIGUOUS_MARGIN, MIN_MATTE_PX, usable_mattes
from roto.v2.splits import build_splits, frame_split

DATA = Path('data/spline_dataset_08_25_26')
DATASET = Path('datasets/v003')
needs_data = pytest.mark.skipif(not DATA.exists(), reason='delivery not present')
needs_built = pytest.mark.skipif(not (DATASET / 'manifest.json').exists(),
                                 reason='datasets/v003 not built')


# ---- ingest ------------------------------------------------------------------

@pytest.mark.parametrize('name,expected', [
    ('project.sfx', True), ('backup.sfx', True), ('backup.9.sfx', True),
    ('project.3.sfx', True), ('autosave.sfx', True),
    ('SINK_SHOTS_sh0390_Roto_V01.sfx', False),
])
def test_autosave_patterns(name, expected):
    assert is_autosave(name) is expected


@pytest.mark.parametrize('name,frame', [
    ('MON_0050_BG1_v001_Roto_V01.1030.exr', 1030),      # dot separator
    ('inn020_split02_v01_Roto_V01_1001.exr', 1001),     # underscore separator
])
def test_frame_regex_accepts_both_separators(name, frame):
    """Two separators are in use. The dot-only pattern in ``roto.exr`` returns no frames for
    the underscore form, which silently removed two Tier-1 shots from the referee."""
    m = FRAME_RE.search(name)
    assert m and int(m.group(1)) == frame


@needs_data
def test_pick_prefers_project_sfx_over_a_larger_autosave():
    """ts_021262's largest file is backup.4.sfx at 7.7 MB; project.sfx is 5.1 MB. Largest-wins
    -- the archive's rule -- picks a mid-session autosave the artist later cut shapes from."""
    shot = find_shot(DATA / 'ts_021262', referee=False)
    assert shot.sfx.name == 'project.sfx'
    assert 'rung 1' in shot.reason
    largest = sfx_candidates(DATA / 'ts_021262')[0]
    assert largest.stat().st_size > shot.sfx.stat().st_size


@needs_data
@pytest.mark.parametrize('name', subset.TIER1)
def test_every_tier1_shot_resolves_and_has_mattes(name):
    shot = find_shot(DATA / name, referee=False)
    assert shot.usable, shot.error
    assert shot.mattes(), f'{name} resolved no delivered mattes'


# ---- manifest ----------------------------------------------------------------

def test_element_id_is_path_safe():
    assert element_id('ts_1', 'Lady and Man matte') == 'ts_1__Lady_and_Man_matte'
    assert element_id('ts_1', '///') == 'ts_1__unnamed'


@needs_data
def test_manifest_tags_rather_than_excludes():
    """The archive manifest dropped over-keyed layers. v2 keeps and tags them (charter S6, D4);
    five Tier-1 elements would have been silently lost to that threshold."""
    els = elements_for(find_shot(DATA / 'ts_021150', referee=False))
    assert len(els) == 1
    assert els[0].tags['over_keyed'] is True
    assert els[0].tags['keys_per_live_frame'] > 0.75


# ---- QC ----------------------------------------------------------------------

@needs_data
def test_degenerate_matte_is_rejected_not_scored():
    """ts_021658 delivers a 1x1 px sequence; scoring it is meaningless and resizing it raises."""
    shot = find_shot(DATA / 'ts_021658', referee=False)
    keep, findings = usable_mattes(shot.mattes(), 2880, 1518)
    assert keep == {}
    assert any(f.startswith('degenerate') for f in findings)


@needs_data
def test_resolution_mismatch_is_rejected_not_cropped():
    """ts_020355 delivers 958x1435 against a 2882x2006 document -- a different format, not a
    scaled copy. Cropping both to the smaller shape would score two misaligned pictures."""
    shot = find_shot(DATA / 'ts_020355', referee=False)
    keep, findings = usable_mattes(shot.mattes(), 2882, 2006)
    assert keep == {}
    assert any(f.startswith('resolution') for f in findings)


@needs_built
def test_collisions_resolve_by_score_and_do_not_fail_a_shot():
    """A small element inside a large one claims the same channel and loses by a wide margin.
    Only a margin under AMBIGUOUS_MARGIN is the sh0230 failure."""
    qc = json.loads((DATASET / 'manifest.json').read_text())['qc']
    collisions = [f for r in qc.values() for f in r['findings'] if f.startswith('collision')]
    assert collisions, 'expected at least one resolved collision in Tier 1'
    assert all(r['passed'] for r in qc.values())


# ---- splits ------------------------------------------------------------------

def test_frame_split_never_holds_an_endpoint():
    train, held = frame_split(40, 7)
    assert 0 not in held and 39 not in held
    assert not set(train) & set(held)
    assert len(train) + len(held) == 40


def test_held_shot_withholds_all_its_elements():
    s = build_splits([('a__x', 'a', range(10)), ('b__y', 'b', range(10)),
                      ('b__z', 'b', range(10))], 7, held_shots=['b'])
    assert s['trained_elements'] == ['a__x']
    assert s['elements']['b__z']['held_because'] == ['shot b']


def test_unknown_held_shot_raises():
    with pytest.raises(ValueError, match='held shots not in the dataset'):
        build_splits([('a__x', 'a', range(5))], 7, held_shots=['nope'])


# ---- ledger ------------------------------------------------------------------

@needs_built
def test_ledger_is_all_green():
    """The Step 1 gate. Charter L3: Bucket A is exactly zero, pinned by a row that fails loudly."""
    from roto.v2.ledger import RED, build_ledger
    rows = build_ledger(DATASET)
    reds = [r for r in rows if r.status == RED]
    assert not reds, '\n'.join(f'{r.item}: {r.measured} -- {r.note}' for r in reds)


@needs_built
def test_scoring_ceiling_is_exactly_one():
    """Plan S1's named gate, and it is *exactly* 1.0 rather than 1.0 to six places -- the
    comparison runs through the same uint16 quantiser that wrote the stored alpha."""
    from roto.v2.ledger import build_ledger
    row = next(r for r in build_ledger(DATASET) if 'render back' in r.item)
    assert float(row.measured) == 1.0
