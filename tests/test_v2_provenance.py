"""A local run must not be able to reach a results table.

The rule is from ``aws-gpu-migration.md`` section 5.4, and the reason it is a test rather than
a convention is in the note: once local smoke tests and staging runs write to the same bucket,
the only thing between a two-shot probe and a quoted number is this check. A convention would
hold until the first hurried afternoon.
"""
from __future__ import annotations

import json

import pytest

from roto.v2.provenance import (ENV_VAR, LOCAL, STAGING, UNSTAMPED, check_quotable,
                                dataset_fingerprint, environment, environment_of, stamp)
from roto.v2.score import frozen_table


def _run(env: str | None, fingerprint: str = 'abc123', **extra):
    """One scored-run record, shaped as ``score_run`` returns it."""
    prov = {} if env is None else {'environment': env, 'dataset_fingerprint': fingerprint}
    return {'run': 's3a', 'dataset': 'datasets/v003', 'provenance': prov,
            'held_shots': [], 'held_shot_elements': [], 'frozen_elements': [], 'trained': {'mean_soft_iou': 0.97}, **extra}


def test_environment_defaults_to_local(monkeypatch):
    """An unset variable means local, because an unstamped shell run is almost always a probe."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert environment() == LOCAL


def test_environment_reads_the_variable(monkeypatch):
    monkeypatch.setenv(ENV_VAR, STAGING)
    assert environment() == STAGING
    assert environment(LOCAL) == LOCAL, 'an explicit argument wins over the environment'


def test_unknown_environment_raises_rather_than_defaulting(monkeypatch):
    """A typo must not silently read as either value -- one hides a probe, one invents a result."""
    monkeypatch.setenv(ENV_VAR, 'stagng')
    with pytest.raises(ValueError, match='unknown environment'):
        environment()


def test_fingerprint_follows_the_manifest(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({'elements': ['a']}))
    first = dataset_fingerprint(tmp_path)
    assert first and len(first) == 16

    (tmp_path / 'manifest.json').write_text(json.dumps({'elements': ['a', 'b']}))
    assert dataset_fingerprint(tmp_path) != first, 'a different build must fingerprint differently'


def test_fingerprint_is_none_without_a_manifest(tmp_path):
    """A fixture dataset has no manifest. That is not an error, it is just unfingerprintable."""
    assert dataset_fingerprint(tmp_path) is None


def test_stamp_carries_environment_and_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, STAGING)
    (tmp_path / 'manifest.json').write_text('{}')
    s = stamp(tmp_path)
    assert s['environment'] == STAGING
    assert s['dataset_fingerprint'] == dataset_fingerprint(tmp_path)
    assert s['dataset'] == str(tmp_path)


def test_check_quotable_refuses_a_local_run():
    with pytest.raises(ValueError, match='refusing to put a local run'):
        check_quotable([_run(STAGING), _run(LOCAL)])


def test_check_quotable_allows_staging_and_unstamped():
    """Unstamped runs predate this and still table -- refusing them would void published work."""
    check_quotable([_run(STAGING), _run(None)])
    assert environment_of(_run(None)) == UNSTAMPED


def test_frozen_table_refuses_a_local_run():
    with pytest.raises(ValueError, match='refusing to put a local run'):
        frozen_table([_run(LOCAL)], label='v2 S3A')


def test_frozen_table_allows_local_when_asked():
    out = frozen_table([_run(LOCAL)], label='v2 S3A', allow_local=True)
    assert 'environment local' in out, 'looking anyway must still say what it is looking at'


def test_frozen_table_warns_on_mixed_fingerprints():
    """Two builds in one table is a cross-dataset comparison wearing one dataset's name."""
    out = frozen_table([_run(STAGING, 'aaa'), _run(STAGING, 'bbb')], label='v2 S3A')
    assert 'different dataset fingerprints' in out


def test_frozen_table_stays_quiet_for_a_clean_staging_table():
    """The frozen format is extended, not rewritten: a normal table gains no header line."""
    out = frozen_table([_run(STAGING), _run(STAGING)], label='v2 S3A')
    assert 'environment' not in out.splitlines()[1]
