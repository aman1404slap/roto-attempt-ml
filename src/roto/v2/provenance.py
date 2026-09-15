"""Which environment produced a run, and which dataset it saw.

Everything in this project rests on not confusing weather for a result. Until now that was
safe by construction: there was one machine, one ``datasets/`` folder, and a number either
came off a 40k-step run or it did not. Once local smoke tests and staging runs write to the
same bucket -- which is the whole point of routing both through one storage layer -- a
two-shot sanity run becomes quotable by accident, and nothing in ``train_log.json`` would say
so.

So every run stamps two things it cannot derive later:

**The environment.** ``local`` is a smoke test: does this code run on a GPU without crashing.
It never produces a number that gets quoted, and :func:`check_quotable` makes that structural
rather than remembered. ``staging`` is where results come from. The stamp comes from the
``ROTO_ENVIRONMENT`` variable rather than from ``TrainConfig`` on purpose -- it is provenance,
not science, and putting it in the config would make two identical runs compare unequal.

**The dataset fingerprint.** A hash of the built dataset's ``manifest.json``, which records
every element, every crop parameter and every split the build made. Two runs with the same
fingerprint saw the same data; two with different fingerprints did not, however alike their
``--dataset`` arguments looked. That is the question a bucket makes hard to answer by eye,
because ``datasets/v003`` is a path and not a promise.

A run from before this existed carries no stamp. It is reported as ``unstamped`` and still
tables -- refusing it would invalidate every number already published, which is the opposite
of the guarantee. Only an explicit ``local`` is refused.
"""
from __future__ import annotations

import hashlib
import os
import platform
from pathlib import Path
from typing import Any

LOCAL, STAGING = 'local', 'staging'
ENVIRONMENTS = (LOCAL, STAGING)

ENV_VAR = 'ROTO_ENVIRONMENT'
"""Set by whatever launches the run. The Django service sets it per environment; a shell that
does not set it gets ``local``, which is the safe default -- an unstamped shell run is far
more likely to be someone's probe than a result."""

UNSTAMPED = 'unstamped'
"""What a run from before this module reports as. Not a third environment."""


def environment(explicit: str | None = None) -> str:
    """The environment this process is running in.

    ``explicit`` wins so a caller that knows can say so; otherwise ``ROTO_ENVIRONMENT``;
    otherwise ``local``. An unrecognised value raises rather than defaulting, because a typo
    that silently reads as ``local`` would be caught by nothing and a typo that silently
    reads as ``staging`` would be worse.
    """
    env = explicit or os.environ.get(ENV_VAR) or LOCAL
    if env not in ENVIRONMENTS:
        raise ValueError(f'unknown environment {env!r}, want one of {ENVIRONMENTS}')
    return env


def dataset_fingerprint(dataset_root: str | Path) -> str | None:
    """16 hex characters of the SHA-256 of the dataset's ``manifest.json``.

    The manifest is the right thing to hash rather than the alphas: it records the source
    root, every element built, the crop config, the QC grades and the counts, so any change
    that could move a number changes it. Hashing the PNGs would be slower and would also
    change on a lossless re-encode.

    ``None`` when there is no manifest -- which happens for a fixture dataset in a test, and
    should not be an error there.
    """
    p = Path(dataset_root) / 'manifest.json'
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def stamp(dataset_root: str | Path, explicit_environment: str | None = None) -> dict[str, Any]:
    """The provenance block written onto a checkpoint and its ``train_log.json``."""
    return {
        'environment': environment(explicit_environment),
        'dataset': str(dataset_root),
        'dataset_fingerprint': dataset_fingerprint(dataset_root),
        'host': platform.node(),
        'run_id': os.environ.get('ROTO_RUN_ID'),
    }


def environment_of(record: dict[str, Any] | None) -> str:
    """The environment a scored run reports, or ``unstamped`` for one from before this."""
    prov = (record or {}).get('provenance') or {}
    return prov.get('environment') or UNSTAMPED


def check_quotable(records: list[dict[str, Any]], what: str = 'this table') -> None:
    """Raise if any run came from ``local``.

    The rule from the migration note, in code: *nothing under ``runs/local/`` may ever be
    quoted as a result*. The message names the runs rather than saying "refused", because the
    person who hits this is usually scoring the right configuration from the wrong place.
    """
    bad = [r for r in records if environment_of(r) == LOCAL]
    if not bad:
        return
    names = ', '.join(str(r.get('run') or r.get('checkpoint') or f'seed {r.get("seed")}')
                      for r in bad)
    raise ValueError(
        f'refusing to put a local run in {what}: {names}.\n'
        'A local run is a smoke test -- a short schedule on a handful of shots, run to prove '
        'the code executes on a GPU. Re-run it in staging before quoting the number, or pass '
        'allow_local=True to look at it knowing it is not a result.')
