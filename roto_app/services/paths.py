"""Where everything lives: the storage root, and the task's local disk.

One place, because these prefixes are the storage contract and a second opinion about them is
how a local smoke test ends up in the staging results directory:

    <source>/                            nobody writes -- the archive: .sfx, EXRs, plates
    <storage>/datasets/<version>/        the build step writes -- derived training data
    <storage>/runs/staging/<run>/        staging writes -- real runs
    <storage>/runs/local/<user>/<run>/   local writes -- smoke tests

``<storage>`` is ``s3://<bucket>`` in staging and a gitignored folder — ``abc`` — locally, and
``<source>`` is ``s3://production-citadel/<prefix>`` or the archive already on disk. **The
layout inside is the same either way**, which is what makes local a smaller staging rather than
a second design. See :mod:`roto_app.helpers.storage`.

**Local runs are namespaced under a user and never share a prefix with staging.** Two people
smoke-testing the same rung would otherwise overwrite each other, and -- the reason that
matters -- a local artifact sitting at a staging path is one directory listing away from being
quoted. The environment is stamped inside the checkpoint too, so the guarantee does not rest
on the path alone (see ``roto.v2.provenance``); this is the first of the two fences.
"""

import os

from django.conf import settings

from roto_app.helpers import storage


def storage_root() -> str:
    """``s3://<bucket>`` or a local directory, per ``STORAGE_ROOT``.

    A relative local root is resolved against the repo rather than the working directory, so a
    run launched from anywhere writes to the same ``abc`` — a detached subprocess does not
    inherit the shell's cwd, and "it wrote somewhere else" is a tedious thing to debug.
    """
    root = settings.STORAGE_ROOT
    if not root:
        raise ValueError(
            "STORAGE_ROOT is not set; there is nowhere to read datasets from or write runs to. "
            "Set it to a local folder (e.g. abc) or to s3://<bucket> in .env."
        )
    if storage.is_s3(root):
        return root
    return root if os.path.isabs(root) else os.path.join(settings.BASE_DIR, root)


def storage_uri(*parts: str) -> str:
    return storage.join(storage_root(), *parts)


def dataset_key(version: str) -> str:
    """The storage-relative key of a dataset version, for existence checks."""
    return f"datasets/{version}"


def dataset_uri(version: str) -> str:
    return storage_uri(dataset_key(version))


def source_uri(shot_prefix: str = "") -> str:
    """The read-only archive. Only the dataset build reads this.

    ``SOURCE_ROOT`` is ``s3://production-citadel/<prefix>`` once the cross-account read exists,
    and the archive already on disk until then. Read-only in both cases: nothing in this service
    writes to it, and the build is the only thing that opens it at all.
    """
    root = settings.SOURCE_ROOT
    if not storage.is_s3(root) and not os.path.isabs(root):
        root = os.path.join(settings.BASE_DIR, root)
    return storage.join(root, shot_prefix) if shot_prefix else str(root).rstrip("/")


def run_key(run_name: str, environment: str, user: str | None = None) -> str:
    """Storage-relative key for one run's artifacts, by environment."""
    if environment == "local":
        return f"runs/local/{user or settings.LOCAL_USER}/{run_name}"
    return f"runs/{environment}/{run_name}"


def run_uri(run_name: str, environment: str, user: str | None = None) -> str:
    return storage_uri(run_key(run_name, environment, user))


def runs_uri(environment: str, user: str | None = None) -> str:
    """The directory a run's siblings live in -- where a score table lands beside them."""
    if environment == "local":
        return storage_uri("runs", "local", user or settings.LOCAL_USER)
    return storage_uri("runs", environment)


# -- local disk on whichever machine is executing ------------------------------------------


def scratch(*parts: str) -> str:
    path = os.path.join(settings.SCRATCH_DIR, *[p.strip("/") for p in parts if p])
    os.makedirs(path, exist_ok=True)
    return path


def local_dataset_dir(version: str) -> str:
    """Datasets are cached by version, outside any single run's directory.

    Deliberately shared: a second seed of the same rung on the same instance should not
    re-download the build, and a dataset version is immutable by construction -- a different
    build is a different version, which is what ``roto.v2.provenance``'s fingerprint checks.
    """
    return scratch("datasets", version)


def local_runs_root() -> str:
    """What ``--runs-root`` is given. One directory per run name underneath."""
    return scratch("runs")


def local_run_dir(run_name: str) -> str:
    return scratch("runs", run_name)


def local_data_root() -> str:
    """Where the raw archive is staged for a dataset build. CPU-only work, never the GPU task's
    concern -- training needs the derived dataset and nothing else, which is why the training
    task never has to hold client footage."""
    return scratch("data")
