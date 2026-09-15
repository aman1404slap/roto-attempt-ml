"""Where everything lives, in S3 and on the task's local disk.

One place, because these four prefixes are the storage contract and a second opinion about
them is how a local smoke test ends up in the staging results directory:

    s3://production-citadel/<prefix>/        nobody writes -- the archive: .sfx, EXRs, plates
    s3://<ours>/datasets/<version>/          the build step writes -- derived training data
    s3://<ours>/runs/staging/<run>/          staging writes -- real runs
    s3://<ours>/runs/local/<user>/<run>/     local writes -- smoke tests

**Local runs are namespaced under a user and never share a prefix with staging.** Two people
smoke-testing the same rung would otherwise overwrite each other, and -- the reason that
matters -- a local artifact sitting at a staging path is one directory listing away from being
quoted. The environment is stamped inside the checkpoint too, so the guarantee does not rest
on the path alone (see ``roto.v2.provenance``); this is the first of the two fences.
"""

import os

from django.conf import settings


def bucket_uri(*parts: str) -> str:
    """``s3://<our bucket>/a/b/c``."""
    if not settings.AWS_DEFAULT_BUCKET:
        raise ValueError(
            "AWS_DEFAULT_BUCKET is not set; there is nowhere to read datasets from or write "
            "runs to. Set it in .env."
        )
    tail = "/".join(p.strip("/") for p in parts if p)
    return (
        f"s3://{settings.AWS_DEFAULT_BUCKET}/{tail}"
        if tail
        else (f"s3://{settings.AWS_DEFAULT_BUCKET}")
    )


def dataset_key(version: str) -> str:
    """The bucket-relative key of a dataset version, for existence checks."""
    return f"datasets/{version}"


def dataset_uri(version: str) -> str:
    return bucket_uri(dataset_key(version))


def source_uri(shot_prefix: str = "") -> str:
    """The read-only archive in the other account. Only the dataset build reads this."""
    parts = [p for p in (settings.SOURCE_S3_PREFIX, shot_prefix) if p]
    tail = "/".join(p.strip("/") for p in parts)
    return (
        f"s3://{settings.SOURCE_S3_BUCKET}/{tail}"
        if tail
        else (f"s3://{settings.SOURCE_S3_BUCKET}")
    )


def run_key(run_name: str, environment: str, user: str | None = None) -> str:
    """Bucket-relative key for one run's artifacts, by environment."""
    if environment == "local":
        return f"runs/local/{user or settings.LOCAL_USER}/{run_name}"
    return f"runs/{environment}/{run_name}"


def run_uri(run_name: str, environment: str, user: str | None = None) -> str:
    return bucket_uri(run_key(run_name, environment, user))


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
    """Where the raw archive is synced for a dataset build. CPU-only work, never the GPU task's
    concern -- training needs the derived dataset and nothing else, which is why the training
    task never has to hold client footage."""
    return scratch("data")
