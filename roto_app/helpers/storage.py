"""One storage interface, two backings: an S3 bucket, or a directory on disk.

Until the infrastructure exists there is no bucket to write to, and waiting for one would mean
the service could not be run at all. So the storage root is a *setting* rather than a fact:
``s3://<bucket>`` in staging, a gitignored folder — ``abc`` — locally.

**The layout inside the root is identical either way.** ``datasets/<version>/``,
``runs/staging/<run>/``, ``runs/local/<user>/<run>/`` — same three prefixes, same rules about
who writes which. That is the whole point: local is staging with a different root, not a
different design, so the code that will run on ECS is the code that runs here, and swapping back
to a real bucket is one environment variable rather than a rewrite.

The two backings differ in exactly one place — this module — and nothing above it knows which
is in play.
"""

import os
import shutil
from pathlib import Path

from roto_app.helpers import aws_helpers
from roto_app.helpers.aws_helpers import S3Unavailable  # NOQA: F401  (re-exported)

S3_SCHEME = "s3://"


def is_s3(uri: str) -> bool:
    return str(uri).startswith(S3_SCHEME)


def join(root: str, *parts: str) -> str:
    """Join onto a root that may be an S3 URI or a local path.

    ``os.path.join`` would do the wrong thing to ``s3://bucket`` on Windows and the right thing
    by accident elsewhere; being explicit costs three lines and removes the question.
    """
    tail = "/".join(str(p).strip("/") for p in parts if p)
    if not tail:
        return str(root).rstrip("/")
    return f"{str(root).rstrip('/')}/{tail}"


def prefix_exists(uri: str) -> bool:
    """Whether anything is stored under ``uri``.

    Raises :class:`S3Unavailable` when the question could not be *asked* — an unset bucket, no
    credentials, a policy that refuses the list. That is distinct from a confident ``False``,
    and conflating them turns a configuration fault into "your dataset does not exist".
    """
    if is_s3(uri):
        bucket, _, prefix = uri[len(S3_SCHEME) :].partition("/")
        return aws_helpers.s3_prefix_exists(bucket=bucket, prefix=prefix)

    path = Path(uri)
    return path.is_dir() and any(path.iterdir())


def sync_in(*, src: str, dest_path: str) -> None:
    """Bring a prefix down to local disk, so the training code sees an ordinary directory."""
    if is_s3(src):
        aws_helpers.sync_s3_to_local(s3_uri=src, dest_path=dest_path)
        return
    _copy_tree(src, dest_path)


def sync_out(*, src_path: str, dest: str) -> None:
    """Push a finished directory to the storage root."""
    if is_s3(dest):
        aws_helpers.sync_local_to_s3(src_path=src_path, s3_uri=dest)
        return
    _copy_tree(src_path, dest)


def copy_file_out(*, src_path: str, dest: str) -> None:
    """Put one file at one key. Used for the score table and summary."""
    if is_s3(dest):
        aws_helpers.copy_local_file_to_s3(src_path=src_path, s3_uri=dest)
        return
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dest)
    print(f"copied {src_path} -> {dest}")


def _copy_tree(src: str, dest: str) -> None:
    """Incremental directory copy — the local stand-in for ``aws s3 sync``.

    Incremental rather than a plain ``copytree`` because ``sync`` is: a dataset already on disk
    from an earlier run should cost a stat per file, not a re-copy of 1.9 GB. Size and mtime are
    what S3 sync compares too, so the two backings skip the same files for the same reason.

    Nothing is ever deleted from the destination. That matches ``sync_in``/``sync_out``'s default
    and it matters more here, where the destination is a folder someone may be keeping things in.
    """
    src_root, dest_root = Path(src), Path(dest)
    if not src_root.is_dir():
        raise FileNotFoundError(f"nothing to copy: {src_root} is not a directory")

    copied = skipped = 0
    for source in src_root.rglob("*"):
        if source.is_dir():
            continue
        target = dest_root / source.relative_to(src_root)
        if _same_file(source, target):
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1

    print(f"synced {src_root} -> {dest_root} ({copied} copied, {skipped} already current)")


def _same_file(source: Path, target: Path) -> bool:
    if not target.exists():
        return False
    a, b = source.stat(), target.stat()
    # int() on mtime because copy2 preserves sub-second precision unevenly across filesystems,
    # and a float comparison would re-copy everything on some of them.
    return a.st_size == b.st_size and int(a.st_mtime) == int(b.st_mtime)


def describe(uri: str) -> str:
    """How to refer to a location in a log line or an error."""
    return str(uri) if is_s3(uri) else os.path.abspath(uri)
