"""S3 in and out, by shelling to the AWS CLI.

**boto3 stays out of the training path on purpose.** The code reads plain local paths and an
S3-native data layer would be a rewrite in exchange for nothing, so the contract is: sync to
local disk, run the existing code unchanged, sync the results back. That is also what keeps
``src/roto`` free of any dependency on this service.

``aws s3 sync`` rather than ``cp --recursive``, because a dataset that is already on disk from
a previous run on the same instance should cost a listing rather than a re-download, and
because a re-sync after a partial failure is then idempotent.
"""

import subprocess  # nosec
import time
from functools import wraps
from typing import Any, Callable

from roto_app.helpers.utils import ensure_suffix


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff_factor: float = 4.0):
    """Retry with exponential backoff. S3 failures here are usually transient or fatal, and
    the fatal ones (no cross-account permission, a KMS key we cannot use) fail identically
    three times, which is itself a useful signal in the log."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            current_delay = delay
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries:
                        print(f"{func.__name__} failed after {max_retries + 1} attempts")
                        raise
                    print(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                    print(f"Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
            raise last_exception

        return wrapper

    return decorator


def _run(args: list[str]) -> subprocess.CompletedProcess:
    print("Command: " + " ".join(args))
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)  # nosec
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or "No stderr"
        print(f"AWS command failed (rc={proc.returncode}): {stderr}")
        raise subprocess.CalledProcessError(
            proc.returncode, args, output=proc.stdout, stderr=proc.stderr
        )
    if proc.stdout.strip():
        print(proc.stdout.strip()[-4000:])
    return proc


@retry_on_failure()
def sync_s3_to_local(*, s3_uri: str, dest_path: str, delete: bool = False) -> None:
    """Pull a prefix down to local disk.

    ``delete=False`` by default: a partially-synced dataset should be completed rather than
    cleared, and nothing local under a dataset version is ours to remove.
    """
    args = ["aws", "s3", "sync", ensure_suffix(s3_uri, "/"), ensure_suffix(dest_path, "/")]
    if delete:
        args.append("--delete")
    _run(args)


@retry_on_failure()
def sync_local_to_s3(*, src_path: str, s3_uri: str) -> None:
    """Push a directory up. Used for a finished run and for a freshly built dataset."""
    _run(["aws", "s3", "sync", ensure_suffix(src_path, "/"), ensure_suffix(s3_uri, "/")])


class S3Unavailable(RuntimeError):
    """S3 could not be asked the question -- no bucket configured, no credentials, no access.

    Distinct from "the prefix is not there", which is a legitimate ``False``. Conflating the two
    is how a missing ``AWS_DEFAULT_BUCKET`` turns into "dataset v003 does not exist", which sends
    whoever hit it looking in the wrong place entirely.
    """


def s3_prefix_exists(*, bucket: str, prefix: str) -> bool:
    """Whether anything exists under a prefix.

    Used before a build to refuse silently overwriting an existing dataset version, and before a
    training run to say "that dataset version is not in the bucket" as a 400 rather than as a
    GPU task that starts, syncs nothing and trains on an empty directory.

    **Not retried, unlike the syncs.** This runs inside a web request, and the failures it
    actually sees are configuration -- an unset bucket, absent credentials, a policy that does
    not allow the list. Those fail identically four times while the caller waits twenty-one
    seconds for an answer that was available immediately.
    """
    if not bucket:
        raise S3Unavailable(
            "AWS_DEFAULT_BUCKET is not set, so there is no bucket to look in. Set it in .env "
            "(see .env.example), or run with RUN_EXECUTOR=local against a dataset already on "
            "disk."
        )

    proc = subprocess.run(  # nosec
        [
            "aws",
            "s3api",
            "list-objects-v2",
            "--bucket",
            bucket,
            "--prefix",
            prefix,
            "--max-items",
            "1",
            "--output",
            "json",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        # The CLI's own stderr, not the exit code. "Unable to locate credentials" tells you what
        # to do; "returned non-zero exit status 252" tells you nothing.
        raise S3Unavailable(
            f"could not list s3://{bucket}/{prefix}: {proc.stderr.strip() or 'no stderr'}"
        )
    return '"Contents"' in proc.stdout


@retry_on_failure()
def copy_local_file_to_s3(*, src_path: str, s3_uri: str) -> None:
    """Put one file at one key.

    Separate from :func:`sync_local_to_s3` because syncing a directory to upload two files in
    it pushes everything else in that directory too -- which, for a runs root, means re-
    uploading every checkpoint already in the bucket.
    """
    _run(["aws", "s3", "cp", src_path, s3_uri])
