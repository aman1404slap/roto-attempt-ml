"""What a training task actually does, start to finish.

Four steps, and the middle one is the existing code called unchanged:

1. sync the dataset version down from the storage root to local disk
2. run ``roto.v2.train.train`` on plain local paths
3. sync the run directory back, under the prefix its environment owns
4. record what happened on the run row

**Step 2 imports torch, and nothing above it does.** The import is inside the function rather
than at module top so the web process -- which only ever launches runs and reads rows -- never
pays for it, and so this module stays importable on a machine with no CUDA. That is the same
lazy-heavy-import convention the sibling services use, for the same reason.

**The storage root may be a bucket or a folder on disk, and nothing here knows which.** That
is :mod:`roto_app.helpers.storage`'s job, and it is what lets this run before the
infrastructure exists without the code that will run on ECS being different code.

**Nothing here reaches into the archive.** Training needs only the derived dataset; the raw
``.sfx`` and EXRs are required to *build* it and never after, which is why the GPU task never
holds client footage. Building is a separate, CPU-only command.
"""

from django.conf import settings
from django.utils import timezone

from roto_app.helpers import storage
from roto_app.helpers.utils import free_disk_gb
from roto_app.models.run import (
    RUN_STAGE_COMPLETED,
    RUN_STAGE_SYNCING_DATASET,
    RUN_STAGE_SYNCING_RESULTS,
    RUN_STAGE_TRAINING,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_IN_PROGRESS,
)
from roto_app.services import paths


def execute(run) -> dict:
    """Run one seed of one rung. Returns the training summary."""
    started = timezone.now()
    run.status = RUN_STATUS_IN_PROGRESS
    run.execution_started_at = started
    run.save(update_fields=["status", "execution_started_at"])
    run.start_stage_logging("Run started.", time_log_key="RUN_STARTED_AT")

    print(f"[{run.name}] scratch={settings.SCRATCH_DIR} free={free_disk_gb('/')} GB")

    dataset_dir = _sync_dataset(run)
    summary = _train(run, dataset_dir)
    artifacts = _sync_results(run)

    run.refresh_from_db()
    run.stage = RUN_STAGE_COMPLETED
    run.status = RUN_STATUS_COMPLETED
    run.completed_at = timezone.now()
    run.result_data = {
        **(run.result_data or {}),
        "artifacts_uri": artifacts,
        "provenance": summary.get("provenance"),
        # The headline cost numbers, on the row rather than only in the synced train_log.
        # These are what the capacity sizing in the migration note is built from, and reading
        # them should not require fetching an object out of S3.
        "n_params": summary.get("n_params"),
        "steps": summary.get("steps"),
        "layers": summary.get("layers"),
        "wall_clock_s": summary.get("wall_clock_s"),
        "steps_per_s": summary.get("steps_per_s"),
        "peak_gpu_mb": summary.get("peak_gpu_mb"),
        "device_name": summary.get("device_name"),
        "final": summary.get("final"),
    }
    run.save(update_fields=["stage", "status", "completed_at", "result_data"])
    run.update_stage_logging(
        start_time=started,
        message_template="Run completed in {completion_time} seconds.",
        time_log_key="RUN_COMPLETED_AT",
    )
    return summary


def _sync_dataset(run) -> str:
    """Bring the derived dataset down. The training code then sees an ordinary local path."""
    t = run.start_stage_logging(
        f"Syncing dataset {run.dataset_version}.",
        time_log_key="DATASET_SYNC_STARTED_AT",
        new_stage=RUN_STAGE_SYNCING_DATASET,
    )
    dataset_dir = paths.local_dataset_dir(run.dataset_version)
    storage.sync_in(src=paths.dataset_uri(run.dataset_version), dest_path=dataset_dir)
    run.update_stage_logging(
        start_time=t,
        message_template="Dataset synced in {completion_time} seconds.",
        time_log_key="DATASET_SYNC_COMPLETED_AT",
    )
    return dataset_dir


def _train(run, dataset_dir: str) -> dict:
    """The existing training loop, on local paths, with the rung's own configuration."""
    t = run.start_stage_logging(
        f"Training {run.rung} seed {run.seed}.",
        time_log_key="TRAINING_STARTED_AT",
        new_stage=RUN_STAGE_TRAINING,
    )

    # Heavy imports stay inside the function: the web process never pays for torch.
    from dataclasses import replace

    from roto.v2.rungs import RUNS
    from roto.v2.train import train

    if run.rung not in RUNS:
        raise ValueError(f"unknown rung {run.rung!r}; known rungs are {sorted(RUNS)}")

    cfg = replace(RUNS[run.rung], seed=run.seed)
    if run.steps:
        cfg = replace(cfg, steps=run.steps)

    out_dir = paths.local_run_dir(run.name)
    summary = train(dataset_dir, out_dir, cfg)

    run.update_stage_logging(
        start_time=t,
        message_template="Training finished in {completion_time} seconds.",
        time_log_key="TRAINING_COMPLETED_AT",
    )
    return summary


def _sync_results(run) -> str:
    """Push the run directory to the prefix this environment owns."""
    t = run.start_stage_logging(
        "Syncing results.",
        time_log_key="RESULT_SYNC_STARTED_AT",
        new_stage=RUN_STAGE_SYNCING_RESULTS,
    )
    uri = paths.run_uri(run.name, run.environment)
    storage.sync_out(src_path=paths.local_run_dir(run.name), dest=uri)
    run.update_stage_logging(
        start_time=t,
        message_template="Results synced in {completion_time} seconds.",
        time_log_key="RESULT_SYNC_COMPLETED_AT",
    )
    return uri
