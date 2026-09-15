"""One training run, from the request that asked for it to the artifacts it left in S3.

**The five fields that make a run reproducible are columns, not payload.** The migration note
says it plainly: the run name, the dataset version, the seed, the step count and the
environment tag are what let a result be traced back to the configuration that produced it, and
they belong in the request rather than in the worker's defaults. Burying them in
``request_payload`` would make them unqueryable and, worse, optional.

Everything else about the run -- what the launcher was told, what the training loop reported,
where the artifacts went -- lands in the JSON columns, which follow the sibling services so the
admin and the callback shape are the ones people already read.
"""

from django.db import models
from django.utils import timezone

from roto_app.helpers.utils import time_difference_in_seconds
from roto_app.models.base_model import BaseModel

RUN_STATUS_SCHEDULED = "SCHEDULED"
RUN_STATUS_IN_PROGRESS = "IN_PROGRESS"
RUN_STATUS_COMPLETED = "COMPLETED"
RUN_STATUS_FAILED = "FAILED"
RUN_STATUS_STOPPED = "STOPPED"

RUN_STAGE_SCHEDULED = "SCHEDULED"
RUN_STAGE_LAUNCHING = "LAUNCHING"
RUN_STAGE_SYNCING_DATASET = "SYNCING_DATASET"
RUN_STAGE_TRAINING = "TRAINING"
RUN_STAGE_SYNCING_RESULTS = "SYNCING_RESULTS"
RUN_STAGE_COMPLETED = "COMPLETED"

TERMINAL_STATUSES = (RUN_STATUS_COMPLETED, RUN_STATUS_FAILED, RUN_STATUS_STOPPED)

ENVIRONMENT_LOCAL = "local"
ENVIRONMENT_STAGING = "staging"
ENVIRONMENTS = (ENVIRONMENT_LOCAL, ENVIRONMENT_STAGING)


class Run(BaseModel):
    """A single seed of a single rung against a single dataset build."""

    reference_id = models.CharField(max_length=255, db_index=True)
    """The caller's own id for this run, so a result can be matched back to whatever asked for
    it. Defaults to ``<rung>_seed<seed>`` when the caller has nothing of its own."""

    rung = models.CharField(max_length=64, db_index=True)
    """Which named configuration in ``roto.v2.rungs`` -- ``s3a`` and so on. Not free-form
    hyperparameters: a run has to name a rung that exists, because a configuration assembled
    per request is one nothing can be compared against."""

    dataset_version = models.CharField(max_length=64, db_index=True)
    seed = models.IntegerField(default=1)
    steps = models.IntegerField(null=True, blank=True)
    """``None`` means the rung's own schedule -- 40k. A shorter one is a probe, not a result,
    and the run log records which it was."""

    environment = models.CharField(max_length=32, db_index=True, default=ENVIRONMENT_LOCAL)
    """``local`` or ``staging``. Decides the S3 prefix the artifacts land under and is stamped
    onto the checkpoint, where ``roto.v2.provenance`` uses it to keep a smoke test out of a
    results table."""

    stage = models.CharField(max_length=255, db_index=True, default=RUN_STAGE_SCHEDULED)
    status = models.CharField(max_length=128, db_index=True, default=RUN_STATUS_SCHEDULED)

    executor = models.CharField(max_length=32, default="ecs")
    ecs_cluster = models.CharField(max_length=255, blank=True, default="")
    ecs_task_arn = models.CharField(max_length=512, blank=True, default="", db_index=True)
    """What ``ecs:DescribeTasks`` and ``ecs:StopTask`` are called with. Blank for a local run,
    which has a pid in ``result_data`` instead."""

    execution_started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    request_payload = models.JSONField(null=False, blank=True, default=dict)
    result_data = models.JSONField(null=False, blank=True, default=dict)
    error_logs = models.JSONField(null=False, blank=True, default=dict)
    time_logs = models.JSONField(null=False, blank=True, default=dict)
    run_logs = models.JSONField(null=False, blank=True, default=dict)

    __REPR__ = ("id", "rung", "seed", "environment", "status")

    class Meta:
        ordering = ("-created",)

    @property
    def name(self) -> str:
        """``s3a_seed1`` -- the run directory name, unchanged from the shell convention."""
        return f"{self.rung}_seed{self.seed}"

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def as_api_dict(self) -> dict:
        """What ``GET /api/runs/<id>`` returns. Flat, and every reproducibility field present."""
        return {
            "id": str(self.external_id),
            "reference_id": self.reference_id,
            "rung": self.rung,
            "dataset_version": self.dataset_version,
            "seed": self.seed,
            "steps": self.steps,
            "environment": self.environment,
            "status": self.status,
            "stage": self.stage,
            "executor": self.executor,
            "ecs_task_arn": self.ecs_task_arn or None,
            "created": self.created.isoformat() if self.created else None,
            "execution_started_at": (
                self.execution_started_at.isoformat() if self.execution_started_at else None
            ),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result_data": self.result_data,
            "error_logs": self.error_logs,
            "time_logs": self.time_logs,
        }

    # -- stage logging, same shape as the sibling services ---------------------------------

    def start_stage_logging(self, message, time_log_key=None, new_stage=None):
        """Open a stage and record when it started. Returns the start time."""
        start_time = timezone.now()

        if new_stage is not None:
            self.stage = new_stage
        if time_log_key is None:
            time_log_key = f"{self.stage.upper()}_STARTED_AT"

        if not isinstance(self.time_logs, dict):
            self.time_logs = {}
        if not isinstance(self.run_logs, dict):
            self.run_logs = {}

        self.time_logs[time_log_key] = start_time.isoformat()
        self.run_logs[self.stage.upper()] = message
        self.save(update_fields=["time_logs", "run_logs", "stage"])
        print(f"[{self.name}] {self.stage}: {message}")
        return start_time

    def update_stage_logging(self, start_time, message_template, time_log_key=None, new_stage=None):
        """Close a stage with how long it took. ``message_template`` takes ``{completion_time}``."""
        current_time = timezone.now()
        completion_time = time_difference_in_seconds(current_time, start_time)

        if time_log_key is None:
            time_log_key = f"{self.stage.upper()}_COMPLETED_AT"

        if not isinstance(self.time_logs, dict):
            self.time_logs = {}
        if not isinstance(self.run_logs, dict):
            self.run_logs = {}

        self.time_logs[time_log_key] = current_time.isoformat()
        self.run_logs[self.stage.upper()] = message_template.format(completion_time=completion_time)

        if new_stage is not None:
            self.stage = new_stage

        self.save(update_fields=["time_logs", "run_logs", "stage"])
        print(f"[{self.name}] {self.stage}: {self.run_logs[self.stage.upper()]}")
        return current_time

    def fail(self, exc: Exception) -> None:
        """Record a failure without losing what the run had already logged."""
        self.status = RUN_STATUS_FAILED
        errors = self.error_logs if isinstance(self.error_logs, dict) else {}
        errors["ERROR_MESSAGE"] = f"{exc}"
        self.error_logs = errors
        self.completed_at = timezone.now()
        self.save(update_fields=["status", "error_logs", "completed_at"])
