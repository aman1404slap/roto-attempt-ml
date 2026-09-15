"""Start a run, on a GPU task or on this machine.

**A local run is a staging run with a smaller argument list, not a different code path.** Both
executors launch the same Django management command with the same arguments; the only thing
that differs is what runs it. That is the whole point of routing local through staging's
bucket too, and it is what makes a local smoke test worth anything -- if local ran different
code, "it worked locally" would say nothing about whether it will work on ECS.

What local is *for* is stated once, here, so it does not have to be remembered: it answers
"does this code run on a GPU without crashing". It never produces a number that gets quoted.
"""

import shlex
import subprocess  # nosec
import sys
from pathlib import Path

from django.conf import settings

from roto_app.helpers import ecs
from roto_app.models.run import RUN_STAGE_LAUNCHING, RUN_STATUS_FAILED

EXECUTOR_ECS = "ecs"
EXECUTOR_LOCAL = "local"


def build_command(run) -> list[str]:
    """The argv both executors run. Every reproducibility field is explicit on it.

    Nothing is left to the worker's defaults: the rung, the dataset version, the seed, the
    step count and the environment all appear here, so the command in the ECS console and the
    row in the database say the same thing and neither has to be trusted over the other.
    """
    command = [
        "python3",
        "./manage.py",
        "train_run",
        "--run-id",
        str(run.external_id),
    ]
    return command


def task_environment(run) -> dict:
    """Variables the task needs that are provenance rather than configuration.

    These are deliberately not command arguments: ``roto.v2.provenance`` reads the environment
    from the process, so that a run launched by anything -- this service, a shell, a future
    scheduler -- is stamped the same way and cannot claim to be staging by passing a flag.
    """
    return {
        "ROTO_ENVIRONMENT": run.environment,
        "ROTO_RUN_ID": str(run.external_id),
    }


def launch(run) -> None:
    """Start ``run`` and record how. Raises on failure, with the run row already marked."""
    executor = run.executor or settings.RUN_EXECUTOR
    command = build_command(run)
    env = task_environment(run)

    run.start_stage_logging(
        message=f"launching on {executor}: {' '.join(shlex.quote(c) for c in command)}",
        time_log_key="LAUNCH_STARTED_AT",
        new_stage=RUN_STAGE_LAUNCHING,
    )

    try:
        if executor == EXECUTOR_ECS:
            task_arn = ecs.run_task(
                command=command, environment=env, started_by=f"roto-{run.external_id}"
            )
            run.ecs_task_arn = task_arn
            run.ecs_cluster = settings.ECS_RUN_TASK_CLUSTER_NAME
            run.save(update_fields=["ecs_task_arn", "ecs_cluster"])
            print(f"[{run.name}] launched as {task_arn}")
        else:
            pid = _launch_local(command, env)
            result = dict(run.result_data or {})
            result["local_pid"] = pid
            run.result_data = result
            run.save(update_fields=["result_data"])
            print(f"[{run.name}] launched locally as pid {pid}")
    except Exception as e:
        run.status = RUN_STATUS_FAILED
        errors = dict(run.error_logs or {})
        errors["LAUNCH_ERROR"] = f"{e}"
        if getattr(e, "failures", None):
            errors["AWS_FAILURES"] = [dict(f) for f in e.failures]
        run.error_logs = errors
        run.save(update_fields=["status", "error_logs"])
        raise


def _launch_local(command: list[str], env: dict) -> int:
    """Detach the command from the web process.

    Detached on purpose: the run outlives the request that asked for it, exactly as the ECS
    task does. ``start_new_session`` keeps it alive when the dev server reloads, which it does
    on every file save and would otherwise kill a half-hour job mid-way.
    """
    import os

    argv = [sys.executable if command[0] == "python3" else command[0], *command[1:]]
    log_path = Path(settings.SCRATCH_DIR) / "local_runs.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a") as log:
        proc = subprocess.Popen(  # nosec
            argv,
            cwd=str(Path(settings.BASE_DIR)),
            env={**os.environ, **{k: str(v) for k, v in env.items()}},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"local run logging to {log_path}")
    return proc.pid
