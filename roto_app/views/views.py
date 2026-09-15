"""The API: trigger a run, ask how it is going, stop it.

Thin on purpose -- validate, create the row, hand off to the launcher. A 30-minute training
job cannot run inside a web request, so this half only ever accepts work and reports on it;
the other half is the management command the task runs.

    POST /api/runs              enqueue a run, returns its id
    GET  /api/runs              list recent runs
    GET  /api/runs/<id>         status
    POST /api/runs/<id>/stop    stop a running task
"""

import json

from django.conf import settings
from django.contrib.auth import get_user_model
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from roto_app.helpers import ecs
from roto_app.helpers.aws_helpers import s3_prefix_exists
from roto_app.helpers.utils import (
    STATUS_CODE_400,
    STATUS_CODE_401,
    STATUS_CODE_404,
    authenticate,
    format_response,
)
from roto_app.models.run import (
    ENVIRONMENTS,
    RUN_STATUS_STOPPED,
    Run,
)
from roto_app.services import launcher, paths


def health_check(request):
    return format_response("Success")


@csrf_exempt
@require_POST
def create_run(request):
    """Create and launch a run.

    Everything that makes the run reproducible has to arrive in the request: the rung, the
    dataset version, the seed, the step count and the environment. They are not the worker's
    defaults to choose, because a run whose parameters came from whatever the worker happened
    to be configured with cannot be reproduced from its own record.
    """
    try:
        authenticate(request)
    except PermissionError:
        return format_response("Invalid credentials", status_code=STATUS_CODE_401)

    try:
        payload = json.loads(request.body or "{}")
    except json.JSONDecodeError as e:
        return format_response("Invalid JSON", data={"errors": str(e)}, status_code=STATUS_CODE_400)

    try:
        from roto.v2.rungs import DEFAULT_RUN, RUNS

        rung = payload.get("rung") or DEFAULT_RUN
        if rung not in RUNS:
            return format_response(
                f"Unknown rung {rung!r}",
                status_code=STATUS_CODE_400,
                data={"known_rungs": sorted(RUNS)},
            )

        environment = payload.get("environment") or settings.ROTO_ENVIRONMENT
        if environment not in ENVIRONMENTS:
            return format_response(
                f"Unknown environment {environment!r}",
                status_code=STATUS_CODE_400,
                data={"known_environments": list(ENVIRONMENTS)},
            )

        seed = int(payload.get("seed", 1))
        steps = payload.get("steps")
        steps = int(steps) if steps else None
        version = payload.get("dataset_version") or settings.DEFAULT_DATASET_VERSION

        # Checked before anything is launched. Otherwise a typo'd version costs a GPU task
        # that starts, syncs an empty prefix and trains on nothing -- which fails slowly and
        # looks like a code problem.
        if not s3_prefix_exists(
            bucket=settings.AWS_DEFAULT_BUCKET, prefix=paths.dataset_key(version)
        ):
            return format_response(
                f"Dataset {version} is not in the bucket",
                status_code=STATUS_CODE_400,
                data={"expected": paths.dataset_uri(version)},
            )

        run = Run.objects.create(
            reference_id=payload.get("reference_id") or f"{rung}_seed{seed}",
            rung=rung,
            dataset_version=version,
            seed=seed,
            steps=steps,
            environment=environment,
            executor=payload.get("executor") or settings.RUN_EXECUTOR,
            request_payload=payload,
        )

        launcher.launch(run)
        return format_response("Run launched", data=run.as_api_dict())
    except Exception as e:
        return format_response("Failed!", data={"errors": str(e)}, status_code=STATUS_CODE_400)


@require_GET
def list_runs(request):
    try:
        authenticate(request)
    except PermissionError:
        return format_response("Invalid credentials", status_code=STATUS_CODE_401)

    runs = Run.objects.all()
    if request.GET.get("rung"):
        runs = runs.filter(rung=request.GET["rung"])
    if request.GET.get("environment"):
        runs = runs.filter(environment=request.GET["environment"])
    if request.GET.get("status"):
        runs = runs.filter(status=request.GET["status"])

    limit = min(int(request.GET.get("limit", 50)), 200)
    return format_response("Success", data=[r.as_api_dict() for r in runs[:limit]])


@require_GET
def run_status(request, run_id):
    """Status of one run, with ECS consulted only while the row is not yet terminal.

    Once the task has written its own outcome the row is the better source: it knows what
    stage the job reached and why it failed, and ECS only knows the container's exit code.
    Tasks also age out of ``DescribeTasks`` within hours, so asking about an old run would
    turn a completed run into an ``UNKNOWN``.
    """
    try:
        authenticate(request)
    except PermissionError:
        return format_response("Invalid credentials", status_code=STATUS_CODE_401)

    try:
        run = Run.objects.get(external_id=run_id)
    except (Run.DoesNotExist, ValueError, TypeError):
        return format_response("Run not found", status_code=STATUS_CODE_404)

    data = run.as_api_dict()
    if run.ecs_task_arn and not run.is_terminal:
        try:
            data["ecs"] = ecs.describe_task(run.ecs_task_arn)
        except Exception as e:
            # A task ECS no longer knows about is normal for an old run; the row still
            # answers the question, so this is reported and not raised.
            data["ecs"] = {"error": str(e)}
    return format_response("Success", data=data)


@csrf_exempt
@require_POST
def stop_run(request, run_id):
    """Stop a running task.

    Worth knowing what this currently costs: the checkpoint is written once, after the
    training loop, so stopping a run discards everything it has done. Mid-run checkpointing
    (migration note 5.2) is what turns this into a pause.
    """
    try:
        authenticate(request)
    except PermissionError:
        return format_response("Invalid credentials", status_code=STATUS_CODE_401)

    try:
        run = Run.objects.get(external_id=run_id)
    except (Run.DoesNotExist, ValueError, TypeError):
        return format_response("Run not found", status_code=STATUS_CODE_404)

    if run.is_terminal:
        return format_response(f"Run is already {run.status}", data=run.as_api_dict())
    if not run.ecs_task_arn:
        return format_response(
            "This run has no ECS task to stop; it was launched locally.",
            data=run.as_api_dict(),
            status_code=STATUS_CODE_400,
        )

    try:
        ecs.stop_task(run.ecs_task_arn, reason=f"stopped via API by request {run.external_id}")
    except Exception as e:
        return format_response(
            "Failed to stop task", data={"errors": str(e)}, status_code=STATUS_CODE_400
        )

    run.status = RUN_STATUS_STOPPED
    run.save(update_fields=["status"])
    return format_response("Run stopped", data=run.as_api_dict())


@csrf_exempt
@require_POST
def seed_data(request):
    try:
        authenticate(request)
    except PermissionError:
        return format_response("Invalid credentials", status_code=STATUS_CODE_401)

    User = get_user_model()
    if User.objects.exists():
        return format_response("Already exists!")

    domain = "slapshot.work" if settings.ENV in ("local", "staging") else "slapshot.ai"
    User.objects.create_superuser(  # nosec
        username="Admin", email=f"admin@{domain}", password="workville"
    )
    return format_response("Success")
