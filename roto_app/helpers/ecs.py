"""Launching, watching and stopping a GPU task on ECS.

A 30-minute training job cannot run inside a web request, so the API is necessarily two
pieces: something that accepts a job and something that runs it. This is the join between
them. ``POST /api/runs`` calls :func:`run_task` and returns; the task runs the same Django
management command the local executor runs, and the only difference between the two
environments is which of them launched it.

The three IAM actions this needs -- ``ecs:RunTask``, ``ecs:DescribeTasks``, ``ecs:StopTask``,
plus ``iam:PassRole`` -- are exactly what ``devops-requirements.md`` asks for.
"""

import boto3
from django.conf import settings


class ECSLaunchError(RuntimeError):
    """A task that could not be placed. Carries the AWS failures so the run row can keep them."""

    def __init__(self, message: str, failures=None):
        super().__init__(message)
        self.failures = failures or []


CAPACITY_FAILURES = (
    "RESOURCE:MEMORY",
    "RESOURCE:CPU",
    "RESOURCE:GPU",
    "RESOURCE:AGENT",
    "MemberOf placement constraint unsatisfied.",
)
"""Failures that mean "no GPU instance is up yet", not "this task is wrong".

Worth separating, because a cluster that scales to zero between runs -- which is what we want,
usage is a few jobs a week -- answers the first request after an idle period with exactly
these while capacity comes up. Retrying is right; rewriting the task definition is not.
"""


def _client():
    return boto3.client("ecs", region_name=settings.AWS_DEFAULT_REGION)


def run_task(*, command: list[str], environment: dict | None = None, started_by: str = ""):
    """Launch one GPU task running ``command``. Returns the task ARN.

    ``command`` overrides the container's, which is how one image serves every job: the same
    image builds a dataset, trains a rung or scores a run depending on the management command
    it is handed.
    """
    if not settings.ECS_RUN_TASK_DEFINITION:
        raise ECSLaunchError(
            "ECS_RUN_TASK_DEFINITION is not set. Set RUN_EXECUTOR=local to run on this "
            "machine instead, or point it at the task definition devops created."
        )

    overrides = {
        "name": settings.ECS_RUN_TASK_CONTAINER_NAME,
        "command": command,
        "resourceRequirements": [{"value": "1", "type": "GPU"}],
    }
    if environment:
        overrides["environment"] = [{"name": k, "value": str(v)} for k, v in environment.items()]

    kwargs = {
        "cluster": settings.ECS_RUN_TASK_CLUSTER_NAME,
        "taskDefinition": settings.ECS_RUN_TASK_DEFINITION,
        "launchType": "EC2",
        "overrides": {"containerOverrides": [overrides]},
        "enableExecuteCommand": True,
    }
    if started_by:
        kwargs["startedBy"] = started_by[:36]
    if settings.ECS_RUN_TASK_SUBNET_IDS:
        kwargs["networkConfiguration"] = {
            "awsvpcConfiguration": {
                "subnets": settings.ECS_RUN_TASK_SUBNET_IDS.split(","),
                "securityGroups": [
                    s for s in settings.ECS_RUN_TASK_SECURITY_GROUP_IDS.split(",") if s
                ],
                "assignPublicIp": "DISABLED",
            }
        }

    response = _client().run_task(**kwargs)

    failures = response.get("failures") or []
    if failures and not response.get("tasks"):
        capacity = all(f.get("reason") in CAPACITY_FAILURES for f in failures)
        raise ECSLaunchError(
            (
                "No GPU capacity available to place the task; the cluster may still be scaling up."
                if capacity
                else f"ECS refused the task: {failures}"
            ),
            failures=failures,
        )
    if failures:
        print(f"run_task succeeded with failures attached: {failures}")

    return response["tasks"][0]["taskArn"]


def describe_task(task_arn: str) -> dict:
    """Last status, stop reason and exit code for one task.

    ``GET /api/runs/<id>`` uses this only while the run row is not yet terminal: once the task
    itself has written its outcome to the row, ECS is the less informative of the two, and
    tasks age out of ``DescribeTasks`` within hours anyway.
    """
    response = _client().describe_tasks(
        cluster=settings.ECS_RUN_TASK_CLUSTER_NAME, tasks=[task_arn]
    )
    tasks = response.get("tasks") or []
    if not tasks:
        return {"lastStatus": "UNKNOWN", "failures": response.get("failures") or []}
    task = tasks[0]
    containers = task.get("containers") or [{}]
    return {
        "lastStatus": task.get("lastStatus"),
        "desiredStatus": task.get("desiredStatus"),
        "stoppedReason": task.get("stoppedReason"),
        "exitCode": containers[0].get("exitCode"),
        "startedAt": task.get("startedAt").isoformat() if task.get("startedAt") else None,
        "stoppedAt": task.get("stoppedAt").isoformat() if task.get("stoppedAt") else None,
    }


def stop_task(task_arn: str, reason: str = "stopped via API") -> dict:
    """Stop a running task.

    Note what this costs today: training writes its checkpoint once, after the loop, so a
    stopped run loses everything it has done. Mid-run checkpointing (migration note 5.2) is
    what makes this a pause rather than a discard, and it is not in yet.
    """
    return _client().stop_task(
        cluster=settings.ECS_RUN_TASK_CLUSTER_NAME, task=task_arn, reason=reason[:255]
    )
