"""Run one training job. This is what the ECS task's command override runs.

It takes a run id and nothing else. Every parameter of the run -- the rung, the dataset
version, the seed, the step count, the environment -- is on the row, put there by the request
that created it. Passing them on the command line as well would give the run two sources of
truth and no way to tell which one produced the checkpoint.

    python manage.py train_run --run-id <uuid>

``--create`` is the shell escape hatch: it makes the row first, so a run can be started
without the API being up. It is how you drive a smoke test on a laptop.

    python manage.py train_run --create --rung s3a --seed 1 --dataset-version v003
"""

import sys

from django.conf import settings
from django.core.management import BaseCommand
from sentry_sdk import capture_exception, new_scope

from roto_app.models.run import Run
from roto_app.services import train_service


class Command(BaseCommand):
    help = "Train one rung, one seed, syncing the dataset in and the results out."

    def add_arguments(self, parser):
        parser.add_argument("--run-id", help="external_id of an existing Run row")
        parser.add_argument(
            "--create",
            action="store_true",
            help="create the run row first, for driving a run without the API",
        )
        parser.add_argument("--rung", default=None, help="with --create: which rung, e.g. s3a")
        parser.add_argument("--seed", type=int, default=1, help="with --create")
        parser.add_argument(
            "--steps",
            type=int,
            default=None,
            help="with --create: override the rung's schedule. A shorter "
            "schedule is a probe, not a result.",
        )
        parser.add_argument("--dataset-version", default=None, help="with --create")
        parser.add_argument("--reference-id", default=None, help="with --create")

    def handle(self, *args, **options):  # NOQA
        run = self._resolve(options)
        print(f"train_run: {run.name} ({run.environment}) run_id={run.external_id}")

        try:
            train_service.execute(run)
        except Exception as e:
            print(f"train_run failed for {run.name}: {e}")
            with new_scope() as scope:
                scope.set_tag("command", "train_run")
                scope.set_tag("rung", run.rung)
                scope.set_tag("environment", run.environment)
                scope.set_extra("run_id", str(run.external_id))
                capture_exception(e)
            run.fail(e)
            # Non-zero so the ECS task stops as failed and the exit code is visible in
            # DescribeTasks, not only in the row.
            sys.exit(1)

        print(f"train_run: {run.name} completed")

    def _resolve(self, options) -> Run:
        from roto.v2.rungs import DEFAULT_RUN, RUNS

        if options.get("run_id"):
            return Run.objects.get(external_id=options["run_id"])

        if not options.get("create"):
            raise SystemExit("pass --run-id, or --create to make the row here")

        rung = options.get("rung") or DEFAULT_RUN
        if rung not in RUNS:
            raise SystemExit(f"unknown rung {rung!r}; known rungs are {sorted(RUNS)}")

        seed = options.get("seed") or 1
        return Run.objects.create(
            reference_id=options.get("reference_id") or f"{rung}_seed{seed}",
            rung=rung,
            dataset_version=options.get("dataset_version") or settings.DEFAULT_DATASET_VERSION,
            seed=seed,
            steps=options.get("steps"),
            environment=settings.ROTO_ENVIRONMENT,
            executor="local",
            request_payload={"source": "manage.py train_run --create"},
        )
