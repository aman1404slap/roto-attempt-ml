"""Score finished runs into the frozen table.

Pulls the seeds' run directories and the dataset down from the storage root, scores them with
the existing code, and puts the table and summary back beside the runs.

    python manage.py score_run --rung s3a --seeds 1 2 --dataset-version v003

**A run stamped ``local`` is refused here, and that refusal is not this command's doing.** It
comes from ``roto.v2.provenance`` by way of ``frozen_table``, so a table built by any route --
this command, ``scripts/score_v2.py``, a notebook -- is governed by the same rule. ``--allow-
local`` exists to look at a smoke test knowing what it is; the table says so in its header.
"""

import json
from pathlib import Path

from django.conf import settings
from django.core.management import BaseCommand, CommandError
from sentry_sdk import capture_exception, new_scope

from roto_app.helpers import storage
from roto_app.services import paths


class Command(BaseCommand):
    help = "Score one or more seeds of a rung into the frozen table."

    def add_arguments(self, parser):
        parser.add_argument("--rung", required=True, help="e.g. s3a")
        parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
        parser.add_argument("--dataset-version", default=None)
        parser.add_argument(
            "--environment",
            default=None,
            help="which prefix to read runs from (default: this environment)",
        )
        parser.add_argument(
            "--allow-local",
            action="store_true",
            help="table a local run. It is a smoke test, not a result.",
        )
        parser.add_argument("--lifespan", choices=("artist", "predicted"), default="artist")
        parser.add_argument("--point-count", choices=("artist", "predicted"), default="artist")
        parser.add_argument("--alive-on", type=float, default=0.5)
        parser.add_argument("--alive-off", type=float, default=0.2)
        parser.add_argument("--alive-min-gap", type=int, default=0)
        parser.add_argument("--no-upload", action="store_true")

    def handle(self, *args, **options):  # NOQA
        rung = options["rung"]
        version = options.get("dataset_version") or settings.DEFAULT_DATASET_VERSION
        environment = options.get("environment") or settings.ROTO_ENVIRONMENT

        dataset_dir = paths.local_dataset_dir(version)
        storage.sync_in(src=paths.dataset_uri(version), dest_path=dataset_dir)

        runs_root = Path(paths.local_runs_root())
        for seed in options["seeds"]:
            name = f"{rung}_seed{seed}"
            storage.sync_in(src=paths.run_uri(name, environment), dest_path=str(runs_root / name))

        try:
            from roto.v2.reconstruct import RebuildConfig
            from roto.v2.score import frozen_table, score_run

            cfg = RebuildConfig(
                lifespan=options["lifespan"],
                point_count=options["point_count"],
                alive_on=options["alive_on"],
                alive_off=options["alive_off"],
                alive_min_gap=options["alive_min_gap"],
            )

            scored = []
            for seed in options["seeds"]:
                checkpoint = runs_root / f"{rung}_seed{seed}" / "model.pt"
                if not checkpoint.exists():
                    print(f"  (no {checkpoint}, skipping seed {seed})")
                    continue
                scored.append(dict(score_run(checkpoint, dataset_dir, cfg, seed=seed), seed=seed))

            if not scored:
                raise CommandError(f"{rung}: nothing to score")

            try:
                table = frozen_table(
                    scored,
                    label=f"v2 {rung.upper()}",
                    allow_local=options.get("allow_local", False),
                )
            except ValueError as refusal:
                # A refusal is an answer, not a crash. It is also not a Sentry event: the
                # guarantee working as designed is the most ordinary thing that can happen here.
                raise CommandError(str(refusal)) from None
        except CommandError:
            raise
        except Exception as e:
            with new_scope() as scope:
                scope.set_tag("command", "score_run")
                scope.set_tag("rung", rung)
                capture_exception(e)
            raise

        print()
        print(table)

        summary_path = runs_root / f"{rung}_summary.json"
        table_path = runs_root / f"{rung}_table.txt"
        summary_path.write_text(json.dumps(scored, indent=2, default=float))
        table_path.write_text(table)

        if not options.get("no_upload"):
            destination = paths.runs_uri(environment)
            for path in (summary_path, table_path):
                storage.copy_file_out(src_path=str(path), dest=storage.join(destination, path.name))
            print(f"scores written beside the runs under {destination}")
