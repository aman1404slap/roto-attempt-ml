"""Build a dataset version from the archive and put it in our bucket.

**This is the only job that reads the archive, and it needs no GPU.** Training reads the
derived dataset and nothing else, so keeping the build separate is what lets the GPU task run
without ever holding client footage. On a large archive the build is a substantial job in its
own right and wants to be its own queued task rather than a step inside a training job.

    python manage.py build_dataset --version v003 --tier tier1

The output is compared against what it replaces rather than trusted: rebuilding an existing
version is refused unless ``--overwrite`` is passed, because a dataset version is the unit
every published number is keyed to, and silently replacing one makes every result that cites
it wrong in a way nothing would report.
"""

from django.conf import settings
from django.core.management import BaseCommand
from sentry_sdk import capture_exception, new_scope

from roto_app.helpers import aws_helpers
from roto_app.helpers.utils import free_disk_gb
from roto_app.services import paths


class Command(BaseCommand):
    help = "Sync the archive down, build a dataset version, sync it to our bucket."

    def add_arguments(self, parser):
        parser.add_argument(
            "--version",
            default=None,
            help="dataset version, e.g. v003 (default: DEFAULT_DATASET_VERSION)",
        )
        parser.add_argument("--tier", default="tier1", choices=("tier1", "tier2"))
        parser.add_argument(
            "--shots", nargs="+", default=None, help="explicit shot names, overriding --tier"
        )
        parser.add_argument("--all", action="store_true", help="every shot in the archive")
        parser.add_argument("--size", type=int, default=256)
        parser.add_argument("--stride", type=int, default=1)
        parser.add_argument(
            "--no-qc-gate",
            action="store_true",
            help="record pairing QC failures instead of dropping the shot",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="replace an existing dataset version in the bucket",
        )
        parser.add_argument(
            "--skip-source-sync", action="store_true", help="the archive is already on local disk"
        )
        parser.add_argument(
            "--no-upload",
            action="store_true",
            help="build locally and stop; nothing is written to the bucket",
        )

    def handle(self, *args, **options):  # NOQA
        version = options.get("version") or settings.DEFAULT_DATASET_VERSION
        upload = not options.get("no_upload")

        if upload and not options.get("overwrite"):
            if aws_helpers.s3_prefix_exists(
                bucket=settings.AWS_DEFAULT_BUCKET, prefix=paths.dataset_key(version)
            ):
                raise SystemExit(
                    f"{paths.dataset_uri(version)} already exists. A dataset version is what "
                    "every published number is keyed to -- build a new version, or pass "
                    "--overwrite if you really mean to replace this one."
                )

        data_root = paths.local_data_root()
        if not options.get("skip_source_sync"):
            print(f"syncing archive from {paths.source_uri()} (read-only, other account)")
            print(f"free disk before sync: {free_disk_gb('/')} GB")
            aws_helpers.sync_s3_to_local(s3_uri=paths.source_uri(), dest_path=data_root)

        out_root = paths.local_dataset_dir(version)

        try:
            # Heavy imports inside the handler: this module is imported by Django's command
            # discovery on every manage.py call, including ones with no numpy present.
            from roto.v2.build import v2_crop_config
            from roto.v2.dataset import build_dataset
            from roto.v2.subset import TIER1, TIER2

            shots = None
            if options.get("shots"):
                shots = tuple(options["shots"])
            elif not options.get("all"):
                shots = TIER1 if options["tier"] == "tier1" else TIER1 + TIER2

            manifest = build_dataset(
                data_root,
                out_root,
                shots,
                cfg=v2_crop_config(size=options["size"], stride=options["stride"]),
                qc_gates=not options.get("no_qc_gate"),
            )
        except Exception as e:
            with new_scope() as scope:
                scope.set_tag("command", "build_dataset")
                scope.set_tag("dataset_version", version)
                capture_exception(e)
            raise

        counts = manifest["counts"]
        print(
            f"{out_root}: {counts['shots']} shots, {counts['elements']} elements "
            f"({counts['gold']} gold, {counts['silver']} silver), {counts['shapes']} shapes, "
            f"{counts['frames']} element-frames"
        )
        if manifest["shots_failed_qc"]:
            print(f"  QC failed: {manifest['shots_failed_qc']}")

        if upload:
            uri = paths.dataset_uri(version)
            print(f"uploading to {uri}")
            aws_helpers.sync_local_to_s3(src_path=out_root, s3_uri=uri)
            print(f"dataset {version} available at {uri}")
        else:
            print(f"--no-upload: dataset left at {out_root}")
