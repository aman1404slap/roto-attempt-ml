"""The API's contract, and the two refusals that matter.

Run with Django's own runner rather than pytest::

    python manage.py test roto_app

Deliberately separate from ``tests/``, which is the science suite and must stay runnable with
no database, no settings module and no Django installed. Two suites because there are two
codebases here, and the boundary between them is the point.
"""

import json
from unittest import mock

from django.test import TestCase, override_settings

from roto_app.helpers.aws_helpers import S3Unavailable, s3_prefix_exists
from roto_app.models.run import Run

API_KEY = "test-key"
HEADERS = {"HTTP_X_TASK_API_KEY": API_KEY}


@override_settings(
    ROTO_APP_TASK_API_KEY=API_KEY,
    AWS_DEFAULT_BUCKET="test-bucket",
    DEFAULT_DATASET_VERSION="v003",
    ROTO_ENVIRONMENT="staging",
    RUN_EXECUTOR="ecs",
)
class CreateRunTests(TestCase):
    def setUp(self):
        self.exists = mock.patch("roto_app.views.views.s3_prefix_exists", return_value=True).start()
        self.launch = mock.patch("roto_app.services.launcher.launch").start()
        self.addCleanup(mock.patch.stopall)

    def post(self, payload, **extra):
        return self.client.post(
            "/api/runs", data=json.dumps(payload), content_type="application/json", **extra
        )

    def test_requires_an_api_key(self):
        """This endpoint starts GPU instances. Unauthenticated, it is a way to spend money."""
        response = self.post({"rung": "s3a"})
        self.assertEqual(response.status_code, 401)
        self.assertFalse(Run.objects.exists())

    def test_creates_and_launches_a_run(self):
        response = self.post({"rung": "s3a", "seed": 2, "dataset_version": "v003"}, **HEADERS)
        self.assertEqual(response.status_code, 200)

        run = Run.objects.get()
        self.assertEqual(run.rung, "s3a")
        self.assertEqual(run.seed, 2)
        self.assertEqual(run.name, "s3a_seed2")
        self.assertEqual(run.environment, "staging")
        self.launch.assert_called_once_with(run)

    def test_defaults_to_the_live_rung_and_configured_dataset(self):
        """A request that names nothing gets s3a on the default build -- not an S0 re-run."""
        self.post({}, **HEADERS)
        run = Run.objects.get()
        self.assertEqual(run.rung, "s3a")
        self.assertEqual(run.dataset_version, "v003")

    def test_rejects_an_unknown_rung(self):
        """A rung is a named configuration, not free-form hyperparameters. One assembled per
        request is one nothing can be compared against."""
        response = self.post({"rung": "s9z"}, **HEADERS)
        self.assertEqual(response.status_code, 400)
        self.assertIn("known_rungs", response.json()["data"])
        self.assertFalse(Run.objects.exists())

    def test_rejects_an_unknown_environment(self):
        response = self.post({"environment": "prod"}, **HEADERS)
        self.assertEqual(response.status_code, 400)
        self.assertFalse(Run.objects.exists())

    def test_an_unreachable_s3_is_503_not_400(self):
        """ "Not there" and "could not ask" are different answers.

        Reporting an unset bucket or absent credentials as "dataset v003 does not exist" sends
        whoever hit it looking in the wrong place. 503 says the fault is ours.
        """
        self.exists.side_effect = S3Unavailable("AWS_DEFAULT_BUCKET is not set")
        response = self.post({}, **HEADERS)
        self.assertEqual(response.status_code, 503)
        self.assertIn("AWS_DEFAULT_BUCKET", response.json()["data"]["errors"])
        self.assertFalse(Run.objects.exists())
        self.launch.assert_not_called()

    def test_rejects_a_dataset_that_is_not_in_the_bucket(self):
        """Checked before launching. A typo'd version otherwise costs a GPU task that starts,
        syncs an empty prefix and fails slowly, looking like a code problem."""
        self.exists.return_value = False
        response = self.post({"dataset_version": "v999"}, **HEADERS)
        self.assertEqual(response.status_code, 400)
        self.assertFalse(Run.objects.exists())
        self.launch.assert_not_called()

    def test_the_five_reproducibility_fields_survive_the_round_trip(self):
        """Rung, dataset version, seed, steps and environment -- the note's five."""
        self.post(
            {
                "rung": "s2c",
                "dataset_version": "v003",
                "seed": 3,
                "steps": 4000,
                "environment": "local",
            },
            **HEADERS,
        )
        data = Run.objects.get().as_api_dict()
        self.assertEqual(
            (
                data["rung"],
                data["dataset_version"],
                data["seed"],
                data["steps"],
                data["environment"],
            ),
            ("s2c", "v003", 3, 4000, "local"),
        )


@override_settings(ROTO_APP_TASK_API_KEY=API_KEY, ECS_RUN_TASK_CLUSTER_NAME="roto")
class RunStatusTests(TestCase):
    def setUp(self):
        self.run = Run.objects.create(
            reference_id="r",
            rung="s3a",
            dataset_version="v003",
            seed=1,
            environment="staging",
            executor="ecs",
            ecs_task_arn="arn:aws:ecs:task/abc",
        )

    def test_status_requires_an_api_key(self):
        self.assertEqual(self.client.get(f"/api/runs/{self.run.external_id}").status_code, 401)

    def test_unknown_run_is_404(self):
        response = self.client.get("/api/runs/00000000-0000-0000-0000-000000000000", **HEADERS)
        self.assertEqual(response.status_code, 404)

    @mock.patch("roto_app.helpers.ecs.describe_task", return_value={"lastStatus": "RUNNING"})
    def test_consults_ecs_while_the_run_is_live(self, describe):
        response = self.client.get(f"/api/runs/{self.run.external_id}", **HEADERS)
        self.assertEqual(response.json()["data"]["ecs"], {"lastStatus": "RUNNING"})
        describe.assert_called_once()

    @mock.patch("roto_app.helpers.ecs.describe_task")
    def test_does_not_consult_ecs_once_the_run_is_terminal(self, describe):
        """The row knows what stage the job reached and why it failed; ECS knows an exit code.
        Old tasks also age out of DescribeTasks, which would turn a completed run into UNKNOWN."""
        self.run.status = "COMPLETED"
        self.run.save(update_fields=["status"])
        response = self.client.get(f"/api/runs/{self.run.external_id}", **HEADERS)
        self.assertNotIn("ecs", response.json()["data"])
        describe.assert_not_called()

    @mock.patch("roto_app.helpers.ecs.describe_task", side_effect=RuntimeError("aged out"))
    def test_an_ecs_error_does_not_fail_the_status_call(self, _describe):
        response = self.client.get(f"/api/runs/{self.run.external_id}", **HEADERS)
        self.assertEqual(response.status_code, 200)
        self.assertIn("error", response.json()["data"]["ecs"])


@override_settings(ROTO_APP_TASK_API_KEY=API_KEY, ECS_RUN_TASK_CLUSTER_NAME="roto")
class StopRunTests(TestCase):
    def setUp(self):
        self.run = Run.objects.create(
            reference_id="r",
            rung="s3a",
            dataset_version="v003",
            seed=1,
            environment="staging",
            executor="ecs",
            ecs_task_arn="arn:aws:ecs:task/abc",
        )

    @mock.patch("roto_app.helpers.ecs.stop_task")
    def test_stops_a_running_task(self, stop):
        response = self.client.post(f"/api/runs/{self.run.external_id}/stop", **HEADERS)
        self.assertEqual(response.status_code, 200)
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, "STOPPED")
        stop.assert_called_once()

    @mock.patch("roto_app.helpers.ecs.stop_task")
    def test_will_not_stop_a_finished_run(self, stop):
        self.run.status = "COMPLETED"
        self.run.save(update_fields=["status"])
        self.client.post(f"/api/runs/{self.run.external_id}/stop", **HEADERS)
        stop.assert_not_called()

    @mock.patch("roto_app.helpers.ecs.stop_task")
    def test_a_local_run_has_no_task_to_stop(self, stop):
        self.run.ecs_task_arn = ""
        self.run.executor = "local"
        self.run.save(update_fields=["ecs_task_arn", "executor"])
        response = self.client.post(f"/api/runs/{self.run.external_id}/stop", **HEADERS)
        self.assertEqual(response.status_code, 400)
        stop.assert_not_called()


class HealthCheckTests(TestCase):
    def test_health_check_is_open(self):
        """The load balancer has no API key."""
        self.assertEqual(self.client.get("/health-check").status_code, 200)


class WebProcessStaysLightTests(TestCase):
    """The API validates and launches. It must never import the training stack to do it.

    Not a style point: if serving a request pulls in torch, the web service needs the GPU
    image, and "validate the rung name" becomes a reason to carry CUDA in a container that
    only ever writes database rows. ``roto.v2.rungs`` imports ``roto.v2.config`` -- a
    dataclass and nothing else -- precisely so this stays true.
    """

    def test_importing_the_rung_catalogue_does_not_import_torch(self):
        import subprocess  # nosec
        import sys

        code = (
            "import sys; import roto.v2.rungs; "
            "print('torch' if 'torch' in sys.modules else 'clean')"
        )
        result = subprocess.run(  # nosec
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "clean")


class S3PreflightTests(TestCase):
    """The pre-flight check itself: it must not turn a configuration fault into a 'no'."""

    def test_an_unset_bucket_raises_rather_than_returning_false(self):
        with self.assertRaises(S3Unavailable) as caught:
            s3_prefix_exists(bucket="", prefix="datasets/v003")
        self.assertIn("AWS_DEFAULT_BUCKET", str(caught.exception))

    @mock.patch("roto_app.helpers.aws_helpers.subprocess.run")
    def test_a_cli_failure_surfaces_its_stderr(self, run):
        """`Unable to locate credentials` tells you what to do; `exit status 252` does not."""
        run.return_value = mock.Mock(
            returncode=252, stdout="", stderr="Unable to locate credentials"
        )
        with self.assertRaises(S3Unavailable) as caught:
            s3_prefix_exists(bucket="a-bucket", prefix="datasets/v003")
        self.assertIn("Unable to locate credentials", str(caught.exception))

    @mock.patch("roto_app.helpers.aws_helpers.subprocess.run")
    def test_it_is_not_retried(self, run):
        """It runs inside a web request, and its failures are configuration -- which fails
        identically four times while the caller waits twenty-one seconds."""
        run.return_value = mock.Mock(returncode=252, stdout="", stderr="boom")
        with self.assertRaises(S3Unavailable):
            s3_prefix_exists(bucket="a-bucket", prefix="datasets/v003")
        self.assertEqual(run.call_count, 1)

    @mock.patch("roto_app.helpers.aws_helpers.subprocess.run")
    def test_a_missing_prefix_is_a_plain_false(self, run):
        run.return_value = mock.Mock(returncode=0, stdout="{}", stderr="")
        self.assertFalse(s3_prefix_exists(bucket="a-bucket", prefix="datasets/v999"))
