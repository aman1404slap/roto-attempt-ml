"""The storage layer: one interface, an S3 bucket or a folder, and the same layout in both.

The point of these is not that copying files works. It is that **the local backing and the S3
backing agree about layout**, because the whole reason local runs against a folder is that the
code which will run on ECS is the code that runs here. A local backing that put runs somewhere
else would make every local test a test of something we are not going to ship.
"""

from pathlib import Path
from unittest import mock

from django.test import TestCase, override_settings

from roto_app.helpers import storage
from roto_app.services import paths


class LayoutTests(TestCase):
    """Same keys under either root. This is the property the local mode exists to preserve."""

    @override_settings(STORAGE_ROOT="s3://roto-staging", BASE_DIR="/code", LOCAL_USER="aman")
    def test_s3_root_builds_bucket_uris(self):
        self.assertEqual(paths.dataset_uri("v003"), "s3://roto-staging/datasets/v003")
        self.assertEqual(
            paths.run_uri("s3a_seed1", "staging"), "s3://roto-staging/runs/staging/s3a_seed1"
        )
        self.assertEqual(
            paths.run_uri("s3a_seed1", "local"),
            "s3://roto-staging/runs/local/aman/s3a_seed1",
        )

    @override_settings(STORAGE_ROOT="abc", BASE_DIR="/code", LOCAL_USER="aman")
    def test_local_root_builds_the_same_keys_under_a_folder(self):
        self.assertEqual(paths.dataset_uri("v003"), "/code/abc/datasets/v003")
        self.assertEqual(paths.run_uri("s3a_seed1", "staging"), "/code/abc/runs/staging/s3a_seed1")
        self.assertEqual(paths.run_uri("s3a_seed1", "local"), "/code/abc/runs/local/aman/s3a_seed1")

    @override_settings(STORAGE_ROOT="abc", BASE_DIR="/code")
    def test_a_relative_root_resolves_against_the_repo_not_the_cwd(self):
        """A detached subprocess does not inherit the shell's cwd. "It wrote somewhere else" is
        a tedious thing to debug, so the root is anchored once."""
        self.assertTrue(paths.storage_root().startswith("/code/"))

    @override_settings(STORAGE_ROOT="")
    def test_an_unset_root_says_so(self):
        with self.assertRaises(ValueError) as caught:
            paths.storage_root()
        self.assertIn("STORAGE_ROOT", str(caught.exception))

    @override_settings(STORAGE_ROOT="s3://roto-staging", LOCAL_USER="aman")
    def test_local_runs_never_share_a_prefix_with_staging(self):
        """The first of the two fences. The second is the stamp inside the checkpoint."""
        self.assertNotEqual(
            paths.run_key("s3a_seed1", "local"), paths.run_key("s3a_seed1", "staging")
        )
        self.assertTrue(paths.run_key("s3a_seed1", "local").startswith("runs/local/aman/"))


class LocalBackingTests(TestCase):
    """The folder backing behaves like ``aws s3 sync`` in the ways that are relied on."""

    def _tree(self, root: Path, **files):
        for name, body in files.items():
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body)

    def test_sync_copies_a_tree(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            src, dest = Path(tmp) / "src", Path(tmp) / "dest"
            self._tree(src, **{"model.pt": "weights", "logs/train_log.json": "{}"})
            storage.sync_out(src_path=str(src), dest=str(dest))
            self.assertEqual((dest / "model.pt").read_text(), "weights")
            self.assertEqual((dest / "logs/train_log.json").read_text(), "{}")

    def test_sync_is_incremental(self):
        """A dataset already on disk should cost a stat per file, not a re-copy of 1.9 GB."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            src, dest = Path(tmp) / "src", Path(tmp) / "dest"
            self._tree(src, **{"a.bin": "x", "b.bin": "y"})
            storage.sync_out(src_path=str(src), dest=str(dest))

            with mock.patch("roto_app.helpers.storage.shutil.copy2") as copy2:
                storage.sync_out(src_path=str(src), dest=str(dest))
                copy2.assert_not_called()

    def test_sync_recopies_a_changed_file(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            src, dest = Path(tmp) / "src", Path(tmp) / "dest"
            self._tree(src, **{"a.bin": "x"})
            storage.sync_out(src_path=str(src), dest=str(dest))
            (src / "a.bin").write_text("changed and longer")
            storage.sync_out(src_path=str(src), dest=str(dest))
            self.assertEqual((dest / "a.bin").read_text(), "changed and longer")

    def test_sync_never_deletes_from_the_destination(self):
        """The destination is a folder someone may be keeping things in."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            src, dest = Path(tmp) / "src", Path(tmp) / "dest"
            self._tree(src, **{"a.bin": "x"})
            self._tree(dest, **{"keep_me.txt": "mine"})
            storage.sync_out(src_path=str(src), dest=str(dest))
            self.assertTrue((dest / "keep_me.txt").exists())

    def test_prefix_exists_is_false_for_missing_and_empty(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty"
            empty.mkdir()
            self.assertFalse(storage.prefix_exists(str(empty)))
            self.assertFalse(storage.prefix_exists(str(Path(tmp) / "nope")))

            self._tree(Path(tmp) / "full", **{"manifest.json": "{}"})
            self.assertTrue(storage.prefix_exists(str(Path(tmp) / "full")))

    def test_a_missing_source_says_what_is_missing(self):
        with self.assertRaises(FileNotFoundError):
            storage.sync_in(src="/definitely/not/here", dest_path="/tmp/whatever")  # nosec


class DispatchTests(TestCase):
    """An s3:// root must still go to the AWS CLI, unchanged."""

    @mock.patch("roto_app.helpers.storage.aws_helpers.sync_s3_to_local")
    def test_s3_sync_in_goes_to_aws(self, sync):
        storage.sync_in(src="s3://b/datasets/v003", dest_path="/scratch/v003")
        sync.assert_called_once_with(s3_uri="s3://b/datasets/v003", dest_path="/scratch/v003")

    @mock.patch("roto_app.helpers.storage.aws_helpers.sync_local_to_s3")
    def test_s3_sync_out_goes_to_aws(self, sync):
        storage.sync_out(src_path="/scratch/run", dest="s3://b/runs/staging/s3a_seed1")
        sync.assert_called_once_with(
            src_path="/scratch/run", s3_uri="s3://b/runs/staging/s3a_seed1"
        )

    @mock.patch("roto_app.helpers.storage.aws_helpers.s3_prefix_exists", return_value=True)
    def test_s3_prefix_exists_splits_bucket_from_key(self, exists):
        self.assertTrue(storage.prefix_exists("s3://roto-staging/datasets/v003"))
        exists.assert_called_once_with(bucket="roto-staging", prefix="datasets/v003")
