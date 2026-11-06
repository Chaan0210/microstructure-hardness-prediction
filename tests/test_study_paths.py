"""Relocated study scripts accept shared data paths without allowing escapes."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[1]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDIES = _repo_path(ROOT, "studies")
sys.path.insert(0, str(STUDIES))
import candidate_pipeline
import frozen_validation
import validate_submission


class StudyPathTests(unittest.TestCase):
    def test_artifact_alias_preserves_study_relative_paths(self):
        for resolve in (
            frozen_validation.project_path,
            lambda path: candidate_pipeline.local(path, artifacts=True),
        ):
            with self.subTest(resolver=resolve):
                path = resolve("artifacts/path-check/output.json")
                self.assertEqual(path.resolve(), ROOT / "artifacts/path-check/output.json")
                self.assertEqual(path.relative_to(STUDIES).as_posix(), "artifacts/path-check/output.json")
                for invalid in ("../artifacts/output.json", "/tmp/output.json", "data/train.csv"):
                    with self.assertRaises(ValueError):
                        resolve(invalid)

    def test_data_and_report_aliases(self):
        for resolve in (candidate_pipeline.local, validate_submission.path):
            expected = {
                "data/sample_submission.csv": ROOT / "data/sample_submission.csv",
                "reports/path-check.json": ROOT / "reports/validation/path-check.json",
            }
            for name, target in expected.items():
                with self.subTest(resolver=resolve, path=name):
                    self.assertEqual(resolve(name).resolve(), target)
            with self.assertRaises(ValueError):
                resolve("../README.md")

    def test_symlink_cannot_escape_repository(self):
        with tempfile.TemporaryDirectory(dir=STUDIES) as directory:
            link = _repo_path(Path(directory), "outside")
            link.symlink_to(tempfile.gettempdir(), target_is_directory=True)
            name = (link / "output.json").relative_to(STUDIES)
            for resolve in (candidate_pipeline.local, validate_submission.path, frozen_validation.project_path):
                with self.assertRaises(ValueError):
                    resolve(name)


if __name__ == "__main__":
    unittest.main()
