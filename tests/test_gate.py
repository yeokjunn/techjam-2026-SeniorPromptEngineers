from __future__ import annotations

import csv
import json
import re
import shutil
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.evaluation.gate import run_gate


REPO_ROOT = Path(__file__).resolve().parents[1]
KIT_DIR = REPO_ROOT / "kuairand-starter-kit"

LOG_HEADER = (
    "user_id,video_id,date,hourmin,time_ms,"
    "is_click,is_like,is_follow,is_comment,is_forward,is_hate,long_view,"
    "play_time_ms,duration_ms,profile_stay_time,comment_stay_time,is_profile_enter,"
    "is_rand,tab"
)


def write_synthetic_data(root: Path) -> Path:
    """Tiny dataset across all three date windows.

    Expected test split (file order after the date filter, early log first):
    (0,u1,v1) (1,u2,v2) (2,u2,v1) (3,u1,v1) — the last pair duplicates (u1,v1),
    mirroring the real test split's 3% duplicate (user, video) rows.
    """
    data_dir = root / "synthetic_data"
    data_dir.mkdir(parents=True)
    (data_dir / "video_features_basic_pure.csv").write_text(
        "video_id,author_id\nv1,a1\nv2,a2\nv3,a3\n", encoding="utf-8", newline=""
    )
    (data_dir / "log_standard_4_08_to_4_21_pure.csv").write_text(
        LOG_HEADER + "\n"
        "u1,v1,20220410,1800,0,0,0,0,0,0,0,1,863,30066,0,0,0,0,1\n"
        "u1,v1,20220430,1800,0,0,0,0,0,0,0,1,863,30066,0,0,0,0,1\n",
        encoding="utf-8",
        newline="",
    )
    (data_dir / "log_standard_4_22_to_5_08_pure.csv").write_text(
        LOG_HEADER + "\n"
        "u3,v3,20220425,1800,0,0,0,0,0,0,0,1,863,30066,0,0,0,0,1\n"
        "u2,v2,20220501,1800,0,0,0,0,0,0,0,1,863,30066,0,0,0,0,1\n"
        "u2,v1,20220502,1800,0,0,0,0,0,0,0,1,863,30066,0,0,0,0,1\n"
        "u1,v1,20220503,1800,0,0,0,0,0,0,0,1,863,30066,0,0,0,0,1\n",
        encoding="utf-8",
        newline="",
    )
    return data_dir


class _GateFixture:
    """run_dir/node_dir/data_dir layout with N=4 persisted test scores."""

    def __init__(self, root: Path) -> None:
        self.run_dir = root / "run"
        self.node_dir = root / "generated" / "001_cand"
        self.node_dir.mkdir(parents=True)
        self.data_dir = write_synthetic_data(root)
        self.scores = np.random.default_rng(0).random(4)
        np.save(self.node_dir / "test_scores.npy", self.scores)

    def call(self):
        return run_gate(self.run_dir, self.node_dir, self.data_dir, KIT_DIR)


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


class GateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fixture = _GateFixture(Path(self._tmp.name))

    def test_gate_writes_a_submission_that_passes_the_kit_check(self):
        result = self.fixture.call()
        self.assertEqual(result.status, "ok")
        submission = self.fixture.run_dir / "submission.csv"
        self.assertTrue(submission.is_file())
        rows = read_rows(submission)
        self.assertEqual(rows[0], ["row_id", "user_id", "video_id", "score"])
        self.assertEqual(len(rows), 5)  # header + 4
        self.assertEqual(
            [(row[0], row[1], row[2]) for row in rows[1:]],
            [("0", "u1", "v1"), ("1", "u2", "v2"), ("2", "u2", "v1"), ("3", "u1", "v1")],
        )
        for cell, expected in zip([row[3] for row in rows[1:]], self.fixture.scores):
            self.assertAlmostEqual(float(cell), float(expected), places=8)
        self.assertTrue((self.fixture.run_dir / "gate_done.json").is_file())
        self.assertEqual(result.details["rows"], 4)
        self.assertFalse(result.details["scored"])
        self.assertIn("✓", result.details["check_stdout"])

    def test_gate_is_idempotent(self):
        first = self.fixture.call()
        submission = self.fixture.run_dir / "submission.csv"
        digest_before = submission.read_bytes()
        second = self.fixture.call()
        self.assertEqual(second.status, "ok")
        self.assertTrue(second.details["reused"])
        self.assertEqual(submission.read_bytes(), digest_before)
        self.assertEqual(second.details.get("sha256"), first.details.get("sha256"))

    def test_scores_resolved_from_the_artifacts_directory(self):
        # The real-run spelling: scores live next to model.npz under
        # run_dir/artifacts/<node name>/, node_dir itself is only a pointer.
        artifacts = self.fixture.run_dir / "artifacts" / self.fixture.node_dir.name
        artifacts.mkdir(parents=True)
        np.save(artifacts / "test_scores.npy", self.fixture.scores)
        (self.fixture.node_dir / "test_scores.npy").unlink()
        result = self.fixture.call()
        self.assertEqual(result.status, "ok")

    def test_missing_test_scores_returns_error(self):
        (self.fixture.node_dir / "test_scores.npy").unlink()
        result = self.fixture.call()
        self.assertEqual(result.status, "error")
        self.assertEqual(result.details["reason"], "missing_test_scores")
        self.assertEqual(len(result.details["searched"]), 2)
        self.assertFalse((self.fixture.run_dir / "submission.csv").exists())

    def test_wrong_length_scores_return_error(self):
        np.save(self.fixture.node_dir / "test_scores.npy", np.zeros(3))
        result = self.fixture.call()
        self.assertEqual(result.status, "error")
        self.assertEqual(result.details["reason"], "bad_test_scores")
        self.assertEqual(result.details["got_rows"], 3)
        self.assertEqual(result.details["expected_rows"], 4)
        self.assertFalse((self.fixture.run_dir / "submission.csv").exists())

    def test_nonfinite_scores_are_rejected_before_the_kit_runs(self):
        np.save(
            self.fixture.node_dir / "test_scores.npy",
            np.asarray([np.nan, 0.1, 0.2, 0.3]),
        )
        result = self.fixture.call()
        self.assertEqual(result.details["reason"], "bad_test_scores")
        self.assertFalse((self.fixture.run_dir / "submission.csv").exists())

    def test_missing_kit_returns_error(self):
        empty_kit = Path(self._tmp.name) / "empty_kit"
        empty_kit.mkdir()
        result = run_gate(
            self.fixture.run_dir, self.fixture.node_dir, self.fixture.data_dir, empty_kit
        )
        self.assertEqual(result.status, "error")
        self.assertEqual(result.details["reason"], "kit_unavailable")

    def test_unexpected_exception_becomes_status_error(self):
        with patch("src.evaluation.gate.np.load", side_effect=RuntimeError("boom")):
            result = self.fixture.call()
        self.assertEqual(result.status, "error")
        self.assertEqual(result.details["reason"], "unexpected")
        self.assertIn("RuntimeError: boom", result.details["error"])

    def test_gate_result_carries_no_test_metric(self):
        result = self.fixture.call()
        self.assertEqual(result.status, "ok")
        payload = json.dumps(asdict(result))
        self.assertIsNone(re.search(r"GAUC|nDCG|primary", payload))

    def test_submission_path_is_repo_relative_when_under_repo_root(self):
        repo_tmp = Path(tempfile.mkdtemp(dir=REPO_ROOT))
        self.addCleanup(shutil.rmtree, repo_tmp, ignore_errors=True)
        fixture = _GateFixture(repo_tmp)
        result = fixture.call()
        self.assertEqual(result.status, "ok")
        self.assertFalse(Path(result.submission_path).is_absolute())
        self.assertTrue(result.submission_path.startswith("tmp"))


if __name__ == "__main__":
    unittest.main()
