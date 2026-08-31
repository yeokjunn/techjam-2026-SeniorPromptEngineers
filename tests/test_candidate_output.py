from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.experiments.contracts import CandidateOutput
from src.experiments.run_candidate import validate_and_persist_output


class CandidateOutputTests(unittest.TestCase):
    def test_trusted_metrics_override_candidate_diagnostics(self):
        output = CandidateOutput(
            validation_scores=np.asarray([1.0, 0.0]),
            checkpoint_state={"weights": np.asarray([1.0])},
            diagnostics={"primary": 99.0, "pairs": 10},
        )
        with tempfile.TemporaryDirectory() as directory:
            payload = validate_and_persist_output(
                output,
                ["user", "user"],
                np.asarray([1, 0], dtype=np.float32),
                Path(directory),
            )
            self.assertAlmostEqual(payload["metrics"]["primary"], 1.0)
            self.assertNotIn("primary", payload["diagnostics"])
            self.assertTrue((Path(directory) / "model.npz").is_file())

    def test_wrong_length_and_nonfinite_scores_are_rejected(self):
        cases = [np.asarray([1.0]), np.asarray([np.nan, 0.0])]
        for scores in cases:
            with self.subTest(scores=scores), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    validate_and_persist_output(
                        CandidateOutput(scores, {}),
                        ["user", "user"],
                        np.asarray([1, 0], dtype=np.float32),
                        Path(directory),
                    )

    def test_nonfinite_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            validate_and_persist_output(
                CandidateOutput(
                    np.asarray([1.0, 0.0]),
                    {"weights": np.asarray([np.inf])},
                ),
                ["user", "user"],
                np.asarray([1, 0], dtype=np.float32),
                Path(directory),
            )

    def test_test_scores_are_persisted_as_float64(self):
        output = CandidateOutput(
            validation_scores=np.asarray([1.0, 0.0]),
            checkpoint_state={"weights": np.asarray([1.0])},
            # float32 in, float64 out: %.9g in the later CSV needs more than
            # float32's ~7 significant digits or the formatting creates ties.
            test_scores=np.asarray([0.25, 0.75], dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as directory:
            payload = validate_and_persist_output(
                output,
                ["user", "user"],
                np.asarray([1, 0], dtype=np.float32),
                Path(directory),
                expected_test_rows=2,
            )
            self.assertEqual(payload["test_scores_status"], "ok")
            saved = np.load(Path(directory) / "test_scores.npy")
            self.assertEqual(saved.dtype, np.float64)
            self.assertEqual(saved.tolist(), [0.25, 0.75])
            self.assertTrue(payload["test_scores_path"].endswith("test_scores.npy"))

    def test_missing_test_scores_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = validate_and_persist_output(
                CandidateOutput(np.asarray([1.0, 0.0]), {}),
                ["user", "user"],
                np.asarray([1, 0], dtype=np.float32),
                Path(directory),
                expected_test_rows=2,
            )
            self.assertEqual(payload["test_scores_status"], "missing")
            self.assertIsNone(payload["test_scores_path"])
            # The checkpoint half still ran: the ledger keeps the number.
            self.assertTrue((Path(directory) / "model.npz").is_file())
            self.assertFalse((Path(directory) / "test_scores.npy").exists())

    def test_wrong_length_test_scores_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = validate_and_persist_output(
                CandidateOutput(
                    np.asarray([1.0, 0.0]), {}, test_scores=np.asarray([1.0])
                ),
                ["user", "user"],
                np.asarray([1, 0], dtype=np.float32),
                Path(directory),
                expected_test_rows=2,
            )
            self.assertEqual(payload["test_scores_status"], "invalid")
            self.assertIsNone(payload["test_scores_path"])

    def test_nonfinite_test_scores_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = validate_and_persist_output(
                CandidateOutput(
                    np.asarray([1.0, 0.0]),
                    {},
                    test_scores=np.asarray([np.nan, 0.0]),
                ),
                ["user", "user"],
                np.asarray([1, 0], dtype=np.float32),
                Path(directory),
                expected_test_rows=2,
            )
            self.assertEqual(payload["test_scores_status"], "invalid")

    def test_not_required_when_expected_rows_is_none(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = validate_and_persist_output(
                CandidateOutput(
                    np.asarray([1.0, 0.0]),
                    {},
                    test_scores=np.asarray([1.0, 0.0]),
                ),
                ["user", "user"],
                np.asarray([1, 0], dtype=np.float32),
                Path(directory),
            )
            self.assertEqual(payload["test_scores_status"], "not_required")
            self.assertIsNone(payload["test_scores_path"])
            self.assertFalse((Path(directory) / "test_scores.npy").exists())

    def test_random_exposure_metrics_are_separate_diagnostics(self):
        output = CandidateOutput(
            validation_scores=np.asarray([0.8, 0.2]),
            checkpoint_state={},
            random_validation_scores=np.asarray([0.1, 0.9]),
        )
        with tempfile.TemporaryDirectory() as directory:
            payload = validate_and_persist_output(
                output,
                ["standard", "standard"],
                np.asarray([1, 0], dtype=np.float32),
                Path(directory),
                random_valid_users=["random", "random"],
                random_valid_y=np.asarray([1, 0], dtype=np.float32),
            )
        self.assertEqual(payload["diagnostics"]["random_exposure_status"], "scored")
        self.assertEqual(payload["metrics"]["primary"], 1.0)
        random_metrics = payload["diagnostic_metrics"]["random_exposure"]
        self.assertAlmostEqual(random_metrics["primary"], 0.31546488404273987)
        self.assertAlmostEqual(random_metrics["robustness_gap"], 0.6845351159572601)
        self.assertNotIn("random_exposure", payload["metrics"])

    def test_topk_diagnostics_are_trusted_validation_metadata(self):
        output = CandidateOutput(
            validation_scores=np.asarray([0.9, 0.8, 0.1, 0.7, 0.6, 0.5]),
            checkpoint_state={},
        )
        with tempfile.TemporaryDirectory() as directory:
            payload = validate_and_persist_output(
                output,
                ["u", "u", "u", "v", "v", "v"],
                np.asarray([1, 0, 1, 0, 1, 0], dtype=np.float32),
                Path(directory),
            )
        report = payload["topk_diagnostics"]
        self.assertEqual(report["topk"], 5)
        self.assertIn("per_user_ndcg", report)
        self.assertIn("ndcg_by_impression_count", report)
        self.assertIn("ndcg_by_positive_count", report)
        self.assertEqual(report["top5_positive_hits"], 3)
        self.assertEqual(payload["diagnostics"]["topk_diagnostics"], report)

    def test_missing_random_scores_is_optional_and_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = validate_and_persist_output(
                CandidateOutput(np.asarray([1.0, 0.0]), {}),
                ["standard", "standard"],
                np.asarray([1, 0], dtype=np.float32),
                Path(directory),
                random_valid_users=["random", "random"],
                random_valid_y=np.asarray([1, 0], dtype=np.float32),
            )
        self.assertEqual(payload["diagnostics"]["random_exposure_status"], "not_scored")
        self.assertEqual(payload["diagnostic_metrics"], {})

    def test_invalid_random_scores_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
            ValueError, "Random-exposure"
        ):
            validate_and_persist_output(
                CandidateOutput(
                    np.asarray([1.0, 0.0]),
                    {},
                    random_validation_scores=np.asarray([np.nan]),
                ),
                ["standard", "standard"],
                np.asarray([1, 0], dtype=np.float32),
                Path(directory),
                random_valid_users=["random", "random"],
                random_valid_y=np.asarray([1, 0], dtype=np.float32),
            )

    def test_ceiling_hit_is_marked_as_leak(self):
        # The canonical perfect-ranking fixture scores primary 1.0 — above the
        # leak ceiling — so it doubles as the ceiling case: the ledger keeps
        # the number, the payload flags it, nothing raises.
        output = CandidateOutput(
            validation_scores=np.asarray([1.0, 0.0]),
            checkpoint_state={"weights": np.asarray([1.0])},
        )
        with tempfile.TemporaryDirectory() as directory:
            payload = validate_and_persist_output(
                output,
                ["user", "user"],
                np.asarray([1, 0], dtype=np.float32),
                Path(directory),
            )
            self.assertEqual(payload["sanity_class"], "leak")

    def test_floor_miss_is_marked_low_score(self):
        output = CandidateOutput(
            validation_scores=np.asarray([0.0, 1.0]),  # perfectly inverted
            checkpoint_state={},
        )
        with tempfile.TemporaryDirectory() as directory:
            payload = validate_and_persist_output(
                output,
                ["user", "user"],
                np.asarray([1, 0], dtype=np.float32),
                Path(directory),
            )
            self.assertEqual(payload["sanity_class"], "low_score")


if __name__ == "__main__":
    unittest.main()
