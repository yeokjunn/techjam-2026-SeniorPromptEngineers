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


if __name__ == "__main__":
    unittest.main()
