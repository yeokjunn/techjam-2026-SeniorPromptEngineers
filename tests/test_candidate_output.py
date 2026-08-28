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


if __name__ == "__main__":
    unittest.main()
