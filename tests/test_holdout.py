"""The selection/reporting split must be user-level, disjoint, balanced and stable.

Rationale is in src/evaluation/holdout.py: the reported score sat +0.0025 above the median of
its own epoch curve while the claimed improvement was +0.0024, so selection noise and the
result were the same size. These tests pin the properties that make the reporting half an
honest estimate rather than a second look at the same data.
"""

from __future__ import annotations

import subprocess
import sys
import unittest

import numpy as np

from src.evaluation.holdout import REPORT_SHARE, selection_mask, split_users


class SplitPropertyTests(unittest.TestCase):
    USERS = [f"u{index // 4}" for index in range(4000)]  # 1000 users, 4 impressions each

    def test_halves_are_complementary(self):
        selection, reporting = split_users(self.USERS)
        self.assertTrue(np.array_equal(selection, ~reporting))
        self.assertEqual(int((selection | reporting).sum()), len(self.USERS))
        self.assertEqual(int((selection & reporting).sum()), 0)

    def test_a_user_never_appears_on_both_sides(self):
        """GAUC is per user and nDCG ranks within a user, so a torn user would be scored twice."""
        selection, reporting = split_users(self.USERS)
        select_users = {u for u, keep in zip(self.USERS, selection) if keep}
        report_users = {u for u, keep in zip(self.USERS, reporting) if keep}
        self.assertEqual(select_users & report_users, set())
        self.assertEqual(len(select_users) + len(report_users), len(set(self.USERS)))

    def test_the_split_is_roughly_balanced(self):
        selection, _ = split_users(self.USERS)
        self.assertAlmostEqual(selection.mean(), 1 - REPORT_SHARE, delta=0.05)

    def test_it_is_deterministic_within_a_process(self):
        self.assertTrue(np.array_equal(selection_mask(self.USERS), selection_mask(self.USERS)))

    def test_it_is_stable_across_processes(self):
        """`hash()` is salted per process, so the split must not depend on it."""
        script = (
            "from src.evaluation.holdout import selection_mask;"
            "print(''.join('1' if b else '0' for b in selection_mask("
            "[f'u{i//4}' for i in range(400)])))"
        )
        runs = {
            subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, check=True,
                env={"PYTHONHASHSEED": seed, "PYTHONPATH": "."},
            ).stdout.strip()
            for seed in ("0", "1", "12345")
        }
        self.assertEqual(len(runs), 1, "split changed with PYTHONHASHSEED")

    def test_order_does_not_matter_only_identity(self):
        """A user's side is a property of the user, not of where the row happens to sit."""
        selection, _ = split_users(self.USERS)
        side = {u: bool(k) for u, k in zip(self.USERS, selection)}
        shuffled = list(reversed(self.USERS))
        reshuffled, _ = split_users(shuffled)
        for user, keep in zip(shuffled, reshuffled):
            self.assertEqual(side[user], bool(keep))

    def test_empty_input_is_handled(self):
        selection, reporting = split_users([])
        self.assertEqual(len(selection), 0)
        self.assertEqual(len(reporting), 0)


class WorkerIntegrationTests(unittest.TestCase):
    def test_the_worker_reports_both_halves_next_to_the_full_metric(self):
        """`primary` must stay full-validation: the baseline and judged delta are measured on it."""
        import tempfile
        from pathlib import Path

        from src.experiments.contracts import CandidateOutput
        from src.experiments.run_candidate import validate_and_persist_output

        rng = np.random.default_rng(0)
        users = [f"u{i // 4}" for i in range(4000)]
        labels = rng.integers(0, 2, size=len(users)).astype(np.float32)
        scores = rng.normal(size=len(users))
        output = CandidateOutput(validation_scores=scores, checkpoint_state={})

        with tempfile.TemporaryDirectory() as directory:
            payload = validate_and_persist_output(
                output, users, labels, Path(directory)
            )
        metrics = payload["metrics"]
        for key in ("primary", "GAUC", "nDCG@5", "select_primary", "report_primary"):
            with self.subTest(key=key):
                self.assertIn(key, metrics)
        # the two halves are genuinely different samples, so they should not coincide exactly
        self.assertNotEqual(metrics["select_primary"], metrics["report_primary"])

    def test_early_stopping_uses_a_strict_subset_of_validation(self):
        users = [f"u{i // 4}" for i in range(4000)]
        selection, _ = split_users(users)
        self.assertLess(int(selection.sum()), len(users))
        self.assertGreater(int(selection.sum()), 0)


if __name__ == "__main__":
    unittest.main()
