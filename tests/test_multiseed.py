"""Candidates are scored on several seeds and reported as a mean with a spread.

The published baseline is a 5-seed mean with std 0.0008 while a candidate was scored on one
seed, so a single-seed result is drawn from a distribution about as wide as the improvement
being claimed. That is why a 0.6039 candidate replicated at 0.6025 and 0.6024. Averaging makes
the number the run selects on, converges on and reports the same denoised quantity, and
`primary_std` is what turns "+0.0024" into "+0.0024 +/- s".
"""

from __future__ import annotations

import unittest

import numpy as np

from src.experiments.contracts import CandidateOutput
from src.experiments.run_candidate import _apply_seed_aggregate, _score_run, _seed_plan


def payload_with(primaries: list[float], extra: dict | None = None) -> tuple[dict, list]:
    runs = [
        (seed, None, {"primary": value, "GAUC": value + 0.05, **(extra or {})})
        for seed, value in enumerate(primaries, start=7)
    ]
    payload = {"metrics": {"primary": primaries[0], "GAUC": primaries[0] + 0.05}}
    return payload, runs


class SeedPlanTests(unittest.TestCase):
    def test_one_seed_reproduces_the_previous_behaviour(self):
        self.assertEqual(_seed_plan({"seed": 42}, 1), [42])

    def test_seeds_are_consecutive_from_the_proposed_one(self):
        """Reproducible from the manifest alone: seed=s means s, s+1, s+2."""
        self.assertEqual(_seed_plan({"seed": 42}, 3), [42, 43, 44])

    def test_a_missing_or_degenerate_count_falls_back_to_one_run(self):
        self.assertEqual(_seed_plan({}, 0), [0])
        self.assertEqual(_seed_plan({"seed": 5}, -3), [5])


class AggregateTests(unittest.TestCase):
    def test_metrics_become_the_mean_across_seeds(self):
        payload, runs = payload_with([0.60, 0.62, 0.64])
        _apply_seed_aggregate(payload, runs, representative_seed=8)
        self.assertAlmostEqual(payload["metrics"]["primary"], 0.62, places=6)
        self.assertAlmostEqual(payload["metrics"]["GAUC"], 0.67, places=6)

    def test_the_spread_is_reported_so_a_delta_can_carry_error_bars(self):
        payload, runs = payload_with([0.6039, 0.6025, 0.6024])
        _apply_seed_aggregate(payload, runs, representative_seed=7)
        self.assertAlmostEqual(payload["metrics"]["primary"], 0.60293, places=5)
        # the real replication spread that motivated this change
        self.assertAlmostEqual(payload["metrics"]["primary_std"], 0.00084, places=5)
        self.assertEqual(payload["metrics"]["seeds_run"], 3.0)

    def test_a_single_seed_is_a_strict_generalisation(self):
        """Mean of one is that one, so nothing downstream shifts when seeds=1."""
        payload, runs = payload_with([0.6039])
        _apply_seed_aggregate(payload, runs, representative_seed=7)
        self.assertAlmostEqual(payload["metrics"]["primary"], 0.6039, places=6)
        self.assertEqual(payload["metrics"]["primary_std"], 0.0)
        self.assertEqual(payload["metrics"]["seeds_run"], 1.0)

    def test_per_seed_values_are_recorded_for_audit(self):
        payload, runs = payload_with([0.60, 0.62])
        _apply_seed_aggregate(payload, runs, representative_seed=8)
        self.assertEqual(payload["seed_primaries"], {"7": 0.60, "8": 0.62})
        self.assertEqual(payload["representative_seed"], 8)

    def test_non_numeric_metrics_are_left_alone(self):
        payload, runs = payload_with([0.60, 0.62], extra={"note": "x"})
        _apply_seed_aggregate(payload, runs, representative_seed=7)
        self.assertNotIn("note", [k for k, v in payload["metrics"].items() if isinstance(v, float) and v == 0])


class ScoreRunTests(unittest.TestCase):
    USERS = [f"u{index // 4}" for index in range(400)]

    def labels(self) -> np.ndarray:
        return np.resize(np.array([1, 0, 1, 0], dtype=np.float32), len(self.USERS))

    def test_a_seed_producing_nan_fails_the_candidate_rather_than_being_averaged(self):
        """A NaN seed must not be silently dropped from the mean."""
        scores = np.zeros(len(self.USERS))
        scores[3] = np.nan
        output = CandidateOutput(validation_scores=scores, checkpoint_state={})
        with self.assertRaises(ValueError):
            _score_run(output, self.USERS, self.labels())

    def test_a_wrong_length_seed_is_rejected(self):
        output = CandidateOutput(validation_scores=np.zeros(3), checkpoint_state={})
        with self.assertRaises(ValueError):
            _score_run(output, self.USERS, self.labels())

    def test_a_valid_run_returns_full_validation_metrics(self):
        rng = np.random.default_rng(0)
        output = CandidateOutput(
            validation_scores=rng.normal(size=len(self.USERS)), checkpoint_state={}
        )
        metrics = _score_run(output, self.USERS, self.labels())
        for key in ("primary", "GAUC", "nDCG@5"):
            self.assertIn(key, metrics)


class RepresentativeSelectionTests(unittest.TestCase):
    def test_the_submitted_checkpoint_is_the_one_nearest_the_mean(self):
        """Submitting the best seed would ship a model scoring above its own reported estimate."""
        runs = [(7, None, {"primary": 0.6039}), (8, None, {"primary": 0.6025}),
                (9, None, {"primary": 0.6024})]
        mean = sum(r[2]["primary"] for r in runs) / len(runs)
        chosen = min(runs, key=lambda run: abs(run[2]["primary"] - mean))
        self.assertEqual(chosen[0], 8)          # 0.6025 is nearest 0.60293
        self.assertNotEqual(chosen[0], 7)       # not the best seed


if __name__ == "__main__":
    unittest.main()
