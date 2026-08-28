from __future__ import annotations

import unittest

from src.agent.convergence import ConvergenceTracker
from src.agent.proposer import ConfigProposer
from src.agent.reflector import reflect
from src.agent.types import ExperimentOutcome, ExperimentSpec


class ConvergenceTests(unittest.TestCase):
    def test_three_non_meaningful_iterations_converge(self):
        tracker = ConvergenceTracker(epsilon=0.002, patience=3)
        self.assertFalse(tracker.observe(0.6000))
        self.assertFalse(tracker.observe(0.6010))
        self.assertFalse(tracker.observe(0.6015))
        self.assertTrue(tracker.observe(0.6019))

    def test_meaningful_improvement_resets_patience(self):
        tracker = ConvergenceTracker(epsilon=0.002, patience=3)
        tracker.observe(0.6000)
        tracker.observe(0.6010)
        self.assertFalse(tracker.observe(0.6030))
        self.assertEqual(tracker.stagnant_iterations, 0)


class ProposerTests(unittest.TestCase):
    def test_config_proposer_exhausts_in_order(self):
        proposer = ConfigProposer(
            [
                {"id": "a", "kind": "random", "hypothesis": "first"},
                {"id": "b", "kind": "popularity", "hypothesis": "second"},
            ]
        )
        self.assertEqual(proposer.propose([]).experiment_id, "a")
        self.assertEqual(proposer.propose([]).experiment_id, "b")
        self.assertIsNone(proposer.propose([]))


class ReflectionTests(unittest.TestCase):
    def test_successful_improvement_is_promoted(self):
        spec = ExperimentSpec("fm", "fm", "baseline")
        outcome = ExperimentOutcome(
            status="success",
            metrics={"GAUC": 0.67, "nDCG@5": 0.54, "primary": 0.605},
            duration_seconds=1.0,
        )
        result = reflect(spec, outcome, previous_best=0.58, official_baseline=0.6016)
        self.assertEqual(result["decision"], "promote_to_best")


if __name__ == "__main__":
    unittest.main()

