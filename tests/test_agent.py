from __future__ import annotations

import random
import unittest

from src.agent.convergence import ConvergenceTracker, official_converged
from src.agent.policy import (
    SearchPolicy,
    exploit_family,
    family_experiment_score,
    next_family_hint,
    required_family,
    scored_primaries,
)
from src.agent.proposer import ConfigProposer
from src.agent.reflector import reflect
from src.agent.types import ExperimentNode, ExperimentOutcome, ExperimentSpec, RunState


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


class PolicyTests(unittest.TestCase):
    def test_family_experiment_score_prefers_a_lower_quality_family(self):
        state = RunState(
            run_id="demo",
            status="running",
            started_at="2026-01-01T00:00:00Z",
            baseline_primary=0.6016,
            nodes=[
                ExperimentNode(
                    iteration=1,
                    experiment_id="bpr_run",
                    hypothesis_id="h1",
                    family="bpr",
                    action="explore",
                    parameters={},
                    status="success",
                    metrics={"GAUC": 0.66, "nDCG@5": 0.52, "primary": 0.61},
                ),
            ],
        )
        self.assertGreater(family_experiment_score(state, "group_softmax"), family_experiment_score(state, "bpr"))

    def test_next_family_hint_prefers_unseen_family_when_none_are_covered(self):
        state = RunState(
            run_id="demo",
            status="running",
            started_at="2026-01-01T00:00:00Z",
            baseline_primary=0.6016,
            nodes=[],
        )
        self.assertEqual(next_family_hint(state), "bpr")

    def test_promising_best_family_is_exploited_before_coverage(self):
        state = RunState(
            run_id="demo",
            status="running",
            started_at="2026-01-01T00:00:00Z",
            baseline_primary=0.6016,
        )
        best = ExperimentNode(
            iteration=1,
            experiment_id="cand_hf_tabcross_prior_days_v1",
            hypothesis_id="hf_tabcross_prior_days_v1",
            family="history_features",
            action="explore",
            parameters={},
            status="success",
            metrics={"GAUC": 0.6695, "nDCG@5": 0.5366, "primary": 0.603063},
        )
        state.nodes.append(best)
        state.best_experiment_id = best.experiment_id
        state.best_metrics = dict(best.metrics or {})

        self.assertEqual(exploit_family(state), "history_features")
        self.assertEqual(required_family(state), "history_features")
        self.assertEqual(next_family_hint(state), "history_features")

    def test_bad_new_family_falls_back_to_best_family(self):
        state = RunState(
            run_id="demo",
            status="running",
            started_at="2026-01-01T00:00:00Z",
            baseline_primary=0.6016,
        )
        best = ExperimentNode(
            iteration=1,
            experiment_id="cand_hf_tabcross_prior_days_v1",
            hypothesis_id="hf_tabcross_prior_days_v1",
            family="history_features",
            action="explore",
            parameters={},
            status="success",
            metrics={"GAUC": 0.6695, "nDCG@5": 0.5366, "primary": 0.603063},
        )
        bpr = ExperimentNode(
            iteration=2,
            experiment_id="cand_bpr_sameuser_v3",
            hypothesis_id="cand_bpr_sameuser_v3",
            family="bpr",
            action="explore",
            parameters={},
            status="success",
            metrics={"GAUC": 0.6553, "nDCG@5": 0.5306, "primary": 0.592949},
        )
        state.nodes.extend([best, bpr])
        state.best_experiment_id = best.experiment_id
        state.best_metrics = dict(best.metrics or {})

        self.assertEqual(exploit_family(state), "history_features")
        self.assertEqual(required_family(state), "history_features")

    def test_convergence_waits_for_best_family_followups(self):
        state = RunState(
            run_id="demo",
            status="running",
            started_at="2026-01-01T00:00:00Z",
            baseline_primary=0.6016,
            stagnant_iterations=3,
        )
        best = ExperimentNode(
            iteration=1,
            experiment_id="cand_hf_tabcross_prior_days_v1",
            hypothesis_id="hf_tabcross_prior_days_v1",
            family="history_features",
            action="explore",
            parameters={},
            status="success",
            metrics={"GAUC": 0.6695, "nDCG@5": 0.5366, "primary": 0.603063},
        )
        state.nodes.append(best)
        state.best_experiment_id = best.experiment_id
        state.best_metrics = dict(best.metrics or {})
        policy = SearchPolicy(0.002, 3, [])

        self.assertFalse(policy.should_stop(state))

        state.nodes.extend(
            [
                ExperimentNode(2, "hf_ablate_1", "h", "history_features", "exploit", {}, "success", {"primary": 0.6020}),
                ExperimentNode(3, "hf_ablate_2", "h", "history_features", "exploit", {}, "success", {"primary": 0.6019}),
            ]
        )
        # Follow-ups are done, but should_stop also gates on coverage: a run may not end while a
        # registered family has never produced a successful candidate. Cover the remaining three
        # so this test keeps measuring what it is named for.
        self.assertFalse(policy.should_stop(state))
        state.nodes.extend(
            [
                ExperimentNode(4, "bpr_c", "h", "bpr", "explore", {}, "success", {"primary": 0.6018}),
                ExperimentNode(5, "gs_c", "h", "group_softmax", "explore", {}, "success", {"primary": 0.6017}),
                ExperimentNode(6, "mt_c", "h", "multi_task", "explore", {}, "success", {"primary": 0.6016}),
            ]
        )
        self.assertTrue(policy.should_stop(state))


class OfficialConvergenceTests(unittest.TestCase):
    """T5 / I7: one implementation, checked against the organizers' formula."""

    def test_official_rule_matches_the_literal_reference_over_random_sequences(self):
        """`official_converged` is the reference rule, not the ratchet.

        The loop below is the spec written out longhand — best over the prefix
        minus best over the prefix `patience` earlier — and it is the authority:
        where the shipped ratchet and this ever disagree, this is what wins.
        """
        rng = random.Random(0)
        epsilon, patience = 0.002, 3
        for _ in range(2000):
            scores = [rng.uniform(0.45, 0.75) for _ in range(rng.randint(1, 20))]
            fired = False
            for k in range(1, len(scores) + 1):
                if k > patience and max(scores[:k]) - max(scores[: k - patience]) <= epsilon:
                    fired = True
                self.assertEqual(
                    official_converged(scores[:k], epsilon, patience), fired, (scores, k)
                )

    def test_stagnation_is_the_only_ratchet(self):
        """The tracker and the policy ratchet through one shared implementation."""
        scores = [0.6015, 0.6016, 0.6018, 0.6060, 0.6061, 0.6062, 0.6063]
        state = RunState("run", "running", "now", scores[0], meaningful_best=scores[0])
        policy = SearchPolicy(0.002, 3, [])
        tracker = ConvergenceTracker(epsilon=0.002, patience=3)
        tracker.observe(scores[0])
        for iteration, score in enumerate(scores[1:], start=1):
            node = ExperimentNode(
                iteration, f"e{iteration}", "h", "bpr", "explore", {}, "success",
                {"primary": score},
            )
            state.nodes.append(node)
            policy.observe_success(state, node)
            tracker.observe(score)
            self.assertEqual(tracker.meaningful_best, state.meaningful_best, iteration)
            self.assertEqual(tracker.stagnant_iterations, state.stagnant_iterations, iteration)
        self.assertEqual(scored_primaries(state), scores[1:])
        self.assertEqual(state.stagnant_iterations, 3)


if __name__ == "__main__":
    unittest.main()

