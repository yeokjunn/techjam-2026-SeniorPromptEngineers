"""The Researcher must be able to see the shape of the training curve, not just its last number.

Every candidate this harness has produced peaks around epoch 3-5 and then decays -- the best one
observed went 0.5990 -> 0.6039 (epoch 4) -> 0.6013 (epoch 7), ending below the 0.6016 baseline.
That fact lived in `ExperimentOutcome.epoch_trace` and never reached a prompt, so the agent kept
proposing epochs=40 and could not attribute a score to where on the curve it landed.
"""

from __future__ import annotations

import json
import unittest

from src.agent.types import ExperimentNode, RunState, summarize_epoch_trace

# The real trace of the best candidate observed (run 20260830T092212798196Z).
REAL_TRACE = [
    {"epoch": index, "primary": value}
    for index, value in enumerate(
        [0.59895, 0.60167, 0.60232, 0.60237, 0.60389, 0.60298, 0.60198, 0.60128]
    )
]


class SummarizeEpochTraceTests(unittest.TestCase):
    def test_it_reports_where_the_run_peaked_and_what_it_gave_back(self):
        summary = summarize_epoch_trace(REAL_TRACE)
        self.assertEqual(summary["epochs_ran"], 8)
        self.assertEqual(summary["peak_epoch"], 4)
        self.assertAlmostEqual(summary["peak"], 0.60389, places=5)
        self.assertAlmostEqual(summary["final"], 0.60128, places=5)
        # the whole point: this run ended 0.0026 below its own peak
        self.assertAlmostEqual(summary["declined_after_peak"], 0.00261, places=5)

    def test_a_still_improving_run_is_distinguishable_from_an_overfit_one(self):
        improving = [{"epoch": i, "primary": p} for i, p in enumerate([0.59, 0.60, 0.61])]
        summary = summarize_epoch_trace(improving)
        self.assertEqual(summary["peak_epoch"], summary["epochs_ran"] - 1)
        self.assertEqual(summary["declined_after_peak"], 0.0)

    def test_missing_or_unusable_traces_omit_the_field_rather_than_report_zeros(self):
        for trace in (None, [], [{"epoch": 0}], [{"epoch": 0, "primary": None}]):
            with self.subTest(trace=trace):
                self.assertIsNone(summarize_epoch_trace(trace))

    def test_a_candidate_that_never_trained_has_no_summary(self):
        """`outcome` is None when repairs are exhausted before training ever runs.

        Calling summarize_epoch_trace(outcome.epoch_trace) unguarded raised AttributeError
        there, which the controller classified as a harness error and which replaced the
        normal iteration ledger record.
        """
        self.assertIsNone(summarize_epoch_trace(None))
        node = ExperimentNode(
            iteration=1, experiment_id="c", hypothesis_id="h", family="bpr",
            action="explore", parameters={}, status="failed",
            trace_summary=summarize_epoch_trace(None),
        )
        self.assertIsNone(node.trace_summary)
        self.assertIn("trace_summary", node.to_dict())

    def test_non_numeric_entries_are_skipped_not_fatal(self):
        mixed = [{"epoch": 0, "primary": 0.6}, {"epoch": 1, "primary": "nan"}, {"epoch": 2, "primary": 0.61}]
        self.assertEqual(summarize_epoch_trace(mixed)["epochs_ran"], 2)


class StateSummaryTests(unittest.TestCase):
    """The summary has to reach the prompt, and has to stay cheap enough to keep sending."""

    def node(self, iteration: int) -> ExperimentNode:
        return ExperimentNode(
            iteration=iteration,
            experiment_id=f"cand_{iteration}",
            hypothesis_id="h",
            family="bpr",
            action="explore",
            parameters={"seed": 0, "epochs": 15},
            status="success",
            metrics={"GAUC": 0.6705, "nDCG@5": 0.5375, "primary": 0.6039},
            trace_summary=summarize_epoch_trace(REAL_TRACE),
        )

    def test_the_trace_summary_reaches_the_researcher_prompt(self):
        from src.agent.roles import ResearchRoles

        state = RunState(
            run_id="r", status="running", started_at="2026-01-01T00:00:00Z", baseline_primary=0.6016
        )
        state.nodes.append(self.node(1))
        rendered = ResearchRoles._state_summary(state)
        self.assertIn("trace_summary", rendered)
        self.assertIn("peak_epoch", rendered)

    def test_the_summary_is_far_cheaper_than_the_raw_trace(self):
        """It rides in the uncached volatile prompt and grows with the run, so size matters."""
        summary_cost = len(json.dumps(summarize_epoch_trace(REAL_TRACE)))
        raw_cost = len(json.dumps(REAL_TRACE))
        self.assertLess(summary_cost, raw_cost / 2)
        # roughly 20 tokens at 4 chars/token, so 50 nodes stays well inside the budget
        self.assertLess(summary_cost, 160)

    def test_nodes_without_a_trace_serialise_cleanly(self):
        state = RunState(
            run_id="r", status="running", started_at="2026-01-01T00:00:00Z", baseline_primary=0.6016
        )
        bare = self.node(1)
        bare.trace_summary = None
        state.nodes.append(bare)
        payload = json.loads(ResearchRolesSummary(state))
        self.assertIsNone(payload["experiments"][0]["trace_summary"])


def ResearchRolesSummary(state: RunState) -> str:
    from src.agent.roles import ResearchRoles

    return ResearchRoles._state_summary(state)


if __name__ == "__main__":
    unittest.main()
