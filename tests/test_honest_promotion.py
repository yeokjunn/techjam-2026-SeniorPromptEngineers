"""The loop's bookkeeping only claims what the measurement can support (W2).

Candidate primaries carry seed noise of about sigma = 9e-4. Four places treated
a bare `>` over that noise as progress, and together they are what produced a
"gain" of roughly the size of the expected maximum of 50 null draws:

* ``policy.SearchPolicy.observe_success`` promoted ``best_*`` on any increase,
  making the recorded best an unprotected running maximum over every draw;
* ``policy.exploit_family`` engaged on any margin over the baseline, and its
  follow-up window is counted from the best node — so every noisy promotion also
  renewed the window and the run could stay pinned on one family;
* ``policy.coverage_complete`` — the family-breadth guarantee — was never called
  by the controller, so a run could report ``stop_reason="converged"`` having
  tried one family out of four;
* ``summary.json`` reported the single best draw even when that draw had been
  replicated across seeds, discarding the one measurement of its own noise the
  run had actually paid for.

Each test below fails against exactly one of those and is written so that the
old behaviour cannot satisfy it.

The last class covers the addendum to the same wave: `k` became a registry grid
key when the capacity axes were unfrozen, so the retired "requires k=16" prose
must no longer be what an off-grid width is rejected with.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from src.agent import families, policy, research_controller
from src.agent.convergence import stagnation
from src.agent.policy import (
    DEFAULT_PROMOTION_MARGIN,
    SearchPolicy,
    exploit_family,
    sanitize_parameters,
)
from src.agent.research_controller import ResearchLoop
from src.agent.types import ExperimentNode, RunState
from src.evaluation.gate import GateResult


REPO_ROOT = Path(__file__).resolve().parents[1]

BASELINE_PRIMARY = 0.6016

BASELINE_SUMMARY = {
    "best": {
        "experiment_id": "official_fm_seed0",
        "metrics": {"GAUC": 0.6671, "nDCG@5": 0.5358, "primary": 0.6015},
        "artifact_path": "baseline.npz",
    }
}


def node(
    iteration: int,
    experiment_id: str,
    primary: float,
    *,
    family: str = "bpr",
    action: str = "explore",
    replicated_from: str | None = None,
) -> ExperimentNode:
    """One successful node. ``artifact_path``/``candidate_dir`` are per-node, so
    a promotion that moves only *some* of the four best fields is visible."""
    return ExperimentNode(
        iteration=iteration,
        experiment_id=experiment_id,
        hypothesis_id=f"h_{experiment_id}",
        family=family,
        action=action,
        parameters={},
        status="success",
        metrics={"GAUC": primary, "nDCG@5": primary, "primary": primary},
        artifact_path=f"{experiment_id}.npz",
        candidate_dir=f"generated_experiments/{experiment_id}",
        replicated_from=replicated_from,
    )


def observe(policy_under_test: SearchPolicy, state: RunState, item: ExperimentNode) -> None:
    """Append and observe, in the order ``run()`` does it (the node is already on
    ``state.nodes`` when ``observe_success`` recomputes the ratchet)."""
    state.nodes.append(item)
    policy_under_test.observe_success(state, item)


def best_fields(state: RunState) -> tuple[Any, ...]:
    return (
        state.best_experiment_id,
        None if state.best_metrics is None else state.best_metrics["primary"],
        state.best_artifact_path,
        state.best_candidate_dir,
    )


class FailingProvider:
    """Any model call raises — a *harness* error (``_error_kind``), so the loop's
    circuit breaker ends the run. Reaching the model at all is the observable
    signal that the loop did not stop at the convergence check."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, **kwargs: Any):
        self.calls += 1
        raise RuntimeError("the loop was not supposed to need another proposal")


def stub_gate(**kwargs: Any) -> GateResult:
    return GateResult(status="ok", submission_path=None, details={"rows": 0})


@contextlib.contextmanager
def honest_loop(**budgets: Any):
    """A real ``ResearchLoop`` over a scratch run root, with a failing provider.

    ``budgets`` overrides the budget block; ``max_wall_clock_seconds=0`` parks
    the loop at the top of ``run()`` so only the end-of-run reporting executes,
    which is how the summary tests observe ``best_replicated`` without training
    anything.
    """
    provider = FailingProvider()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = {
            "mode": "research",
            "name": "honest-promotion",
            "data_dir": str(REPO_ROOT / "data" / "KuaiRand-Pure" / "data"),
            "run_root": str(root / "runs"),
            "generated_root": str(root / "generated"),
            "method_catalog": str(REPO_ROOT / "research" / "methods"),
            "discovery_store": str(root / "discoveries.json"),
            "campaign_log": str(root / "campaign_log.md"),
            "official_validation_baseline": BASELINE_PRIMARY,
            "llm": {"max_total_tokens": 1000},
            "budgets": {
                "max_iterations": 5,
                "max_wall_clock_seconds": 60,
                "experiment_timeout_seconds": 10,
                "test_timeout_seconds": 10,
                "max_debug_repairs": 2,
                # One harness error is enough to end the run, so a loop that
                # keeps going stops on the *first* provider call.
                "max_consecutive_harness_errors": 1,
                **budgets,
            },
            "convergence": {"epsilon": 0.002, "patience": 3},
            "replication_seeds": [1, 2],
        }
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        yield ResearchLoop(
            config, config_path, provider=provider, baseline_summary=BASELINE_SUMMARY
        ), provider


def run_summary(loop: ResearchLoop) -> dict[str, Any]:
    with patch.object(research_controller, "run_gate", stub_gate):
        with contextlib.redirect_stdout(io.StringIO()):
            run_dir = loop.run()
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 1. Promotion margin
# --------------------------------------------------------------------------- #


class PromotionMarginTests(unittest.TestCase):
    def test_a_sub_margin_score_does_not_displace_the_best(self):
        """``best_*`` moves only for a difference the measurement can resolve.

        The three scores are the point: +0.0014 over the incumbent is a real
        improvement, +0.0009 is inside the seed noise. A bare ``>`` promotes
        both; the margin promotes only the first, and all four best fields move
        together or not at all.
        """
        search = SearchPolicy(0.002, 3, [])
        self.assertEqual(search.promotion_margin, DEFAULT_PROMOTION_MARGIN)
        state = RunState("run", "running", "now", BASELINE_PRIMARY)

        first = node(1, "cand_a", 0.6030)
        observe(search, state, first)
        self.assertEqual(
            best_fields(state),
            ("cand_a", 0.6030, "cand_a.npz", "generated_experiments/cand_a"),
        )

        # +0.0009: better, but by less than the margin — the incumbent stands.
        observe(search, state, node(2, "cand_noise", 0.6039))
        self.assertEqual(
            best_fields(state),
            ("cand_a", 0.6030, "cand_a.npz", "generated_experiments/cand_a"),
        )

        # +0.0014 over the same incumbent: over the margin, so it is promoted —
        # which is what makes the assertion above about the margin rather than
        # about promotion being broken outright.
        observe(search, state, node(3, "cand_real", 0.6044))
        self.assertEqual(
            best_fields(state),
            ("cand_real", 0.6044, "cand_real.npz", "generated_experiments/cand_real"),
        )

    def test_the_stagnation_ratchet_is_untouched_by_the_margin(self):
        """T5's ratchet reads the score sequence, not the promotion decision.

        The margin gates the four ``best_*`` fields and nothing else: the
        ``meaningful_best``/``stagnant_iterations`` pair must still be exactly
        ``convergence.stagnation`` over the baseline-seeded sequence.
        """
        search = SearchPolicy(0.002, 3, [])
        state = RunState("run", "running", "now", BASELINE_PRIMARY)
        scores = [0.6030, 0.6039, 0.6044, 0.6045, 0.6100]
        for index, score in enumerate(scores, start=1):
            observe(search, state, node(index, f"cand_{index}", score))
            expected = stagnation([BASELINE_PRIMARY] + scores[:index], 0.002)
            self.assertEqual(
                (state.meaningful_best, state.stagnant_iterations), expected, index
            )

    def test_the_margin_is_configurable_through_the_convergence_block(self):
        """Plumbed exactly like epsilon/patience, and optional."""
        with honest_loop(max_wall_clock_seconds=0) as (loop, _):
            self.assertEqual(loop.policy.promotion_margin, DEFAULT_PROMOTION_MARGIN)
        with honest_loop(max_wall_clock_seconds=0) as (loop, _):
            loop.convergence["promotion_margin"] = 0.05
            rebuilt = SearchPolicy(
                epsilon=float(loop.convergence["epsilon"]),
                patience=int(loop.convergence["patience"]),
                replication_seeds=[],
                promotion_margin=float(loop.convergence["promotion_margin"]),
            )
            self.assertEqual(rebuilt.promotion_margin, 0.05)


# --------------------------------------------------------------------------- #
# 2. The exploit lock keys on the same margin
# --------------------------------------------------------------------------- #


class ExploitLockTests(unittest.TestCase):
    def test_a_sub_margin_lead_over_the_baseline_does_not_engage_the_lock(self):
        """+0.0004 over the baseline is noise, and noise is not a lead.

        The old threshold was ``> baseline``, so this state locked the run onto
        ``bpr`` — and, because ``should_stop`` is gated on the lock being clear,
        it also blocked the run from ever ending on ``converged``.
        """
        state = RunState("run", "running", "now", BASELINE_PRIMARY, stagnant_iterations=3)
        state.nodes.append(node(1, "cand_noise", 0.6020))
        state.best_experiment_id = "cand_noise"
        state.best_metrics = {"primary": 0.6020}

        self.assertIsNone(exploit_family(state))
        self.assertTrue(SearchPolicy(0.002, 3, []).should_stop(state))

        # Non-vacuity: +0.0015 is over the margin and does engage the lock.
        state.nodes.append(node(2, "cand_lead", 0.6031))
        state.best_experiment_id = "cand_lead"
        state.best_metrics = {"primary": 0.6031}
        self.assertEqual(exploit_family(state), "bpr")
        self.assertFalse(SearchPolicy(0.002, 3, []).should_stop(state))

    def test_a_sub_margin_best_does_not_renew_the_follow_up_window(self):
        """The window is counted from the best node, so promotion is the reset.

        Driven through ``observe_success`` because that is where the two rules
        meet: under the old bare ``>`` the +0.0004 candidate below became the
        best, the follow-up count restarted at zero, and the lock re-engaged —
        indefinitely, for as long as noise kept producing new "bests".
        """
        search = SearchPolicy(0.002, 3, [])
        state = RunState("run", "running", "now", BASELINE_PRIMARY)

        observe(search, state, node(1, "cand_lead", 0.6031))
        self.assertEqual(exploit_family(state), "bpr")

        # The two controlled follow-ups the lead is owed; after them the run is
        # free to diversify.
        observe(search, state, node(2, "cand_followup_1", 0.6032))
        observe(search, state, node(3, "cand_followup_2", 0.6033))
        self.assertIsNone(exploit_family(state))

        # +0.0004 over the incumbent: not a promotion, so not a reset either.
        observe(search, state, node(4, "cand_noise", 0.6035))
        self.assertEqual(state.best_experiment_id, "cand_lead")
        self.assertIsNone(exploit_family(state))

        # +0.0014 over the incumbent is a promotion, and a promotion *does*
        # earn a fresh window — the lock still works, it is only harder to fool.
        observe(search, state, node(5, "cand_better", 0.6045))
        self.assertEqual(state.best_experiment_id, "cand_better")
        self.assertEqual(exploit_family(state), "bpr")


# --------------------------------------------------------------------------- #
# 3. The family-breadth guarantee is connected to the stop path
# --------------------------------------------------------------------------- #


def converged_state(loop: ResearchLoop, *covered: str) -> None:
    """Make ``loop``'s state look converged, with one node per named family.

    Every score sits below ``baseline + margin``, so no exploit lead is
    outstanding and ``SearchPolicy.should_stop`` is True on its own.
    """
    for index, family in enumerate(covered, start=1):
        loop.state.nodes.append(node(index, f"cand_{family}", 0.6010, family=family))
    loop.state.iteration_count = len(covered)
    loop.state.stagnant_iterations = int(loop.convergence["patience"])


class CoverageStopTests(unittest.TestCase):
    def test_a_converged_run_with_an_uncovered_family_does_not_stop(self):
        """One family tried out of four is not a verdict about the search space.

        ``should_stop`` is True here — the harness ratchet is satisfied and no
        lead is outstanding — and the old loop wrote ``stop_reason="converged"``
        on the strength of it. The provider is the discriminator: a loop that
        stops never calls it, so ``harness_error_breaker`` is proof that the run
        went back for another proposal instead.
        """
        with honest_loop() as (loop, provider):
            converged_state(loop, "bpr")
            self.assertTrue(loop.policy.should_stop(loop.state))
            self.assertFalse(policy.coverage_complete(loop.state))
            self.assertFalse(loop._may_stop_for_convergence())

            summary = run_summary(loop)

        self.assertNotEqual(summary["stop_reason"], "converged")
        self.assertEqual(summary["stop_reason"], "harness_error_breaker")
        self.assertEqual(provider.calls, 1)

    def test_a_covered_run_still_stops_on_converged(self):
        """Non-vacuity: with every coverage family tried, the verdict stands."""
        with honest_loop() as (loop, provider):
            converged_state(loop, *sorted(families.coverage_families()))
            self.assertTrue(loop._may_stop_for_convergence())

            summary = run_summary(loop)

        self.assertEqual(summary["stop_reason"], "converged")
        self.assertEqual(provider.calls, 0)

    def test_an_exhausted_proposal_budget_releases_the_coverage_gate(self):
        """Coverage must not be able to hold a run past its budget.

        Once no proposal is left, an uncovered family is unreachable and
        blocking on it would only spin — so the gate opens and the run reports
        the convergence it did reach.
        """
        with honest_loop(max_proposals=2) as (loop, provider):
            converged_state(loop, "bpr")
            loop.state.proposal_attempts = 2
            self.assertFalse(policy.coverage_complete(loop.state))
            self.assertTrue(loop._may_stop_for_convergence())

            summary = run_summary(loop)

        self.assertEqual(summary["stop_reason"], "converged")
        self.assertEqual(provider.calls, 0)


def resume(loop: ResearchLoop, provider: FailingProvider) -> ResearchLoop:
    """Persist ``loop``'s state and reopen its run directory, as ``--resume`` does."""
    loop._save()
    return ResearchLoop(
        loop.config,
        loop.config_path,
        provider=provider,
        resume_dir=loop.run_dir,
        baseline_summary=BASELINE_SUMMARY,
    )


class ResumedCoverageStopTests(unittest.TestCase):
    """The stop gate reads the *restored* ``proposal_attempts``.

    Resume rewrites only ``status`` and ``stop_reason``; every counter comes back
    verbatim from ``state.json``. Both halves of ``_may_stop_for_convergence``
    therefore have to keep meaning the same thing after a resume as before one,
    and which half applies is decided by that restored counter — so both are
    pinned here.
    """

    def test_a_resume_with_proposals_left_keeps_the_breadth_guarantee(self):
        with honest_loop(max_proposals=4) as (loop, _):
            converged_state(loop, "bpr")
            loop.state.proposal_attempts = 1
            loop.state.stop_reason = "wall_clock_budget_reached"

            resumed_provider = FailingProvider()
            resumed = resume(loop, resumed_provider)
            self.assertEqual(resumed.state.proposal_attempts, 1)
            self.assertIsNone(resumed.state.stop_reason)
            self.assertTrue(resumed.policy.should_stop(resumed.state))
            self.assertFalse(policy.coverage_complete(resumed.state))
            self.assertFalse(resumed._may_stop_for_convergence())

            summary = run_summary(resumed)

        self.assertNotEqual(summary["stop_reason"], "converged")
        self.assertEqual(summary["stop_reason"], "harness_error_breaker")
        self.assertEqual(resumed_provider.calls, 1)

    def test_a_resume_after_the_proposal_budget_still_releases_the_gate(self):
        """The restored total is the *whole* budget, so the gate stays open.

        This is the deliberate reading: an uncovered family is no more reachable
        after a resume than before one, and blocking on it would only spin.
        """
        with honest_loop(max_proposals=2) as (loop, _):
            converged_state(loop, "bpr")
            loop.state.proposal_attempts = 2
            loop.state.stop_reason = "proposal_budget_reached"

            resumed_provider = FailingProvider()
            resumed = resume(loop, resumed_provider)
            self.assertEqual(resumed.state.proposal_attempts, 2)
            self.assertFalse(policy.coverage_complete(resumed.state))
            self.assertTrue(resumed._may_stop_for_convergence())

            summary = run_summary(resumed)

        self.assertEqual(summary["stop_reason"], "converged")
        self.assertEqual(resumed_provider.calls, 0)


# --------------------------------------------------------------------------- #
# 4. The summary reports the replicated spread, not just the winning draw
# --------------------------------------------------------------------------- #


class ReplicationHonestyTests(unittest.TestCase):
    def test_the_summary_reports_the_spread_of_the_best_experiment(self):
        """Replicating a result and then reporting only its best seed is the
        same selection bias the promotion margin exists to stop."""
        with honest_loop(max_wall_clock_seconds=0) as (loop, _):
            loop.state.nodes.extend(
                [
                    node(1, "cand_bpr", 0.6050),
                    node(2, "cand_bpr_s1", 0.6030, action="replicate",
                         replicated_from="cand_bpr"),
                    node(3, "cand_bpr_s2", 0.6040, action="replicate",
                         replicated_from="cand_bpr"),
                ]
            )
            loop.state.best_experiment_id = "cand_bpr"
            loop.state.best_metrics = {"primary": 0.6050}

            summary = run_summary(loop)

        replicated = summary["best_replicated"]
        self.assertEqual(replicated["n"], 3)
        self.assertAlmostEqual(replicated["median_primary"], 0.6040)
        self.assertAlmostEqual(replicated["spread"], 0.0020)
        # The headline is untouched — the spread is reported beside it, not
        # instead of it.
        self.assertEqual(summary["best"]["experiment_id"], "cand_bpr")
        self.assertEqual(summary["best"]["metrics"]["primary"], 0.6050)

    def test_the_summary_reports_null_when_the_best_was_never_replicated(self):
        """"Unknown" is the honest answer; a zero spread would not be."""
        with honest_loop(max_wall_clock_seconds=0) as (loop, _):
            loop.state.nodes.append(node(1, "cand_bpr", 0.6050))
            loop.state.best_experiment_id = "cand_bpr"
            loop.state.best_metrics = {"primary": 0.6050}

            summary = run_summary(loop)

        self.assertIn("best_replicated", summary)
        self.assertIsNone(summary["best_replicated"])

    def test_the_group_is_found_from_a_best_that_is_itself_a_replica(self):
        """A replica can outscore its source, so the group is keyed on the root."""
        with honest_loop(max_wall_clock_seconds=0) as (loop, _):
            loop.state.nodes.extend(
                [
                    node(1, "cand_bpr", 0.6030),
                    node(2, "cand_bpr_s1", 0.6050, action="replicate",
                         replicated_from="cand_bpr"),
                ]
            )
            loop.state.best_experiment_id = "cand_bpr_s1"
            loop.state.best_metrics = {"primary": 0.6050}

            self.assertEqual(
                loop._best_replication(),
                {"n": 2, "median_primary": 0.6040, "spread": 0.6050 - 0.6030},
            )


# --------------------------------------------------------------------------- #
# 5. The margin gates the claim; the gate submits the argmax
# --------------------------------------------------------------------------- #


class SubmissionHonestyTests(unittest.TestCase):
    def test_the_gate_submits_the_argmax_not_the_margin_gated_best(self):
        """A monotone chain of sub-margin gains never promotes, but is real.

        0.6030 -> 0.6035 -> 0.6039: every score is measured against the standing
        *incumbent*, which never moves off `cand_a` because none of them clears
        it by the margin — the right answer for what the run claims. The
        organizers score the file, though, so the gate must be handed `cand_c`,
        and the summary must carry both numbers so the gap is visible instead of
        silent.
        """
        calls: list[dict[str, Any]] = []

        def recorder(**kwargs: Any) -> GateResult:
            calls.append(dict(kwargs))
            return GateResult(status="ok", submission_path=None, details={"rows": 0})

        with honest_loop(max_wall_clock_seconds=0) as (loop, _):
            search = SearchPolicy(0.002, 3, [])
            for index, (experiment_id, primary) in enumerate(
                (("cand_a", 0.6030), ("cand_b", 0.6035), ("cand_c", 0.6039)), start=1
            ):
                observe(search, loop.state, node(index, experiment_id, primary))
            self.assertEqual(loop.state.best_experiment_id, "cand_a")

            with patch.object(research_controller, "run_gate", recorder):
                with contextlib.redirect_stdout(io.StringIO()):
                    run_dir = loop.run()
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0]["node_dir"], REPO_ROOT / "generated_experiments/cand_c"
        )
        self.assertAlmostEqual(summary["max_scored_primary"], 0.6039)
        # The claim is untouched: the margin still holds `best_*` at `cand_a`.
        self.assertEqual(summary["best"]["experiment_id"], "cand_a")
        self.assertAlmostEqual(summary["best"]["metrics"]["primary"], 0.6030)

    def test_a_run_with_no_successful_node_still_falls_back(self):
        """Non-vacuity for the fallback chain: no argmax, no candidate dir, run dir."""
        calls: list[dict[str, Any]] = []

        def recorder(**kwargs: Any) -> GateResult:
            calls.append(dict(kwargs))
            return GateResult(status="ok", submission_path=None, details={"rows": 0})

        with honest_loop(max_wall_clock_seconds=0) as (loop, _):
            with patch.object(research_controller, "run_gate", recorder):
                with contextlib.redirect_stdout(io.StringIO()):
                    run_dir = loop.run()
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(calls[0]["node_dir"], loop.run_dir)

        self.assertIsNone(summary["max_scored_primary"])


# --------------------------------------------------------------------------- #
# 6. Addendum: `k` is a registry grid key, not a hard-coded pin
# --------------------------------------------------------------------------- #


class WidenedCapacityGridTests(unittest.TestCase):
    def test_an_off_grid_k_is_rejected_in_the_grids_own_wording(self):
        """The retired pin's prose must not outlive the pin.

        `k` was fixed at 16 with "Ranking-loss attribution requires k=16 in the
        first research run."; the capacity unfreeze widened it to (8, 16, 32,
        64) on the ranking families. The legacy message was still what the
        sanitiser raised for an off-grid width — telling the Builder, in a
        re-prompt it is expected to act on, to use a value the grid no longer
        singles out and giving no hint of the widths it does allow.
        """
        for family in ("bpr", "group_softmax"):
            with self.subTest(family=family):
                defaults = families.FAMILIES[family].defaults
                allowed = families.FAMILIES[family].grid["k"]
                with self.assertRaises(ValueError) as rejected:
                    sanitize_parameters(family, {**defaults, "k": 128})
                message = str(rejected.exception)
                self.assertEqual(
                    message,
                    f"{family} k=128 is outside the registry grid {allowed!r}.",
                )
                self.assertNotIn("Ranking-loss attribution", message)
                self.assertNotIn("k=16", message)

                # Non-vacuity: every width the widened grid names is accepted.
                for width in allowed:
                    self.assertEqual(
                        sanitize_parameters(family, {**defaults, "k": width})["k"], width
                    )


if __name__ == "__main__":
    unittest.main()
