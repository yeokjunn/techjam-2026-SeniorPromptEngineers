"""What the run actually submits, and what it spends its leftover budget on.

Three pipeline changes from report 6 (`.superpowers/sdd/A-loop-robustness/
investigation/6-new-methods.md`, sections P1-P3), all of them about the number
the organizers compute rather than about process hygiene:

* **P1 — the submission is an ensemble.** Every successful node already wrote a
  validated, row-order-checked ``test_scores.npy``, so the rank-mean of them
  costs no extra training: measured 0.60363 against 0.60339 for the better of
  the two artifacts on disk. The variance argument is the larger half — the
  submitted number is one draw from sigma ~= 9e-4, and averaging K members
  divides the independent part of that by sqrt(K).
* **P2 — selection is by replication-group median.** The old ``_argmax_candidate``
  maxed over every successful node *including replicas*, which made replication
  make the selection bias worse: two extra seeds of a good config are two extra
  draws in the maximum, so the submission rode the luckiest seed.
* **P3 — leftover budget becomes seeds.** The last full run used 33 of 360
  minutes and 177k of 600k tokens and stopped on ``converged``. Replication
  fires only on a gain of more than epsilon over the *baseline*, so most runs
  reach the gate with no replication group at all — which leaves P2 with nothing
  to take a median of and P1 with one member.

Every test here is written so the previous behaviour cannot satisfy it: the
argmax picks a different node from the group median (class 3), the single-node
gate call is a different directory from the ensemble one (class 2), and a loop
that stops at the first convergence check never schedules a replication
(class 4).
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

import numpy as np

from src.agent import families, research_controller
from src.agent.research_controller import (
    ResearchLoop,
    _average_ranks,
    _rank_mean,
)
from src.agent.types import ExperimentNode
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


class FailingProvider:
    """Any model call raises: reaching the model at all is the observable signal
    that the loop went back for another proposal instead of stopping."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, **kwargs: Any):
        self.calls += 1
        raise RuntimeError("the loop was not supposed to need another proposal")


@contextlib.contextmanager
def loop_over(root: Path, **budgets: Any):
    """A real ``ResearchLoop`` over a scratch run root, with a failing provider."""
    provider = FailingProvider()
    config = {
        "mode": "research",
        "name": "submission-ensemble",
        "data_dir": str(REPO_ROOT / "data" / "KuaiRand-Pure" / "data"),
        "run_root": str(root / "runs"),
        "generated_root": str(root / "generated"),
        "method_catalog": str(REPO_ROOT / "research" / "methods"),
        "discovery_store": str(root / "discoveries.json"),
        "campaign_log": str(root / "campaign_log.md"),
        "official_validation_baseline": BASELINE_PRIMARY,
        "llm": {"max_total_tokens": 1000},
        "budgets": {
            "max_iterations": 20,
            "max_wall_clock_seconds": 600,
            "experiment_timeout_seconds": 10,
            "test_timeout_seconds": 10,
            "max_debug_repairs": 2,
            # One harness error ends the run, so a loop that keeps going stops
            # on the first provider call rather than spinning.
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


def node(
    loop: ResearchLoop,
    iteration: int,
    experiment_id: str,
    primary: float,
    *,
    family: str = "bpr",
    action: str = "explore",
    replicated_from: str | None = None,
    test_scores: np.ndarray | None = None,
    workspace: bool = False,
) -> ExperimentNode:
    """One successful node, with its artifact directory materialised on disk.

    The linkage the ensemble follows is the real one: the worker saves
    ``test_scores.npy`` next to ``model.npz`` in the run's artifact directory
    (``run_candidate.py:103,128``), and the node records that checkpoint in
    ``artifact_path``. ``workspace=True`` also lays down the candidate workspace
    a replication rebuilds from.
    """
    artifact_dir = loop.run_dir / "artifacts" / f"{iteration:03d}_{experiment_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "model.npz").write_bytes(b"")
    if test_scores is not None:
        np.save(artifact_dir / "test_scores.npy", np.asarray(test_scores, dtype=np.float64))
    candidate_dir = Path(loop.generated_root) / loop.state.run_id / experiment_id
    if workspace:
        candidate_dir.mkdir(parents=True, exist_ok=True)
        (candidate_dir / "manifest.json").write_text(
            json.dumps({"parameters": {"seed": 0}}), encoding="utf-8"
        )
        (candidate_dir / "candidate.py").write_text("", encoding="utf-8")
        (candidate_dir / "test_candidate.py").write_text("", encoding="utf-8")
    item = ExperimentNode(
        iteration=iteration,
        experiment_id=experiment_id,
        hypothesis_id=f"h_{experiment_id}",
        family=family,
        action=action,
        parameters={},
        status="success",
        metrics={"GAUC": primary, "nDCG@5": primary, "primary": primary},
        artifact_path=str(artifact_dir / "model.npz"),
        candidate_dir=str(candidate_dir),
        replicated_from=replicated_from,
    )
    loop.state.nodes.append(item)
    return item


class Recorder:
    """A ``run_gate`` double that records every call and reports success."""

    def __init__(self, status: str = "ok") -> None:
        self.calls: list[dict[str, Any]] = []
        self.status = status

    def __call__(self, **kwargs: Any) -> GateResult:
        self.calls.append(dict(kwargs))
        return GateResult(status=self.status, submission_path=None, details={"rows": 0})


def run_with(loop: ResearchLoop, gate) -> dict[str, Any]:
    with patch.object(research_controller, "run_gate", gate):
        with contextlib.redirect_stdout(io.StringIO()):
            run_dir = loop.run()
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 1. Rank averaging
# --------------------------------------------------------------------------- #


class RankMeanTests(unittest.TestCase):
    def test_ranks_are_one_based_and_ties_share_their_average(self):
        """``scipy.stats.rankdata(method="average")`` without scipy.

        The tie rule is the point: a member that cannot separate two rows must
        not break the tie by row order, because that would smuggle the row index
        into the ensemble as a tiebreaker no member voted for.
        """
        np.testing.assert_allclose(_average_ranks(np.array([10.0, 30.0, 20.0])), [1, 3, 2])
        np.testing.assert_allclose(_average_ranks(np.array([5.0, 5.0, 1.0])), [2.5, 2.5, 1.0])
        # A whole block of ties: ranks 2,3,4 -> 3.0 each, and the block's total
        # is conserved (2+3+4 == 9).
        ranks = _average_ranks(np.array([0.0, 7.0, 7.0, 7.0, 9.0]))
        np.testing.assert_allclose(ranks, [1.0, 3.0, 3.0, 3.0, 5.0])
        self.assertAlmostEqual(float(ranks.sum()), 15.0)
        # Every element tied: all share (1+2+3)/3.
        np.testing.assert_allclose(_average_ranks(np.array([4.0, 4.0, 4.0])), [2.0, 2.0, 2.0])
        np.testing.assert_allclose(_average_ranks(np.array([2.0])), [1.0])

    def test_the_ensemble_is_the_mean_of_the_members_ranks(self):
        first = np.array([1.0, 2.0, 3.0, 4.0])
        second = np.array([4.0, 3.0, 2.0, 1.0])
        # ranks [1,2,3,4] and [4,3,2,1]: two members that disagree completely
        # average to a flat vector, which no mean of the raw scores would give.
        np.testing.assert_allclose(_rank_mean([first, second]), [2.5, 2.5, 2.5, 2.5])
        # One member is the identity.
        np.testing.assert_allclose(_rank_mean([first]), [1.0, 2.0, 3.0, 4.0])

    def test_the_ensemble_is_scale_free(self):
        """Rank-averaging needs no per-member calibration — the reason report 6
        prefers it to a z-score mean it measures as indistinguishable (0.60363
        vs 0.60362). A member whose scores are 1000x another's must not
        dominate, which is exactly what a mean of the raw vectors would do."""
        small = np.array([0.1, 0.2, 0.3])
        huge = np.array([-500.0, 900.0, 100.0])
        np.testing.assert_allclose(
            _rank_mean([small, huge]), _rank_mean([small, np.array([1.0, 3.0, 2.0])])
        )
        # Non-vacuity: the raw mean *is* dominated by the large member, so the
        # two operations genuinely differ.
        raw = (small + huge) / 2.0
        self.assertNotEqual(list(np.argsort(raw)), list(np.argsort(_rank_mean([small, huge]))))

    def test_a_tied_member_still_contributes_its_half_vote(self):
        """A member with no opinion between two rows leaves the decision to the
        others rather than to row order."""
        opinionated = np.array([1.0, 2.0])
        indifferent = np.array([7.0, 7.0])
        np.testing.assert_allclose(
            _rank_mean([opinionated, indifferent]), [(1 + 1.5) / 2, (2 + 1.5) / 2]
        )

    def test_an_empty_member_is_not_a_crash(self):
        np.testing.assert_allclose(_average_ranks(np.array([])), [])
        with self.assertRaises(ValueError):
            _rank_mean([])


# --------------------------------------------------------------------------- #
# 2. P1 — the gate is handed the ensemble
# --------------------------------------------------------------------------- #


class EnsembleSubmissionTests(unittest.TestCase):
    def test_two_members_are_rank_averaged_and_handed_to_the_gate(self):
        """The gate submits ``<run_dir>/artifacts/_ensemble``, not one node.

        That directory is the whole gate-layout decision: ``gate.py:129-132``
        searches ``node_dir/test_scores.npy`` first and
        ``run_dir/artifacts/<node_dir.name>/test_scores.npy`` second, and this
        one file satisfies both — so B's module needs no edit at all.
        """
        recorder = Recorder()
        first = np.array([0.9, 0.1, 0.5, 0.3])
        second = np.array([0.1, 0.9, 0.4, 0.6])
        with tempfile.TemporaryDirectory() as directory:
            with loop_over(Path(directory), max_wall_clock_seconds=0) as (loop, _):
                node(loop, 1, "cand_a", 0.6040, test_scores=first)
                node(loop, 2, "cand_b", 0.6030, family="group_softmax", test_scores=second)
                summary = run_with(loop, recorder)
                run_dir = loop.run_dir

                self.assertEqual(len(recorder.calls), 1)
                node_dir = recorder.calls[0]["node_dir"]
                self.assertEqual(node_dir, run_dir / "artifacts" / "_ensemble")
                self.assertTrue(node_dir.is_absolute())
                # Both of the gate's two search paths resolve to the one file.
                self.assertTrue((node_dir / "test_scores.npy").is_file())
                self.assertTrue(
                    (run_dir / "artifacts" / node_dir.name / "test_scores.npy").is_file()
                )
                np.testing.assert_allclose(
                    np.load(node_dir / "test_scores.npy"), _rank_mean([first, second])
                )
                manifest = json.loads(
                    (node_dir / "ensemble_manifest.json").read_text(encoding="utf-8")
                )

        self.assertEqual(manifest["method"], "rank_mean")
        self.assertEqual(manifest["weights"], "equal")
        self.assertEqual(manifest["n_members"], 2)
        self.assertEqual(manifest["rows"], 4)
        self.assertEqual(
            [member["experiment_id"] for member in manifest["members"]],
            ["cand_a", "cand_b"],
        )
        self.assertEqual(
            summary["submission"], {"method": "rank_mean_ensemble", "n_members": 2}
        )
        # The claim is untouched: only what is *submitted* changed.
        self.assertAlmostEqual(summary["max_scored_primary"], 0.6040)

    def test_a_single_member_keeps_the_previous_behaviour(self):
        """One node is not an ensemble: the gate gets that node's own directory
        and no ``_ensemble`` directory is written at all."""
        recorder = Recorder()
        with tempfile.TemporaryDirectory() as directory:
            with loop_over(Path(directory), max_wall_clock_seconds=0) as (loop, _):
                only = node(loop, 1, "cand_a", 0.6040, test_scores=np.array([0.2, 0.4]))
                summary = run_with(loop, recorder)
                ensemble_dir = loop.run_dir / "artifacts" / "_ensemble"
                self.assertFalse(ensemble_dir.exists())

        self.assertEqual(len(recorder.calls), 1)
        self.assertEqual(recorder.calls[0]["node_dir"], Path(only.candidate_dir))
        self.assertEqual(summary["submission"], {"method": "single", "n_members": 1})

    def test_a_member_of_the_wrong_length_is_dropped(self):
        """Ensembling must never turn a passing gate into ``bad_test_scores``.

        The gate validates the submitted vector against ``load_test_meta``
        (``gate.py:141-152``), so a member of a different length is dropped
        rather than reconciled — and with only one member left, the single-node
        path stands.
        """
        recorder = Recorder()
        with tempfile.TemporaryDirectory() as directory:
            with loop_over(Path(directory), max_wall_clock_seconds=0) as (loop, _):
                best = node(loop, 1, "cand_a", 0.6040, test_scores=np.array([0.2, 0.4, 0.6]))
                node(loop, 2, "cand_short", 0.6030, test_scores=np.array([0.1, 0.2]))
                summary = run_with(loop, recorder)

        self.assertEqual(recorder.calls[0]["node_dir"], Path(best.candidate_dir))
        self.assertEqual(summary["submission"], {"method": "single", "n_members": 1})

    def test_a_node_without_scores_on_disk_is_not_a_member(self):
        recorder = Recorder()
        with tempfile.TemporaryDirectory() as directory:
            with loop_over(Path(directory), max_wall_clock_seconds=0) as (loop, _):
                best = node(loop, 1, "cand_a", 0.6040, test_scores=np.array([0.2, 0.4]))
                node(loop, 2, "cand_no_scores", 0.6030)
                summary = run_with(loop, recorder)

        self.assertEqual(recorder.calls[0]["node_dir"], Path(best.candidate_dir))
        self.assertEqual(summary["submission"], {"method": "single", "n_members": 1})

    def test_a_materially_weaker_member_is_kept_out_of_the_vote(self):
        """Report 6's membership guard: *within epsilon of the argmax*.

        Equal weights make a member's vote independent of its quality, so a
        successful-but-weak node would take a third to a half of the vote on the
        file the organizers score. 0.6030 is inside `epsilon = 0.002` of the
        0.6040 argmax and votes; 0.5850 is not and is dropped — the two
        directions are asserted together, so a filter that excluded everything
        would fail this as surely as one that excluded nothing.
        """
        recorder = Recorder()
        strong = np.array([0.9, 0.1, 0.5, 0.3])
        near = np.array([0.1, 0.9, 0.4, 0.6])
        weak = np.array([0.5, 0.5, 0.9, 0.1])
        with tempfile.TemporaryDirectory() as directory:
            with loop_over(Path(directory), max_wall_clock_seconds=0) as (loop, _):
                node(loop, 1, "cand_a", 0.6040, test_scores=strong)
                node(loop, 2, "cand_b", 0.6030, family="group_softmax", test_scores=near)
                node(loop, 3, "cand_weak", 0.5850, family="history_features",
                     test_scores=weak)
                summary = run_with(loop, recorder)
                node_dir = recorder.calls[0]["node_dir"]
                manifest = json.loads(
                    (node_dir / "ensemble_manifest.json").read_text(encoding="utf-8")
                )
                # The vote is the two survivors, and *only* them: the weak
                # member's scores would have moved every value here.
                np.testing.assert_allclose(
                    np.load(node_dir / "test_scores.npy"), _rank_mean([strong, near])
                )

        self.assertEqual(
            [member["experiment_id"] for member in manifest["members"]],
            ["cand_a", "cand_b"],
        )
        self.assertEqual(
            summary["submission"], {"method": "rank_mean_ensemble", "n_members": 2}
        )

    def test_the_quality_floor_can_leave_a_single_member(self):
        """"Fall back to today's single-node path if fewer than two qualify" —
        the other half of report 6's guard. Both members here have usable scores
        on disk, so only the floor can produce the single-node path."""
        recorder = Recorder()
        with tempfile.TemporaryDirectory() as directory:
            with loop_over(Path(directory), max_wall_clock_seconds=0) as (loop, _):
                best = node(loop, 1, "cand_a", 0.6040, test_scores=np.array([0.2, 0.4]))
                node(loop, 2, "cand_weak", 0.5850, test_scores=np.array([0.4, 0.2]))
                summary = run_with(loop, recorder)
                self.assertFalse((loop.run_dir / "artifacts" / "_ensemble").exists())

        self.assertEqual(recorder.calls[0]["node_dir"], Path(best.candidate_dir))
        self.assertEqual(summary["submission"], {"method": "single", "n_members": 1})

    def test_a_failed_ensemble_build_still_leaves_a_summary(self):
        """The ensemble does filesystem work between the final state and
        ``summary.json``; an OSError there must cost the ensemble and nothing
        else. This is the same guarantee the gate call's own ``try`` exists for,
        and P1 would otherwise have introduced the hole inside A's own module.
        """
        recorder = Recorder()

        def explode(self, candidate):
            raise OSError("disk full")

        with tempfile.TemporaryDirectory() as directory:
            with loop_over(Path(directory), max_wall_clock_seconds=0) as (loop, _):
                best = node(loop, 1, "cand_a", 0.6040, test_scores=np.array([0.2, 0.4]))
                node(loop, 2, "cand_b", 0.6030, test_scores=np.array([0.4, 0.2]))
                with patch.object(ResearchLoop, "_ensemble_submission", explode):
                    summary = run_with(loop, recorder)
                written = sorted(path.name for path in loop.run_dir.iterdir() if path.is_file())
                memory = [
                    json.loads(line)
                    for line in (loop.run_dir / "research_memory.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]

        # Everything the run owes still landed, and the submission is the one it
        # would have made before P1 existed.
        for name in ("summary.json", "best.json", "results.json", "state.json"):
            self.assertIn(name, written)
        self.assertEqual(recorder.calls[0]["node_dir"], Path(best.candidate_dir))
        self.assertEqual(summary["submission"], {"method": "single", "n_members": 1})
        self.assertEqual(summary["gate"]["status"], "ok")
        failures = [item for item in memory if item["type"] == "ensemble_error"]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["error_type"], "OSError")

    def test_a_rejected_ensemble_falls_back_to_the_single_node(self):
        """P1 must not be able to cost the run its submission.

        The gate is the only judge of whether the file is submittable, so if it
        refuses the ensemble the loop retries the node it would have submitted
        before P1 existed, and says so in the summary.
        """

        class RefuseEnsemble(Recorder):
            def __call__(self, **kwargs: Any) -> GateResult:
                self.calls.append(dict(kwargs))
                if kwargs["node_dir"].name == "_ensemble":
                    return GateResult(status="error", details={"reason": "bad_test_scores"})
                return GateResult(status="ok", submission_path="s.csv", details={"rows": 2})

        gate = RefuseEnsemble()
        with tempfile.TemporaryDirectory() as directory:
            with loop_over(Path(directory), max_wall_clock_seconds=0) as (loop, _):
                best = node(loop, 1, "cand_a", 0.6040, test_scores=np.array([0.2, 0.4]))
                node(loop, 2, "cand_b", 0.6030, test_scores=np.array([0.4, 0.2]))
                summary = run_with(loop, gate)

        self.assertEqual(len(gate.calls), 2)
        self.assertEqual(gate.calls[0]["node_dir"].name, "_ensemble")
        self.assertEqual(gate.calls[1]["node_dir"], Path(best.candidate_dir))
        self.assertEqual(summary["gate"]["status"], "ok")
        self.assertEqual(
            summary["submission"],
            {"method": "single", "n_members": 1, "ensemble_rejected": True},
        )


# --------------------------------------------------------------------------- #
# 3. P2 — replicas count once, at their group median
# --------------------------------------------------------------------------- #


class GroupMedianSelectionTests(unittest.TestCase):
    def test_a_lucky_replica_does_not_win_the_submission(self):
        """Three seeds of one config lose to a single node they out-*max*.

        ``cand_rep``: 0.6050 / 0.6031 / 0.6030, median **0.6031**.
        ``cand_solo``: 0.6040, a group of one, median **0.6040**.
        The old global argmax submits ``cand_rep``'s luckiest seed (0.6050); the
        median says the configuration is worth 0.6031 and ``cand_solo`` wins. The
        two rules therefore disagree here, which is what makes the test able to
        fail against the old one.
        """
        recorder = Recorder()
        with tempfile.TemporaryDirectory() as directory:
            with loop_over(Path(directory), max_wall_clock_seconds=0) as (loop, _):
                node(loop, 1, "cand_rep", 0.6050)
                node(loop, 2, "cand_rep_s1", 0.6031, action="replicate",
                     replicated_from="cand_rep")
                node(loop, 3, "cand_rep_s2", 0.6030, action="replicate",
                     replicated_from="cand_rep")
                solo = node(loop, 4, "cand_solo", 0.6040, family="group_softmax")
                summary = run_with(loop, recorder)

        self.assertEqual(recorder.calls[0]["node_dir"], Path(solo.candidate_dir))
        # Both numbers are reported: the raw maximum stays visible so the gap the
        # median creates is legible rather than silent.
        self.assertAlmostEqual(summary["max_scored_primary"], 0.6050)
        self.assertAlmostEqual(summary["best_group_median_primary"], 0.6040)

    def test_a_winning_group_submits_its_original_not_its_best_replica(self):
        """The original is the artifact the replicas were copied from, so it is
        the directory certain to exist — and picking the best replica instead
        would reintroduce exactly the max-over-seeds bias the median removes."""
        recorder = Recorder()
        with tempfile.TemporaryDirectory() as directory:
            with loop_over(Path(directory), max_wall_clock_seconds=0) as (loop, _):
                original = node(loop, 1, "cand_rep", 0.6040)
                node(loop, 2, "cand_rep_s1", 0.6060, action="replicate",
                     replicated_from="cand_rep")
                node(loop, 3, "cand_rep_s2", 0.6045, action="replicate",
                     replicated_from="cand_rep")
                node(loop, 4, "cand_solo", 0.6041, family="group_softmax")
                summary = run_with(loop, recorder)

        # Group median 0.6045 beats the solo 0.6041, and the node handed over is
        # the original, not the 0.6060 replica the argmax would have chosen.
        self.assertEqual(recorder.calls[0]["node_dir"], Path(original.candidate_dir))
        self.assertAlmostEqual(summary["max_scored_primary"], 0.6060)
        self.assertAlmostEqual(summary["best_group_median_primary"], 0.6045)

    def test_an_unreplicated_run_selects_exactly_what_the_argmax_did(self):
        """Non-vacuity: with no replicas every group is a group of one, so the
        median *is* the score and the previous behaviour is unchanged."""
        recorder = Recorder()
        with tempfile.TemporaryDirectory() as directory:
            with loop_over(Path(directory), max_wall_clock_seconds=0) as (loop, _):
                node(loop, 1, "cand_a", 0.6030)
                node(loop, 2, "cand_b", 0.6035, family="group_softmax")
                top = node(loop, 3, "cand_c", 0.6039, family="history_features")
                summary = run_with(loop, recorder)

        self.assertEqual(recorder.calls[0]["node_dir"], Path(top.candidate_dir))
        self.assertAlmostEqual(summary["max_scored_primary"], 0.6039)
        self.assertAlmostEqual(summary["best_group_median_primary"], 0.6039)


# --------------------------------------------------------------------------- #
# 4. P3 — leftover budget becomes seeds of the top-2 configs
# --------------------------------------------------------------------------- #


def converged_state(loop: ResearchLoop, primaries: dict[str, float]) -> None:
    """Make ``loop``'s state look converged and covered, with real workspaces.

    Every score sits below ``baseline + margin``, so no exploit lead is
    outstanding and ``SearchPolicy.should_stop`` is True on its own; one node per
    coverage family opens ``_may_stop_for_convergence``. The workspaces exist
    because a replication rebuilds the candidate from them.
    """
    for index, family in enumerate(sorted(families.coverage_families()), start=1):
        node(
            loop,
            index,
            f"cand_{family}",
            primaries[family],
            family=family,
            workspace=True,
        )
    loop.state.iteration_count = len(primaries)
    loop.state.stagnant_iterations = int(loop.convergence["patience"])


class ReplicationDouble:
    """Stands in for ``_replication``: records the task and adds the node it
    would have produced, so a second pass sees the work as already done."""

    def __init__(self, loop: ResearchLoop) -> None:
        self.loop = loop
        self.tasks: list[dict[str, Any]] = []

    def __call__(self, task: dict[str, Any]) -> None:
        self.tasks.append(dict(task))
        source = str(task["source_experiment"])
        seed = int(task["seed"])
        self.loop.state.iteration_count += 1
        node(
            self.loop,
            self.loop.state.iteration_count,
            f"{source[:65]}_seed{seed}",
            0.6010,
            family=next(
                item.family for item in self.loop.state.nodes
                if item.experiment_id == source
            ),
            action="replicate",
            replicated_from=source,
        )


class FinalReplicationTests(unittest.TestCase):
    PRIMARIES = {"bpr": 0.6010, "group_softmax": 0.6020, "history_features": 0.6005}

    def test_leftover_budget_re_seeds_the_top_two_configs_and_still_converges(self):
        """A converged-ready run with 0 tokens spent and no clock burnt.

        `group_softmax` (0.6020) and `bpr` (0.6010) are the top two group
        medians; `history_features` (0.6005) is not re-seeded. The stop the run
        eventually takes is still ``converged`` — P3 defers it, it does not
        rename it — and the provider is never called, so the deferral bought
        replications and not another proposal.
        """
        with tempfile.TemporaryDirectory() as directory:
            with loop_over(Path(directory)) as (loop, provider):
                converged_state(loop, self.PRIMARIES)
                self.assertTrue(loop.policy.should_stop(loop.state))
                self.assertTrue(loop._may_stop_for_convergence())
                double = ReplicationDouble(loop)
                with patch.object(loop, "_replication", double):
                    summary = run_with(loop, Recorder())
                memory = [
                    json.loads(line)
                    for line in (loop.run_dir / "research_memory.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]

        self.assertEqual(
            double.tasks,
            [
                {"source_experiment": "cand_group_softmax", "seed": 1},
                {"source_experiment": "cand_group_softmax", "seed": 2},
                {"source_experiment": "cand_bpr", "seed": 1},
                {"source_experiment": "cand_bpr", "seed": 2},
            ],
        )
        self.assertEqual(summary["stop_reason"], "converged")
        self.assertEqual(provider.calls, 0)
        # Exactly one re-seeding pass: the second visit finds every seed already
        # measured and enqueues nothing, which is what bounds the loop.
        enqueued = [item for item in memory if item["type"] == "final_replications_enqueued"]
        self.assertEqual(len(enqueued), 1)
        self.assertEqual(enqueued[0]["pass"], 1)

    def test_the_queue_never_outruns_the_iteration_budget(self):
        """A deferral must stay a deferral.

        Each drained task charges an iteration and ``run()`` checks
        ``max_iterations`` *before* the convergence branch, so an untrimmed queue
        exits through ``candidate_budget_reached`` with tasks still pending — a
        different stop reason, not a later one. With three nodes already counted
        and a cap of five, exactly one task fits: one for the best config, and one
        iteration reserved for the pass that takes the ``converged`` exit.
        """
        with tempfile.TemporaryDirectory() as directory:
            with loop_over(Path(directory), max_iterations=5) as (loop, provider):
                converged_state(loop, self.PRIMARIES)
                self.assertEqual(loop.state.iteration_count, 3)
                double = ReplicationDouble(loop)
                with patch.object(loop, "_replication", double):
                    summary = run_with(loop, Recorder())
                leftovers = list(loop.state.pending_replications)

        self.assertEqual(
            double.tasks, [{"source_experiment": "cand_group_softmax", "seed": 1}]
        )
        self.assertEqual(summary["stop_reason"], "converged")
        self.assertEqual(leftovers, [])
        self.assertEqual(provider.calls, 0)

    def test_a_spent_iteration_budget_enqueues_nothing(self):
        """The boundary: with no iteration to spare the enqueue declines, and the
        run stops exactly as it did before P3."""
        with tempfile.TemporaryDirectory() as directory:
            with loop_over(Path(directory), max_iterations=4) as (loop, provider):
                converged_state(loop, self.PRIMARIES)
                double = ReplicationDouble(loop)
                with patch.object(loop, "_replication", double):
                    summary = run_with(loop, Recorder())

        self.assertEqual(double.tasks, [])
        self.assertEqual(summary["stop_reason"], "converged")
        self.assertEqual(provider.calls, 0)

    def test_a_spent_token_budget_stops_immediately(self):
        """60 % of the token budget is the line: past it the run stops as it
        always did, with no replication scheduled."""
        with tempfile.TemporaryDirectory() as directory:
            with loop_over(Path(directory)) as (loop, provider):
                converged_state(loop, self.PRIMARIES)
                loop.state.token_usage.total_tokens = 700  # of 1000
                double = ReplicationDouble(loop)
                with patch.object(loop, "_replication", double):
                    summary = run_with(loop, Recorder())

        self.assertEqual(double.tasks, [])
        self.assertEqual(summary["stop_reason"], "converged")
        self.assertEqual(provider.calls, 0)

    def test_a_spent_wall_clock_stops_immediately(self):
        """50 % of the wall clock is the other line, and it binds on its own —
        the token budget here is untouched."""
        with tempfile.TemporaryDirectory() as directory:
            with loop_over(Path(directory), max_wall_clock_seconds=600) as (loop, provider):
                converged_state(loop, self.PRIMARIES)
                loop.state.wall_clock_seconds = 400.0  # of 600
                double = ReplicationDouble(loop)
                with patch.object(loop, "_replication", double):
                    summary = run_with(loop, Recorder())

        self.assertEqual(double.tasks, [])
        self.assertEqual(summary["stop_reason"], "converged")
        self.assertEqual(provider.calls, 0)

    def test_a_source_without_a_workspace_is_never_enqueued(self):
        """A replication rebuilds the candidate from its workspace, so enqueuing
        a source whose workspace is gone would turn a converged run into
        ``harness_error_breaker`` at the very last step."""
        with tempfile.TemporaryDirectory() as directory:
            with loop_over(Path(directory)) as (loop, provider):
                for index, family in enumerate(
                    sorted(families.coverage_families()), start=1
                ):
                    node(loop, index, f"cand_{family}", 0.6010, family=family)
                loop.state.iteration_count = 3
                loop.state.stagnant_iterations = int(loop.convergence["patience"])
                double = ReplicationDouble(loop)
                with patch.object(loop, "_replication", double):
                    summary = run_with(loop, Recorder())

        self.assertEqual(double.tasks, [])
        self.assertEqual(summary["stop_reason"], "converged")
        self.assertEqual(provider.calls, 0)


if __name__ == "__main__":
    unittest.main()
