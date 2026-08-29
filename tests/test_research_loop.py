from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.agent.llm import ScriptedProvider
from src.agent.research_controller import ResearchLoop
from src.agent.types import ExperimentOutcome


REPO_ROOT = Path(__file__).resolve().parents[1]


def parameters(family: str) -> dict:
    return {
        "seed": 0,
        "k": 16,
        "learning_rate": 0.001,
        "epochs": 5,
        "batch_size": 2048 if family == "bpr" else 1024,
        "patience": 2,
        "negatives_per_positive": 1 if family == "bpr" else None,
        "negatives_per_group": 4 if family == "group_softmax" else None,
        "temperature": 1.0 if family == "group_softmax" else None,
    }


def research(family: str) -> dict:
    return {
        "hypothesis_id": f"h_{family}",
        "family": family,
        "action": "explore",
        "hypothesis": f"test {family}",
        "rationale": "approved method card",
        "parameters": parameters(family),
        "evidence": [
            {
                "title": "Primary paper",
                "url": "https://arxiv.org/abs/1205.2618",
                "method_card_id": family,
            }
        ],
        "needs_web_search": False,
        "parent_experiment": None,
    }


def critic() -> dict:
    return {
        "approved": True,
        "decision": "proceed",
        "rationale": "safe controlled experiment",
        "concerns": [],
        "next_focus": "compare trusted metrics",
    }


def code(family: str) -> str:
    sampler = "sample_bpr_pairs" if family == "bpr" else "sample_softmax_groups"
    final_argument = "1" if family == "bpr" else "4"
    return f'''import numpy as np
from src.experiments.contracts import CandidateOutput
from src.models.sampling import {sampler}

def run(context, parameters):
    {sampler}(context.train_users, context.train_y, np.random.default_rng(0), {final_argument})
    return CandidateOutput(np.zeros(len(context.valid_x)), {{"weights": np.zeros(1)}}, [], {{"pairs": 1}})
'''

TESTS = """import unittest
import candidate

class CandidateTests(unittest.TestCase):
    def test_contract(self):
        self.assertTrue(callable(candidate.run))
"""


def manifest(family: str) -> dict:
    return {
        "candidate_id": f"candidate_{family}",
        "hypothesis_id": f"h_{family}",
        "family": family,
        "code": code(family),
        "tests": TESTS,
        "parameters": parameters(family),
    }


class FakeExecutor:
    def test(self, workspace):
        return True, "ok"

    def train(self, iteration, manifest, workspace, run_dir):
        primary = 0.601 if manifest.family == "bpr" else 0.602
        return ExperimentOutcome(
            status="success",
            metrics={"GAUC": primary, "nDCG@5": primary, "primary": primary},
            duration_seconds=0.01,
            artifact_path=f"artifact-{manifest.family}.npz",
            diagnostics={"eligible_users": 10},
        )


class ResearchLoopTests(unittest.TestCase):
    def test_mocked_loop_covers_both_families_and_persists_resume_state(self):
        responses = [
            research("bpr"),
            critic(),
            manifest("bpr"),
            critic(),
            research("group_softmax"),
            critic(),
            manifest("group_softmax"),
            critic(),
        ]
        provider = ScriptedProvider(responses)
        baseline = {
            "best": {
                "experiment_id": "official_fm_seed0",
                "metrics": {"GAUC": 0.6671, "nDCG@5": 0.5358, "primary": 0.6015},
                "artifact_path": "baseline.npz",
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "mode": "research",
                "name": "test",
                "data_dir": str(REPO_ROOT / "data" / "KuaiRand-Pure" / "data"),
                "run_root": str(root / "runs"),
                "generated_root": str(root / "generated"),
                "method_catalog": str(REPO_ROOT / "research" / "methods"),
                "official_validation_baseline": 0.6016,
                "llm": {"max_total_tokens": 1000},
                "budgets": {
                    "max_iterations": 2,
                    "max_wall_clock_seconds": 60,
                    "experiment_timeout_seconds": 10,
                    "test_timeout_seconds": 10,
                    "max_debug_repairs": 2,
                },
                "convergence": {"epsilon": 0.002, "patience": 3},
                "replication_seeds": [1, 2],
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            loop = ResearchLoop(
                config,
                config_path,
                provider=provider,
                baseline_summary=baseline,
            )
            loop.executor = FakeExecutor()
            run_dir = loop.run()
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
            error = (
                (run_dir / "error.json").read_text(encoding="utf-8")
                if (run_dir / "error.json").is_file()
                else ""
            )
            self.assertNotEqual(summary["stop_reason"], "controller_error", error)
            self.assertEqual(summary["training_attempts"], 2)
            self.assertEqual({node["family"] for node in state["nodes"]}, {"bpr", "group_softmax"})
            self.assertEqual(summary["manual_interventions"], 0)
            self.assertTrue((run_dir / "experiment_tree.json").is_file())
            activity = [
                json.loads(line)
                for line in (run_dir / "activity.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            for stage in (
                "researcher",
                "critic_preflight",
                "builder",
                "safety_tests",
                "training_evaluation",
                "critic_postflight",
                "persistence",
            ):
                statuses = {item["status"] for item in activity if item["stage"] == stage}
                self.assertIn("active", statuses, stage)
                self.assertIn("completed", statuses, stage)
            self.assertTrue((run_dir / "changes" / "001_candidate_bpr.patch").is_file())
            self.assertEqual(len(provider.calls), 8)

    def test_debugger_repairs_are_capped_at_two(self):
        invalid_manifest = manifest("bpr")
        invalid_manifest["code"] = "import os\n"
        debug = {
            "preserve_hypothesis": True,
            "diagnosis": "forbidden import remains",
            "replacement_code": "import os\n",
            "replacement_tests": TESTS,
        }
        provider = ScriptedProvider(
            [research("bpr"), critic(), invalid_manifest, dict(debug), dict(debug)]
        )
        baseline = {
            "best": {
                "experiment_id": "official_fm_seed0",
                "metrics": {"GAUC": 0.6671, "nDCG@5": 0.5358, "primary": 0.6015},
                "artifact_path": "baseline.npz",
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "mode": "research",
                "name": "repair-test",
                "data_dir": str(REPO_ROOT / "data" / "KuaiRand-Pure" / "data"),
                "run_root": str(root / "runs"),
                "generated_root": str(root / "generated"),
                "method_catalog": str(REPO_ROOT / "research" / "methods"),
                "official_validation_baseline": 0.6016,
                "llm": {"max_total_tokens": 1000},
                "budgets": {
                    "max_iterations": 1,
                    "max_wall_clock_seconds": 60,
                    "experiment_timeout_seconds": 10,
                    "test_timeout_seconds": 10,
                    "max_debug_repairs": 2,
                },
                "convergence": {"epsilon": 0.002, "patience": 3},
                "replication_seeds": [1, 2],
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            loop = ResearchLoop(
                config,
                config_path,
                provider=provider,
                baseline_summary=baseline,
            )
            run_dir = loop.run()
            record = json.loads(
                (run_dir / "iterations.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(record["repairs"], 2)
            self.assertEqual(record["status"], "failed")
            self.assertEqual(loop.state.training_attempts, 0)
            self.assertEqual(len(provider.calls), 5)

    def test_iteration_training_and_proposal_caps_are_independent(self):
        """Each of the three caps is its own knob (I5).

        The candidate cap is deliberately loose (5) and the training-attempt cap
        tight (1), so only a loop that reads the two keys separately can stop on
        the training cap here. Three clean iterations are scripted even though a
        correct loop consumes one: before the split, ``max_iterations`` drove all
        three caps and the run carried on to ``converged`` on iteration 3 — the
        conflation this test pins. The proposal cap is left out of the config so
        the ``max_iterations * 2`` default is exercised too.
        """
        responses = [
            research("bpr"), critic(), manifest("bpr"), critic(),
            research("group_softmax"), critic(), manifest("group_softmax"), critic(),
            research("bpr"), critic(), manifest("bpr"), critic(),
        ]
        provider = ScriptedProvider(responses)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, config_path = research_config(
                root, max_iterations=5, max_training_attempts=1
            )
            loop = ResearchLoop(
                config,
                config_path,
                provider=provider,
                baseline_summary=BASELINE_SUMMARY,
            )
            loop.executor = FakeExecutor()
            run_dir = loop.run()
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

            self.assertEqual(summary["stop_reason"], "training_attempt_budget_reached")
            self.assertEqual(summary["training_attempts"], 1)
            # The other two caps are nowhere near their limits, which is what
            # makes the stop above attributable to the training cap alone.
            self.assertEqual(summary["iterations"], 1)
            self.assertEqual(loop.state.proposal_attempts, 1)
            self.assertEqual(loop.max_training_attempts, 1)
            self.assertEqual(loop.max_proposals, 10)
            self.assertEqual(len(provider.calls), 4)

            # Cheap complement to the two end-to-end pins below: an explicit
            # max_proposals beats the max_iterations * 2 default, and an absent
            # max_training_attempts falls back to max_iterations.
            explicit, explicit_path = research_config(
                root, max_iterations=5, max_proposals=3
            )
            explicit_loop = ResearchLoop(
                explicit,
                explicit_path,
                provider=ScriptedProvider([]),
                baseline_summary=BASELINE_SUMMARY,
            )
            self.assertEqual(explicit_loop.max_proposals, 3)
            self.assertEqual(explicit_loop.max_training_attempts, 5)

    def test_training_attempt_cap_bounds_retraining_within_one_candidate(self):
        """`_execute`'s own guard reads the training cap, not the candidate cap.

        The loop-top check cannot pin `_execute`'s guard: `_execute` is entered
        with `training_attempts == 0`, so the guard is only reached when training
        fails and is repaired *inside* one candidate. Here every training run
        fails and every repair succeeds, so attempts accumulate until a cap
        stops them — at 2 (`max_training_attempts`) if the guard is correct, at 5
        (`max_iterations`) if it is reverted to the shared knob. Both spellings
        end the run with the same `stop_reason`, so the attempt and repair counts
        are what discriminate.
        """
        script = [research("bpr"), critic(), manifest("bpr")]
        script += [debug_patch("bpr") for _ in range(6)]  # only 2 are consumed
        provider = ScriptedProvider(script)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, config_path = research_config(
                root,
                max_iterations=5,
                max_training_attempts=2,
                max_debug_repairs=5,
            )
            loop = ResearchLoop(
                config,
                config_path,
                provider=provider,
                baseline_summary=BASELINE_SUMMARY,
            )
            loop.executor = TrainingFailureExecutor()
            run_dir = loop.run()
            record = json.loads(
                (run_dir / "iterations.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )

            self.assertEqual(loop.state.training_attempts, 2)
            self.assertEqual(record["repairs"], 2)
            self.assertEqual(record["status"], "failed")
            self.assertEqual([node.status for node in loop.state.nodes], ["failed"])
            # Three role calls plus one debugger call per failed training run.
            self.assertEqual(len(provider.calls), 5)
            self.assertEqual(
                json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["stop_reason"],
                "training_attempt_budget_reached",
            )

    def test_proposal_cap_comes_from_the_config_not_from_max_iterations(self):
        """The proposal cap is a config key, not `max_iterations * 2`.

        Every proposal is malformed and re-prompting is switched off, so one
        provider call is one charged proposal and the run can only end on the
        proposal cap. It ends at 1 with `max_proposals: 1`; were the cap still
        derived from `max_iterations` it would end at 10, under the same
        `stop_reason` — so the count is the discriminator again.
        """
        broken = research("bpr")
        broken["family"] = "nope"  # types.py -> ValueError, a proposal fault
        provider = ScriptedProvider([dict(broken) for _ in range(12)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, config_path = research_config(
                root,
                max_iterations=5,
                max_proposals=1,
                max_role_reprompts=0,
            )
            loop = ResearchLoop(
                config,
                config_path,
                provider=provider,
                baseline_summary=BASELINE_SUMMARY,
            )
            loop.executor = FakeExecutor()
            run_dir = loop.run()
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

            self.assertEqual(loop.state.proposal_attempts, 1)
            self.assertEqual(len(provider.calls), 1)
            self.assertEqual(summary["stop_reason"], "proposal_budget_reached")
            # The abandoned proposal became no candidate and no training run, so
            # neither of the other two caps could have ended this run.
            self.assertEqual(summary["iterations"], 0)
            self.assertEqual(summary["training_attempts"], 0)


# --------------------------------------------------------------------------- #
# Shared fixtures for the three cap tests above (I5 / T3).
#
# These live *below* the class rather than beside the builders at the top of the
# file: the inline configs at :138-147 and :202-208 are cited by line number in
# other owners' plans and hand-off notes, and inserting anything above them would
# move those citations. Module scope is evaluated at import, so the tests see
# these names regardless of the order they appear in.
# --------------------------------------------------------------------------- #


BASELINE_SUMMARY = {
    "best": {
        "experiment_id": "official_fm_seed0",
        "metrics": {"GAUC": 0.6671, "nDCG@5": 0.5358, "primary": 0.6015},
        "artifact_path": "baseline.npz",
    }
}


def research_config(root: Path, **budgets: int) -> tuple[dict, Path]:
    """A research config under ``root`` plus the path it was frozen to."""
    config = {
        "mode": "research",
        "name": "cap-split-test",
        "data_dir": str(REPO_ROOT / "data" / "KuaiRand-Pure" / "data"),
        "run_root": str(root / "runs"),
        "generated_root": str(root / "generated"),
        "method_catalog": str(REPO_ROOT / "research" / "methods"),
        "official_validation_baseline": 0.6016,
        "llm": {"max_total_tokens": 1000},
        "budgets": {
            "max_wall_clock_seconds": 60,
            "experiment_timeout_seconds": 10,
            "test_timeout_seconds": 10,
            "max_debug_repairs": 2,
            **budgets,
        },
        "convergence": {"epsilon": 0.002, "patience": 3},
        "replication_seeds": [1, 2],
    }
    path = root / f"config_{len(list(root.glob('config_*.json')))}.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return config, path


def debug_patch(family: str) -> dict:
    """A debugger reply that hands back working code, so the repair succeeds."""
    return {
        "preserve_hypothesis": True,
        "diagnosis": "training diverged; the candidate contract itself is sound",
        "replacement_code": code(family),
        "replacement_tests": TESTS,
    }


class TrainingFailureExecutor:
    """Tests always pass, training always fails — so repairs and retraining loop."""

    def test(self, workspace):
        return True, "ok"

    def train(self, iteration, manifest, workspace, run_dir):
        return ExperimentOutcome(
            status="failed",
            metrics=None,
            duration_seconds=0.01,
            error="training diverged before the first epoch completed",
        )


if __name__ == "__main__":
    unittest.main()
