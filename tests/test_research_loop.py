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


if __name__ == "__main__":
    unittest.main()
