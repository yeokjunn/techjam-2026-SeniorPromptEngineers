"""Tests for the journal + results renderer (T2 / I16)."""
from __future__ import annotations

import glob
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from src.agent.report import render_reports

REPO_ROOT = Path(__file__).resolve().parents[1]

PARENT_CODE = (
    "# BPR candidate — parent\n"
    "import numpy as np\n"
    "from src.models.sampling import sample_bpr_pairs\n"
    "\n"
    "def run(context, parameters):\n"
    "    pairs = sample_bpr_pairs(context, parameters)\n"
    "    return None\n"
)

CHILD_CODE = (
    "# BPR candidate — seed1 replication\n"
    "import numpy as np\n"
    "from src.models.sampling import sample_bpr_pairs\n"
    "\n"
    "def run(context, parameters):\n"
    "    # replication with different seed\n"
    "    pairs = sample_bpr_pairs(context, parameters)\n"
    "    return None\n"
)

ITERATION_REJECTED = {
    "iteration": 1,
    "proposal": {
        "hypothesis_id": "h1",
        "family": "bpr",
        "action": "explore",
        "hypothesis": "test BPR loss",
        "rationale": "ranking metrics",
        "parameters": {"seed": 0, "learning_rate": 0.001},
        "evidence": [],
    },
    "preflight": {
        "approved": False,
        "decision": "reject",
        "rationale": "Already tried",
    },
    "status": "critic_rejected",
    "manual_intervention": False,
}

ITERATION_SUCCESS = {
    "iteration": 2,
    "proposal": {
        "hypothesis_id": "h2",
        "family": "bpr",
        "action": "explore",
        "hypothesis": "BPR with lower lr",
        "rationale": "converge better",
        "parameters": {"seed": 0, "learning_rate": 0.0005},
        "evidence": [{"title": "BPR paper", "url": "http://example.com"}],
        "parent_experiment": None,
    },
    "preflight": {
        "approved": True,
        "decision": "approve",
        "rationale": "novel approach",
    },
    "manifest": {
        "candidate_id": "bpr_lr0005",
        "hypothesis_id": "h2",
        "family": "bpr",
        "parameters": {"seed": 0, "learning_rate": 0.0005},
        "code_sha256": "abc123",
        "tests_sha256": "def456",
    },
    "repairs": 0,
    "outcome": {
        "status": "success",
        "metrics": {"GAUC": 0.6700, "nDCG@5": 0.5400, "primary": 0.6050},
        "duration_seconds": 12.5,
        "failure_class": None,
        "error": None,
        "recovery": None,
        "epoch_trace": [],
        "diagnostics": {},
    },
    "postflight": {
        "approved": True,
        "decision": "supported",
        "rationale": "Improvement over baseline",
    },
    "status": "success",
    "manual_intervention": False,
}

ITERATION_FAILED = {
    "iteration": 3,
    "proposal": {
        "hypothesis_id": "h3",
        "family": "group_softmax",
        "action": "explore",
        "hypothesis": "Group softmax loss",
        "rationale": "listwise",
        "parameters": {"seed": 0, "learning_rate": 0.001, "temperature": 1.0},
        "evidence": [],
    },
    "preflight": {
        "approved": True,
        "decision": "approve",
        "rationale": "new family",
    },
    "manifest": {
        "candidate_id": "gsoftmax_t1",
        "hypothesis_id": "h3",
        "family": "group_softmax",
        "parameters": {"seed": 0},
        "code_sha256": "ghi789",
        "tests_sha256": "jkl012",
    },
    "repairs": 1,
    "outcome": {
        "status": "failed",
        "metrics": None,
        "duration_seconds": 5.0,
        "failure_class": "timeout",
        "error": "Training exceeded 900s",
        "recovery": "reduced epochs",
        "epoch_trace": [],
        "diagnostics": {},
    },
    "postflight": None,
    "status": "failed",
    "manual_intervention": False,
}

ITERATION_REPLICATION = {
    "iteration": 4,
    "proposal": {
        "hypothesis_id": "h2",
        "family": "bpr",
        "action": "replicate",
        "hypothesis": "Replication of bpr_lr0005 with seed 1",
        "rationale": "Deterministic replication",
        "parameters": {"seed": 1, "learning_rate": 0.0005},
        "evidence": [],
        "parent_experiment": "bpr_lr0005",
    },
    "preflight": {
        "approved": True,
        "decision": "replicate",
        "rationale": "inherited",
    },
    "manifest": {
        "candidate_id": "bpr_lr0005_seed1",
        "hypothesis_id": "h2",
        "family": "bpr",
        "parameters": {"seed": 1, "learning_rate": 0.0005},
        "code_sha256": "abc123",
        "tests_sha256": "def456",
    },
    "repairs": 0,
    "outcome": {
        "status": "success",
        "metrics": {"GAUC": 0.6680, "nDCG@5": 0.5380, "primary": 0.6030},
        "duration_seconds": 11.0,
        "failure_class": None,
        "error": None,
        "recovery": None,
        "epoch_trace": [],
        "diagnostics": {},
    },
    "postflight": {
        "approved": True,
        "decision": "supported",
        "rationale": "Consistent with original",
    },
    "status": "success",
    "manual_intervention": False,
}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")


def _build_fixture(base_dir: Path) -> Path:
    """Build a complete fixture run directory under base_dir."""
    run_id = "test_run"
    run_dir = base_dir / "runs" / run_id
    run_dir.mkdir(parents=True)

    gen_root = base_dir / "generated_experiments"

    # run_config.json
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "name": "test-research",
                "generated_root": str(gen_root),
                "official_validation_baseline": 0.6016,
                "budgets": {"max_iterations": 50},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # iterations.jsonl
    _write_jsonl(
        run_dir / "iterations.jsonl",
        [ITERATION_REJECTED, ITERATION_SUCCESS, ITERATION_FAILED, ITERATION_REPLICATION],
    )

    # summary.json
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "completed",
                "stop_reason": "converged",
                "iterations": 4,
                "training_attempts": 3,
                "manual_interventions": 1,
                "converged_official": True,
                "token_usage": {
                    "input_tokens": 5000,
                    "output_tokens": 2000,
                    "total_tokens": 7000,
                    "cached_tokens": 1000,
                    "web_search_calls": 0,
                },
                "wall_clock_seconds": 120.5,
                "best": {
                    "experiment_id": "bpr_lr0005",
                    "metrics": {
                        "GAUC": 0.6700,
                        "nDCG@5": 0.5400,
                        "primary": 0.6050,
                    },
                    "artifact_path": None,
                    "candidate_dir": None,
                },
                "gate": {
                    "status": "not_implemented",
                    "submission_path": None,
                    "details": {},
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # results.json
    (run_dir / "results.json").write_text(
        json.dumps(
            [
                {
                    "iteration": 2,
                    "experiment_id": "bpr_lr0005",
                    "family": "bpr",
                    "action": "explore",
                    "status": "success",
                    "metrics": {"GAUC": 0.6700, "nDCG@5": 0.5400, "primary": 0.6050},
                    "delta_vs_baseline": 0.6050 - 0.6016,
                },
                {
                    "iteration": 3,
                    "experiment_id": "gsoftmax_t1",
                    "family": "group_softmax",
                    "action": "explore",
                    "status": "failed",
                    "metrics": None,
                    "delta_vs_baseline": None,
                },
                {
                    "iteration": 4,
                    "experiment_id": "bpr_lr0005_seed1",
                    "family": "bpr",
                    "action": "replicate",
                    "status": "success",
                    "metrics": {"GAUC": 0.6680, "nDCG@5": 0.5380, "primary": 0.6030},
                    "delta_vs_baseline": 0.6030 - 0.6016,
                },
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # resources.json
    (run_dir / "resources.json").write_text(
        json.dumps(
            {
                "token_usage": {"total_tokens": 7000},
                "wall_clock_seconds": 120.5,
                "training_attempts": 3,
                "iteration_count": 4,
                "manual_interventions": 1,
                "gpu_hours": 0.0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # research_memory.jsonl — records with role, iteration, usage
    memory_records = [
        {"role": "researcher", "iteration": 2, "usage": {"total_tokens": 1000}},
        {"role": "critic", "iteration": 2, "usage": {"total_tokens": 500}},
        {"role": "builder", "iteration": 2, "usage": {"total_tokens": 2000}},
        {"role": "critic", "iteration": 2, "usage": {"total_tokens": 300}},
        {"role": "researcher", "iteration": 3, "usage": {"total_tokens": 1200}},
        {"role": "critic", "iteration": 3, "usage": {"total_tokens": 400}},
        {"role": "builder", "iteration": 3, "usage": {"total_tokens": 1600}},
    ]
    _write_jsonl(run_dir / "research_memory.jsonl", memory_records)

    # passes/002_builder_0.json
    passes_dir = run_dir / "passes"
    passes_dir.mkdir()
    (passes_dir / "002_builder_0.json").write_text(
        json.dumps(
            {
                "prompt": "Build a BPR candidate...",
                "result": {
                    "role": "builder",
                    "data": {
                        "code": PARENT_CODE,
                    },
                    "usage": {"total_tokens": 2000},
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # interventions.json
    (run_dir / "interventions.json").write_text(
        json.dumps([{"reason": "manual seed override"}], indent=2) + "\n",
        encoding="utf-8",
    )

    # Generated candidate source files
    parent_dir = gen_root / run_id / "002_bpr_lr0005"
    parent_dir.mkdir(parents=True)
    (parent_dir / "candidate.py").write_text(PARENT_CODE, encoding="utf-8")

    child_dir = gen_root / run_id / "004_bpr_lr0005_seed1"
    child_dir.mkdir(parents=True)
    (child_dir / "candidate.py").write_text(CHILD_CODE, encoding="utf-8")

    return run_dir


class ReportTests(unittest.TestCase):

    def test_empty_run_directory_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as d:
            result = render_reports(Path(d))
            self.assertIsNone(result)
            self.assertFalse((Path(d) / "journal.md").exists())
            self.assertFalse((Path(d) / "results.md").exists())

    def test_fixture_run_renders_both_reports(self):
        with tempfile.TemporaryDirectory() as d:
            run_dir = _build_fixture(Path(d))
            render_reports(run_dir)
            journal = run_dir / "journal.md"
            results = run_dir / "results.md"
            self.assertTrue(journal.exists(), "journal.md not created")
            self.assertTrue(results.exists(), "results.md not created")
            self.assertGreater(len(journal.read_text(encoding="utf-8")), 0)
            self.assertGreater(len(results.read_text(encoding="utf-8")), 0)

    def test_journal_contains_a_unified_diff_between_parent_and_child(self):
        with tempfile.TemporaryDirectory() as d:
            run_dir = _build_fixture(Path(d))
            render_reports(run_dir)
            journal = (run_dir / "journal.md").read_text(encoding="utf-8")
            self.assertIn("```diff", journal)
            self.assertIn("+# BPR candidate", journal)
            self.assertIn("+def run(context, parameters):", journal)

    def test_journal_falls_back_to_the_builder_pass_when_the_directory_is_gone(self):
        with tempfile.TemporaryDirectory() as d:
            run_dir = _build_fixture(Path(d))
            gen_root = Path(d) / "generated_experiments"
            shutil.rmtree(gen_root)
            render_reports(run_dir)
            journal = (run_dir / "journal.md").read_text(encoding="utf-8")
            self.assertIn("def run(context, parameters)", journal)
            self.assertIn("builder pass", journal.lower())

    def test_journal_prints_failure_class_and_repairs(self):
        with tempfile.TemporaryDirectory() as d:
            run_dir = _build_fixture(Path(d))
            render_reports(run_dir)
            journal = (run_dir / "journal.md").read_text(encoding="utf-8")
            self.assertIn("timeout", journal.lower())
            repair_mentioned = "repair" in journal.lower() or "1 repair" in journal.lower()
            self.assertTrue(repair_mentioned, "repairs not mentioned in journal")

    def test_results_reports_gate_deltas_tokens_and_interventions(self):
        with tempfile.TemporaryDirectory() as d:
            run_dir = _build_fixture(Path(d))
            render_reports(run_dir)
            results = (run_dir / "results.md").read_text(encoding="utf-8")
            self.assertIn("not_implemented", results)
            self.assertIn("0.6016", results)
            self.assertIn("researcher", results.lower())
            self.assertIn("of 50", results)
            self.assertIn("converged", results)
            self.assertIn("manual seed override", results)
            self.assertIn("0.0", results)

    def test_partial_summary_does_not_raise(self):
        with tempfile.TemporaryDirectory() as d:
            run_dir = _build_fixture(Path(d))
            summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
            summary.pop("gate", None)
            summary.pop("converged_official", None)
            (run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            (run_dir / "interventions.json").unlink(missing_ok=True)
            render_reports(run_dir)
            results = (run_dir / "results.md").read_text(encoding="utf-8")
            lowered = results.lower()
            has_fallback = "not reported" in lowered or "none recorded" in lowered
            self.assertTrue(
                has_fallback,
                "Missing fields should produce 'not reported' or 'none recorded'",
            )

    def test_committed_baseline_run_renders(self):
        matches = sorted(glob.glob(str(REPO_ROOT / "runs" / "*" / "summary.json")))
        if not matches:
            self.skipTest("No committed baseline run found under runs/")

        source_run = Path(matches[0]).parent
        best_path = source_run / "best.json"
        if not best_path.exists():
            self.skipTest(f"No best.json in {source_run.name}")

        best = json.loads(best_path.read_text(encoding="utf-8"))
        best_primary = best["metrics"]["primary"]

        with tempfile.TemporaryDirectory() as d:
            copy_dir = Path(d) / source_run.name
            shutil.copytree(source_run, copy_dir)
            render_reports(copy_dir)
            journal = copy_dir / "journal.md"
            results = copy_dir / "results.md"
            self.assertTrue(journal.exists(), "journal.md not created for baseline run")
            self.assertTrue(results.exists(), "results.md not created for baseline run")
            journal_text = journal.read_text(encoding="utf-8")
            results_text = results.read_text(encoding="utf-8")
            primary_str = f"{best_primary:.4f}"
            found = primary_str in journal_text or primary_str in results_text
            if not found:
                primary_str_long = str(best_primary)
                found = primary_str_long in journal_text or primary_str_long in results_text
            self.assertTrue(
                found,
                f"best.metrics.primary ({best_primary}) not found in rendered reports",
            )


if __name__ == "__main__":
    unittest.main()
