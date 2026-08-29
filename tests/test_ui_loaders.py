from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.ui.loaders import (
    discover_runs,
    load_activity_timeline,
    load_candidate_files,
    load_dashboard_config,
    load_gate_result,
    load_journal_reports,
    load_role_passes,
    load_run_snapshot,
    validate_submission,
)


class DashboardLoaderTests(unittest.TestCase):
    def test_normalizes_baseline_and_research_run_schemas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "runs" / "baseline"
            research = root / "runs" / "research"
            baseline.mkdir(parents=True)
            research.mkdir(parents=True)
            (baseline / "summary.json").write_text(
                json.dumps(
                    {
                        "run_id": "baseline",
                        "best": {
                            "experiment_id": "official_fm_seed0",
                            "metrics": {"GAUC": 0.6671, "nDCG@5": 0.5358, "primary": 0.6015},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (baseline / "iterations.jsonl").write_text(
                json.dumps(
                    {
                        "iteration": 1,
                        "experiment_id": "official_fm_seed0",
                        "kind": "fm",
                        "hypothesis": "reproduce",
                        "outcome": {"status": "success", "metrics": {"primary": 0.6015}},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (research / "state.json").write_text(
                json.dumps(
                    {
                        "run_id": "research",
                        "status": "running",
                        "baseline_primary": 0.6015,
                        "best_experiment_id": "candidate_bpr",
                        "best_metrics": {"GAUC": 0.66, "nDCG@5": 0.55, "primary": 0.605},
                        "nodes": [],
                    }
                ),
                encoding="utf-8",
            )
            baseline_snapshot = load_run_snapshot(baseline)
            research_snapshot = load_run_snapshot(research)
            self.assertEqual(baseline_snapshot.status, "completed")
            self.assertAlmostEqual(baseline_snapshot.best_metrics["primary"], 0.6015)
            self.assertEqual(research_snapshot.status, "running")
            self.assertAlmostEqual(research_snapshot.baseline_primary, 0.6015)
            self.assertEqual(len(discover_runs(root / "runs")), 2)

    def test_partial_final_jsonl_line_is_tolerated(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()
            complete = {
                "event_id": "1",
                "iteration": 1,
                "stage": "researcher",
                "status": "active",
                "started_at": "2026-08-29T00:00:00+00:00",
                "updated_at": "2026-08-29T00:00:00+00:00",
            }
            (run / "activity.jsonl").write_text(
                json.dumps(complete) + "\n{\"event_id\":", encoding="utf-8"
            )
            timeline, warnings = load_activity_timeline(run)
            self.assertEqual(len(timeline), 1)
            self.assertIn("partial final line", warnings[0])

    def test_config_rejects_any_judge_owned_path(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "ui.json"
            config_path.write_text(
                json.dumps({"run_root": "data/judge/runs"}), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                load_dashboard_config(config_path)

    def test_submission_validator_is_truthful_about_unchecked_alignment(self):
        payload = (
            "row_id,user_id,video_id,score\n"
            "0,u1,v1,0.2\n"
            "1,u1,v1,0.3\n"
        )
        check = validate_submission(payload)
        self.assertTrue(check.valid)
        self.assertEqual(check.duplicate_pairs, 1)
        self.assertFalse(check.alignment_checked)
        self.assertIn("not checked", check.warnings[0])

    def test_submission_validator_rejects_bad_order_ids_and_scores(self):
        payload = "user_id,row_id,video_id,score\nu1,2,v1,nan\n"
        check = validate_submission(payload)
        self.assertFalse(check.valid)
        self.assertGreaterEqual(len(check.errors), 3)

    def test_load_role_passes_and_candidate_files(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            passes_dir = run_dir / "passes"
            cand_dir = run_dir / "gen" / "001_candidate"
            passes_dir.mkdir(parents=True)
            cand_dir.mkdir(parents=True)

            pass_file = passes_dir / "001_researcher_0.json"
            pass_file.write_text(
                json.dumps(
                    {
                        "prompt": "ROLE: Researcher",
                        "result": {
                            "role": "researcher",
                            "model": "gpt-5.5",
                            "latency_seconds": 1.25,
                            "usage": {"total_tokens": 100, "input_tokens": 80, "output_tokens": 20},
                            "data": {"hypothesis": "Test BPR"},
                            "sources": [{"title": "BPR", "url": "https://arxiv.org/abs/1205.2618"}],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (cand_dir / "candidate.py").write_text("def run(c, p): pass", encoding="utf-8")
            (cand_dir / "test_candidate.py").write_text("def test_ok(): pass", encoding="utf-8")

            passes = load_role_passes(run_dir, 1)
            self.assertEqual(len(passes), 1)
            self.assertEqual(passes[0].role, "researcher")
            self.assertEqual(passes[0].model, "gpt-5.5")
            self.assertEqual(len(passes[0].sources), 1)

            code, tests = load_candidate_files(run_dir, cand_dir)
            self.assertEqual(code, "def run(c, p): pass")
            self.assertEqual(tests, "def test_ok(): pass")

    def test_load_gate_result_and_journal_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            (run_dir / "gate_done.json").write_text(
                json.dumps({"status": "ok", "details": {"rows": 170588}}), encoding="utf-8"
            )
            (run_dir / "journal.md").write_text("# Journal\nContent", encoding="utf-8")
            (run_dir / "results.md").write_text("# Results\nContent", encoding="utf-8")

            gate = load_gate_result(run_dir)
            self.assertIsNotNone(gate)
            self.assertEqual(gate["status"], "ok")

            journal, results = load_journal_reports(run_dir)
            self.assertEqual(journal, "# Journal\nContent")
            self.assertEqual(results, "# Results\nContent")


if __name__ == "__main__":
    unittest.main()
