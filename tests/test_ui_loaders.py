from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.ui.loaders import (
    discover_runs,
    load_activity_timeline,
    load_dashboard_config,
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


if __name__ == "__main__":
    unittest.main()
