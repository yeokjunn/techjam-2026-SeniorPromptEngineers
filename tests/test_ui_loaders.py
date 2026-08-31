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
            discovered = discover_runs(root / "runs")
            self.assertEqual(len(discovered), 1)
            self.assertEqual(discovered[0].run_id, "research")

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

            role_payloads = (
                ("builder", {"candidate_id": "candidate_bpr"}),
                ("critic_postflight", {"decision": "keep"}),
                ("critic_preflight", {"approved": True}),
                ("researcher", {"hypothesis": "Test BPR"}),
            )
            for role, data in role_payloads:
                (passes_dir / f"001_{role}_0.json").write_text(
                    json.dumps(
                        {
                            "prompt": f"ROLE: {role}",
                            "result": {
                                "role": role,
                                "model": "GLM-5.3-Flash",
                                "latency_seconds": 1.25,
                                "usage": {
                                    "total_tokens": 100,
                                    "input_tokens": 80,
                                    "output_tokens": 20,
                                },
                                "data": data,
                                "sources": [
                                    {
                                        "title": "BPR",
                                        "url": "https://arxiv.org/abs/1205.2618",
                                    }
                                ],
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            (cand_dir / "candidate.py").write_text("def run(c, p): pass", encoding="utf-8")
            (cand_dir / "test_candidate.py").write_text("def test_ok(): pass", encoding="utf-8")

            passes = load_role_passes(run_dir, 1)
            self.assertEqual(len(passes), 4)
            self.assertEqual(
                [item.role for item in passes],
                ["researcher", "critic_preflight", "builder", "critic_postflight"],
            )
            self.assertEqual(passes[0].role, "researcher")
            self.assertEqual(passes[0].model, "GLM-5.3-Flash")
            self.assertEqual(len(passes[0].sources), 1)

            code, tests = load_candidate_files(run_dir, Path("gen") / "001_candidate")
            self.assertEqual(code, "def run(c, p): pass")
            self.assertEqual(tests, "def test_ok(): pass")

    def test_snapshot_joins_real_iteration_manifest_to_node_candidate_dir_lazily(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            candidate_dir = run_dir / "generated" / "001_candidate_bpr"
            candidate_dir.mkdir(parents=True)
            (candidate_dir / "candidate.py").write_text("CODE", encoding="utf-8")
            (candidate_dir / "test_candidate.py").write_text("TESTS", encoding="utf-8")
            (run_dir / "state.json").write_text(
                json.dumps(
                    {
                        "run_id": "run",
                        "status": "completed",
                        "nodes": [
                            {
                                "iteration": 1,
                                "experiment_id": "candidate_bpr",
                                "hypothesis_id": "h1",
                                "candidate_dir": "generated/001_candidate_bpr",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "iterations.jsonl").write_text(
                json.dumps(
                    {
                        "iteration": 1,
                        "proposal": {"hypothesis_id": "h1", "hypothesis": "Try BPR"},
                        "manifest": {
                            "candidate_id": "candidate_bpr",
                            "code_sha256": "abc",
                            "tests_sha256": "def",
                        },
                        "status": "success",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = load_run_snapshot(run_dir)
            iteration = snapshot.iterations[0]
            self.assertEqual(iteration.candidate_dir, "generated/001_candidate_bpr")
            self.assertEqual(
                load_candidate_files(run_dir, iteration.candidate_dir), ("CODE", "TESTS")
            )

    def test_snapshot_tolerates_null_iteration_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            (run_dir / "iterations.jsonl").write_text(
                json.dumps(
                    {
                        "iteration": 1,
                        "status": "success",
                        "proposal": {"family": "bpr", "parameters": None},
                        "configuration": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = load_run_snapshot(run_dir)
            self.assertEqual(snapshot.iterations[0].parameters, {})


    def test_load_gate_result_and_journal_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            (run_dir / "gate_done.json").write_text(
                json.dumps({"status": "ok", "details": {"rows": 170588}}), encoding="utf-8"
            )
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "gate": {
                            "status": "error",
                            "details": {"reason": "missing_test_scores"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "journal.md").write_text("# Journal\nContent", encoding="utf-8")
            (run_dir / "results.md").write_text("# Results\nContent", encoding="utf-8")

            gate = load_gate_result(run_dir)
            self.assertIsNotNone(gate)
            self.assertEqual(gate["status"], "error")
            self.assertEqual(gate["details"]["reason"], "missing_test_scores")

            journal, results = load_journal_reports(run_dir)
            self.assertEqual(journal, "# Journal\nContent")
            self.assertEqual(results, "# Results\nContent")

    def test_snapshot_deduplicates_iterations_by_iteration_number(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            (run_dir / "iterations.jsonl").write_text(
                json.dumps({"iteration": 1, "status": "proposal_failed", "proposal": {"family": "bpr"}})
                + "\n"
                + json.dumps({"iteration": 1, "status": "success", "proposal": {"family": "bpr"}})
                + "\n",
                encoding="utf-8",
            )
            snapshot = load_run_snapshot(run_dir)
            self.assertEqual(len(snapshot.iterations), 1)
            self.assertEqual(snapshot.iterations[0].status, "success")

    def test_load_live_eda_and_eda_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            eda_dir = run_dir / "eda"
            eda_dir.mkdir(parents=True)

            (eda_dir / "latest.json").write_text(
                json.dumps(
                    {
                        "iteration": 1,
                        "status": "in_progress",
                        "plan": {"objective": "Investigate duration features"},
                        "report": {"summary": "Duration buckets show signal"},
                    }
                ),
                encoding="utf-8",
            )
            (eda_dir / "001_eda.json").write_text(
                json.dumps(
                    {
                        "iteration": 1,
                        "status": "completed",
                        "plan": {"objective": "Completed plan"},
                        "report": {
                            "summary": "Completed report",
                            "findings": [{"key": "f1"}],
                            "feature_candidates": [{"name": "dur_bucket"}],
                        },
                    }
                ),
                encoding="utf-8",
            )

            from src.ui.loaders import load_eda_artifacts, load_live_eda_latest

            live_eda = load_live_eda_latest(run_dir)
            self.assertIsNotNone(live_eda)
            self.assertEqual(live_eda.iteration, 1)
            self.assertEqual(live_eda.status, "in_progress")
            self.assertEqual(live_eda.plan.get("objective"), "Investigate duration features")

            artifacts = load_eda_artifacts(run_dir)
            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0].status, "completed")
            self.assertEqual(len(artifacts[0].feature_candidates), 1)

    def test_load_debugger_events_from_debugger_and_research_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()

            (run_dir / "debugger_memory.jsonl").write_text(
                json.dumps(
                    {
                        "iteration": 2,
                        "stage": "safety_tests",
                        "candidate_id": "candidate_bpr",
                        "error_type": "AssertionError",
                        "error": "Shapes mismatch",
                        "lesson": "Align batch dimensions before matrix multiply",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "research_memory.jsonl").write_text(
                json.dumps(
                    {
                        "type": "role_retry",
                        "iteration": 2,
                        "label": "builder",
                        "error_type": "SyntaxError",
                        "error": "invalid syntax",
                        "reprompt": "Fix syntax error on line 12",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            from src.ui.loaders import load_debugger_events

            events = load_debugger_events(run_dir)
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0].event_type, "debugger_memory")
            self.assertEqual(events[0].candidate_id, "candidate_bpr")
            self.assertEqual(events[1].event_type, "role_retry")
            self.assertEqual(events[1].stage, "builder")
            self.assertEqual(events[1].lesson, "Re-prompt #Fix syntax error on line 12: invalid syntax")



            snapshot = load_run_snapshot(run_dir)
            self.assertEqual(len(snapshot.debugger_events), 2)

    def test_load_llm_calls_tolerates_partial_and_foreign_files(self):
        from src.ui.loaders import load_llm_calls

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            passes = run_dir / "passes"
            passes.mkdir(parents=True)
            (passes / "001_researcher_0.json").write_text(
                json.dumps(
                    {
                        "recorded_at": "2026-08-30T16:05:00+00:00",
                        "prompt": "You are one role.",
                        "result": {
                            "role": "researcher",
                            "model": "deepseek-v4-pro",
                            "latency_seconds": 42.5,
                            "retries": 1,
                            "usage": {
                                "input_tokens": 6000,
                                "output_tokens": 4000,
                                "total_tokens": 10000,
                                "cached_tokens": 500,
                            },
                            "data": {"family": "bpr", "hypothesis": "Pairwise loss helps."},
                            "sources": [{"title": "BPR paper", "url": "https://arxiv.org/abs/1205.2618"}],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (passes / "001_critic_preflight_0.json").write_text(
                json.dumps(
                    {
                        "recorded_at": "2026-08-30T16:06:00+00:00",
                        "result": {
                            "role": "critic_preflight",
                            "data": {
                                "approved": True,
                                "decision": "approved_as_proposed",
                                "concerns": ["Guard the empty-pair case."],
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            # A file the harness is mid-writing, and one with a foreign name.
            (passes / "002_builder_0.json").write_text('{"recorded_at": "2026-', encoding="utf-8")
            (passes / "notes.json").write_text("{}", encoding="utf-8")

            calls = load_llm_calls(run_dir)
            self.assertEqual([c.role for c in calls], ["researcher", "critic_preflight"])
            first = calls[0]
            self.assertEqual(first.iteration, 1)
            self.assertEqual(first.family, "researcher")
            self.assertEqual(first.total_tokens, 10000)
            self.assertEqual(first.cached_tokens, 500)
            self.assertEqual(first.retries, 1)
            self.assertIn("[bpr]", first.gist)
            self.assertEqual(first.sources[0]["title"], "BPR paper")
            second = calls[1]
            self.assertEqual(second.family, "critic")
            self.assertEqual(second.model, "unknown")
            self.assertIn("approved_as_proposed", second.gist)
            self.assertIn("Guard the empty-pair case.", second.gist)

    def test_load_memory_events_filters_typed_records(self):
        from src.ui.loaders import load_memory_events

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            lines = [
                json.dumps({"role": "researcher", "iteration": 1, "usage": {}}),
                json.dumps(
                    {
                        "type": "role_retry",
                        "label": "researcher",
                        "iteration": 2,
                        "reprompt": 1,
                        "error": "missing field: hypothesis",
                        "error_type": "RoleOutputInvalid",
                    }
                ),
                json.dumps(
                    {
                        "type": "controller_error",
                        "iteration": 3,
                        "kind": "harness",
                        "error": "Request timed out.",
                        "error_type": "APITimeoutError",
                    }
                ),
                json.dumps({"type": "convergence", "iteration": 4}),
            ]
            (run_dir / "research_memory.jsonl").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            events = load_memory_events(run_dir)
            self.assertEqual([e.kind for e in events], ["role_retry", "controller_error"])
            self.assertEqual(events[0].reprompt, 1)
            self.assertEqual(events[0].label, "researcher")
            self.assertEqual(events[1].label, "harness")
            self.assertEqual(events[1].error_type, "APITimeoutError")

    def test_snapshot_surfaces_honest_story_fields_and_iteration_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "run_id": "honest",
                        "status": "completed",
                        "stop_reason": "iteration_budget_reached",
                        "converged_official": True,
                        "converged_official_iteration": 3,
                        "max_scored_primary": 0.6042,
                        "best_replicated": {"n": 3, "median_primary": 0.6033, "spread": 0.0004},
                        "best": {
                            "experiment_id": "bpr_best",
                            "metrics": {"GAUC": 0.67, "nDCG@5": 0.5368, "primary": 0.6034},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "run_config.json").write_text(
                json.dumps({"llm": {"max_total_tokens": 600000}, "budgets": {"max_iterations": 50}}),
                encoding="utf-8",
            )
            (run_dir / "baseline_selection.json").write_text(
                json.dumps(
                    {
                        "selected": "runs/prior/summary.json",
                        "skipped": [{"path": "runs/old/summary.json", "reason": "revision_mismatch"}],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "interventions.jsonl").write_text("", encoding="utf-8")
            (run_dir / "DATA_CARD.md").write_text("# Dataset Profile\n", encoding="utf-8")
            (run_dir / "iterations.jsonl").write_text(
                json.dumps(
                    {
                        "iteration": 1,
                        "repairs": 2,
                        "proposal": {"family": "bpr", "hypothesis": "h"},
                        "outcome": {
                            "status": "failed",
                            "failure_class": "timeout",
                            "error": "Experiment exceeded 900s",
                        },
                        "status": "failed",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = load_run_snapshot(run_dir)
            self.assertIs(snapshot.converged_official, True)
            self.assertEqual(snapshot.converged_official_iteration, 3)
            self.assertAlmostEqual(snapshot.max_scored_primary, 0.6042)
            self.assertEqual(snapshot.best_replicated["n"], 3)
            self.assertEqual(snapshot.token_cap, 600000)
            self.assertEqual(snapshot.baseline_selection["selected"], "runs/prior/summary.json")
            self.assertEqual(snapshot.interventions, ())
            self.assertTrue(snapshot.interventions_recorded)
            self.assertIn("Dataset Profile", snapshot.data_card_markdown)
            item = snapshot.iterations[0]
            self.assertEqual(item.failure_class, "timeout")
            self.assertEqual(item.error, "Experiment exceeded 900s")
            self.assertEqual(item.repairs, 2)

    def test_snapshot_derives_max_scored_primary_for_running_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            (run_dir / "state.json").write_text(
                json.dumps(
                    {
                        "run_id": "live",
                        "status": "running",
                        "baseline_primary": 0.6015,
                        "nodes": [
                            {"status": "success", "metrics": {"primary": 0.6021}},
                            {"status": "success", "metrics": {"primary": 0.6034}},
                            {"status": "failed", "metrics": {"primary": 0.9999}},
                            {"status": "success", "metrics": {}},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            snapshot = load_run_snapshot(run_dir)
            # summary.json does not exist yet: the raw max comes from scored nodes
            # only (the failed node's number must not leak into the story).
            self.assertAlmostEqual(snapshot.max_scored_primary, 0.6034)
            self.assertIsNone(snapshot.converged_official)
            self.assertFalse(snapshot.interventions_recorded)
            self.assertIsNone(snapshot.data_card_markdown)

    def test_mid_run_convergence_signal_comes_from_research_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            (run_dir / "state.json").write_text(
                json.dumps({"run_id": "live", "status": "running", "baseline_primary": 0.6015}),
                encoding="utf-8",
            )
            (run_dir / "research_memory.jsonl").write_text(
                json.dumps({"iteration": 3, "official": True, "type": "convergence"}) + "\n",
                encoding="utf-8",
            )
            snapshot = load_run_snapshot(run_dir)
            # summary.json does not exist yet, but the verdict is on disk and
            # must not be reported as pending.
            self.assertIs(snapshot.converged_official, True)
            self.assertEqual(snapshot.converged_official_iteration, 3)

    def test_snapshot_falls_back_to_state_meters_without_resources_file(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            (run_dir / "state.json").write_text(
                json.dumps(
                    {
                        "run_id": "old",
                        "status": "running",
                        "baseline_primary": 0.6015,
                        "iteration_count": 4,
                        "training_attempts": 5,
                        "wall_clock_seconds": 120.5,
                        "token_usage": {"total_tokens": 9000},
                    }
                ),
                encoding="utf-8",
            )
            snapshot = load_run_snapshot(run_dir, include_details=False)
            self.assertEqual(snapshot.resources.get("iteration_count"), 4)
            self.assertEqual(snapshot.resources.get("token_usage", {}).get("total_tokens"), 9000)

    def test_submission_validator_aggregates_row_errors(self):
        header = "row_id,user_id,video_id,score\n"
        body = "".join(f"{i},u{i},v{i},\n" for i in range(500))
        check = validate_submission(header + body)
        self.assertFalse(check.valid)
        # One aggregated line for 500 bad scores, not 500 error lines.
        self.assertLessEqual(len(check.errors), 3)
        self.assertTrue(any("500 row(s) have a non-finite score" in e for e in check.errors))

    def test_load_interventions_distinguishes_missing_empty_and_filled(self):
        from src.ui.loaders import load_interventions

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            entries, recorded = load_interventions(run_dir)
            self.assertEqual((entries, recorded), ((), False))

            (run_dir / "interventions.jsonl").write_text("", encoding="utf-8")
            entries, recorded = load_interventions(run_dir)
            self.assertEqual(entries, ())
            self.assertTrue(recorded)

            (run_dir / "interventions.jsonl").write_text(
                json.dumps({"at": "2026-08-30T00:00:00Z", "action": "restarted worker"}) + "\n",
                encoding="utf-8",
            )
            entries, recorded = load_interventions(run_dir)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["action"], "restarted worker")


if __name__ == "__main__":
    unittest.main()
