from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


STREAMLIT_AVAILABLE = importlib.util.find_spec("streamlit") is not None


@unittest.skipUnless(STREAMLIT_AVAILABLE, "optional Streamlit dependency is not installed")
class DashboardAppTests(unittest.TestCase):
    def test_experiment_dag_uses_distinct_ids_escapes_labels_and_skips_missing_parents(self):
        from src.ui.app import _experiment_dag_dot

        dot = _experiment_dag_dot(
            (
                {
                    "experiment_id": "fm-v2",
                    "family": 'pairwise "BPR"',
                    "status": "success",
                    "metrics": {"primary": 0.61},
                },
                {
                    "experiment_id": "fm_v2",
                    "parent_experiment": "missing-parent",
                    "status": "failed",
                },
                {
                    "experiment_id": None,
                    "parent_experiment": "fm-v2",
                    "status": "success",
                },
            ),
            "fm-v2",
        )

        self.assertIn('n0 [label="fm-v2\\n[pairwise \\"BPR\\"]', dot)
        self.assertIn('n1 [label="fm_v2', dot)
        self.assertIn('n2 [label="unknown_2', dot)
        self.assertIn("n0 -> n2;", dot)
        self.assertNotIn("missing-parent ->", dot)

    def test_both_dashboard_entry_points_render_without_exceptions(self):
        from streamlit.testing.v1 import AppTest

        root = Path(__file__).resolve().parents[1]
        for app_path in (root / "streamlit_app.py", root / "src" / "ui" / "app.py"):
            with self.subTest(entry_point=app_path.name):
                app = AppTest.from_file(str(app_path), default_timeout=30).run()
                self.assertEqual(list(app.exception), [])
                self.assertEqual(
                    [tab.label for tab in app.tabs],
                    ["Story", "Activity", "Evidence", "Compare", "Judge sheet"],
                )
                # Each destination's summoned sections exist and default sanely.
                by_key = {radio.key: radio for radio in app.radio}
                self.assertEqual(by_key["section-activity"].value, "Loop stages")
                self.assertEqual(by_key["section-evidence"].value, "Trust & audit")

    def test_live_stream_and_diagnostics_render(self):
        from src.ui.models import DebuggerEvent, EDAArtifact, RolePass, RunSnapshot, StageTransition
        from src.ui.app import _render_live_role_stream, _render_live_diagnostics, _feature_lab, _eda
        from pathlib import Path

        snapshot = RunSnapshot(
            run_id="live_test",
            path=Path("."),
            status="running",
            stop_reason=None,
            started_at="2026-08-30T00:00:00Z",
            best_experiment_id="cand_1",
            best_metrics={"primary": 0.605},
            baseline_primary=0.6016,
            activity=StageTransition(
                event_id="e1",
                iteration=2,
                stage="debugger",
                status="active",
                started_at="2026-08-30T00:00:00Z",
                updated_at="2026-08-30T00:01:00Z",
                attempt=2,
                objective="Debug BPR candidate",
                error="Shapes mismatch in loss",
                repair="Added squeeze() before dot product",
            ),
            live_role_passes=(
                RolePass(
                    sequence=0,
                    role="eda_researcher",
                    model="GLM-5.3-Flash",
                    latency_seconds=1.0,
                    data={"objective": "Plan EDA", "questions": ["Q1?"], "leakage_risks": ["Risk1"]},
                ),
                RolePass(
                    sequence=1,
                    role="eda_builder",
                    model="GLM-5.3-Flash",
                    latency_seconds=1.5,
                    data={
                        "summary": "EDA Done",
                        "findings": [{"insight": "High correlation"}],
                        "feature_candidates": [{"name": "dur_bucket"}],
                    },
                ),
            ),
            live_eda=EDAArtifact(
                iteration=2,
                path=Path("eda/latest.json"),
                status="in_progress",
                plan={"objective": "Plan EDA"},
                report={"summary": "Live findings", "feature_candidates": [{"name": "live_feat"}]},
                feature_candidates=({"name": "live_feat"},),
            ),
            debugger_events=(
                DebuggerEvent(
                    iteration=2,
                    stage="safety_tests",
                    candidate_id="cand_1",
                    error_type="AssertionError",
                    error="Shapes mismatch",
                    lesson="Squeeze tensor dimension",
                ),
            ),
        )

        # Ensure these rendering helper functions execute without throwing any exceptions
        _render_live_role_stream(snapshot)
        _render_live_diagnostics(snapshot)
        _feature_lab(snapshot)

    def _snapshot(self, **overrides):
        from src.ui.models import RunSnapshot

        base = dict(
            run_id="ledger_test",
            path=Path("."),
            status="completed",
            stop_reason="iteration_budget_reached",
            started_at="2026-08-30T00:00:00Z",
            best_experiment_id="bpr_best",
            best_metrics={"GAUC": 0.67, "nDCG@5": 0.5368, "primary": 0.6034},
            baseline_primary=0.6015,
        )
        base.update(overrides)
        return RunSnapshot(**base)

    def test_verdict_ledger_pairs_stop_reason_with_official_rule(self):
        from src.ui.app import _verdict_ledger_html
        from src.ui.models import IterationSnapshot

        iterations = tuple(
            IterationSnapshot(iteration=i, experiment_id=f"e{i}", status="success")
            for i in range(1, 6)
        )
        snapshot = self._snapshot(
            iterations=iterations,
            converged_official=True,
            converged_official_iteration=3,
            max_scored_primary=0.6042,
            best_replicated={"n": 3, "median_primary": 0.6033, "spread": 0.0004},
            interventions=(),
            interventions_recorded=True,
            run_config={
                "convergence": {"epsilon": 0.002, "patience": 3},
                "budgets": {"max_iterations": 50, "max_wall_clock_seconds": 21600},
                "llm": {"max_total_tokens": 600000},
            },
            token_cap=600000,
            resources={
                "token_usage": {"total_tokens": 267145},
                "iteration_count": 5,
                "wall_clock_seconds": 2455.0,
            },
        )
        markup = _verdict_ledger_html(snapshot)
        self.assertIn("fired @ iter 3", markup)
        self.assertIn("harness continued to iter 5 for coverage", markup)
        self.assertIn("iteration budget reached", markup)
        self.assertIn("0.6034", markup)  # margin-gated claim
        self.assertIn("0.6042", markup)  # raw max, beside the claim
        self.assertIn("above the claim", markup)
        self.assertIn("n = 3", markup)
        self.assertIn("spread 0.0004", markup)
        self.assertIn("ledger present and empty", markup)
        self.assertIn("267,145 / 600,000", markup)

    def test_verdict_ledger_is_honest_about_pending_and_missing_evidence(self):
        from src.ui.app import _verdict_ledger_html

        snapshot = self._snapshot(
            status="running",
            stop_reason=None,
            best_metrics=None,
            best_experiment_id=None,
            interventions_recorded=False,
        )
        markup = _verdict_ledger_html(snapshot)
        self.assertIn("RUNNING", markup)
        self.assertIn("pending", markup)
        self.assertIn("no promoted candidate yet", markup)
        self.assertIn("not replicated", markup)
        self.assertIn("no ledger file recorded", markup)
        self.assertIn("cap not recorded", markup)

    def test_agent_trace_helpers_summarize_calls_by_family(self):
        from src.ui.app import _burn_down_chart, _role_summary_rows, _trace_rows
        from src.ui.models import LLMCall

        calls = tuple(
            LLMCall(
                iteration=1,
                role=role,
                family=family,
                sequence=0,
                recorded_at=f"2026-08-30T16:0{i}:00+00:00",
                model="deepseek-v4-pro",
                latency_seconds=10.0 * (i + 1),
                input_tokens=100,
                output_tokens=200,
                total_tokens=300,
                cached_tokens=50,
                retries=i,
                gist=f"gist {i}",
            )
            for i, (role, family) in enumerate(
                [("researcher", "researcher"), ("builder", "builder"), ("builder", "builder")]
            )
        )
        snapshot = self._snapshot(llm_calls=calls, token_cap=1000)
        rows = _role_summary_rows(snapshot)
        by_family = {row["role family"]: row for row in rows}
        self.assertEqual(by_family["Builder"]["calls"], 2)
        self.assertEqual(by_family["Builder"]["tokens"], 600)
        self.assertEqual(by_family["Builder"]["share"], "66.7%")
        self.assertEqual(by_family["Researcher"]["tokens"], 300)

        trace = _trace_rows(calls)
        self.assertEqual(len(trace), 3)
        self.assertEqual(trace[0]["role"], "Researcher")
        self.assertEqual(trace[1]["total"], 300)
        self.assertEqual(trace[2]["retries"], 2)

        chart = _burn_down_chart(snapshot)
        self.assertIsNotNone(chart)
        self.assertIsNone(_burn_down_chart(self._snapshot(llm_calls=())))

    def test_agents_and_memory_chains_render_without_exceptions(self):
        from src.ui.app import _agents, _render_memory_chains
        from src.ui.models import LLMCall, MemoryEvent

        snapshot = self._snapshot(
            llm_calls=(
                LLMCall(
                    iteration=1,
                    role="builder",
                    family="builder",
                    sequence=0,
                    recorded_at="2026-08-30T16:00:00+00:00",
                    model="deepseek-v4-pro",
                    latency_seconds=200.0,
                    total_tokens=25000,
                    gist="candidate · 5,450 chars code",
                    prompt="You are one role in an autonomous agent.",
                    data={"candidate_id": "bpr_first"},
                ),
            ),
            memory_events=(
                MemoryEvent(
                    kind="role_retry",
                    iteration=1,
                    label="researcher",
                    error="missing field: hypothesis",
                    error_type="RoleOutputInvalid",
                    reprompt=1,
                ),
                MemoryEvent(
                    kind="controller_error",
                    iteration=2,
                    label="harness",
                    error="Request timed out.",
                    error_type="APITimeoutError",
                ),
            ),
            token_cap=600000,
        )
        _agents(snapshot)
        _render_memory_chains(snapshot)
        # An empty snapshot renders the honest empty states, not an exception.
        _agents(self._snapshot(llm_calls=(), memory_events=()))

    def test_verdict_ledger_flags_a_stale_running_run_and_inherited_claims(self):
        from src.ui.app import _verdict_ledger_html
        from src.ui.models import StageTransition

        stale = self._snapshot(
            status="running",
            stop_reason=None,
            best_metrics={"GAUC": 0.6671, "nDCG@5": 0.5358, "primary": 0.6015},
            best_experiment_id="official_fm_seed0",
            max_scored_primary=None,
            activity=StageTransition(
                event_id="e1",
                iteration=1,
                stage="builder",
                status="active",
                started_at="2026-08-29T00:00:00Z",
                updated_at="2026-08-29T00:00:00Z",
            ),
        )
        markup = _verdict_ledger_html(stale, stale_after=1200)
        # A run quiet for a day is not presented as confidently live...
        self.assertIn("RUNNING?", markup)
        self.assertIn("possibly interrupted", markup)
        # ...and an inherited baseline artifact is never dressed as a result.
        self.assertIn("adopted baseline artifact", markup)
        self.assertNotIn("Δ +0.0000", markup)

    def test_journal_renders_without_unsafe_html(self):
        from src.ui.app import _render_journal_markdown

        # Must not raise, and must not need unsafe_allow_html: the diff fence
        # becomes an expander + code block, the rest plain markdown.
        _render_journal_markdown(
            "# Journal\n<script>alert('x')</script>\n"
            "```diff\n- old\n+ new\n```\ntail text"
        )

    def test_compare_and_judge_view_render_without_exceptions(self):
        from src.ui.app import _compare, _judge_view
        from src.ui.models import IterationSnapshot

        finished = self._snapshot(
            iterations=(
                IterationSnapshot(
                    iteration=1,
                    experiment_id="bpr_best",
                    status="success",
                    family="bpr",
                    parameters={"k": 16, "l2": 1e-6, "learning_rate": 0.001},
                    metrics={"GAUC": 0.67, "nDCG@5": 0.5368, "primary": 0.6034},
                ),
            ),
            max_scored_primary=0.6034,
            interventions_recorded=True,
        )
        running = self._snapshot(run_id="live_run", status="running", stop_reason=None)
        _compare([finished, running], 0.6016)
        _judge_view(finished, 0.6016)
        _judge_view(self._snapshot(best_metrics=None, best_experiment_id=None), 0.6016)

    def test_theme_is_config_plus_live_currentcolor_derivation_only(self):
        """The architecture that makes the in-menu theme toggle instant:
        config.toml is the only palette authority (Streamlit repaints it
        live), and the custom CSS names no mode color and never branches —
        every tone derives from currentColor via color-mix at the element
        where it is used. Any regression back to a server-picked palette
        (st.context.theme, _PALETTES, prefers-color-scheme, or a config hex
        baked into the CSS) breaks the no-reload toggle again."""
        import tomllib

        from src.ui import app

        root = Path(__file__).resolve().parents[1]
        source = (root / "src" / "ui" / "app.py").read_text(encoding="utf-8")
        config = tomllib.loads(
            (root / ".streamlit" / "config.toml").read_text(encoding="utf-8")
        )
        for mode in ("light", "dark"):
            section = config["theme"][mode]
            for key in (
                "backgroundColor",
                "secondaryBackgroundColor",
                "textColor",
                "primaryColor",
            ):
                self.assertIn(key, section)
            # No configured theme color may be baked into the page CSS.
            for value in section.values():
                self.assertNotIn(str(value).lower(), app._CSS.lower())
        # No server-side theme branching of any kind.
        self.assertNotIn("st.context.theme", source)
        self.assertNotIn("_PALETTES", source)
        self.assertNotIn("prefers-color-scheme", app._CSS)
        # The live mechanism itself.
        self.assertIn("currentColor", app._CSS)
        self.assertIn("color-mix", app._CSS)

    def test_story_band_tells_the_60_second_story_honestly(self):
        from src.ui.app import _story_band_html
        from src.ui.models import StageTransition

        finished = self._snapshot(
            stop_reason="converged",
            converged_official=True,
            converged_official_iteration=3,
            max_scored_primary=0.6035,
            gate_info={"status": "ok"},
            interventions_recorded=True,
        )
        markup = _story_band_html(finished, 0.6016)
        self.assertIn("Completed", markup)
        self.assertIn("fired @ iteration 3", markup)
        self.assertIn("loop ran untouched", markup)
        self.assertIn("Δ +0.0018", markup)
        self.assertIn("not replicated — single measurement", markup)
        self.assertIn("raw max 0.6035", markup)
        # No run_config on file: the budget cell says so instead of inventing caps.
        self.assertIn("No caps on record", markup)

        # An inherited baseline artifact is never dressed as this run's result.
        inherited = self._snapshot(max_scored_primary=None)
        markup = _story_band_html(inherited, 0.6016)
        self.assertIn("adopted baseline artifact", markup)
        self.assertNotIn("Δ +", markup)

        # A run quiet past the stale threshold is not presented as healthy-live.
        stale = self._snapshot(
            status="running",
            stop_reason=None,
            activity=StageTransition(
                event_id="e1",
                iteration=2,
                stage="builder",
                status="active",
                started_at="2026-08-29T00:00:00Z",
                updated_at="2026-08-29T00:00:00Z",
            ),
        )
        markup = _story_band_html(stale, 0.6016, stale_after=1200)
        self.assertIn("Quiet", markup)
        self.assertIn("possibly interrupted", markup)

        # With caps on record, the headline is what is LEFT of the tightest one.
        capped = self._snapshot(
            max_scored_primary=0.6035,
            token_cap=1000,
            resources={"token_usage": {"total_tokens": 700}},
            run_config={"budgets": {"max_iterations": 10}},
        )
        markup = _story_band_html(capped, 0.6016)
        self.assertIn("30% left", markup)
        self.assertIn("llm tokens is the tightest budget", markup)

    def test_story_band_survives_hostile_and_degenerate_artifacts(self):
        """Half-written or hand-edited run artifacts must degrade honestly:
        no crash, no unescaped HTML, no caps invented from junk values."""
        from src.ui.app import _story_band_html

        # A null token count and junk caps: render, don't crash.
        degenerate = self._snapshot(
            max_scored_primary=0.6035,
            resources={"token_usage": {"total_tokens": None}},
            run_config={
                "budgets": {"max_wall_clock_seconds": "0", "max_iterations": -5}
            },
        )
        markup = _story_band_html(degenerate, 0.6016)
        # Zero and negative caps are "no cap on record", never a division.
        self.assertIn("No caps on record", markup)

        # Usage beyond a real cap clamps at 0% left, never negative.
        overrun = self._snapshot(
            max_scored_primary=0.6035,
            token_cap=1000,
            resources={"token_usage": {"total_tokens": 1600}},
        )
        markup = _story_band_html(overrun, 0.6016)
        self.assertIn("0% left", markup)
        self.assertNotIn("-", markup.split("% left")[0][-6:])

        # Artifact-supplied replication counts are escaped before rendering.
        hostile = self._snapshot(
            max_scored_primary=0.6035,
            best_replicated={"n": "<script>alert(1)</script>", "median_primary": 0.6},
        )
        markup = _story_band_html(hostile, 0.6016)
        self.assertNotIn("<script>", markup)
        self.assertIn("&lt;script&gt;", markup)

    def test_story_landing_renders_without_exceptions(self):
        from src.ui.app import _story

        _story(
            self._snapshot(max_scored_primary=0.6035, interventions_recorded=True),
            0.6016,
            1200,
        )
        _story(self._snapshot(best_metrics=None, best_experiment_id=None), 0.6016, 1200)

    def test_iteration_summary_is_a_projection_of_the_full_ledger(self):
        from src.ui.app import _iteration_overview_rows, _iteration_summary_rows
        from src.ui.models import IterationSnapshot

        snapshot = self._snapshot(
            iterations=(
                IterationSnapshot(
                    iteration=1,
                    experiment_id="bpr_best",
                    status="success",
                    family="bpr",
                    parameters={"k": 16, "l2": 1e-6, "learning_rate": 0.001},
                    metrics={"GAUC": 0.67, "nDCG@5": 0.5368, "primary": 0.6034},
                ),
            ),
        )
        summary = _iteration_summary_rows(snapshot)
        full = _iteration_overview_rows(snapshot)
        self.assertEqual(len(summary), len(full))
        # The glanceable table shows fewer columns, and every one of them is
        # the same value the full record holds — a summary, never a rewrite.
        self.assertLess(len(summary[0]), len(full[0]))
        for key, value in summary[0].items():
            self.assertEqual(value, full[0][key])
        self.assertIn("primary", summary[0])
        self.assertIn("status", summary[0])


if __name__ == "__main__":
    unittest.main()
