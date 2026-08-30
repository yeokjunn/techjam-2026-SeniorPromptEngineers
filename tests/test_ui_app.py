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
                app = AppTest.from_file(str(app_path), default_timeout=20).run()
                self.assertEqual(list(app.exception), [])
                self.assertEqual(
                    [tab.label for tab in app.tabs],
                    ["Pipeline", "EDA", "Feature Lab", "Iterations", "Results"],
                )

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



if __name__ == "__main__":
    unittest.main()
