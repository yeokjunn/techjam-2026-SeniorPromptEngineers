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


if __name__ == "__main__":
    unittest.main()
