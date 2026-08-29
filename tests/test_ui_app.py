from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


STREAMLIT_AVAILABLE = importlib.util.find_spec("streamlit") is not None


@unittest.skipUnless(STREAMLIT_AVAILABLE, "optional Streamlit dependency is not installed")
class DashboardAppTests(unittest.TestCase):
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
