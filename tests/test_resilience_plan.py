import unittest

from src.agent.families import AUX_HEADS, HISTORY_GROUPS
from src.agent.policy import sanitize_parameters
from src.agent.types import EDAReport


class ResiliencePlanTests(unittest.TestCase):
    def test_history_groups_require_explicit_opt_in(self):
        result = sanitize_parameters("history_features", {"use_recency": True})
        self.assertTrue(result["use_recency"])
        for group in HISTORY_GROUPS:
            if group != "recency":
                self.assertFalse(result[f"use_{group}"])

    def test_history_features_rejects_empty_signal_set(self):
        with self.assertRaisesRegex(ValueError, "at least one candidate-dependent"):
            sanitize_parameters(
                "history_features",
                {f"use_{group}": False for group in HISTORY_GROUPS},
            )

    def test_multi_task_rejects_empty_auxiliary_set(self):
        with self.assertRaisesRegex(ValueError, "at least one auxiliary"):
            sanitize_parameters(
                "multi_task", {f"use_{head}": False for head in AUX_HEADS}
            )

    def test_eda_report_truncates_schema_strings(self):
        report = EDAReport.from_dict({
            "summary": "s" * 400,
            "findings": [{"title": "t" * 90, "observation": "o" * 230,
                          "implication": "i" * 230, "evidence": "e" * 190}],
            "feature_candidates": [{"name": "n" * 90, "description": "d" * 230,
                "family": "f" * 50, "expected_impact": "x" * 170,
                "implementation_scope": "p" * 190, "leakage_risk": "r" * 190}],
            "recommended_next_focus": "z" * 310, "ui_notes": ["u" * 170],
        })
        self.assertEqual(len(report.summary), 360)
        self.assertEqual(len(report.findings[0].title), 80)
        self.assertEqual(len(report.feature_candidates[0].description), 220)
        self.assertEqual(len(report.recommended_next_focus), 300)
        self.assertEqual(len(report.ui_notes[0]), 160)


if __name__ == "__main__":
    unittest.main()
