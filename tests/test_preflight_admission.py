import unittest

from src.agent.types import CriticDecision


class PreflightAdmissionTests(unittest.TestCase):
    def test_legacy_approved_decision_defaults_to_approved(self):
        decision = CriticDecision.from_dict({
            "approved": True,
            "decision": "approve",
            "rationale": "safe",
        })
        self.assertEqual(decision.admission, "approved")

    def test_soft_review_is_distinct_from_hard_rejection(self):
        decision = CriticDecision.from_dict({
            "approved": False,
            "decision": "review",
            "rationale": "novelty concern",
            "concerns": ["repeats a prior configuration"],
            "admission": "borderline",
        })
        self.assertFalse(decision.approved)
        self.assertEqual(decision.admission, "borderline")

    def test_hard_rejection_is_explicit(self):
        decision = CriticDecision.from_dict({
            "approved": False,
            "decision": "hard_reject",
            "rationale": "leakage",
            "admission": "hard_reject",
        })
        self.assertEqual(decision.admission, "hard_reject")


if __name__ == "__main__":
    unittest.main()
