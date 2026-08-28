from __future__ import annotations

import unittest

from src.evaluation.official import official_evaluate


class OfficialEvaluationTests(unittest.TestCase):
    def test_perfect_two_item_user_scores_one(self):
        metrics = official_evaluate(
            user_ids=["user" , "user"],
            labels=[1, 0],
            scores=[1.0, 0.0],
        )
        self.assertAlmostEqual(metrics["GAUC"], 1.0)
        self.assertAlmostEqual(metrics["nDCG@5"], 1.0)
        self.assertAlmostEqual(metrics["primary"], 1.0)


if __name__ == "__main__":
    unittest.main()

