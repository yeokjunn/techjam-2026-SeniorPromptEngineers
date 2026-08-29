from __future__ import annotations

import math
import unittest

from src.evaluation.official import official_evaluate


class OfficialEvaluationTests(unittest.TestCase):
    def test_perfect_two_item_user_scores_one(self):
        metrics = official_evaluate(
            user_ids=["user", "user"],
            labels=[1, 0],
            scores=[1.0, 0.0],
        )
        self.assertAlmostEqual(metrics["GAUC"], 1.0)
        self.assertAlmostEqual(metrics["nDCG@5"], 1.0)
        self.assertAlmostEqual(metrics["primary"], 1.0)

    def test_zero_positive_user_is_counted_in_ndcg_and_excluded_from_gauc(self):
        # a: perfect one-positive user; b: zero-positive user -> GAUC skips b,
        # nDCG records 0.0 for b and includes it in the mean.
        metrics = official_evaluate(
            user_ids=["a", "a", "b", "b"],
            labels=[1, 0, 0, 0],
            scores=[1.0, 0.0, 1.0, 0.0],
        )
        self.assertAlmostEqual(metrics["GAUC"], 1.0)
        self.assertAlmostEqual(metrics["nDCG@5"], 0.5)
        self.assertAlmostEqual(metrics["primary"], 0.75)
        self.assertEqual(metrics["users"], 2.0)
        self.assertEqual(metrics["rows"], 4.0)

    def test_all_positive_user_is_excluded_from_gauc(self):
        # evaluate.py only accumulates GAUC for 0 < positives < impressions, so
        # an all-positive user ranked arbitrarily changes nothing.
        base = official_evaluate(
            user_ids=["a", "a", "c", "c"],
            labels=[1, 0, 1, 1],
            scores=[1.0, 0.0, 0.9, 0.1],
        )
        self.assertAlmostEqual(base["GAUC"], 1.0)
        self.assertEqual(base["users"], 2.0)

    def test_gauc_is_weighted_by_positive_count(self):
        # p: 3 rows, labels [1,1,0], perfect order -> AUC 1.0, weight 2.
        # q: 2 rows, positive ranked last -> AUC 0.0, weight 1.
        # Weighted mean 2/3, not the unweighted 0.5.
        metrics = official_evaluate(
            user_ids=["p", "p", "p", "q", "q"],
            labels=[1, 1, 0, 0, 1],
            scores=[3.0, 2.0, 1.0, 1.0, 0.0],
        )
        self.assertAlmostEqual(metrics["GAUC"], 2.0 / 3.0, places=12)

    def test_ties_are_broken_by_row_order(self):
        # Identical scores: the sort is stable, so the original row order is
        # the ranked order. Positive second -> 1/log2(3); positive first -> 1.0;
        # tie-corrected AUC is 0.5 both ways.
        labels_second = official_evaluate(
            user_ids=["u", "u"], labels=[0, 1], scores=[0.5, 0.5]
        )
        self.assertAlmostEqual(
            labels_second["nDCG@5"], 1.0 / math.log2(3), places=12
        )
        labels_first = official_evaluate(
            user_ids=["u", "u"], labels=[1, 0], scores=[0.5, 0.5]
        )
        self.assertAlmostEqual(labels_first["nDCG@5"], 1.0, places=12)
        self.assertAlmostEqual(labels_second["GAUC"], 0.5, places=12)
        self.assertAlmostEqual(labels_first["GAUC"], 0.5, places=12)

    def test_ndcg_truncates_at_k_5(self):
        # Six impressions, one positive: at rank 5 the discount is log2(6)
        # against an ideal of log2(2); at rank 6 the gain falls off the truncation.
        at_five = official_evaluate(
            user_ids=["u"] * 6,
            labels=[0, 0, 0, 0, 1, 0],
            scores=[6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        )
        self.assertAlmostEqual(
            at_five["nDCG@5"], 1.0 / math.log2(6), places=12
        )
        at_six = official_evaluate(
            user_ids=["u"] * 6,
            labels=[0, 0, 0, 0, 0, 1],
            scores=[6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        )
        self.assertAlmostEqual(at_six["nDCG@5"], 0.0, places=12)

    def test_users_and_rows_are_reported(self):
        metrics = official_evaluate(
            user_ids=["a", "a", "b"],
            labels=[1, 0, 1],
            scores=[0.9, 0.1, 0.5],
        )
        self.assertEqual(metrics["users"], 2.0)
        self.assertEqual(metrics["rows"], 3.0)


if __name__ == "__main__":
    unittest.main()
