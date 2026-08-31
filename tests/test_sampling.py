from __future__ import annotations

import unittest

import numpy as np

from src.models.sampling import (
    sample_bpr_pairs,
    sample_constrained_hard_bpr_pairs,
    sample_hard_bpr_pairs,
    sample_softmax_groups,
)


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.users = ["u", "u", "u", "v", "v", "only_positive"]
        self.labels = np.asarray([1, 0, 0, 1, 0, 1], dtype=np.float32)

    def test_bpr_pairs_are_same_user_and_opposite_label(self):
        positives, negatives = sample_bpr_pairs(
            self.users, self.labels, np.random.default_rng(0), 2
        )
        self.assertGreater(len(positives), 0)
        for positive, negative in zip(positives, negatives):
            self.assertEqual(self.users[positive], self.users[negative])
            self.assertEqual(self.labels[positive], 1)
            self.assertEqual(self.labels[negative], 0)
            self.assertNotEqual(self.users[positive], "only_positive")

    def test_softmax_groups_have_expected_shape_and_same_user(self):
        positives, negatives = sample_softmax_groups(
            self.users, self.labels, np.random.default_rng(0), 2
        )
        self.assertEqual(negatives.shape, (len(positives), 2))
        for row, positive in enumerate(positives):
            for negative in negatives[row]:
                self.assertEqual(self.users[positive], self.users[negative])
                self.assertEqual(self.labels[negative], 0)

    def test_hard_bpr_pairs_stay_same_user_and_prefer_high_score_negatives(self):
        hardness = np.asarray([0.0, 0.9, 0.1, 0.0, 0.8, 0.0], dtype=np.float32)
        positives, negatives = sample_hard_bpr_pairs(
            self.users,
            self.labels,
            np.random.default_rng(0),
            hardness,
            negatives_per_positive=1,
            top_fraction=0.5,
        )
        self.assertGreater(len(positives), 0)
        for positive, negative in zip(positives, negatives):
            self.assertEqual(self.users[positive], self.users[negative])
            self.assertEqual(self.labels[positive], 1)
            self.assertEqual(self.labels[negative], 0)
        self.assertIn(1, negatives)
        self.assertIn(4, negatives)

    def test_constrained_hard_pairs_prefer_matching_key(self):
        hardness = np.asarray([0.0, 0.9, 0.1, 0.0, 0.8, 0.0], dtype=np.float32)
        constraints = np.asarray(["a", "a", "b", "x", "x", "z"])
        positives, negatives = sample_constrained_hard_bpr_pairs(
            self.users,
            self.labels,
            np.random.default_rng(1),
            hardness,
            constraints,
            negatives_per_positive=1,
        )
        for positive, negative in zip(positives, negatives):
            self.assertEqual(self.users[positive], self.users[negative])
            self.assertEqual(constraints[positive], constraints[negative])


if __name__ == "__main__":
    unittest.main()

