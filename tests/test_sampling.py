from __future__ import annotations

import unittest

import numpy as np

from src.models.sampling import sample_bpr_pairs, sample_softmax_groups


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


if __name__ == "__main__":
    unittest.main()

