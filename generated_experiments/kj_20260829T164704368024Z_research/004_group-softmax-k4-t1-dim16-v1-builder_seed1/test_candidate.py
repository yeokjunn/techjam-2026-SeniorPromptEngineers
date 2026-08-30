import unittest
import numpy as np
from candidate import run, _dedupe_negatives
from src.models.sampling import sample_softmax_groups


class FakeContext:
    def __init__(self, with_test=True, seed=0):
        rng = np.random.default_rng(seed)
        n_rows, n_users, n_fields, field_dim = 300, 20, 5, 60
        self.field_dimension = field_dim
        self.train_x = rng.integers(0, field_dim, size=(n_rows, n_fields))
        self.train_y = rng.integers(0, 2, size=n_rows).astype(float)
        self.train_users = rng.integers(0, n_users, size=n_rows)
        for u in range(n_users):
            rows = np.nonzero(self.train_users == u)[0]
            if len(rows) >= 2:
                self.train_y[rows[0]] = 1.0
                self.train_y[rows[1]] = 0.0
        self.valid_x = rng.integers(0, field_dim, size=(80, n_fields))
        self.valid_users = rng.integers(0, n_users, size=80)
        self.test_x = rng.integers(0, field_dim, size=(50, n_fields)) if with_test else None

    def evaluate_validation(self, scores):
        scores = np.asarray(scores, dtype=float)
        return {"gauc": 0.5 + 0.001 * float(scores.mean()), "ndcg@5": 0.5}


class TestGroupSoftmax(unittest.TestCase):
    def _params(self):
        return {"batch_size": 512, "epochs": 2, "k": 16, "learning_rate": 0.0005,
                "negatives_per_group": 4, "patience": 2, "seed": 0, "temperature": 1.0}

    def test_end_to_end(self):
        ctx = FakeContext()
        out = run(ctx, self._params())
        self.assertIsInstance(out.validation_scores, np.ndarray)
        self.assertEqual(out.validation_scores.shape[0], 80)
        self.assertTrue(np.all(np.isfinite(out.validation_scores)))
        self.assertIsNotNone(out.test_scores)
        self.assertEqual(out.test_scores.shape[0], 50)
        self.assertTrue(np.all(np.isfinite(out.test_scores)))
        self.assertGreaterEqual(len(out.training_trace), 1)
        for arr in out.checkpoint_state.values():
            self.assertIsInstance(arr, np.ndarray)
        self.assertIn("groups_per_epoch", out.diagnostics)
        self.assertIn("duplicate_negatives_resampled", out.diagnostics)

    def test_test_scores_none_when_test_x_none(self):
        ctx = FakeContext(with_test=False)
        out = run(ctx, self._params())
        self.assertIsNone(out.test_scores)

    def test_dedupe_negatives_removes_pos_and_intra_group_dups(self):
        rng = np.random.default_rng(1)
        users = np.array([0] * 8 + [1] * 8)
        pos = np.array([0, 8])
        neg = np.array([[0, 0, 1, 2], [8, 9, 9, 10]])
        fixed, n_dup = _dedupe_negatives(users, pos, neg, rng)
        # Row 0: columns 0 and 1 equal the positive (2 flags).
        # Row 1: column 0 equals the positive; columns 1 and 2 equal each
        # other and both are flagged (3 flags). Total = 5 flagged entries.
        self.assertEqual(n_dup, 5)
        # Clean negatives must not be resampled when a sibling is a duplicate:
        # row 0 keeps its clean columns 2 and 3 unchanged.
        self.assertEqual(int(fixed[0, 2]), 1)
        self.assertEqual(int(fixed[0, 3]), 2)
        for row in range(fixed.shape[0]):
            self.assertNotIn(int(pos[row]), fixed[row].tolist())
            self.assertEqual(len(set(fixed[row].tolist())), fixed.shape[1])

    def test_dedupe_negatives_noop_when_clean(self):
        rng = np.random.default_rng(2)
        users = np.array([0] * 8 + [1] * 8)
        pos = np.array([0, 8])
        neg = np.array([[1, 2, 3, 4], [9, 10, 11, 12]])
        fixed, n_dup = _dedupe_negatives(users, pos, neg, rng)
        self.assertEqual(n_dup, 0)
        self.assertTrue(np.array_equal(fixed, neg))

    def test_sample_softmax_groups_shapes(self):
        rng = np.random.default_rng(3)
        users = np.arange(40) % 8
        labels = np.tile(np.array([1.0, 0.0, 1.0, 0.0, 1.0]), 8)
        pos, neg = sample_softmax_groups(users, labels, rng, negatives_per_group=4)
        self.assertEqual(neg.shape[1], 4)
        self.assertTrue(np.all(labels[np.asarray(pos, dtype=int)] > 0))


if __name__ == "__main__":
    unittest.main()