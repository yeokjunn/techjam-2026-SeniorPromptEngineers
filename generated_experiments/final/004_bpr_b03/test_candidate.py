import unittest
import types
import numpy as np
import candidate
from src.models.sampling import sample_bpr_pairs
from src.experiments.contracts import CandidateOutput


class TestCandidate(unittest.TestCase):
    def test_sample_bpr_pairs_same_user(self):
        users = ['u1', 'u1', 'u2', 'u2', 'u3', 'u3']
        labels = np.array([1, 0, 1, 0, 1, 0], dtype=np.int64)
        pos, neg = sample_bpr_pairs(users, labels, np.random.default_rng(7), 2)
        self.assertEqual(len(pos), len(neg))
        self.assertTrue(np.all(pos >= 0))
        self.assertTrue(np.all(neg >= 0))
        for p, n in zip(pos.tolist(), neg.tolist()):
            self.assertEqual(users[p], users[n])
            self.assertEqual(int(labels[p]), 1)
            self.assertEqual(int(labels[n]), 0)

    def test_run_on_synthetic_context(self):
        ctx = types.SimpleNamespace(
            train_x=np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], dtype=np.int64),
            train_y=np.array([1, 0, 1, 0, 1, 0], dtype=np.float32),
            train_users=['u1', 'u1', 'u2', 'u2', 'u3', 'u3'],
            valid_x=np.array([[2, 3], [4, 5]], dtype=np.int64),
            valid_users=['u1', 'u2'],
            field_dimension=100,
            evaluate_validation=lambda scores: {'primary': float(np.mean(np.asarray(scores, dtype=np.float64))), 'gauc': 0.5, 'nDCG@5': 0.5},
            test_x=None,
        )
        params = {'seed': 0, 'k': 2, 'learning_rate': 0.001, 'epochs': 1, 'batch_size': 2, 'patience': 2, 'negatives_per_positive': 1, 'negatives_per_group': None, 'temperature': None}
        output = candidate.run(ctx, params)
        self.assertIsInstance(output, CandidateOutput)
        self.assertEqual(len(output.validation_scores), len(ctx.valid_x))
        self.assertTrue(np.all(np.isfinite(output.validation_scores)))
        self.assertIsNotNone(output.checkpoint_state)
        self.assertIsInstance(output.checkpoint_state, dict)
        self.assertIsNone(output.test_scores)
        self.assertIsInstance(output.training_trace, list)


if __name__ == '__main__':
    unittest.main()