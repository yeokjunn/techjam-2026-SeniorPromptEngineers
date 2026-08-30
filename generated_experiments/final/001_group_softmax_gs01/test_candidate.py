import unittest
import numpy as np
from types import SimpleNamespace
from unittest.mock import patch
from candidate import run

class TestGroupSoftmaxCandidate(unittest.TestCase):

    def _make_context(self):
        rng = np.random.default_rng(0)
        n_rows = 40
        n_users = 5
        n_fields = 8
        users = [f'u{i%n_users}' for i in range(n_rows)]
        # Feature indices must be integer (categorical) for FMRanker
        train_x = rng.integers(0, 10, size=(n_rows, n_fields)).astype(np.int64)
        train_y = np.array([1 if i % 3 != 0 else 0 for i in range(n_rows)], dtype=np.float32)
        valid_x = rng.integers(0, 10, size=(10, n_fields)).astype(np.int64)
        valid_users = [f'u{i%n_users}' for i in range(10)]
        context = SimpleNamespace(
            train_x=train_x,
            train_y=train_y,
            train_users=users,
            valid_x=valid_x,
            valid_users=valid_users,
            field_dimension=n_fields,
            test_x=None,
            evaluate_validation=lambda scores: 0.5  # provides the required callable
        )
        return context

    def test_run_returns_valid_output(self):
        context = self._make_context()
        params = {
            'seed': 42,
            'k': 16,
            'learning_rate': 0.0005,
            'epochs': 2,
            'batch_size': 8,
            'patience': 2,
            'negatives_per_group': 4,
            'temperature': 1.0
        }
        result = run(context, params)
        self.assertEqual(len(result.validation_scores), len(context.valid_x))
        self.assertTrue(np.isfinite(result.validation_scores).all())
        self.assertIsInstance(result.checkpoint_state, dict)
        self.assertIn('V', result.checkpoint_state)
        self.assertIn('W', result.checkpoint_state)
        self.assertIn('b', result.checkpoint_state)

    def test_sample_softmax_groups_shape(self):
        from src.models.sampling import sample_softmax_groups
        users = ['u1', 'u1', 'u1', 'u1', 'u2', 'u2', 'u3']
        labels = np.array([1, 0, 0, 0, 1, 0, 1], dtype=np.float32)
        rng = np.random.default_rng(0)
        pos, neg = sample_softmax_groups(users, labels, rng, negatives_per_group=3)
        self.assertEqual(neg.ndim, 2)
        self.assertEqual(neg.shape[1], 3)
        self.assertEqual(pos.shape[0], neg.shape[0])

if __name__ == '__main__':
    unittest.main()
