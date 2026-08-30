import unittest
import numpy as np
from src.models.sampling import sample_softmax_groups
from candidate import run


class FakeContext:
    def __init__(self):
        rng = np.random.default_rng(0)
        self.field_dimension = 20
        n = 200
        self.train_x = rng.integers(0, self.field_dimension, size=(n, 5))
        self.valid_x = rng.integers(0, self.field_dimension, size=(n, 5))
        self.train_users = np.repeat(np.arange(20), 10)
        self.valid_users = np.repeat(np.arange(20), 10)
        self.train_y = (rng.random(n) < 0.5).astype(np.int64)
        for u in range(20):
            idx = np.where(self.train_users == u)[0]
            self.train_y[idx[0]] = 1
            self.train_y[idx[1]] = 0
        self.valid_y = (rng.random(n) < 0.4).astype(np.int64)
        self.test_x = None

    def evaluate_validation(self, scores):
        return {'primary': float(np.mean(scores)), 'gauc': 0.5, 'ndcg5': 0.5}


PARAMS = {'batch_size': 512, 'epochs': 2, 'k': 16, 'learning_rate': 0.0003,
          'negatives_per_group': 4, 'patience': 1, 'seed': 0, 'temperature': 1.0}


class TestSamplingGroups(unittest.TestCase):
    def test_same_user_group_construction(self):
        ctx = FakeContext()
        rng = np.random.default_rng(7)
        pos_idx, neg_groups = sample_softmax_groups(
            ctx.train_users, ctx.train_y, rng, negatives_per_group=4)
        self.assertEqual(neg_groups.ndim, 2)
        self.assertEqual(neg_groups.shape[1], 4)
        self.assertEqual(pos_idx.shape[0], neg_groups.shape[0])
        self.assertTrue(np.all(ctx.train_y[pos_idx] == 1))
        self.assertTrue(np.all(ctx.train_y[neg_groups.reshape(-1)] == 0))
        self.assertTrue(np.all(
            ctx.train_users[pos_idx][:, None] == ctx.train_users[neg_groups]))


class TestRun(unittest.TestCase):
    def test_run_returns_outputs(self):
        ctx = FakeContext()
        out = run(ctx, dict(PARAMS, negatives_per_group=8))
        self.assertEqual(out.validation_scores.shape[0], ctx.valid_x.shape[0])
        self.assertTrue(np.all(np.isfinite(out.validation_scores)))
        self.assertIsNone(out.test_scores)
        self.assertIn('best_valid_scores', out.checkpoint_state)
        self.assertTrue(len(out.training_trace) >= 1)
        self.assertEqual(out.diagnostics['negatives_per_group'], 8)
        self.assertEqual(out.diagnostics['steps_per_epoch'], 1)


if __name__ == '__main__':
    unittest.main()
