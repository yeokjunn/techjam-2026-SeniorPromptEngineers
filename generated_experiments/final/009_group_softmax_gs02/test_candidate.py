import unittest
import numpy as np
from src.experiments.contracts import CandidateOutput
from candidate import run

class TestRun(unittest.TestCase):
    def test_run_tiny_with_test(self):
        class MockContext:
            pass
        ctx = MockContext()
        n_train, n_valid, n_test, field_dim = 60, 20, 10, 100
        ctx.field_dimension = field_dim
        rng = np.random.default_rng(0)
        ctx.train_x = rng.integers(0, field_dim, size=(n_train, 5)).astype(np.int64)
        ctx.train_y = rng.integers(0, 2, size=n_train).astype(np.float32)
        ctx.train_users = np.arange(n_train) % 10
        ctx.valid_x = rng.integers(0, field_dim, size=(n_valid, 5)).astype(np.int64)
        ctx.valid_users = np.arange(n_valid) % 10
        ctx.test_x = rng.integers(0, field_dim, size=(n_test, 5)).astype(np.int64)
        def eval_func(scores):
            return (float(np.mean(scores)), 0.0, 0.0)
        ctx.evaluate_validation = eval_func
        params = {
            'seed': 0, 'k': 4, 'learning_rate': 0.01, 'epochs': 2,
            'batch_size': 8, 'negatives_per_group': 2, 'patience': 2, 'temperature': 1.0
        }
        result = run(ctx, params)
        self.assertIsInstance(result, CandidateOutput)
        self.assertEqual(len(result.validation_scores), n_valid)
        self.assertTrue(np.all(np.isfinite(result.validation_scores)))
        self.assertIsNotNone(result.test_scores)
        self.assertEqual(len(result.test_scores), n_test)
        self.assertTrue(np.all(np.isfinite(result.test_scores)))
        self.assertIsInstance(result.checkpoint_state, dict)
        for v in result.checkpoint_state.values():
            self.assertIsInstance(v, np.ndarray)
        self.assertTrue(len(result.training_trace) > 0)

    def test_run_tiny_no_test(self):
        class MockContext:
            pass
        ctx = MockContext()
        n_train, n_valid, field_dim = 60, 20, 100
        ctx.field_dimension = field_dim
        rng = np.random.default_rng(1)
        ctx.train_x = rng.integers(0, field_dim, size=(n_train, 5)).astype(np.int64)
        ctx.train_y = rng.integers(0, 2, size=n_train).astype(np.float32)
        ctx.train_users = np.arange(n_train) % 10
        ctx.valid_x = rng.integers(0, field_dim, size=(n_valid, 5)).astype(np.int64)
        ctx.valid_users = np.arange(n_valid) % 10
        ctx.test_x = None
        def eval_func(scores):
            return (float(np.mean(scores)),)
        ctx.evaluate_validation = eval_func
        params = {
            'seed': 1, 'k': 4, 'learning_rate': 0.01, 'epochs': 1,
            'batch_size': 8, 'negatives_per_group': 2, 'patience': 2, 'temperature': 1.0
        }
        result = run(ctx, params)
        self.assertIsNone(result.test_scores)
        self.assertTrue(np.all(np.isfinite(result.validation_scores)))

if __name__ == '__main__':
    unittest.main()
