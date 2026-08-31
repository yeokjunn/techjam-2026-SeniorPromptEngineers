import unittest
import numpy as np
from unittest.mock import patch
from src.experiments.contracts import CandidateOutput
import candidate

class TestCandidate(unittest.TestCase):

    def test_run_output(self):
        n_train = 100
        n_valid = 20
        n_test = 15
        n_fields = 5
        field_dim = 50
        rng = np.random.RandomState(0)
        train_users = np.repeat(np.arange(10), 10)
        train_y = np.zeros(n_train, dtype=np.int32)
        for u in range(10):
            idx = np.where(train_users == u)[0]
            mid = len(idx) // 2
            train_y[idx[:mid]] = 1
        train_x = rng.randint(0, field_dim, size=(n_train, n_fields)).astype(np.int32)
        valid_x = rng.randint(0, field_dim, size=(n_valid, n_fields)).astype(np.int32)
        test_x = rng.randint(0, field_dim, size=(n_test, n_fields)).astype(np.int32)

        def evaluate_validation(scores):
            return {'gauc': 0.5, 'ndcg@5': 0.5}

        class FakeContext:
            pass
        ctx = FakeContext()
        ctx.train_x = train_x
        ctx.train_y = train_y
        ctx.train_users = train_users
        ctx.valid_x = valid_x
        ctx.valid_users = np.arange(10)
        ctx.field_dimension = field_dim
        ctx.evaluate_validation = evaluate_validation
        ctx.test_x = test_x
        ctx.random_valid_x = None
        parameters = {'seed': 42, 'k': 16, 'learning_rate': 0.001, 'epochs': 2, 'batch_size': 16, 'patience': 1, 'negatives_per_positive': 1, 'aux_weight': 0.05, 'use_is_click': True, 'use_is_like': False, 'use_is_follow': False, 'use_is_comment': False, 'use_is_forward': False, 'use_play_time': False}

        def fake_aux(rows, spec):
            return np.full((len(rows), 1), 0.5, dtype=np.float32)
        with patch('candidate.build_aux_labels', side_effect=fake_aux):
            result = candidate.run(ctx, parameters)
        self.assertIsInstance(result, CandidateOutput)
        self.assertEqual(len(result.validation_scores), n_valid)
        self.assertTrue(np.all(np.isfinite(result.validation_scores)))
        self.assertEqual(len(result.test_scores), n_test)
        self.assertTrue(np.all(np.isfinite(result.test_scores)))
        self.assertIsNone(result.random_validation_scores)
        self.assertIn('V', result.checkpoint_state)
        self.assertIn('W', result.checkpoint_state)
        self.assertIn('b', result.checkpoint_state)
        self.assertGreaterEqual(len(result.training_trace), 1)
        self.assertIn('epoch', result.training_trace[0])
        self.assertIn('primary', result.training_trace[0])
if __name__ == '__main__':
    unittest.main()
