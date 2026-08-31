import unittest
import numpy as np
from unittest.mock import patch
from src.experiments.contracts import CandidateOutput

class FakeContext:

    def __init__(self):
        self.train_x = np.array([[0, 1], [1, 1], [0, 2], [2, 1], [2, 2]], dtype=np.int32)
        self.train_y = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        self.train_users = np.array([0, 0, 1, 1, 1], dtype=np.int32)
        self.valid_x = np.array([[1, 2], [0, 1]], dtype=np.int32)
        self.valid_users = np.array([0, 1], dtype=np.int32)
        self.field_dimension = 10
        self.test_x = None
        self.random_valid_x = None

    def evaluate_validation(self, scores):
        return {'gauc': 0.5, 'ndcg@5': 0.5, 'primary': 0.5}

class TestCandidate(unittest.TestCase):

    def test_run(self):
        params = {'seed': 42, 'learning_rate': 0.001, 'epochs': 1, 'batch_size': 2, 'negatives_per_positive': 1, 'patience': 1, 'k': 16, 'spec': {'columns': ['user', 'item']}}
        ctx = FakeContext()
        import candidate

        def fake_build_features(data, spec):
            return np.asarray(data, dtype=np.int32)

        def fake_sample_bpr_pairs(users, labels, rng, negatives_per_positive):
            return (np.array([0, 2], dtype=np.int32), np.array([1, 3], dtype=np.int32))
        with patch('candidate.build_features', side_effect=fake_build_features), patch('candidate.sample_bpr_pairs', side_effect=fake_sample_bpr_pairs):
            out = candidate.run(ctx, params)
        self.assertIsInstance(out, CandidateOutput)
        self.assertTrue(np.all(np.isfinite(out.validation_scores)))
        self.assertIsNone(out.test_scores)
        self.assertIsNone(out.random_validation_scores)
        self.assertEqual(len(out.validation_scores), len(ctx.valid_x))
        self.assertIn('V', out.checkpoint_state)
        self.assertIn('W', out.checkpoint_state)
        self.assertIn('b', out.checkpoint_state)
        self.assertEqual(len(out.training_trace), 1)
        self.assertTrue(np.all(np.isfinite(out.validation_scores)))
if __name__ == '__main__':
    unittest.main()
