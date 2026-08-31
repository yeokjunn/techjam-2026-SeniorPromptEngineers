import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from src.experiments.contracts import CandidateOutput

class TinyContext:

    def __init__(self):
        self.spec = None
        self.field_dimension = 5
        np.random.seed(0)
        self.train_x = np.random.randint(0, 4, size=(10, 2))
        self.train_y = np.random.randint(0, 2, size=10)
        self.valid_x = np.random.randint(0, 4, size=(10, 2))
        self.valid_y = np.random.randint(0, 2, size=10)
        self.test_x = np.random.randint(0, 4, size=(10, 2))
        self.random_valid_x = None

    def evaluate_validation(self, scores):
        return {'primary': 0.5, 'gauc': 0.4, 'ndcg': 0.6}

class TestCandidateRun(unittest.TestCase):

    def test_run_returns_valid_candidate_output(self):
        ctx = TinyContext()
        params = {'seed': 42, 'k': 16, 'learning_rate': 0.0005, 'epochs': 2, 'batch_size': 512, 'patience': 2, 'negatives_per_group': 4, 'temperature': 1.0}
        from candidate import run
        with patch('candidate.build_features', side_effect=lambda x, s: x.astype(np.int64)), patch('candidate.sample_softmax_groups') as mock_sampler:
            mock_sampler.return_value = (np.array([0, 1, 2, 3]), np.array([[4, 5, 6, 7], [8, 9, 0, 1], [2, 3, 4, 5], [6, 7, 8, 9]]))
            result = run(ctx, params)
        self.assertIsInstance(result, CandidateOutput)
        self.assertEqual(len(result.validation_scores), 10)
        self.assertIsNotNone(result.test_scores)
        self.assertEqual(len(result.test_scores), 10)
        self.assertIsNone(result.random_validation_scores)
        self.assertIsInstance(result.checkpoint_state, dict)
        self.assertTrue(all(np.isfinite(result.validation_scores)))
        self.assertTrue(all(np.isfinite(result.test_scores)))
        self.assertEqual(len(result.training_trace), 2)
if __name__ == '__main__':
    unittest.main()
