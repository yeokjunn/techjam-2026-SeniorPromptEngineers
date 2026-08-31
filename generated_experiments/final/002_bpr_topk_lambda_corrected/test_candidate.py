import unittest
import numpy as np
import math
from unittest.mock import patch, MagicMock
from src.models.fm_core import FMRanker
from src.models.sampling import sample_bpr_pairs
from src.experiments.contracts import CandidateOutput
from candidate import run

class DummyContext:

    def __init__(self):
        self.field_dimension = 20
        self.train_users = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
        self.train_y = np.array([1, 0, 1, 1, 0, 0], dtype=np.float32)
        self.train_x = np.random.randint(0, self.field_dimension, size=(len(self.train_users), 1)).astype(np.int32)
        self.valid_x = np.random.randint(0, self.field_dimension, size=(4, 1)).astype(np.int32)
        self.valid_users = np.array([0, 1, 2, 0], dtype=np.int64)
        self.test_x = np.random.randint(0, self.field_dimension, size=(3, 1)).astype(np.int32)
        self.random_valid_x = np.random.randint(0, self.field_dimension, size=(5, 1)).astype(np.int32)

    def evaluate_validation(self, scores):
        return (0.5, 0.5, 0.5)

class TestCandidate(unittest.TestCase):

    def test_run_outputs(self):
        ctx = DummyContext()
        params = {'seed': 42, 'k': 16, 'learning_rate': 0.0005, 'epochs': 2, 'batch_size': 4, 'patience': 1, 'negatives_per_positive': 1, 'negatives_per_group': None, 'temperature': None}
        with patch('candidate.sample_bpr_pairs') as mock_sampler:
            pos = np.array([0, 2])
            neg = np.array([1, 4])
            mock_sampler.return_value = (pos, neg)
            result = run(ctx, params)
        self.assertIsInstance(result, CandidateOutput)
        self.assertEqual(len(result.validation_scores), len(ctx.valid_x))
        self.assertTrue(np.all(np.isfinite(result.validation_scores)))
        self.assertEqual(len(result.test_scores), len(ctx.test_x))
        self.assertTrue(np.all(np.isfinite(result.test_scores)))
        self.assertEqual(len(result.random_validation_scores), len(ctx.random_valid_x))
        self.assertTrue(np.all(np.isfinite(result.random_validation_scores)))
        self.assertIsInstance(result.checkpoint_state, dict)
        self.assertTrue(all((isinstance(v, np.ndarray) for v in result.checkpoint_state.values())))
        self.assertIsInstance(result.training_trace, list)
        self.assertTrue(all(('epoch' in d and 'loss' in d for d in result.training_trace)))
if __name__ == '__main__':
    unittest.main()
