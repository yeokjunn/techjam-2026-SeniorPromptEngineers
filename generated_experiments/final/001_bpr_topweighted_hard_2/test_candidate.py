import unittest
import numpy as np
from unittest.mock import patch
from src.experiments.contracts import CandidateOutput
import candidate

class TestCandidate(unittest.TestCase):

    @patch('candidate.feature_dimension')
    @patch('candidate.build_features')
    def test_run_returns_valid_outputs(self, mock_build, mock_feat_dim):
        """Verify run returns a CandidateOutput with finite scores."""
        mock_feat_dim.return_value = 0
        mock_build.side_effect = lambda x, spec: x
        n_train = 200
        n_valid = 20
        n_fields = 3
        field_dim = 20
        rng = np.random.RandomState(0)
        train_x = rng.randint(0, field_dim, size=(n_train, n_fields))
        valid_x = rng.randint(0, field_dim, size=(n_valid, n_fields))
        n_users = 10
        users = np.repeat(np.arange(n_users), n_train // n_users)
        labels = np.zeros(n_train, dtype=np.int64)
        for u in range(n_users):
            mask = users == u
            idx = np.where(mask)[0]
            labels[idx[::2]] = 1

        def evaluate_validation(scores):
            return float(np.mean(scores))

        class Context:
            pass
        ctx = Context()
        ctx.train_x = train_x
        ctx.train_y = labels
        ctx.train_users = users
        ctx.valid_x = valid_x
        ctx.valid_users = np.arange(n_valid) % n_users
        ctx.field_dimension = field_dim
        ctx.evaluate_validation = evaluate_validation
        ctx.test_x = None
        ctx.random_valid_x = None
        params = {'batch_size': 32, 'epochs': 1, 'k': 16, 'learning_rate': 0.0005, 'negatives_per_positive': 1, 'patience': 1, 'seed': 42}
        result = candidate.run(ctx, params)
        self.assertIsInstance(result, CandidateOutput)
        self.assertEqual(len(result.validation_scores), n_valid)
        self.assertTrue(np.all(np.isfinite(result.validation_scores)))
        self.assertIsNone(result.test_scores)
        self.assertIsNone(result.random_validation_scores)
        self.assertIsInstance(result.checkpoint_state, dict)
        self.assertIn('V', result.checkpoint_state)
        self.assertIn('W', result.checkpoint_state)
        self.assertIn('b', result.checkpoint_state)
        self.assertTrue(len(result.training_trace) > 0)
if __name__ == '__main__':
    unittest.main()
