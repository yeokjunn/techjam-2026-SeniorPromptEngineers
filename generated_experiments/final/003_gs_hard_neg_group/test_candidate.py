import unittest
from unittest.mock import patch
import numpy as np
import candidate

class FakeContext:

    def __init__(self):
        n_train = 1000
        n_valid = 200
        self.field_dimension = 300
        self.train_x = np.random.randint(0, 100, size=(n_train, 2))
        self.train_y = (np.random.rand(n_train) > 0.5).astype(np.float32)
        self.train_users = self.train_x[:, 0].astype(np.int64)
        self.valid_x = np.random.randint(0, 100, size=(n_valid, 2))
        self.valid_users = self.valid_x[:, 0].astype(np.int64)
        self.test_x = np.random.randint(0, 100, size=(100, 2))
        self.random_valid_x = None

        def evaluate_validation(scores):
            return float(np.mean(scores))
        self.evaluate_validation = evaluate_validation

class TestCandidate(unittest.TestCase):

    def test_run_returns_valid_output(self):
        ctx = FakeContext()
        params = {'seed': 42, 'learning_rate': 0.0005, 'epochs': 2, 'batch_size': 64, 'negatives_per_group': 4, 'temperature': 1.0, 'patience': 2}

        def fake_build_features(data, spec):
            return np.zeros((len(data), 2), dtype=np.int32)

        def fake_sample_softmax_groups(users, labels, rng, K):
            uniq = np.unique(users)
            pos_idx = []
            neg_idx = []
            for u in uniq:
                idx_u = np.where(users == u)[0]
                pos_u = idx_u[labels[idx_u] > 0.5]
                neg_u = idx_u[labels[idx_u] <= 0.5]
                if len(pos_u) > 0 and len(neg_u) >= K:
                    for p in pos_u:
                        negs = rng.choice(neg_u, size=K, replace=False)
                        pos_idx.append(p)
                        neg_idx.append(negs)
            if len(pos_idx) == 0:
                return (np.array([], dtype=np.int64), np.array([], dtype=np.int64).reshape(0, K))
            return (np.array(pos_idx, dtype=np.int64), np.array(neg_idx, dtype=np.int64))
        with patch('candidate.build_features', side_effect=fake_build_features), patch('candidate.sample_softmax_groups', side_effect=fake_sample_softmax_groups):
            out = candidate.run(ctx, params)
        self.assertEqual(len(out.validation_scores), len(ctx.valid_x))
        self.assertTrue(np.all(np.isfinite(out.validation_scores)))
        self.assertIsNotNone(out.checkpoint_state)
        self.assertIsNotNone(out.test_scores)
        self.assertTrue(np.all(np.isfinite(out.test_scores)))
if __name__ == '__main__':
    unittest.main()
