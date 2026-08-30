import unittest
import numpy as np
from candidate import run
from src.models.sampling import sample_bpr_pairs

class SampleBprTest(unittest.TestCase):
    def test_same_user_pairs(self):
        users = np.array([0,0,0,1,1,1])
        labels = np.array([1,0,1,0,1,0])
        rng = np.random.default_rng(0)
        pos, neg = sample_bpr_pairs(users, labels, rng, negatives_per_positive=1)
        self.assertEqual(len(pos), len(neg))
        for p, n in zip(pos, neg):
            self.assertEqual(users[p], users[n])
            self.assertEqual(labels[p], 1)
            self.assertEqual(labels[n], 0)

class RunTest(unittest.TestCase):
    def test_run_returns_candidate(self):
        field_dim = 100
        class Ctx:
            def __init__(self):
                self.train_x = np.random.randint(0, field_dim, size=(10, 5)).astype(np.int64)
                self.train_y = np.array([1,0,1,0,1,0,1,0,1,0])
                self.train_users = np.array([0,0,0,0,0,1,1,1,1,1])
                self.valid_x = np.random.randint(0, field_dim, size=(4, 5)).astype(np.int64)
                self.valid_users = np.array([0,0,1,1])
                self.field_dimension = field_dim
                self.test_x = None
            def evaluate_validation(self, scores):
                return 0.5
        ctx = Ctx()
        params = {
            'seed': 42,
            'k': 16,
            'learning_rate': 0.0005,
            'batch_size': 2048,
            'epochs': 20,
            'negatives_per_positive': 1,
            'patience': 5
        }
        output = run(ctx, params)
        self.assertEqual(len(output.validation_scores), len(ctx.valid_x))
        self.assertIsNone(output.test_scores)
        self.assertIsNotNone(output.checkpoint_state)
        self.assertIsInstance(output.training_trace, list)

if __name__ == '__main__':
    unittest.main()