import unittest
import numpy as np
from src.models.fm_core import FMRanker
from src.models.sampling import sample_bpr_pairs

class TestBPR(unittest.TestCase):
    def test_sample_bpr_pairs_same_user_and_labels(self):
        users = ['u1', 'u1', 'u1', 'u2', 'u2', 'u3', 'u3', 'u3', 'u3']
        labels = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1])
        rng = np.random.default_rng(0)
        pos, neg = sample_bpr_pairs(users, labels, rng, negatives_per_positive=2)
        self.assertEqual(len(pos), len(neg))
        for p, n in zip(pos, neg):
            self.assertEqual(users[p], users[n])
            self.assertEqual(labels[p], 1)
            self.assertEqual(labels[n], 0)

    def test_bpr_gradient_updates_model(self):
        rng = np.random.default_rng(0)
        model = FMRanker(dimension=100, embedding_dim=2, learning_rate=0.1, l2=0.0, seed=0)
        x = rng.integers(0, 100, size=(4, 5))
        s0 = model.predict(x)
        pos_x = x[:2]
        neg_x = x[2:]
        pos_scores = model.logits(pos_x)[0]
        neg_scores = model.logits(neg_x)[0]
        diff = pos_scores - neg_scores
        grad = 1.0 / (1.0 + np.exp(-diff)) - 1.0
        grad = grad / 2
        gv_p, gw_p, gb_p = model.gradients(pos_x, grad)
        gv_n, gw_n, gb_n = model.gradients(neg_x, -grad)
        model.apply_gradients(gv_p + gv_n, gw_p + gw_n, gb_p + gb_n)
        s1 = model.predict(x)
        self.assertFalse(np.allclose(s0, s1))

if __name__ == '__main__':
    unittest.main()
