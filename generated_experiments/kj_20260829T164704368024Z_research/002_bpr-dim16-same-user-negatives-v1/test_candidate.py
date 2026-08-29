import unittest
import numpy as np
from src.experiments.contracts import CandidateOutput
from src.models.sampling import sample_bpr_pairs
import candidate


class FakeContext:
    def __init__(self, with_test):
        rng = np.random.default_rng(0)
        self.field_dimension = 50
        self.train_x = rng.integers(0, 50, size=(200, 5))
        self.train_y = rng.integers(0, 2, size=200).astype(np.float64)
        self.train_users = rng.integers(0, 20, size=200)
        self.valid_x = rng.integers(0, 50, size=(80, 5))
        self.valid_users = rng.integers(0, 20, size=80)
        self.valid_y = rng.integers(0, 2, size=80).astype(np.float64)
        self.test_x = rng.integers(0, 50, size=(60, 5)) if with_test else None

    def evaluate_validation(self, scores):
        scores = np.asarray(scores, dtype=np.float64)
        self.last_shape = scores.shape
        return {"gauc": 0.5, "ndcg@5": 0.6, "primary": 0.55}


PARAMS = {"batch_size": 2048, "epochs": 2, "k": 16, "learning_rate": 0.0005,
          "negatives_per_positive": 1, "patience": 1, "seed": 0}


class TestBprPairs(unittest.TestCase):
    def test_pairs_are_same_user_and_label_aligned(self):
        users = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        labels = np.array([1, 0, 1, 0, 1, 1, 0, 0], dtype=np.float64)
        rng = np.random.default_rng(7)
        pos, neg = sample_bpr_pairs(users, labels, rng, negatives_per_positive=1)
        self.assertEqual(pos.shape, neg.shape)
        self.assertTrue(np.all(users[pos] == users[neg]))
        self.assertTrue(np.all(labels[pos] == 1))
        self.assertTrue(np.all(labels[neg] == 0))


class TestRun(unittest.TestCase):
    def test_run_basic_contract(self):
        context = FakeContext(with_test=False)
        out = candidate.run(context, PARAMS)
        self.assertIsInstance(out, CandidateOutput)
        scores = np.asarray(out.validation_scores, dtype=np.float64)
        self.assertEqual(scores.shape[0], 80)
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertGreater(len(out.training_trace), 0)
        self.assertIsNone(out.test_scores)
        for value in out.checkpoint_state.values():
            self.assertIsInstance(value, np.ndarray)
        self.assertIn("best_primary", out.diagnostics)

    def test_run_with_test_scores(self):
        context = FakeContext(with_test=True)
        out = candidate.run(context, PARAMS)
        test_scores = np.asarray(out.test_scores, dtype=np.float64)
        self.assertEqual(test_scores.shape[0], 60)
        self.assertTrue(np.all(np.isfinite(test_scores)))

    def test_run_two_negatives(self):
        context = FakeContext(with_test=False)
        params = dict(PARAMS)
        params["negatives_per_positive"] = 2
        out = candidate.run(context, params)
        scores = np.asarray(out.validation_scores, dtype=np.float64)
        self.assertEqual(scores.shape[0], 80)
        self.assertTrue(np.all(np.isfinite(scores)))
