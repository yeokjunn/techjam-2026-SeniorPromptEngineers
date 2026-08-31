import unittest
import numpy as np
import candidate

class TestGroupSoftmaxHelpers(unittest.TestCase):

    def test_gradients_sum_to_zero(self):
        logits = np.array([[1.0, 0.5, -0.2]], dtype=np.float32)
        grads, losses = candidate._group_softmax(logits, 2.0)
        self.assertTrue(np.all(np.isfinite(grads)))
        self.assertAlmostEqual(float(np.sum(grads[0])), 0.0, places=6)

    def test_loss_positive(self):
        logits = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        grads, losses = candidate._group_softmax(logits, 1.0)
        expected = float(np.log(3.0))
        self.assertAlmostEqual(float(losses[0]), expected, places=5)
        self.assertTrue(np.all(np.isfinite(grads)))

    def test_primary_from_metrics(self):
        metrics = {'GAUC': 0.6, 'nDCG@5': 0.4}
        self.assertAlmostEqual(candidate._primary(metrics), 0.5)

    def test_primary_scalar(self):
        self.assertAlmostEqual(candidate._primary(0.7), 0.7)
