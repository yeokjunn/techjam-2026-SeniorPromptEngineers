import unittest
import numpy as np
import candidate

class TestCandidateContract(unittest.TestCase):

    def test_run_is_callable(self):
        self.assertTrue(callable(candidate.run))

    def test_sigmoid_stable_extremes(self):
        x = np.array([1000.0, -1000.0])
        y = candidate._sigmoid(x)
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertAlmostEqual(float(y[0]), 1.0, places=6)
        self.assertAlmostEqual(float(y[1]), 0.0, places=6)

    def test_eval_metrics_parses_tuple(self):
        scores = np.zeros(3)
        record = {}

        def ev(s):
            record['called'] = True
            return (0.73, 0.41)
        g, n = candidate._eval_metrics(ev, scores)
        self.assertTrue(record['called'])
        self.assertAlmostEqual(g, 0.73, places=6)
        self.assertAlmostEqual(n, 0.41, places=6)

    def test_eval_metrics_parses_dict(self):
        scores = np.zeros(3)
        record = {}

        def ev(s):
            record['called'] = True
            return {'GAUC': 0.73, 'nDCG@5': 0.41}
        g, n = candidate._eval_metrics(ev, scores)
        self.assertTrue(record['called'])
        self.assertAlmostEqual(g, 0.73, places=6)
        self.assertAlmostEqual(n, 0.41, places=6)

    def test_concat_preserves_order(self):
        rows = np.array([[0, 1], [2, 3]], dtype=np.int32)
        extra = np.array([[5], [6]], dtype=np.int32)
        out = candidate._concat(rows, extra)
        np.testing.assert_array_equal(out, np.array([[0, 1, 5], [2, 3, 6]], dtype=np.int32))
if __name__ == '__main__':
    unittest.main()
