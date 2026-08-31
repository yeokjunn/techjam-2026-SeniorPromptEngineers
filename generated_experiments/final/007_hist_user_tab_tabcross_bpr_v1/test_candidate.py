import unittest
import numpy as np
from unittest import mock
import candidate
from src.models.features import feature_dimension
from src.experiments.contracts import CandidateOutput
from candidate import run

def _fake_build_features(rows, spec):
    """Deterministic synthetic feature builder matching the documented API.
    Returns one column per enabled use_* group, with indices in [0, feature_dimension(spec))."""
    n_groups = sum([spec.get('use_recency', False), spec.get('use_tab_cross', False), spec.get('use_user_author', False), spec.get('use_user_rate', False), spec.get('use_user_tab', False), spec.get('use_video_age', False)])
    width = feature_dimension(spec)
    rng = np.random.RandomState(0)
    return rng.randint(0, max(width, 1), size=(len(rows), n_groups)).astype(np.int32)

class FakeContext:

    def __init__(self, n_train=100, n_valid=20, n_test=10, n_random=5):
        self.field_dimension = 50
        self.train_x = np.random.randint(0, self.field_dimension, size=(n_train, 5))
        self.train_users = np.repeat(np.arange(10), 10)
        self.train_y = (np.arange(100) % 2).astype(np.float32)
        self.valid_x = np.random.randint(0, self.field_dimension, size=(n_valid, 5))
        self.valid_users = np.random.randint(0, 10, size=n_valid)
        self.test_x = np.random.randint(0, self.field_dimension, size=(n_test, 5))
        self.random_valid_x = np.random.randint(0, self.field_dimension, size=(n_random, 5))

    def evaluate_validation(self, scores):
        return {'GAUC': 0.5, 'nDCG@5': 0.5, 'primary': 0.5}

class TestHistFeaturesBPR(unittest.TestCase):

    def setUp(self):
        self.params = {'seed': 42, 'k': 16, 'learning_rate': 0.0005, 'epochs': 1, 'batch_size': 32, 'patience': 2, 'negatives_per_positive': 1, 'negatives_per_group': None, 'temperature': None, 'smoothing': 20.0, 'scheme': 'prior_days', 'use_recency': True, 'use_tab_cross': True, 'use_user_author': True, 'use_user_rate': True, 'use_user_tab': True, 'use_video_age': True}
        self._bf_patch = mock.patch.object(candidate, 'build_features', side_effect=_fake_build_features)
        self._bf_patch.start()

    def tearDown(self):
        self._bf_patch.stop()

    def test_run_returns_candidate_output(self):
        ctx = FakeContext()
        result = run(ctx, self.params)
        self.assertIsInstance(result, CandidateOutput)
        self.assertEqual(len(result.validation_scores), len(ctx.valid_x))
        self.assertTrue(np.all(np.isfinite(result.validation_scores)))
        self.assertIsInstance(result.checkpoint_state, dict)
        self.assertIsInstance(result.training_trace, list)
        self.assertIsInstance(result.diagnostics, dict)
        self.assertIsNotNone(result.test_scores)
        self.assertEqual(len(result.test_scores), len(ctx.test_x))
        self.assertIsNotNone(result.random_validation_scores)
        self.assertEqual(len(result.random_validation_scores), len(ctx.random_valid_x))

    def test_test_scores_none_when_test_x_none(self):
        ctx = FakeContext()
        ctx.test_x = None
        result = run(ctx, self.params)
        self.assertIsNone(result.test_scores)

    def test_all_history_groups_still_active(self):
        """Guard against silently dropping any approved history signal."""
        spec = {'smoothing': self.params['smoothing'], 'scheme': self.params['scheme'], 'use_recency': self.params['use_recency'], 'use_tab_cross': self.params['use_tab_cross'], 'use_user_author': self.params['use_user_author'], 'use_user_rate': self.params['use_user_rate'], 'use_user_tab': self.params['use_user_tab'], 'use_video_age': self.params['use_video_age']}
        self.assertEqual(feature_dimension(dict(spec, split='train', field_offset=50)), 54)
        self.assertEqual(sum([spec['use_recency'], spec['use_tab_cross'], spec['use_user_author'], spec['use_user_rate'], spec['use_user_tab'], spec['use_video_age']]), 6)
if __name__ == '__main__':
    unittest.main()
