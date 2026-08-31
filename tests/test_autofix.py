import unittest
import json
import tempfile
from pathlib import Path

from src.agent.autofix import fix_candidate_source, fix_test_source
from src.agent.candidate_runner import CandidateWorkspace
from src.agent.safety import validate_source
from src.agent.types import CandidateManifest


class AutofixTests(unittest.TestCase):
    def test_all_candidate_repairs_are_safe_and_idempotent(self):
        source = '''import unittest
from types import SimpleNamespace
from src.models.sampling import sample_bpr_pairs
def run(context):
    value = getattr(context, "test_x", None)
    present = hasattr(context, "test_x")
    dimension = context.field_dimension
    primary = float(context.evaluate_validation(value))
    sample_bpr_pairs([], [], None, 1)
    return present, dimension, primary
'''
        fixed = fix_candidate_source(source)
        validate_source(fixed)
        self.assertNotIn("import unittest", fixed)
        self.assertNotIn("from types", fixed)
        self.assertNotIn("getattr(", fixed)
        self.assertNotIn("hasattr(", fixed)
        self.assertIn("_field_dim", fixed)
        self.assertEqual(fixed, fix_candidate_source(fixed))

    def test_test_imports_keep_unittest(self):
        fixed = fix_test_source("import unittest\nfrom types import SimpleNamespace\n")
        self.assertIn("import unittest", fixed)
        self.assertNotIn("from types", fixed)
        validate_source(fixed, test_file=True)

    def test_validation_metrics_tuple_unpack_is_repaired(self):
        source = """def run(context):
    primary, gauc, ndcg = context.evaluate_validation([0.1])
    return primary, gauc, ndcg
"""
        fixed = fix_candidate_source(source)
        self.assertIn("_autofix_metrics_tuple", fixed)
        namespace = {}
        exec(fixed, namespace)

        class Context:
            def evaluate_validation(self, scores):
                return {"primary": 0.6, "GAUC": 0.7, "nDCG@5": 0.5}

        self.assertEqual(namespace["run"](Context()), (0.6, 0.7, 0.5))
    def test_fm_ranker_shape_dimension_is_repaired(self):
        source = """from src.models.fm_core import FMRanker
def run(context):
    model = FMRanker(dimension=context.train_x.shape[1], embedding_dim=16)
    return model
"""
        fixed = fix_candidate_source(source)
        self.assertNotIn("context.train_x.shape[1]", fixed)
        self.assertIn("_field_dim", fixed)
        self.assertEqual(fixed, fix_candidate_source(fixed))

    def test_embedded_unittest_class_is_stripped_from_candidate(self):
        source = """import unittest
from src.experiments.contracts import CandidateOutput
class TestCandidate(unittest.TestCase):
    def test_something(self):
        pass
def run(context):
    return CandidateOutput([0.1], {}, [], {})
if __name__ == "__main__":
    unittest.main()
"""
        fixed = fix_candidate_source(source)
        self.assertNotIn("TestCandidate", fixed)
        self.assertNotIn("unittest.main()", fixed)
        validate_source(fixed)
        self.assertEqual(fixed, fix_candidate_source(fixed))


if __name__ == "__main__":
    unittest.main()
