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
        self.assertEqual(fixed, fix_candidate_source(fixed))

    def test_workspace_manifest_records_repaired_source(self):
        source = "from src.models.sampling import sample_bpr_pairs\ndef run(context):\n    sample_bpr_pairs([], [], None, 1)\n    return getattr(context, 'x', None)\n"
        manifest = CandidateManifest("candidate", "hypothesis", "bpr", source, "import unittest\n", {})
        with tempfile.TemporaryDirectory() as directory:
            workspace = CandidateWorkspace(Path(directory), "run", 1, manifest.candidate_id)
            workspace.write(manifest)
            saved = json.loads((workspace.directory / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["code"], workspace.code_path.read_text(encoding="utf-8"))
            self.assertNotIn("getattr(", saved["code"])


if __name__ == "__main__":
    unittest.main()
