from __future__ import annotations

import unittest
import tempfile

from src.agent.candidate_runner import CandidateExecutor, CandidateWorkspace
from src.agent.safety import (
    ALLOWED_IMPORTS,
    FORBIDDEN_CALLS,
    SAFE_BUILTIN_NAMES,
    SafetyViolation,
    contained_path,
    is_allowed_import,
    restricted_builtins,
    validate_source,
)
from src.agent.types import CandidateManifest
from pathlib import Path


class SafetyTests(unittest.TestCase):
    def test_safe_candidate_is_accepted(self):
        validate_source(
            "import numpy as np\n"
            "from src.experiments.contracts import CandidateOutput\n"
            "def run(context, parameters):\n"
            "    return CandidateOutput(np.zeros(len(context.valid_x)), {}, [], {})\n"
        )

    def test_judge_reference_is_rejected(self):
        with self.assertRaises(SafetyViolation):
            validate_source("SECRET = 'data/judge/test_truth.jsonl'\n")

    def test_filesystem_and_process_imports_are_rejected(self):
        for source in ("import os\n", "import subprocess\n", "open('x')\n"):
            with self.subTest(source=source), self.assertRaises(SafetyViolation):
                validate_source(source)

    def test_evaluator_import_is_rejected(self):
        with self.assertRaises(SafetyViolation):
            validate_source("from src.evaluation.official import official_evaluate\n")

    def test_path_traversal_is_rejected(self):
        with self.assertRaises(SafetyViolation):
            contained_path(Path("generated"), "..", "outside.py")

    def test_safe_generated_unit_test_runs_in_isolated_workspace(self):
        code = (
            "import numpy as np\n"
            "from src.experiments.contracts import CandidateOutput\n"
            "from src.models.sampling import sample_bpr_pairs\n"
            "def run(context, parameters):\n"
            "    sample_bpr_pairs(context.train_users, context.train_y, np.random.default_rng(0), 1)\n"
            "    return CandidateOutput(np.zeros(len(context.valid_x)), {}, [], {})\n"
        )
        tests = (
            "import unittest\n"
            "import candidate\n"
            "class ContractTests(unittest.TestCase):\n"
            "    def test_callable(self):\n"
            "        self.assertTrue(callable(candidate.run))\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = CandidateWorkspace(root / "generated", "run", 1, "candidate")
            workspace.write(
                CandidateManifest("candidate", "h", "bpr", code, tests, {})
            )
            executor = CandidateExecutor(
                Path(__file__).resolve().parents[1], root, 10, 10
            )
            passed, output = executor.test(workspace)
            self.assertTrue(passed, output)

    def test_builtins_open_subscript_is_rejected(self):
        with self.assertRaises(SafetyViolation):
            validate_source("__builtins__['open']('x.npz')")

    def test_builtins_import_subscript_is_rejected(self):
        with self.assertRaises(SafetyViolation):
            validate_source("__builtins__['__import__']('os')")

    def test_bare_dunder_name_is_rejected(self):
        with self.assertRaises(SafetyViolation):
            validate_source("leaked = __builtins__")

    def test_aliased_forbidden_attribute_is_rejected(self):
        source = """
import numpy as np
f = np.load
f('x.npz')
"""
        with self.assertRaises(SafetyViolation):
            validate_source(source)

    def test_literal_dataset_path_is_rejected(self):
        for source in (
            "P = 'data/KuaiRand-Pure/data/log_standard_4_22_to_5_08_pure.csv'",
            "P = 'log_random_4_22_to_5_08_pure'",
            "P = 'KuaiRand-Pure'",
            "P = '/data/'",
        ):
            with self.subTest(source=source), self.assertRaises(SafetyViolation):
                validate_source(source)

    def test_relative_import_is_rejected(self):
        with self.assertRaises(SafetyViolation):
            validate_source("from . import candidate", test_file=True)

    def test_main_guard_and_string_methods_are_accepted(self):
        source = """
from __future__ import annotations

import unittest

import numpy.random


def normalise(name):
    return name.replace('_', '-')


if __name__ == '__main__':
    unittest.main()
"""
        validate_source(source, test_file=True)


class RestrictedBuiltinsTests(unittest.TestCase):
    def namespace(self, *, test_file=False):
        """Mirror the namespace run_candidate.py builds via module_from_spec."""
        return {
            "__builtins__": restricted_builtins(test_file=test_file),
            "__name__": "generated_candidate",
        }

    def test_restricted_builtins_block_open_and_import(self):
        namespace = self.namespace()
        with self.assertRaises(NameError):
            exec("open('x')", namespace)
        with self.assertRaises(SafetyViolation):
            exec("import os", namespace)
        with self.assertRaises(SafetyViolation):
            exec("__import__('os')", namespace)

    def test_guarded_import_rejects_relative_imports(self):
        with self.assertRaises(SafetyViolation):
            exec("from . import candidate", self.namespace(test_file=True))

    def test_guarded_import_allows_numpy_and_project_modules(self):
        namespace = self.namespace()
        exec("import numpy as np", namespace)
        exec("import numpy.random", namespace)
        exec("from src.models.sampling import sample_bpr_pairs", namespace)
        exec("from src.experiments.contracts import CandidateOutput", namespace)

    def test_test_only_imports_need_the_test_file_flag(self):
        exec("import unittest", self.namespace(test_file=True))
        with self.assertRaises(SafetyViolation):
            exec("import unittest", self.namespace())

    def test_restricted_builtins_support_class_definitions(self):
        namespace = self.namespace()
        source = """
class Trainer:
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs

    def total(self):
        return sum(range(self.epochs))


result = Trainer(4).total()
"""
        exec(source, namespace)
        self.assertEqual(namespace["result"], 6)

    def test_safe_builtins_exclude_every_forbidden_call(self):
        self.assertEqual(SAFE_BUILTIN_NAMES & FORBIDDEN_CALLS, set())
        self.assertNotIn("open", restricted_builtins())

    def test_static_and_dynamic_import_rules_agree(self):
        """The guarded __import__ and validate_source share is_allowed_import by construction."""
        for name in ("os", "subprocess", "pathlib", "src.evaluation.official"):
            with self.subTest(module=name):
                self.assertFalse(is_allowed_import(name, ALLOWED_IMPORTS))
        for name in ("numpy", "numpy.random", "src.models.sampling", "__future__"):
            with self.subTest(module=name):
                self.assertTrue(is_allowed_import(name, ALLOWED_IMPORTS))


if __name__ == "__main__":
    unittest.main()
