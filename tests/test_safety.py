from __future__ import annotations

import unittest
import tempfile

from src.agent.candidate_runner import CandidateExecutor, CandidateWorkspace
from src.agent.safety import SafetyViolation, contained_path, validate_source
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


if __name__ == "__main__":
    unittest.main()
