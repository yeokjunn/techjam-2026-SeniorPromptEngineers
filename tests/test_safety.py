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


if __name__ == "__main__":
    unittest.main()
