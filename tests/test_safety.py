from __future__ import annotations

import unittest
import tempfile

from src.agent.candidate_runner import CandidateExecutor, CandidateWorkspace
from unittest.mock import patch

from src.agent.families import (
    FAMILIES,
    Family,
    builder_brief,
    coverage_families,
    family_names,
)
from src.agent.runtime_contracts import runtime_contract_prompt
from src.agent.policy import sanitize_parameters
from src.agent.safety import (
    ALLOWED_IMPORTS,
    FORBIDDEN_CALLS,
    SAFE_BUILTIN_NAMES,
    SafetyViolation,
    contained_path,
    is_allowed_import,
    restricted_builtins,
    validate_family_contract,
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


class FamilyContractTests(unittest.TestCase):
    BPR_SOURCE = """
import numpy as np
from src.models.sampling import sample_bpr_pairs


def run(context, parameters):
    sample_bpr_pairs(context.train_users, context.train_y, np.random.default_rng(0), 1)
"""

    def test_family_contract_reads_the_registry(self):
        validate_family_contract(self.BPR_SOURCE, "bpr")
        with self.assertRaises(SafetyViolation):
            validate_family_contract("def run(context, parameters):\n    return None\n", "bpr")
        with self.assertRaises(SafetyViolation):
            validate_family_contract(self.BPR_SOURCE, "listwise_magic")

    def test_one_of_group_accepts_either_member(self):
        entry = Family(
            name="probe",
            method_card="research/methods/bpr.md",
            trusted_sampler="sample_bpr_pairs",
            required_calls=(("sample_bpr_pairs", "sample_softmax_groups"), ("build_features",)),
        )
        with patch.dict(FAMILIES, {"probe": entry}):
            both = "sample_softmax_groups(a, b, c)\nbuild_features(rows, spec)\n"
            validate_family_contract(both, "probe")
            with self.assertRaises(SafetyViolation) as raised:
                validate_family_contract("sample_bpr_pairs(a, b, c)\n", "probe")
            self.assertIn("build_features()", str(raised.exception))

    def test_registry_defaults_reproduce_todays_sanitiser(self):
        """The promise to A: pointing sanitize_parameters at the registry changes no behaviour."""
        for name in ("bpr", "group_softmax"):
            with self.subTest(family=name):
                self.assertEqual(sanitize_parameters(name, {}), FAMILIES[name].defaults)

    def test_every_grid_value_is_accepted_by_todays_sanitiser(self):
        """Each grid value must be reachable, holding the other parameters at their defaults.

        One exception is deliberate rather than a gap. `multi_task` requires at least one
        auxiliary head, and its defaults enable exactly one (`use_is_click`), so switching that
        single head off empties the set and the sanitiser rightly rejects it. The grid entry
        (True, False) and the "at least one head" rule are both correct; they simply interact.
        Where a toggle would empty the set, another head is enabled so the value under test is
        still exercised rather than skipped.
        """
        for name, entry in FAMILIES.items():
            toggles = [key for key in entry.grid if key.startswith("use_")]
            for key, allowed in entry.grid.items():
                values = (0, 999) if key == "seed" else tuple(allowed)
                for value in values:
                    with self.subTest(family=name, parameter=key, value=value):
                        proposed = {**entry.defaults, key: value}
                        if (
                            key in toggles
                            and value is False
                            and not any(proposed.get(other) for other in toggles)
                        ):
                            keep_on = next(other for other in toggles if other != key)
                            proposed[keep_on] = True
                        parameters = sanitize_parameters(name, proposed)
                        self.assertEqual(parameters[key], value)

    def test_multi_task_rejects_disabling_every_auxiliary_head(self):
        """The constraint the test above works around must actually be enforced."""
        entry = FAMILIES["multi_task"]
        toggles = [key for key in entry.grid if key.startswith("use_")]
        all_off = {**entry.defaults, **{key: False for key in toggles}}
        with self.assertRaises(ValueError) as raised:
            sanitize_parameters("multi_task", all_off)
        self.assertIn("auxiliary target head", str(raised.exception))

    def test_coverage_families_is_the_minimum_set_not_every_family(self):
        self.assertEqual(
            coverage_families(),
            frozenset({"bpr", "group_softmax", "history_features", "multi_task"}),
        )
        self.assertTrue(coverage_families().issubset(family_names()))

    def test_family_entries_stay_hashable_despite_grid_dicts(self):
        self.assertEqual(len({FAMILIES["bpr"], FAMILIES["bpr"]}), 1)
        self.assertEqual(len(set(FAMILIES.values())), len(FAMILIES))

    def test_builder_brief_names_the_mandatory_calls_and_the_grid(self):
        brief = builder_brief("bpr")
        self.assertIn(
            "src.models.sampling.sample_bpr_pairs(users, labels, rng, negatives_per_positive)",
            brief,
        )
        self.assertIn("learning_rate: 0.0003, 0.0005, 0.001", brief)
        self.assertIn("seed: 0-999", brief)
        self.assertIn("negatives_per_positive: 1, 2", brief)

    def test_feature_runtime_contracts_forbid_raw_data_access(self):
        for family in ("history_features", "multi_task"):
            with self.subTest(family=family):
                prompt = runtime_contract_prompt(family)
                self.assertIn("read raw CSVs", prompt)
                self.assertIn("dataset files directly", prompt)


if __name__ == "__main__":
    unittest.main()
