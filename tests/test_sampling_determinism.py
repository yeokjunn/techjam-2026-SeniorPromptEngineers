"""Sampling must not depend on PYTHONHASHSEED (review I-determinism).

``eligible_user_indices`` used to iterate ``positives.keys() & negatives.keys()``
-- a set of ``str`` user ids whose iteration order Python permutes per process
when ``PYTHONHASHSEED`` is unset.  ``rng.choice`` then consumed randomness in a
different order, so byte-identical candidates at seed 0 scored 0.601954 /
0.603570 / 0.602029.  These tests pin both halves of the fix: the sorted
iteration order in ``src/models/sampling.py`` and the explicit
``PYTHONHASHSEED`` in the candidate subprocess environment.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

from src.agent.candidate_runner import CandidateExecutor, CandidateWorkspace
from src.models.sampling import sample_bpr_pairs, sample_softmax_groups


REPO_ROOT = Path(__file__).resolve().parents[1]

# Many distinct string ids: with hash randomisation on, the set intersection in
# eligible_user_indices lands in a different order for essentially every seed.
DRIVER = textwrap.dedent(
    """
    import importlib.util
    import json
    import sys

    import numpy as np

    spec = importlib.util.spec_from_file_location("sampling_under_test", sys.argv[1])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    users = []
    labels = []
    for index in range(40):
        user = f"user-{index:03d}-{index * 7919}"
        users.extend([user] * 4)
        labels.extend([1.0, 0.0, 1.0, 0.0])
    labels = np.asarray(labels, dtype=np.float32)

    positives, negatives = module.sample_bpr_pairs(
        users, labels, np.random.default_rng(0), 2
    )
    group_positives, group_negatives = module.sample_softmax_groups(
        users, labels, np.random.default_rng(0), 4
    )
    print(
        json.dumps(
            {
                "bpr": [positives.tolist(), negatives.tolist()],
                "softmax": [group_positives.tolist(), group_negatives.tolist()],
            }
        )
    )
    """
)


def _run_driver(sampling_path: Path, hash_seed: str, work_dir: Path) -> str:
    driver_path = work_dir / "driver.py"
    driver_path.write_text(DRIVER, encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = hash_seed
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(driver_path), str(sampling_path)],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


class SamplingDeterminismTests(unittest.TestCase):
    def setUp(self):
        self.users = [f"user-{index:03d}-{index * 7919}" for index in range(40)]
        self.users = [user for user in self.users for _ in range(4)]
        self.labels = np.asarray([1.0, 0.0, 1.0, 0.0] * 40, dtype=np.float32)

    def test_same_seed_reproduces_identical_bpr_pairs(self):
        first = sample_bpr_pairs(self.users, self.labels, np.random.default_rng(0), 2)
        second = sample_bpr_pairs(self.users, self.labels, np.random.default_rng(0), 2)
        self.assertGreater(len(first[0]), 0)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_same_seed_reproduces_identical_softmax_groups(self):
        first = sample_softmax_groups(
            self.users, self.labels, np.random.default_rng(0), 4
        )
        second = sample_softmax_groups(
            self.users, self.labels, np.random.default_rng(0), 4
        )
        self.assertGreater(len(first[0]), 0)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_sampling_is_identical_across_hash_seeds_in_subprocesses(self):
        """The strong one: two processes, two PYTHONHASHSEED values, one output.

        This is what actually failed before the fix -- in-process repetition
        cannot see it, because a single process has a single hash seed.
        """
        sampling_path = REPO_ROOT / "src" / "models" / "sampling.py"
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            first = _run_driver(sampling_path, "1", work_dir)
            second = _run_driver(sampling_path, "2", work_dir)
        self.assertTrue(json.loads(first)["bpr"][0])
        self.assertEqual(first, second)

    def test_pre_fix_sampling_did_vary_across_hash_seeds(self):
        """RED evidence, kept executable: the committed pre-fix module differs.

        ``git show`` the version of sampling.py that iterated the raw set, run
        the same driver against it under two hash seeds, and assert the outputs
        disagree.  If this ever starts passing-as-equal the regression guard
        above has stopped proving anything.

        The ``5bce4bc`` object can vanish once this wave is squashed or rebased,
        and the ``skipTest`` below is the deliberate response: this test is
        *evidence* that the fix was needed, not a guard on current behaviour, so
        degrading to a skip is acceptable — the guard above is what must stay
        green.
        """
        completed = subprocess.run(
            ["git", "show", "5bce4bc:src/models/sampling.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or "positives.keys() & negatives.keys()" not in completed.stdout:
            self.skipTest("pre-fix sampling.py revision is not reachable from here")
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            legacy_path = work_dir / "legacy_sampling.py"
            legacy_path.write_text(completed.stdout, encoding="utf-8")
            first = _run_driver(legacy_path, "1", work_dir)
            second = _run_driver(legacy_path, "2", work_dir)
        self.assertNotEqual(
            first, second, "pre-fix sampling was expected to vary with PYTHONHASHSEED"
        )


class CandidateHashSeedTests(unittest.TestCase):
    def test_candidate_environment_pins_pythonhashseed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = CandidateWorkspace(root / "generated", "run0001", 1, "cand01")
            executor = CandidateExecutor(
                repo_root=REPO_ROOT,
                data_dir=root / "data_dir",
                experiment_timeout_seconds=60,
                test_timeout_seconds=60,
            )
            environment = executor._environment(workspace)
        self.assertEqual(environment["PYTHONHASHSEED"], "0")


if __name__ == "__main__":
    unittest.main()
