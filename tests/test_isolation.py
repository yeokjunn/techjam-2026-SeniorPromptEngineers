from __future__ import annotations

import ast
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent.candidate_runner import CandidateExecutor, CandidateWorkspace
from src.agent.types import CandidateManifest
from src.evaluation.official import (
    LABEL_PLACEHOLDER,
    TEST_ROWS,
    classify_primary,
    load_test_meta,
    load_train_valid,
    starter_modules,
    within_baseline_tolerance,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "KuaiRand-Pure" / "data"
REAL_DATA = (DATA_DIR / "log_standard_4_22_to_5_08_pure.csv").is_file()

LOG_HEADER = (
    "user_id,video_id,date,hourmin,time_ms,"
    "is_click,is_like,is_follow,is_comment,is_forward,is_hate,long_view,"
    "play_time_ms,duration_ms,profile_stay_time,comment_stay_time,is_profile_enter,"
    "is_rand,tab"
)


def write_synthetic_dir(root: Path) -> Path:
    """Tiny data dir whose test-window label column is the poison string 'LEAK'.

    If load_test_meta ever reads that column the placeholder contract breaks:
    'LEAK' != '0' would coerce to 1 instead of LABEL_PLACEHOLDER.
    """
    data_dir = root / "synthetic_data"
    data_dir.mkdir(parents=True)
    (data_dir / "video_features_basic_pure.csv").write_text(
        "video_id,author_id\n1,auth1\n2,auth2\n", encoding="utf-8", newline=""
    )
    # Both standard logs exist; the loader must read them in this fixed order.
    (data_dir / "log_standard_4_08_to_4_21_pure.csv").write_text(
        LOG_HEADER + "\n"
        # train window — must not appear in the test split
        "u1,1,20220410,1800,0,0,0,0,0,0,0,LEAK,863,30066,0,0,0,0,1\n"
        # test window — one row from the early file
        "u1,1,20220429,1800,0,0,0,0,0,0,0,LEAK,863,30066,0,0,0,0,1\n",
        encoding="utf-8",
        newline="",
    )
    (data_dir / "log_standard_4_22_to_5_08_pure.csv").write_text(
        LOG_HEADER + "\n"
        # valid window — must not appear in the test split
        "u3,1,20220425,1800,0,0,0,0,0,0,0,LEAK,863,30066,0,0,0,0,1\n"
        # test window — two rows from the late file, one with an unknown video
        "u2,2,20220508,1800,0,0,0,0,0,0,0,LEAK,863,30066,0,0,0,0,1\n"
        "u2,999,20220501,1800,0,0,0,0,0,0,0,LEAK,863,30066,0,0,0,0,1\n",
        encoding="utf-8",
        newline="",
    )
    return data_dir


class TestSplitLoaderTests(unittest.TestCase):
    def test_synthetic_poisoned_label_is_never_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = write_synthetic_dir(Path(tmp))
            split = load_test_meta(data_dir)
        self.assertEqual(len(split.rows), 3)
        self.assertTrue(all(row[6] == LABEL_PLACEHOLDER for row in split.rows))
        # Only test-window rows, in file order (early log first).
        self.assertEqual(
            [row[:5] for row in split.rows],
            [
                (20220429, "u1", "1", "auth1", "1"),
                (20220508, "u2", "2", "auth2", "1"),
                (20220501, "u2", "999", "UNK", "1"),
            ],
        )
        self.assertEqual(
            split.meta,
            ((0, "u1", "1"), (1, "u2", "2"), (2, "u2", "999")),
        )

    def test_expected_rows_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = write_synthetic_dir(Path(tmp))
            with self.assertRaises(ValueError):
                load_test_meta(data_dir, expected_rows=2)
            load_test_meta(data_dir, expected_rows=3)  # exact match is fine

    def test_test_loader_source_never_names_the_label_column(self):
        self.assertNotIn("long_view", inspect.getsource(load_test_meta))

    @unittest.skipUnless(REAL_DATA, "KuaiRand-Pure not present")
    def test_test_split_row_count_and_date_window(self):
        split = load_test_meta(DATA_DIR, expected_rows=TEST_ROWS)
        self.assertEqual(len(split.rows), TEST_ROWS)
        self.assertEqual(len(split.meta), TEST_ROWS)
        self.assertTrue(
            all(20220429 <= row[0] <= 20220508 for row in split.rows)
        )
        self.assertEqual(
            split.meta,
            tuple(
                (index, row[1], row[2]) for index, row in enumerate(split.rows)
            ),
        )

    @unittest.skipUnless(REAL_DATA, "KuaiRand-Pure not present")
    def test_test_rows_match_the_kit_loader_element_for_element(self):
        data_module, _, _ = starter_modules()
        kit_rows = data_module.load(str(DATA_DIR))["test"]
        split = load_test_meta(DATA_DIR)
        self.assertEqual(len(split.rows), len(kit_rows))
        self.assertEqual(
            [row[:6] for row in split.rows], [row[:6] for row in kit_rows]
        )


class CandidateEnvironmentTests(unittest.TestCase):
    def _fixture(self, root: Path):
        workspace = CandidateWorkspace(root / "generated", "run0001", 1, "cand01")
        executor = CandidateExecutor(
            repo_root=REPO_ROOT,
            data_dir=root / "data_dir",
            experiment_timeout_seconds=60,
            test_timeout_seconds=60,
        )
        return workspace, executor

    def test_candidate_environment_drops_provider_keys(self):
        thread_caps = (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "sentinel", "ANTHROPIC_API_KEY": "sentinel"},
        ):
            workspace, executor = self._fixture(Path(tmp))
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import os,json;print(json.dumps(dict(os.environ)))",
                ],
                env=executor._environment(workspace),
                cwd=workspace.directory,
                capture_output=True,
                text=True,
                check=True,
            )
            seen = json.loads(completed.stdout)
        self.assertFalse(
            any(key.startswith(("OPENAI_", "ANTHROPIC_")) for key in seen)
        )
        self.assertNotIn("sentinel", json.dumps(seen))
        self.assertEqual(seen["PYTHONDONTWRITEBYTECODE"], "1")
        for name in thread_caps:
            self.assertEqual(seen[name], "1")
        self.assertEqual(seen["PYTHONPATH"], str(REPO_ROOT))
        self.assertEqual(seen["KUAIRAND_DATA_DIR"], str(Path(tmp) / "data_dir"))
        self.assertEqual(seen["HOME"], str(workspace.directory))

    def test_candidate_subprocess_cwd_is_the_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace, executor = self._fixture(Path(tmp))
            completed = subprocess.run(
                [sys.executable, "-c", "import os;print(os.getcwd())"],
                env=executor._environment(workspace),
                cwd=workspace.directory,
                capture_output=True,
                text=True,
                check=True,
            )
        self.assertEqual(
            Path(completed.stdout.strip()).resolve(), workspace.directory.resolve()
        )


FAILURE_CLASSES = {
    "timeout",
    "crash",
    "bad_output",
    "low_score",
    "leak",
    "missing_test_scores",
}

TRIVIAL_CANDIDATE_CODE = (
    "import numpy as np\n"
    "from src.experiments.contracts import CandidateOutput\n"
    "from src.models.sampling import sample_bpr_pairs\n"
    "def run(context, parameters):\n"
    "    sample_bpr_pairs(context.train_users, context.train_y, np.random.default_rng(0), 1)\n"
    "    return CandidateOutput(np.zeros(len(context.valid_x)), {}, [], {})\n"
)
TRIVIAL_CANDIDATE_TESTS = (
    "import unittest\n"
    "import candidate\n"
    "class ContractTests(unittest.TestCase):\n"
    "    def test_callable(self):\n"
    "        self.assertTrue(callable(candidate.run))\n"
)


class TrainValidSplitTests(unittest.TestCase):
    @unittest.skipUnless(REAL_DATA, "KuaiRand-Pure not present")
    def test_train_valid_split_sizes(self):
        splits = load_train_valid(DATA_DIR)
        self.assertEqual(set(splits), {"train", "valid"})
        self.assertEqual(len(splits["train"]), 1_141_112)
        self.assertEqual(len(splits["valid"]), 124_909)

    @unittest.skipUnless(REAL_DATA, "KuaiRand-Pure not present")
    def test_no_train_or_valid_row_is_dated_after_20220428(self):
        splits = load_train_valid(DATA_DIR)
        dates = [row[0] for rows in splits.values() for row in rows]
        # The plan expected min 20220408 (the nominal window start), but the
        # dataset's earliest standard log row is 20220409 — verified identical
        # in the kit's own loader. Containment plus the actual bounds:
        self.assertTrue(all(20220408 <= date <= 20220428 for date in dates))
        self.assertEqual(min(dates), 20220409)
        self.assertEqual(max(dates), 20220428)

    @unittest.skipUnless(REAL_DATA, "KuaiRand-Pure not present")
    def test_train_and_valid_match_the_kit_loader_row_for_row(self):
        data_module, _, _ = starter_modules()
        kit = data_module.load(str(DATA_DIR))
        splits = load_train_valid(DATA_DIR)
        for name in ("train", "valid"):
            self.assertEqual(len(splits[name]), len(kit[name]))
            # Row equality implies encoding equality: the kit's encode is a
            # pure function of the rows.
            self.assertEqual(splits[name][:5000], kit[name][:5000])


class FailureClassTests(unittest.TestCase):
    def test_failure_classes_cover_every_return_path(self):
        source = (
            REPO_ROOT / "src" / "agent" / "candidate_runner.py"
        ).read_text(encoding="utf-8")
        failed_calls = 0
        for node in ast.walk(ast.parse(source)):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "ExperimentOutcome"
            ):
                continue
            status = None
            failure_class = None
            keyword_present = False
            for keyword in node.keywords:
                if keyword.arg == "status" and isinstance(keyword.value, ast.Constant):
                    status = keyword.value.value
                if keyword.arg == "failure_class":
                    keyword_present = True
                    if isinstance(keyword.value, ast.Constant):
                        failure_class = keyword.value.value
            if status != "failed":
                continue
            failed_calls += 1
            # Every failed outcome carries the keyword; literal values must be
            # members of the frozen six-value set (the sanity branch passes a
            # runtime value whose domain classify_primary's bounds test pins).
            self.assertTrue(keyword_present, "failed outcome without failure_class")
            if failure_class is not None:
                self.assertIn(failure_class, FAILURE_CLASSES)
        self.assertGreaterEqual(failed_calls, 4)

    def test_timeout_outcome_is_classified(self):
        # train() reports stdout paths relative to the repo root, so the run
        # dir must live under it (the controller guarantees this invariant).
        run_root = Path(tempfile.mkdtemp(dir=REPO_ROOT))
        self.addCleanup(shutil.rmtree, run_root, ignore_errors=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = CandidateWorkspace(root / "generated", "run0001", 1, "cand01")
            manifest = CandidateManifest(
                candidate_id="cand01",
                hypothesis_id="h1",
                family="bpr",
                code=TRIVIAL_CANDIDATE_CODE,
                tests=TRIVIAL_CANDIDATE_TESTS,
                parameters={},
            )
            workspace.write(manifest)
            executor = CandidateExecutor(
                repo_root=REPO_ROOT,
                data_dir=root / "data_dir",
                experiment_timeout_seconds=0,
                test_timeout_seconds=60,
            )
            outcome = executor.train(1, manifest, workspace, run_root)
        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.failure_class, "timeout")


class SanityBoundsTests(unittest.TestCase):
    def test_classify_primary_bounds(self):
        self.assertEqual(classify_primary(0.4699), "low_score")
        self.assertIsNone(classify_primary(0.47))
        self.assertIsNone(classify_primary(0.80))
        self.assertEqual(classify_primary(0.8001), "leak")
        self.assertIsNone(classify_primary(0.6015))

    def test_within_baseline_tolerance_is_two_sided(self):
        for value in (0.5986, 0.6046):
            self.assertTrue(within_baseline_tolerance(value), value)
        for value in (0.5985, 0.6047, 0.85):
            self.assertFalse(within_baseline_tolerance(value), value)


if __name__ == "__main__":
    unittest.main()
