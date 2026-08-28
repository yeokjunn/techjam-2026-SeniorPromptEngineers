from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

from src.evaluation.official import (
    LABEL_PLACEHOLDER,
    TEST_ROWS,
    load_test_meta,
    starter_modules,
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


if __name__ == "__main__":
    unittest.main()
