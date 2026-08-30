from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.evaluation.official import load_random_validation


HEADER = (
    "user_id,video_id,date,hourmin,time_ms,is_click,is_like,is_follow,"
    "is_comment,is_forward,is_hate,long_view,play_time_ms,duration_ms,"
    "profile_stay_time,comment_stay_time,is_profile_enter,is_rand,tab\n"
)


class RandomValidationLoaderTests(unittest.TestCase):
    def test_loads_only_official_validation_dates_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "video_features_basic_pure.csv").write_text(
                "video_id,author_id\nv1,a1\nv2,a2\n", encoding="utf-8"
            )
            rows = [
                "u0,v1,20220421,0,0,0,0,0,0,0,0,1,0,10,0,0,0,1,1\n",
                "u1,v1,20220422,0,0,0,0,0,0,0,0,1,0,10,0,0,0,1,1\n",
                "u2,v2,20220428,0,0,0,0,0,0,0,0,0,0,20,0,0,0,1,2\n",
                "u3,v2,20220429,0,0,0,0,0,0,0,0,1,0,20,0,0,0,1,2\n",
            ]
            (root / "log_random_4_22_to_5_08_pure.csv").write_text(
                HEADER + "".join(rows), encoding="utf-8"
            )
            loaded = load_random_validation(root)

        self.assertEqual([row[1] for row in loaded], ["u1", "u2"])
        self.assertEqual([row[3] for row in loaded], ["a1", "a2"])
        self.assertEqual([row[6] for row in loaded], [1, 0])


if __name__ == "__main__":
    unittest.main()
