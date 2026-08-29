"""Tests for the data card renderer (T1 / I15)."""
from __future__ import annotations

import os
import time
import unittest
from pathlib import Path

from src.agent.safety import FORBIDDEN_TEXT
from src.evaluation.datacard import render_data_card

REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_DATA_DIR = REPO_ROOT / "data" / "KuaiRand-Pure" / "data"

LOG_HEADER = (
    "user_id,video_id,date,hourmin,time_ms,"
    "is_click,is_like,is_follow,is_comment,is_forward,is_hate,"
    "long_view,play_time_ms,duration_ms,"
    "profile_stay_time,comment_stay_time,is_profile_enter,is_rand,tab"
)

VIDEO_FEAT_HEADER = (
    "video_id,author_id,video_type,upload_dt,upload_type,"
    "visible_status,video_duration,server_width,server_height,"
    "music_id,music_type,tag"
)

USER_FEAT_HEADER = (
    "user_id,user_active_degree,is_lowactive_period,is_live_streamer,"
    "is_video_author,follow_user_num,follow_user_num_range,"
    "fans_user_num,fans_user_num_range,friend_user_num,"
    "friend_user_num_range,register_days,register_days_range,"
    + ",".join(f"onehot_feat{i}" for i in range(18))
)

VIDEO_STAT_HEADER = (
    "video_id,counts,show_cnt,show_user_num,play_cnt,play_user_num,"
    "play_duration,complete_play_cnt,complete_play_user_num,"
    "valid_play_cnt,valid_play_user_num,long_time_play_cnt,"
    "long_time_play_user_num,short_time_play_cnt,"
    "short_time_play_user_num,play_progress,comment_stay_duration,"
    "like_cnt,like_user_num,click_like_cnt,double_click_cnt,"
    "cancel_like_cnt,cancel_like_user_num,comment_cnt,comment_user_num,"
    "direct_comment_cnt,reply_comment_cnt,delete_comment_cnt,"
    "delete_comment_user_num,comment_like_cnt,comment_like_user_num,"
    "follow_cnt,follow_user_num,cancel_follow_cnt,cancel_follow_user_num,"
    "share_cnt,share_user_num,download_cnt,download_user_num,"
    "report_cnt,report_user_num,reduce_similar_cnt,"
    "reduce_similar_user_num,collect_cnt,collect_user_num,"
    "cancel_collect_cnt,cancel_collect_user_num,"
    "direct_comment_user_num,reply_comment_user_num,"
    "share_all_cnt,share_all_user_num,outsite_share_all_cnt"
)

ONEHOT_ZEROS = ",".join(["0"] * 18)


def _log_row(
    user_id, video_id, date, tab=1,
    is_click=0, long_view=0, duration_ms=5000,
    is_like=0, is_follow=0, is_comment=0,
    is_forward=0, is_hate=0, is_profile_enter=0,
    play_time_ms=1000, hourmin=1200,
):
    return (
        f"{user_id},{video_id},{date},{hourmin},1650000000000,"
        f"{is_click},{is_like},{is_follow},{is_comment},{is_forward},{is_hate},"
        f"{long_view},{play_time_ms},{duration_ms},"
        f"0,0,{is_profile_enter},0,{tab}"
    )


def _video_feat_row(video_id, author_id=100, video_type="NORMAL",
                    upload_type="LongImport"):
    return (
        f"{video_id},{author_id},{video_type},2022-04-10,{upload_type},"
        f"0.0,87433.0,720.0,1280.0,9155697,9.0,39"
    )


def _user_feat_row(user_id, active_degree="full_active", is_live_streamer=1):
    return (
        f"{user_id},{active_degree},0,{is_live_streamer},1,"
        f"100,100+,50,[30\\,60),10,[1\\,30),200,180+,{ONEHOT_ZEROS}"
    )


def _video_stat_row(video_id, counts=50, show_cnt=1000.0):
    zeros = ",".join(["0.0"] * 49)
    return f"{video_id},{counts},{show_cnt},{zeros}"


def _write_fixture(data_dir: Path, *, include_test_bomb: bool = False):
    """Write a minimal synthetic dataset with known properties.

    Creates:
    - 10 train rows (dates 20220408-20220421) across 3 users and 4 videos
    - 5 valid rows (dates 20220422-20220428) across 2 users and 3 videos
    - Optionally, test-dated rows with unparseable values (BOOM)
    """
    train_rows = [
        _log_row(0, 100, 20220408, is_click=1, long_view=1, tab=1),
        _log_row(0, 101, 20220409, is_click=1, long_view=0, tab=1),
        _log_row(0, 102, 20220410, is_click=0, long_view=1, tab=0),
        _log_row(1, 100, 20220411, is_click=1, long_view=1, tab=1),
        _log_row(1, 101, 20220412, is_click=0, long_view=0, tab=1),
        _log_row(1, 103, 20220413, is_click=1, long_view=0, tab=4,
                 duration_ms=0),
        _log_row(2, 100, 20220414, is_click=0, long_view=0, tab=1),
        _log_row(2, 102, 20220415, is_click=1, long_view=1, tab=0),
        _log_row(2, 103, 20220416, is_click=0, long_view=0, tab=4),
        _log_row(2, 100, 20220417, is_click=1, long_view=0, tab=1),
    ]

    valid_rows = [
        _log_row(0, 100, 20220422, is_click=1, long_view=1, tab=1),
        _log_row(0, 101, 20220423, is_click=0, long_view=0, tab=1),
        _log_row(1, 102, 20220424, is_click=1, long_view=1, tab=0),
        _log_row(1, 103, 20220425, is_click=0, long_view=0, tab=4),
        _log_row(2, 100, 20220426, is_click=1, long_view=0, tab=1),
    ]

    test_rows = []
    if include_test_bomb:
        test_rows.append(
            f"3,200,20220429,1200,1650000000000,"
            f"1,0,0,0,0,0,BOOM,1000,BOOM,0,0,0,0,1"
        )
        test_rows.append(
            f"4,201,20220430,1300,1650000000000,"
            f"0,0,0,0,0,0,BOOM,2000,BOOM,0,0,0,0,0"
        )

    log1 = [LOG_HEADER] + train_rows
    log2 = [LOG_HEADER] + valid_rows + test_rows

    (data_dir / "log_standard_4_08_to_4_21_pure.csv").write_text(
        "\n".join(log1) + "\n", encoding="utf-8"
    )
    (data_dir / "log_standard_4_22_to_5_08_pure.csv").write_text(
        "\n".join(log2) + "\n", encoding="utf-8"
    )

    video_ids = {100, 101, 102, 103}
    if include_test_bomb:
        video_ids |= {200, 201}
    video_feats = [VIDEO_FEAT_HEADER] + [
        _video_feat_row(v) for v in sorted(video_ids)
    ]
    (data_dir / "video_features_basic_pure.csv").write_text(
        "\n".join(video_feats) + "\n", encoding="utf-8"
    )

    user_feats = [USER_FEAT_HEADER] + [
        _user_feat_row(u) for u in range(5 if include_test_bomb else 3)
    ]
    (data_dir / "user_features_pure.csv").write_text(
        "\n".join(user_feats) + "\n", encoding="utf-8"
    )

    video_stats = [VIDEO_STAT_HEADER] + [
        _video_stat_row(v) for v in sorted(video_ids)
    ]
    (data_dir / "video_features_statistic_pure.csv").write_text(
        "\n".join(video_stats) + "\n", encoding="utf-8"
    )


class DataCardTests(unittest.TestCase):

    def test_missing_files_render_an_empty_card(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(render_data_card(Path(d)), "")

    def test_test_dated_rows_are_counted_but_never_parsed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            data_dir = Path(d)
            _write_fixture(data_dir, include_test_bomb=True)
            card = render_data_card(data_dir)
            self.assertNotEqual(card, "")
            self.assertIn("| test | 2 |", card)
            self.assertNotIn("BOOM", card)

    def test_train_and_valid_rates_match_hand_computed_values(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            data_dir = Path(d)
            _write_fixture(data_dir)
            card = render_data_card(data_dir)
            # Train: 10 rows, 4 long_view positives = 40.0000%
            self.assertIn("40.0000 %", card)
            # Train: 10 rows, 6 is_click positives = 60.0000%
            self.assertIn("60.0000 %", card)

    def test_card_is_under_two_hundred_lines(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            data_dir = Path(d)
            _write_fixture(data_dir)
            card = render_data_card(data_dir)
            self.assertNotEqual(card, "")
            lines = card.strip().split("\n")
            self.assertLessEqual(len(lines), 200,
                                 f"Card has {len(lines)} lines, max is 200")

    def test_card_is_deterministic(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            data_dir = Path(d)
            _write_fixture(data_dir)
            card1 = render_data_card(data_dir)
            card2 = render_data_card(data_dir)
            self.assertEqual(card1, card2)

    def test_card_avoids_the_generated_source_blocklist(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            data_dir = Path(d)
            _write_fixture(data_dir)
            card = render_data_card(data_dir)
            lowered = card.lower()
            for forbidden in FORBIDDEN_TEXT:
                self.assertNotIn(
                    forbidden.lower(), lowered,
                    f"Card contains forbidden text: {forbidden!r}"
                )
            extra_blocklist = {"log_standard", "KuaiRand", ".csv", "/data/"}
            for forbidden in extra_blocklist:
                self.assertNotIn(
                    forbidden.lower(), lowered,
                    f"Card contains forbidden text: {forbidden!r}"
                )

    @unittest.skipUnless(
        os.environ.get("DATACARD_FULL"),
        "requires full dataset (set DATACARD_FULL=1)"
    )
    def test_full_dataset_card_matches_the_profiler(self):
        start = time.monotonic()
        card = render_data_card(FULL_DATA_DIR)
        elapsed = time.monotonic() - start

        self.assertLess(elapsed, 60, f"Render took {elapsed:.1f}s, max is 60s")
        self.assertNotEqual(card, "")

        lines = card.strip().split("\n")
        self.assertLessEqual(len(lines), 200)

        # Split sizes
        self.assertIn("1,141,112", card)
        self.assertIn("26,210", card)
        self.assertIn("7,538", card)
        self.assertIn("124,909", card)
        self.assertIn("170,588", card)

        # Train label rates
        self.assertIn("33.66", card)   # long_view
        self.assertIn("46.34", card)   # is_click
        self.assertIn("1.86", card)    # is_like (1.8677)
        self.assertIn("0.10", card)    # is_follow (0.1007) — also is_forward (0.0996)
        self.assertIn("0.25", card)    # is_comment (0.2568)
        self.assertIn("0.04", card)    # is_hate (0.0421)
        self.assertIn("2.53", card)    # is_profile_enter (2.5391)

        # Tab breakdown
        self.assertIn("73.1", card)    # tab 1 share
        self.assertIn("52.9", card)    # tab 1 click rate
        self.assertIn("38.6", card)    # tab 1 long_view rate
        self.assertIn("13.1", card)    # tab 0 share
        self.assertIn("4.2", card)     # tab 0 long_view rate
        self.assertIn("6.6", card)     # tab 4 share
        self.assertIn("48.9", card)    # tab 4 long_view rate

        # Quality
        self.assertIn("239", card)     # zero-duration videos
        self.assertIn("24,076", card)  # zero-duration train rows
        self.assertIn("15,609", card)  # duplicate rows
        self.assertIn("21,127", card)  # is_live_streamer == -124
        self.assertIn("27,285", card)  # total user rows

        # No absolute paths or timestamps
        self.assertNotIn("/Users/", card)
        self.assertNotIn("20260", card)

        # Blocklist
        lowered = card.lower()
        for forbidden in FORBIDDEN_TEXT:
            self.assertNotIn(forbidden.lower(), lowered)
        for forbidden in {"log_standard", "KuaiRand", ".csv", "/data/"}:
            self.assertNotIn(forbidden.lower(), lowered)


if __name__ == "__main__":
    unittest.main()
