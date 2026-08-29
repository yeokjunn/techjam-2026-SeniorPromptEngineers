"""Leakage and layout guarantees for the trusted history feature builder (review I8)."""

from __future__ import annotations

import datetime as dt
import unittest

import numpy as np

from src.models.features import (
    AUX_HEADS,
    GROUPS,
    SLOTS_PER_GROUP,
    UNKNOWN_SLOT,
    aux_dimension,
    build_aux_labels,
    build_features,
    enabled_groups,
    feature_dimension,
)


def row(date: int, user: str, video: str, author: str, tab: str, duration: float, label):
    """A kit-shaped row: (date, user_id, video_id, author_id, tab, duration_ms, long_view)."""
    return (date, user, video, author, tab, duration, label)


def upload_dates(**mapping: str) -> dict[str, int]:
    return {video: dt.date.fromisoformat(day).toordinal() for video, day in mapping.items()}


class ExplosiveLabel:
    """Raises on any use, so a test fails loudly if a non-train label is ever read."""

    def __bool__(self):
        raise AssertionError("a non-train label was read")

    def __float__(self):
        raise AssertionError("a non-train label was read")

    def __int__(self):
        raise AssertionError("a non-train label was read")

    def __eq__(self, other):
        raise AssertionError("a non-train label was read")

    def __gt__(self, other):
        raise AssertionError("a non-train label was read")

    def __hash__(self):
        return 0

    def __repr__(self):
        return "<ExplosiveLabel>"


class FeatureBuilderTests(unittest.TestCase):
    TRAIN = [
        row(20220408, "u1", "v1", "a1", "1", 1000.0, 0),
        row(20220408, "u1", "v2", "a1", "1", 2000.0, 1),
        row(20220409, "u1", "v3", "a2", "2", 3000.0, 1),
        row(20220410, "u2", "v1", "a1", "1", 1500.0, 0),
        row(20220410, "u1", "v4", "a1", "1", 2500.0, 0),
    ]
    UPLOADS = upload_dates(v1="2022-04-01", v2="2022-04-02", v3="2022-04-03", v4="2022-04-04")

    def spec(self, split="train", *, valid=None, test=None, **overrides):
        history = {"train": self.TRAIN}
        if valid is not None:
            history["valid"] = valid
        if test is not None:
            history["test"] = test
        base = {
            "split": split,
            "field_offset": 100,
            "history_rows": history,
            "video_upload_dates": self.UPLOADS,
        }
        base.update(overrides)
        return base

    def build(self, rows, spec):
        return build_features(np.zeros((len(rows), 5)), spec)

    def slot(self, features, group, spec):
        """Recover the within-group slot of a column, undoing offset and group base."""
        groups = enabled_groups(spec)
        column = groups.index(group)
        return features[:, column] - spec["field_offset"] - column * SLOTS_PER_GROUP

    # -- leakage ------------------------------------------------------------------------

    def test_train_features_use_only_strictly_earlier_days(self):
        spec = self.spec()
        features = self.build(self.TRAIN, spec)
        rates = self.slot(features, "user_rate", spec)
        # u1's first day (rows 0-1) and u2's first appearance (row 3) have no prior history.
        self.assertEqual(rates[0], UNKNOWN_SLOT)
        self.assertEqual(rates[1], UNKNOWN_SLOT)
        self.assertEqual(rates[3], UNKNOWN_SLOT)
        # By day two u1 has a populated bucket.
        self.assertNotEqual(rates[2], UNKNOWN_SLOT)
        self.assertNotEqual(rates[4], UNKNOWN_SLOT)

    def test_same_day_rows_cannot_see_each_others_labels(self):
        """Row 1 is u1's only day-one positive; row 0 must not benefit from it."""
        spec = self.spec()
        rates = self.slot(self.build(self.TRAIN, spec), "user_rate", spec)
        self.assertEqual(rates[0], UNKNOWN_SLOT)

    def test_valid_rows_use_all_of_train(self):
        valid = [row(20220422, "u1", "v1", "a1", "1", 1000.0, 0)]
        spec = self.spec("valid", valid=valid)
        rates = self.slot(self.build(valid, spec), "user_rate", spec)
        # u1 is seen throughout train, so a valid row is never "no history".
        self.assertNotEqual(rates[0], UNKNOWN_SLOT)

    def test_labels_of_non_train_rows_are_never_read(self):
        valid = [
            row(20220422, "u1", "v1", "a1", "1", 1000.0, ExplosiveLabel()),
            row(20220423, "u2", "v2", "a1", "1", 2000.0, ExplosiveLabel()),
        ]
        spec = self.spec("valid", valid=valid)
        features = self.build(valid, spec)  # must not raise
        self.assertEqual(features.shape, (2, len(GROUPS)))

    def test_test_split_labels_are_never_read(self):
        test = [row(20220429, "u1", "v1", "a1", "1", 1000.0, ExplosiveLabel())]
        spec = self.spec("test", test=test)
        self.assertEqual(self.build(test, spec).shape, (1, len(GROUPS)))

    # -- layout -------------------------------------------------------------------------

    def test_output_is_deterministic_and_within_range(self):
        spec = self.spec()
        first = self.build(self.TRAIN, spec)
        second = self.build(self.TRAIN, spec)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.dtype, np.int32)
        low = spec["field_offset"]
        self.assertTrue((first >= low).all())
        self.assertTrue((first < low + feature_dimension(spec)).all())

    def test_realistic_field_offset_does_not_overflow(self):
        """KuaiRand-Pure's field_dimension is 40260, well past int16."""
        spec = self.spec()
        spec["field_offset"] = 40260
        features = self.build(self.TRAIN, spec)
        self.assertEqual(features.dtype, np.int32)
        self.assertTrue((features >= 40260).all())
        self.assertTrue((features < 40260 + feature_dimension(spec)).all())

    def test_feature_dimension_matches_enabled_groups(self):
        self.assertEqual(feature_dimension({}), SLOTS_PER_GROUP * len(GROUPS))
        for disabled in range(len(GROUPS) + 1):
            spec = {f"use_{name}": False for name in GROUPS[:disabled]}
            with self.subTest(disabled=disabled):
                self.assertEqual(
                    feature_dimension(spec), SLOTS_PER_GROUP * (len(GROUPS) - disabled)
                )

    def test_disabled_groups_leave_the_remaining_columns_contiguous(self):
        spec = self.spec(use_user_rate=False, use_recency=False)
        features = self.build(self.TRAIN, spec)
        self.assertEqual(features.shape[1], len(GROUPS) - 2)
        self.assertTrue((features < spec["field_offset"] + feature_dimension(spec)).all())

    def test_unknown_keys_fall_in_the_reserved_slot(self):
        valid = [row(20220422, "unseen", "unseen", "unseen", "unseen", 99999.0, 0)]
        spec = self.spec("valid", valid=valid)
        features = self.build(valid, spec)
        for group in GROUPS:
            with self.subTest(group=group):
                self.assertEqual(self.slot(features, group, spec)[0], UNKNOWN_SLOT)

    def test_missing_upload_date_is_unknown_not_a_value_bucket(self):
        valid = [row(20220422, "u1", "no_upload_record", "a1", "1", 1000.0, 0)]
        spec = self.spec("valid", valid=valid)
        self.assertEqual(self.slot(self.build(valid, spec), "video_age", spec)[0], UNKNOWN_SLOT)

    # -- contract -----------------------------------------------------------------------

    def test_schemes_differ_on_train_rows(self):
        prior = self.build(self.TRAIN, self.spec(scheme="prior_days"))
        loo = self.build(self.TRAIN, self.spec(scheme="leave_one_out"))
        self.assertFalse(np.array_equal(prior, loo))

    def test_row_count_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            build_features(np.zeros((len(self.TRAIN) + 1, 5)), self.spec())

    def test_invalid_spec_values_are_rejected(self):
        for overrides in ({"split": "holdout"}, {"scheme": "magic"}, {"smoothing": 0.0}):
            with self.subTest(**overrides), self.assertRaises(ValueError):
                spec = self.spec()
                spec.update(overrides)
                self.build(self.TRAIN, spec)

    def test_smoothing_pulls_sparse_keys_toward_the_prior(self):
        """A heavier prior must not change the layout, only the values inside it."""
        light = self.build(self.TRAIN, self.spec(smoothing=1.0))
        heavy = self.build(self.TRAIN, self.spec(smoothing=1000.0))
        self.assertEqual(light.shape, heavy.shape)
        self.assertEqual(light.dtype, heavy.dtype)

class AuxiliaryLabelTests(unittest.TestCase):
    """The multi_task family's auxiliary targets: (is_click, is_like, play_time_ms) per train row."""

    AUX = [(1, 0, 500.0), (0, 0, 0.0), (1, 1, 120000.0), (0, 0, 250.0), (1, 0, 9000.0)]

    def spec(self, split="train", **overrides):
        base = {"split": split, "aux_rows": self.AUX}
        base.update(overrides)
        return base

    def build(self, spec, rows=None):
        return build_aux_labels(np.zeros((len(rows or self.AUX), 5)), spec)

    def test_aux_labels_are_train_only_and_finite(self):
        targets = self.build(self.spec())
        self.assertEqual(targets.shape, (len(self.AUX), len(AUX_HEADS)))
        self.assertEqual(targets.dtype, np.float32)
        self.assertTrue(np.isfinite(targets).all())
        self.assertTrue(((targets >= 0.0) & (targets <= 1.0)).all())
        for split in ("valid", "test"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                self.build(self.spec(split))

    def test_binary_heads_are_passed_through_unchanged(self):
        targets = self.build(self.spec())
        np.testing.assert_array_equal(targets[:, 0], [1.0, 0.0, 1.0, 0.0, 1.0])
        np.testing.assert_array_equal(targets[:, 1], [0.0, 0.0, 1.0, 0.0, 0.0])

    def test_play_time_is_log_compressed_and_min_max_scaled(self):
        play = self.build(self.spec())[:, 2]
        self.assertAlmostEqual(float(play.min()), 0.0, places=6)
        self.assertAlmostEqual(float(play.max()), 1.0, places=6)
        # log1p compression: the 120000ms row must not dwarf everything else linearly.
        self.assertGreater(float(play[4]), 120000.0 / 120000.0 * 0.5)

    def test_constant_play_time_does_not_divide_by_zero(self):
        spec = self.spec(aux_rows=[(0, 0, 42.0), (1, 0, 42.0)])
        play = build_aux_labels(np.zeros((2, 5)), spec)[:, 2]
        self.assertTrue(np.isfinite(play).all())
        np.testing.assert_array_equal(play, [0.0, 0.0])

    def test_aux_dimension_tracks_enabled_heads(self):
        self.assertEqual(aux_dimension({}), len(AUX_HEADS))
        spec = self.spec(use_is_like=False)
        self.assertEqual(aux_dimension(spec), len(AUX_HEADS) - 1)
        self.assertEqual(self.build(spec).shape[1], len(AUX_HEADS) - 1)

    def test_disabling_every_head_is_rejected(self):
        spec = self.spec(**{f"use_{head}": False for head in AUX_HEADS})
        with self.assertRaises(ValueError):
            self.build(spec)

    def test_row_count_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            build_aux_labels(np.zeros((len(self.AUX) + 1, 5)), self.spec())


if __name__ == "__main__":
    unittest.main()
