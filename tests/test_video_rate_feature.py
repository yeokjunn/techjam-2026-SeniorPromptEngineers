"""The ``video_rate`` history group: train-window only, smoothed, and toggleable.

``video_rate`` is the per-video train-window ``long_view`` rate -- the strongest single measured
signal on KuaiRand-Pure, and the leakage-clean replacement for the kit's
``video_features_statistic_pure.csv``, whose counting window spans the test dates. The point of
this file is that the replacement stays clean: a video's own valid/test rows must never reach its
rate, whatever the scheme, and an unseen video must land in the reserved unknown slot rather than
in a bucket it earned from the future.

Kept separate from ``tests/test_features.py`` (Owner E's) on purpose; the fixtures here are sized
for the video-side key rather than the user-side ones.
"""

from __future__ import annotations

import datetime as dt
import unittest

import numpy as np

from src.agent.families import FAMILIES
from src.agent.families import HISTORY_GROUPS as fam_history_groups
from src.models.features import (
    GROUPS,
    RATE_GROUPS,
    SLOTS_PER_GROUP,
    UNKNOWN_SLOT,
    _compute_train_state,
    _smoothed,
    build_features,
    enabled_groups,
    feature_dimension,
)


def row(date: int, user: str, video: str, author: str, tab: str, duration: float, label):
    """A kit-shaped row: (date, user_id, video_id, author_id, tab, duration_ms, long_view)."""
    return (date, user, video, author, tab, duration, label)


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


class VideoRateFeatureTests(unittest.TestCase):
    """Fixture design.

    Four videos over two train days, chosen so the smoothing question is isolated: ``hot`` and
    ``rare`` both have a *raw* rate of 1.0 and differ only in count (4 rows against 1), so any
    difference between their features is smoothing and nothing else. ``dud`` (4 rows, no
    positives) and ``cold`` (1 row, no positive) balance the split to a global prior of exactly
    0.5. ``ghost`` never appears in train at all.
    """

    TRAIN = [
        row(20220408, "u1", "hot", "a1", "1", 1000.0, 1),
        row(20220408, "u2", "hot", "a1", "1", 1100.0, 1),
        row(20220408, "u3", "cold", "a2", "1", 1200.0, 0),
        row(20220408, "u1", "dud", "a2", "2", 2000.0, 0),
        row(20220408, "u2", "dud", "a2", "2", 2100.0, 0),
        row(20220409, "u3", "hot", "a1", "1", 1300.0, 1),
        row(20220409, "u1", "hot", "a1", "1", 1400.0, 1),
        row(20220409, "u2", "dud", "a2", "2", 2200.0, 0),
        row(20220409, "u3", "rare", "a3", "1", 1500.0, 1),
        row(20220409, "u1", "dud", "a2", "2", 2300.0, 0),
    ]
    UPLOADS = {
        video: dt.date.fromisoformat(day).toordinal()
        for video, day in {
            "hot": "2022-04-01",
            "cold": "2022-04-02",
            "dud": "2022-04-03",
            "rare": "2022-04-04",
        }.items()
    }
    PRIOR = 0.5  # five positives in ten train rows

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
        column = enabled_groups(spec).index(group)
        return features[:, column] - spec["field_offset"] - column * SLOTS_PER_GROUP

    def train_state(self, *, scheme="prior_days", smoothing=20.0):
        return _compute_train_state(self.TRAIN, self.UPLOADS, scheme, smoothing)

    # -- train-window discipline --------------------------------------------------------

    def test_the_group_is_registered_as_a_video_keyed_rate(self):
        self.assertIn("video_rate", GROUPS)
        self.assertIn("video_rate", RATE_GROUPS)

    def test_the_rate_table_counts_train_rows_only(self):
        """Exact counts, so a valid/test row folded in anywhere would change them."""
        table = self.train_state().tables["video_rate"]
        self.assertEqual(table["hot"], [4.0, 4.0])
        self.assertEqual(table["dud"], [0.0, 4.0])
        self.assertEqual(table["rare"], [1.0, 1.0])
        self.assertEqual(table["cold"], [0.0, 1.0])
        self.assertAlmostEqual(self.train_state().prior, self.PRIOR)

    def test_a_video_seen_only_in_validation_never_earns_a_rate(self):
        """The leakage case the kit's statistic file fails: `ghost` is all-positive in valid."""
        valid = [
            row(20220422, "u1", "ghost", "a9", "1", 1000.0, 1),
            row(20220423, "u2", "ghost", "a9", "1", 1000.0, 1),
        ]
        spec = self.spec("valid", valid=valid)
        slots = self.slot(self.build(valid, spec), "video_rate", spec)
        self.assertTrue((slots == UNKNOWN_SLOT).all())
        # And it is absent from the table entirely, not merely unscored.
        self.assertNotIn("ghost", self.train_state().tables["video_rate"])

    def test_a_video_seen_only_in_test_never_earns_a_rate(self):
        test = [row(20220429, "u1", "ghost", "a9", "1", 1000.0, ExplosiveLabel())]
        spec = self.spec("test", test=test)
        self.assertEqual(self.slot(self.build(test, spec), "video_rate", spec)[0], UNKNOWN_SLOT)

    def test_non_train_labels_are_never_read_for_this_group(self):
        valid = [
            row(20220422, "u1", "hot", "a1", "1", 1000.0, ExplosiveLabel()),
            row(20220423, "u2", "ghost", "a9", "1", 1000.0, ExplosiveLabel()),
        ]
        spec = self.spec("valid", valid=valid)
        features = self.build(valid, spec)  # must not raise
        self.assertEqual(features.shape, (2, len(GROUPS)))

    def test_a_train_row_cannot_see_its_own_day(self):
        """`hot`'s day-one rows are its first appearance, so they have no video history."""
        spec = self.spec()
        slots = self.slot(self.build(self.TRAIN, spec), "video_rate", spec)
        for index in range(5):  # every day-one row
            with self.subTest(index=index):
                self.assertEqual(slots[index], UNKNOWN_SLOT)
        self.assertNotEqual(slots[5], UNKNOWN_SLOT)  # day two: `hot` now has day-one history
        self.assertEqual(slots[8], UNKNOWN_SLOT)  # `rare`'s first and only appearance

    def test_validation_rows_injected_as_history_do_not_change_the_train_rates(self):
        """`history_rows` is a write channel into the train table, so it is bounded.

        A candidate can reach labelled *valid* rows and hand them over as "train"
        history; `ghost` is all-positive there and `hot`'s extra positives would
        move its count. Both are outside `TRAIN_START..TRAIN_END`, so the features
        must come out byte-identical to the clean fixture's.
        """
        valid = [
            row(20220422, "u1", "hot", "a1", "1", 1000.0, 0),
            row(20220422, "u2", "ghost", "a9", "1", 1000.0, 0),
        ]
        clean = self.spec("valid", valid=valid)
        smuggled = self.spec("valid", valid=valid)
        smuggled["history_rows"] = {
            "train": self.TRAIN + valid + [row(20220429, "u3", "ghost", "a9", "1", 1.0, 1)],
            "valid": valid,
        }
        self.assertTrue(
            np.array_equal(self.build(valid, smuggled), self.build(valid, clean))
        )
        # Non-vacuity: `ghost` earned nothing at all, and `hot` still scores.
        slots = self.slot(self.build(valid, smuggled), "video_rate", smuggled)
        self.assertNotEqual(slots[0], UNKNOWN_SLOT)
        self.assertEqual(slots[1], UNKNOWN_SLOT)

    def test_valid_rows_score_against_all_of_train(self):
        valid = [row(20220422, "u1", "hot", "a1", "1", 1000.0, 0)]
        spec = self.spec("valid", valid=valid)
        self.assertNotEqual(self.slot(self.build(valid, spec), "video_rate", spec)[0], UNKNOWN_SLOT)

    # -- smoothing ----------------------------------------------------------------------

    def test_a_rare_video_shrinks_further_toward_the_prior_than_a_frequent_one(self):
        """`hot` and `rare` share a raw rate of 1.0, so only the count separates them."""
        state = self.train_state(smoothing=20.0)
        table = state.tables["video_rate"]
        hot = _smoothed(*table["hot"], state.prior, 20.0)
        rare = _smoothed(*table["rare"], state.prior, 20.0)
        self.assertLess(state.prior, rare)
        self.assertLess(rare, hot)
        self.assertLess(hot, 1.0)
        self.assertLess(abs(rare - state.prior), abs(hot - state.prior))

    def test_heavier_smoothing_pulls_a_rare_video_closer_to_the_prior(self):
        distances = []
        for smoothing in (1.0, 5.0, 20.0, 100.0):
            state = self.train_state(smoothing=smoothing)
            value = _smoothed(*state.tables["video_rate"]["rare"], state.prior, smoothing)
            distances.append(abs(value - state.prior))
        self.assertEqual(distances, sorted(distances, reverse=True))

    def test_the_frequent_video_outranks_the_rare_one_in_the_emitted_buckets(self):
        valid = [
            row(20220422, "u1", "rare", "a3", "1", 1500.0, 0),
            row(20220422, "u1", "hot", "a1", "1", 1000.0, 0),
        ]
        spec = self.spec("valid", valid=valid)
        slots = self.slot(self.build(valid, spec), "video_rate", spec)
        self.assertLess(slots[0], slots[1])
        self.assertNotEqual(slots[0], UNKNOWN_SLOT)

    # -- determinism and layout ---------------------------------------------------------

    def test_output_is_deterministic(self):
        spec = self.spec()
        first = self.build(self.TRAIN, spec)
        second = self.build(self.TRAIN, spec)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.dtype, np.int32)

    def test_the_group_appears_only_when_toggled_on(self):
        on = self.spec()
        off = self.spec(use_video_rate=False)
        self.assertIn("video_rate", enabled_groups(on))
        self.assertNotIn("video_rate", enabled_groups(off))

        with_group = self.build(self.TRAIN, on)
        without = self.build(self.TRAIN, off)
        self.assertEqual(with_group.shape[1], len(GROUPS))
        self.assertEqual(without.shape[1], len(GROUPS) - 1)
        self.assertEqual(feature_dimension(off), feature_dimension(on) - SLOTS_PER_GROUP)
        # Appended, not inserted: turning it off leaves every other column byte-identical.
        self.assertTrue(np.array_equal(with_group[:, :-1], without))

    def test_slots_stay_inside_the_declared_index_space(self):
        spec = self.spec()
        spec["field_offset"] = 40260  # KuaiRand-Pure's real field_dimension, well past int16
        features = self.build(self.TRAIN, spec)
        self.assertEqual(features.dtype, np.int32)
        self.assertTrue((features >= 40260).all())
        self.assertTrue((features < 40260 + feature_dimension(spec)).all())

    def test_both_schemes_produce_a_video_rate_column(self):
        for scheme in ("prior_days", "leave_one_out"):
            spec = self.spec(scheme=scheme)
            with self.subTest(scheme=scheme):
                slots = self.slot(self.build(self.TRAIN, spec), "video_rate", spec)
                self.assertEqual(len(slots), len(self.TRAIN))
                self.assertTrue((slots <= UNKNOWN_SLOT).all())
                # `<= UNKNOWN_SLOT` is true by construction, so it only pins the
                # layout. The column is a *rate* under each scheme only if some
                # row actually earned a bucket.
                self.assertTrue((slots != UNKNOWN_SLOT).any())


class VideoRateRegistrationTests(unittest.TestCase):
    """The toggle has to be *proposable*: the schema and sanitiser both read the grid."""

    def test_history_features_carries_the_toggle(self):
        entry = FAMILIES["history_features"]
        self.assertEqual(entry.grid["use_video_rate"], (True, False))
        self.assertIs(entry.defaults["use_video_rate"], True)
        self.assertIn(entry.defaults["use_video_rate"], entry.grid["use_video_rate"])

    def test_the_registry_group_list_matches_the_feature_module(self):
        self.assertEqual(fam_history_groups, GROUPS)
        self.assertIn("video_rate", fam_history_groups)

    def test_the_toggle_did_not_leak_into_the_loss_families(self):
        for name in ("bpr", "group_softmax", "multi_task"):
            with self.subTest(family=name):
                self.assertNotIn("use_video_rate", FAMILIES[name].grid)


if __name__ == "__main__":
    unittest.main()
