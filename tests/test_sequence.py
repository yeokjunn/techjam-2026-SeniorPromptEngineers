"""Leakage and layout guarantees for the trusted user-sequence builder.

Mirrors ``tests/test_features.py``: synthetic rows via the ``history_rows``
override so the suite needs no dataset, and ``ExplosiveLabel`` fails loudly if a
non-train label is ever read.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.models.sequence import DEFAULT_SEQ_LEN, UserSequences, build_user_sequences


def row(date: int, user: str, video: str, author: str = "a1", tab: str = "1",
        duration: float = 1000.0, label=0):
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


class SequenceBuilderTests(unittest.TestCase):
    # Indices for self.TRAIN: 0,1 = u1 day 4/08; 2 = u1 day 4/09;
    # 3 = u2 day 4/10; 4 = u1 day 4/10; 5 = u1 day 4/21.
    TRAIN = [
        row(20220408, "u1", "v1", label=0),
        row(20220408, "u1", "v2", label=1),
        row(20220409, "u1", "v3", label=1),
        row(20220410, "u2", "v1", label=0),
        row(20220410, "u1", "v4", label=0),
        row(20220421, "u1", "v5", label=1),
    ]

    def spec(self, split="train", *, valid=None, test=None, **overrides):
        history = {"train": self.TRAIN}
        if valid is not None:
            history["valid"] = valid
        if test is not None:
            history["test"] = test
        base = {"split": split, "seq_len": 50, "history_rows": history}
        base.update(overrides)
        return base

    def build(self, target_rows, spec) -> UserSequences:
        return build_user_sequences(target_rows, spec)

    def _history_ids(self, seqs: UserSequences, i: int):
        """The real (unmasked) history video vocab indices for target row i.

        Vocab for self.TRAIN is assigned in train-row order: v1=0, v2=1, v3=2,
        v4=3, v5=4 (UNK=5). Tests assert on these integer indices.
        """
        mask = seqs.history_mask[i] > 0.5
        return list(seqs.history_items[i, mask])

    # -- leakage ----------------------------------------------------------------

    def test_train_rows_use_only_strictly_earlier_days(self):
        # Build for the full train set; assert per-row by index.
        seqs = self.build(self.TRAIN, self.spec())
        # u1's first day has no prior history.
        self.assertEqual(self._history_ids(seqs, 0), [])
        self.assertEqual(self._history_ids(seqs, 1), [])
        # On day 4/09, prior days = day 4/08 only -> v1(0), v2(1) (NOT v3 itself).
        self.assertEqual(set(self._history_ids(seqs, 2)), {0, 1})
        self.assertNotIn(2, self._history_ids(seqs, 2))
        # u2 first appearance: no history.
        self.assertEqual(self._history_ids(seqs, 3), [])
        # u1 day 4/10: prior days < 4/10 -> v1, v2, v3 -> {0,1,2}.
        self.assertEqual(set(self._history_ids(seqs, 4)), {0, 1, 2})
        # u1 day 4/21: all earlier train days -> v1, v2, v3, v4 -> {0,1,2,3}.
        self.assertEqual(set(self._history_ids(seqs, 5)), {0, 1, 2, 3})

    def test_same_day_rows_cannot_see_each_other(self):
        # Two rows on day 4/10 neither sees the other; prior days < 4/10 -> v1..v4.
        # extra vocab: v1=0,v2=1,v3=2,v4=3,v5=4,v9=5 (UNK=6).
        extra = self.TRAIN + [row(20220410, "u1", "v9", label=1)]
        target = [row(20220410, "u1", "v1"), row(20220410, "u1", "v9")]
        spec = {"split": "train", "seq_len": 50, "history_rows": {"train": extra}}
        seqs = self.build(target, spec)
        # Prior days < 4/10 -> v1, v2, v3 -> {0,1,2}; v4 (day 4/10) and v9 (day 4/10) excluded.
        self.assertEqual(set(self._history_ids(seqs, 0)), {0, 1, 2})
        self.assertEqual(set(self._history_ids(seqs, 1)), {0, 1, 2})
        self.assertNotIn(5, self._history_ids(seqs, 1))  # v9 (index 5) excluded

    def test_valid_rows_use_all_of_train(self):
        valid = [row(20220422, "u1", "v1")]
        seqs = self.build(valid, self.spec("valid", valid=valid))
        # u1 saw v1..v5 across train; all of train is visible to a valid row.
        self.assertEqual(set(self._history_ids(seqs, 0)), {0, 1, 2, 3, 4})

    def test_test_rows_use_all_of_train_only(self):
        test = [row(20220429, "u1", "v1")]
        seqs = self.build(test, self.spec("test", test=test))
        self.assertEqual(set(self._history_ids(seqs, 0)), {0, 1, 2, 3, 4})

    def test_candidate_never_in_own_history_for_train(self):
        # A same-day duplicate candidate must not see itself (strictly-earlier rule).
        # extra vocab: v1=0,v2=1,v3=2,v4=3,v5=4,v9=5 (UNK=6).
        extra = self.TRAIN + [row(20220421, "u1", "v9", label=1)]
        target = [row(20220421, "u1", "v9")]
        spec = {"split": "train", "seq_len": 50, "history_rows": {"train": extra}}
        seqs = self.build(target, spec)
        self.assertNotIn(5, self._history_ids(seqs, 0))  # v9 (index 5) excluded

    def test_labels_of_non_train_rows_are_never_read(self):
        # valid/test rows carry ExplosiveLabel; the builder only reads date/user/video.
        valid = [row(20220422, "u1", "v1", label=ExplosiveLabel())]
        test = [row(20220429, "u2", "v2", label=ExplosiveLabel())]
        sv = self.build(valid, self.spec("valid", valid=valid))  # must not raise
        st = self.build(test, self.spec("test", test=test))      # must not raise
        self.assertEqual(sv.history_items.shape, (1, 50))
        self.assertEqual(st.history_items.shape, (1, 50))

    def test_vocab_is_train_only(self):
        # A video that never appears in train maps to UNK as the candidate; the
        # valid row's history items are all real train videos (in-vocab).
        valid = [row(20220422, "u1", "vNEVER")]
        spec = self.spec("valid", valid=valid)
        seqs = self.build(valid, spec)
        unk = seqs.field_dimension - 1
        self.assertEqual(seqs.candidate_items[0], unk)
        # Every unmasked history item is a real train video (strictly < unk).
        mask = seqs.history_mask[0] > 0.5
        self.assertTrue((seqs.history_items[0, mask] < unk).all())

    # -- layout -----------------------------------------------------------------

    def test_output_shapes_and_dtypes(self):
        seqs = self.build(self.TRAIN, self.spec())
        self.assertEqual(seqs.history_items.shape, (6, 50))
        self.assertEqual(seqs.history_mask.shape, (6, 50))
        self.assertEqual(seqs.candidate_items.shape, (6,))
        self.assertEqual(seqs.history_items.dtype, np.int32)
        self.assertEqual(seqs.history_mask.dtype, np.float32)
        self.assertEqual(seqs.candidate_items.dtype, np.int32)

    def test_mask_marks_real_history_only_and_padding_is_zero(self):
        # Row 5 (u1 day 4/21) has 4 prior train items.
        seqs = self.build(self.TRAIN, self.spec())
        mask = seqs.history_mask[5]
        self.assertEqual(int(mask.sum()), 4)
        self.assertTrue((mask[:4] == 1.0).all())
        self.assertTrue((mask[4:] == 0.0).all())
        # padding positions hold zero item ids.
        self.assertTrue((seqs.history_items[5, 4:] == 0).all())

    def test_seq_len_caps_history_to_most_recent(self):
        # 8 train items for u1; cap to 3 -> the last 3 chronologically (v5, v6, v7).
        train = [row(20220408 + d, "u1", f"v{d}") for d in range(8)]
        target = [row(20220430, "u1", "vX")]
        spec = {"split": "valid", "seq_len": 3,
                "history_rows": {"train": train, "valid": target}}
        seqs = self.build(target, spec)
        vocab = {f"v{d}": d for d in range(8)}
        unk = len(vocab)  # vX is unseen in train
        self.assertEqual(seqs.candidate_items[0], unk)
        self.assertEqual(int(seqs.history_mask[0].sum()), 3)
        self.assertEqual(list(seqs.history_items[0]),
                         [vocab["v5"], vocab["v6"], vocab["v7"]])

    def test_history_is_chronological_within_unmasked_prefix(self):
        train = [row(20220410, "u1", "v_old"),
                 row(20220415, "u1", "v_new")]
        target = [row(20220420, "u1", "vX")]
        spec = {"split": "valid", "seq_len": 50,
                "history_rows": {"train": train, "valid": target}}
        seqs = self.build(target, spec)
        ids = self._history_ids(seqs, 0)
        # vocab: v_old=0, v_new=1; oldest first within the unmasked prefix.
        self.assertEqual(ids, [0, 1])

    def test_users_with_no_train_history_get_empty_history(self):
        valid = [row(20220422, "uNEW", "v1")]
        seqs = self.build(valid, self.spec("valid", valid=valid))
        self.assertEqual(int(seqs.history_mask[0].sum()), 0)
        self.assertTrue((seqs.history_items[0] == 0).all())

    def test_field_dimension_is_vocab_size_plus_unk(self):
        seqs = self.build(self.TRAIN, self.spec())
        # 5 unique train videos (v1..v5) + 1 UNK = 6.
        self.assertEqual(seqs.field_dimension, 6)

    def test_default_seq_len_constant_is_reasonable(self):
        self.assertGreaterEqual(DEFAULT_SEQ_LEN, 20)


if __name__ == "__main__":
    unittest.main()
