"""Plumbing contract for the trusted DIN trainer (Phase 1).

Locks the return-type contract ``run_candidate.py:validate_and_persist_output``
relies on (finite validation_scores/test_scores of the right dtype/length,
checkpoint_state a finite dict[str, np.ndarray] <=50M elements) WITHOUT needing
the dataset or torch: ``_sequences_for_splits`` is monkeypatched to synthetic
``UserSequences``. Phase 2 replaces the zero-filled scores with real torch math;
this test keeps guarding the contract then.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.experiments.contracts import CandidateContext
from src.models import din_trainer
from src.models.din_trainer import run_din_trainer
from src.models.sequence import UserSequences


def _fake_seqs(n: int, field_dim: int = 7, seq_len: int = 20) -> UserSequences:
    rng = np.random.default_rng(abs(hash((n, field_dim))) % (2**31))
    return UserSequences(
        history_items=rng.integers(0, field_dim, size=(n, seq_len), dtype=np.int32),
        history_mask=rng.integers(0, 2, size=(n, seq_len)).astype(np.float32),
        candidate_items=rng.integers(0, field_dim, size=n, dtype=np.int32),
        field_dimension=field_dim,
    )


class _FakeContext(CandidateContext):
    """Minimal context with train/valid/test arrays sized to match the fake sequences."""

    def __init__(self, field_dimension: int, n_train=64, n_valid=37, n_test=50,
                 evaluate_validation=None):
        rng = np.random.default_rng(42)
        super().__init__(
            train_x=rng.integers(0, field_dimension, size=(n_train, 5), dtype=np.int32),
            train_y=rng.integers(0, 2, size=n_train).astype(np.float32),
            train_users=tuple(f"u{i % 10}" for i in range(n_train)),
            valid_x=rng.integers(0, field_dimension, size=(n_valid, 5), dtype=np.int32),
            valid_users=tuple(f"u{i % 10}" for i in range(n_valid)),
            field_dimension=field_dimension,
            evaluate_validation=evaluate_validation or (lambda s: {"GAUC": 0.5, "nDCG@5": 0.5, "primary": 0.5}),
            test_x=rng.integers(0, field_dimension, size=(n_test, 5), dtype=np.int32) if n_test else None,
        )


class DinTrainerPlumbingTests(unittest.TestCase):
    N_TRAIN = 64
    N_VALID = 37
    N_TEST = 50

    def setUp(self):
        self._orig = din_trainer._sequences_for_splits
        self._train = _fake_seqs(self.N_TRAIN)
        self._valid = _fake_seqs(self.N_VALID)
        self._test = _fake_seqs(self.N_TEST)
        din_trainer._sequences_for_splits = lambda context, seq_len: (
            self._train, self._valid, self._test,
        )

    def tearDown(self):
        din_trainer._sequences_for_splits = self._orig

    def _params(self, **over):
        base = {
            "seed": 0, "k": 16, "learning_rate": 0.001, "epochs": 1,
            "batch_size": 4096, "patience": 4, "embedding_dim": 32, "seq_len": 20,
            "attention_dim": 32, "dropout": 0.2, "aux_weight": 0.1,
            "use_is_click": True, "use_play_time": False,
        }
        base.update(over)
        return base

    def test_return_contract_shapes_dtypes_finiteness(self):
        ctx = _FakeContext(field_dimension=40260)
        val, test, ckpt, trace, diag = run_din_trainer(ctx, self._params())
        # validation_scores: float32 1-D, len == n_valid, finite
        self.assertIsInstance(val, np.ndarray)
        self.assertEqual(val.dtype, np.float32)
        self.assertEqual(val.ndim, 1)
        self.assertEqual(len(val), len(self._valid.candidate_items))
        self.assertTrue(np.all(np.isfinite(val)))
        # test_scores: float64 1-D, len == n_test (170588), finite
        self.assertIsInstance(test, np.ndarray)
        self.assertEqual(test.dtype, np.float64)
        self.assertEqual(test.ndim, 1)
        self.assertEqual(len(test), len(self._test.candidate_items))
        self.assertTrue(np.all(np.isfinite(test)))
        # checkpoint_state: dict[str, np.ndarray], finite, <=50M elements
        self.assertIsInstance(ckpt, dict)
        total = 0
        for key, array in ckpt.items():
            self.assertTrue(key.replace("_", "").isalnum(), key)
            self.assertIsInstance(array, np.ndarray)
            self.assertTrue(np.all(np.isfinite(array)))
            total += int(array.size)
        self.assertLessEqual(total, 50_000_000)
        # training_trace + diagnostics are JSON-safe records
        self.assertIsInstance(trace, list)
        self.assertIsInstance(diag, dict)
        self.assertEqual(diag["family"], "din")

    def test_diagnostics_carry_field_and_seq_dimensions(self):
        ctx = _FakeContext(field_dimension=40260)
        _, _, _, _, diag = run_din_trainer(ctx, self._params())
        self.assertEqual(diag["field_dimension"], 40260)
        self.assertEqual(diag["seq_field_dimension"], self._train.field_dimension)

    def test_negative_seq_len_is_rejected(self):
        ctx = _FakeContext(field_dimension=100)
        with self.assertRaises(ValueError):
            run_din_trainer(ctx, self._params(seq_len=0))

    def test_non_finite_return_is_rejected(self):
        ctx = _FakeContext(field_dimension=100)
        # Inject a NaN into the skeleton path via a patched _train_din.
        orig_train = din_trainer._train_din

        def bad_train(context, parameters, train_seqs, valid_seqs, test_seqs, started):
            n_valid = len(valid_seqs.candidate_items)
            n_test = len(test_seqs.candidate_items)
            val = np.full(n_valid, np.nan, dtype=np.float32)
            test = np.zeros(n_test, dtype=np.float64)
            return val, test, {"stub": np.zeros(1, dtype=np.float32)}, [], {}

        din_trainer._train_din = bad_train
        try:
            with self.assertRaises(ValueError):
                run_din_trainer(ctx, self._params())
        finally:
            din_trainer._train_din = orig_train


if __name__ == "__main__":
    unittest.main()
