"""L2 must not eat the embedding rows a batch never touched.

Folding `l2 * V` into the gradient handed to Adam makes the decay scale-free:
Adam divides by sqrt(v), so once the real gradient has washed out of the moment
estimates the decay alone moves a row by ~lr per step whatever `l2` is, and an id
that stops appearing in batches lands on exactly 0. The decay is decoupled
(AdamW-style) and applied only to the rows a batch actually touched.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.models.fm_core import FMRanker, sigmoid

RARE = 0  # feature id that only ever appears in the first batch
DIMENSION = 24
EMBEDDING_DIM = 4
FIELDS = 3
BATCH = 8


def _train_step(model: FMRanker, features: np.ndarray, labels: np.ndarray) -> None:
    scores, _, _ = model.logits(features)
    residual = ((sigmoid(scores) - labels) / len(features)).astype(np.float32)
    grad_v, grad_w, grad_b = model.gradients(features, residual)
    model.apply_gradients(grad_v, grad_w, grad_b)


def _train_with_one_rare_id(model: FMRanker, steps: int, seed: int = 0) -> np.ndarray:
    """First batch carries RARE; every later batch draws from the common ids only."""
    rng = np.random.default_rng(seed)
    common = np.arange(1, DIMENSION)
    first = np.column_stack(
        [np.full(BATCH, RARE), rng.choice(common, size=(BATCH, FIELDS - 1))]
    )
    _train_step(model, first, rng.integers(0, 2, BATCH).astype(np.float32))
    after_first_batch = model.V[RARE].copy()
    for _ in range(steps):
        features = rng.choice(common, size=(BATCH, FIELDS))
        _train_step(model, features, rng.integers(0, 2, BATCH).astype(np.float32))
    return after_first_batch


class DecayTests(unittest.TestCase):
    def test_id_seen_only_in_the_first_batch_is_not_decayed_to_zero(self):
        """The bug: rare ids kept collapsing to 0 while the batches ignored them."""
        model = FMRanker(
            DIMENSION, embedding_dim=EMBEDDING_DIM, learning_rate=0.01, l2=1e-2, seed=1
        )
        model.V[RARE] = 0.5
        after_first_batch = _train_with_one_rare_id(model, steps=300)

        final = model.V[RARE]
        self.assertGreater(
            np.linalg.norm(final), 0.5 * np.linalg.norm(after_first_batch)
        )
        self.assertGreater(np.abs(final).min(), 0.1)

    def test_rows_a_batch_never_touched_are_left_exactly_alone(self):
        """Sparse decay: no gradient for a row means no decay for that row."""
        model = FMRanker(
            DIMENSION, embedding_dim=EMBEDDING_DIM, learning_rate=0.01, l2=0.5, seed=1
        )
        model.V += 0.5
        model.W += 0.5
        touched = [1, 2, 3]
        untouched = [index for index in range(DIMENSION) if index not in touched]
        features = np.asarray([touched], dtype=np.int64)
        grad_v, grad_w, grad_b = model.gradients(
            features, np.asarray([0.5], dtype=np.float32)
        )
        before_v, before_w = model.V.copy(), model.W.copy()

        model.apply_gradients(grad_v, grad_w, grad_b)

        np.testing.assert_array_equal(model.V[untouched], before_v[untouched])
        np.testing.assert_array_equal(model.W[untouched], before_w[untouched])
        self.assertLess(np.abs(model.V[touched]).max(), np.abs(before_v[touched]).max())
        self.assertLess(np.abs(model.W[touched]).max(), np.abs(before_w[touched]).max())

    def test_repeatedly_touched_rows_shrink_more_as_l2_grows(self):
        """Decay still happens, and now its size tracks l2 instead of just lr."""
        rows = np.arange(8)
        rng = np.random.default_rng(7)
        batches = []
        for _ in range(200):
            grad_v = np.zeros((DIMENSION, EMBEDDING_DIM), dtype=np.float32)
            grad_w = np.zeros(DIMENSION, dtype=np.float32)
            grad_v[rows] = rng.normal(0, 0.01, (len(rows), EMBEDDING_DIM))
            grad_w[rows] = rng.normal(0, 0.01, len(rows))
            batches.append((grad_v, grad_w))

        norms = []
        for l2 in (0.0, 1e-2, 1e-1):
            model = FMRanker(
                DIMENSION,
                embedding_dim=EMBEDDING_DIM,
                learning_rate=0.01,
                l2=l2,
                seed=1,
            )
            model.V[rows] = 0.5
            for grad_v, grad_w in batches:
                model.apply_gradients(grad_v, grad_w)
            norms.append(float(np.linalg.norm(model.V[rows])))

        self.assertGreater(norms[0], norms[1])
        self.assertGreater(norms[1], norms[2])
        self.assertGreater(norms[0], 0.0)


if __name__ == "__main__":
    unittest.main()
