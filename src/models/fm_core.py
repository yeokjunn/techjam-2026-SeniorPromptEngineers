from __future__ import annotations

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


class FMRanker:
    """FM parameters plus Adam updates for agent-generated ranking losses."""

    def __init__(
        self,
        dimension: int,
        embedding_dim: int = 16,
        learning_rate: float = 0.001,
        l2: float = 1e-6,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dimension, embedding_dim)).astype(np.float32)
        self.W = np.zeros(dimension, dtype=np.float32)
        self.b = np.float32(0.0)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, features: np.ndarray):
        embeddings = self.V[features]
        summed = embeddings.sum(axis=1)
        interactions = 0.5 * (
            (summed**2).sum(axis=1) - (embeddings**2).sum(axis=(1, 2))
        )
        scores = self.b + self.W[features].sum(axis=1) + interactions
        return scores, embeddings, summed

    def gradients(self, features: np.ndarray, score_gradients: np.ndarray):
        _, embeddings, summed = self.logits(features)
        grad_v = np.zeros_like(self.V)
        grad_w = np.zeros_like(self.W)
        np.add.at(grad_w, features, score_gradients[:, None])
        np.add.at(
            grad_v,
            features,
            score_gradients[:, None, None] * (summed[:, None, :] - embeddings),
        )
        return grad_v, grad_w, float(score_gradients.sum())

    def apply_gradients(
        self, grad_v: np.ndarray, grad_w: np.ndarray, grad_b: float = 0.0
    ) -> None:
        grad_v = grad_v + self.l2 * self.V
        grad_w = grad_w + self.l2 * self.W
        self.t += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for parameter, gradient, mean, variance in (
            (self.V, grad_v, self.mV, self.vV),
            (self.W, grad_w, self.mW, self.vW),
        ):
            mean *= beta1
            mean += (1 - beta1) * gradient
            variance *= beta2
            variance += (1 - beta2) * (gradient * gradient)
            mean_hat = mean / (1 - beta1**self.t)
            variance_hat = variance / (1 - beta2**self.t)
            parameter -= self.learning_rate * mean_hat / (
                np.sqrt(variance_hat) + epsilon
            )
        self.b -= np.float32(self.learning_rate * grad_b)

    def predict(self, features: np.ndarray, batch_size: int = 200_000) -> np.ndarray:
        chunks = [
            self.logits(features[offset : offset + batch_size])[0]
            for offset in range(0, len(features), batch_size)
        ]
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"V": self.V.copy(), "W": self.W.copy(), "b": np.asarray(self.b)}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        self.V = np.asarray(state["V"], dtype=np.float32).copy()
        self.W = np.asarray(state["W"], dtype=np.float32).copy()
        self.b = np.float32(np.asarray(state["b"]).item())
