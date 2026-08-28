from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConvergenceTracker:
    epsilon: float = 0.002
    patience: int = 3
    best_score: float | None = None
    meaningful_best: float | None = None
    stagnant_iterations: int = 0

    def observe(self, score: float) -> bool:
        """Record a successful score and return whether the run has converged."""
        if self.best_score is None:
            self.best_score = score
            self.meaningful_best = score
            self.stagnant_iterations = 0
            return False

        self.best_score = max(self.best_score, score)
        assert self.meaningful_best is not None
        if score > self.meaningful_best + self.epsilon:
            self.meaningful_best = score
            self.stagnant_iterations = 0
        else:
            self.stagnant_iterations += 1
        return self.stagnant_iterations >= self.patience

