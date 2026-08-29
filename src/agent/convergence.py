from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


def official_converged(
    scores: Sequence[float], epsilon: float = 0.002, patience: int = 3
) -> bool:
    """The organizers' rule: has the best score stopped improving? (I7)

    Converged once some prefix of length ``k > patience`` satisfies
    ``max(scores[:k]) - max(scores[:k - patience]) <= epsilon``. The spec writes
    the guard as ``k >= patience``, which needs an undefined ``best_0``; ``k >
    patience`` is the reading that gives ``patience`` consecutive deltas, and it
    is the shipped behaviour — the tracker converges on the 4th observation.

    Any firing prefix counts, not just the last one, so the verdict latches: a
    run that met the rule at iteration k has met it, whatever a later score does
    to the trailing window. That is what makes "the smallest k whose prefix
    fires" well defined for ``summary.json``.
    """
    return any(
        max(scores[:k]) - max(scores[: k - patience]) <= epsilon
        for k in range(patience + 1, len(scores) + 1)
    )


def stagnation(scores: Sequence[float], epsilon: float) -> tuple[float | None, int]:
    """The harness ratchet: ``(meaningful_best, stagnant_iterations)``.

    Separate from the rule above and deliberately so — this is the loop's own
    impatience, the organizers' formula is the verdict. One implementation each,
    both here.
    """
    meaningful_best: float | None = None
    stagnant = 0
    for score in scores:
        if meaningful_best is None or score > meaningful_best + epsilon:
            meaningful_best = score
            stagnant = 0
        else:
            stagnant += 1
    return meaningful_best, stagnant


@dataclass
class ConvergenceTracker:
    epsilon: float = 0.002
    patience: int = 3
    best_score: float | None = None
    meaningful_best: float | None = None
    stagnant_iterations: int = 0
    scores: list[float] = field(default_factory=list)

    def observe(self, score: float) -> bool:
        """Record a successful score and return whether the run has converged."""
        self.scores.append(score)
        self.best_score = max(self.scores)
        self.meaningful_best, self.stagnant_iterations = stagnation(
            self.scores, self.epsilon
        )
        return official_converged(self.scores, self.epsilon, self.patience)
