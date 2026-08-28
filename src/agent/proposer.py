from __future__ import annotations

from collections.abc import Iterable

from .types import ExperimentSpec


class ConfigProposer:
    """Deterministic first proposer.

    The controller depends only on ``propose``. A later LLM-backed proposer can
    implement the same interface without changing execution, logging, or budget
    enforcement.
    """

    def __init__(self, experiments: Iterable[dict]):
        self._experiments = [ExperimentSpec.from_dict(item) for item in experiments]
        self._next_index = 0

    def propose(self, history: list[dict]) -> ExperimentSpec | None:
        del history  # Reserved for a reflective/LLM proposer.
        if self._next_index >= len(self._experiments):
            return None
        proposal = self._experiments[self._next_index]
        self._next_index += 1
        return proposal

