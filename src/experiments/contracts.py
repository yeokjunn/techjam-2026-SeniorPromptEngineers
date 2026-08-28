from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class CandidateContext:
    train_x: np.ndarray
    train_y: np.ndarray
    train_users: tuple[str, ...]
    valid_x: np.ndarray
    valid_users: tuple[str, ...]
    field_dimension: int
    evaluate_validation: Callable[[np.ndarray], dict[str, float]]
    test_x: np.ndarray | None = None


@dataclass
class CandidateOutput:
    validation_scores: np.ndarray
    checkpoint_state: dict[str, np.ndarray]
    training_trace: list[dict] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    test_scores: np.ndarray | None = None

