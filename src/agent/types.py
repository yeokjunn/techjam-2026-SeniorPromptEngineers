from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    kind: str
    hypothesis: str
    parameters: dict[str, Any] = field(default_factory=dict)
    code_change: str = "none; predefined experiment implementation"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExperimentSpec":
        return cls(
            experiment_id=str(value["id"]),
            kind=str(value["kind"]),
            hypothesis=str(value["hypothesis"]),
            parameters=dict(value.get("parameters", {})),
            code_change=str(
                value.get("code_change", "none; predefined experiment implementation")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentOutcome:
    status: str
    metrics: dict[str, float] | None
    duration_seconds: float
    artifact_path: str | None = None
    epoch_trace: list[dict[str, float]] = field(default_factory=list)
    error: str | None = None
    recovery: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    command: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
