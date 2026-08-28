from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .families import family_names


def _required(value: dict[str, Any], name: str, expected_type):
    if name not in value or not isinstance(value[name], expected_type):
        raise ValueError(f"{name!r} is required and must be {expected_type}")
    return value[name]


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
    diagnostics: dict[str, Any] = field(default_factory=dict)
    failure_class: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    web_search_calls: int = 0

    def add(self, other: "TokenUsage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.cached_tokens += other.cached_tokens
        self.web_search_calls += other.web_search_calls

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "TokenUsage":
        value = value or {}
        return cls(**{name: int(value.get(name, 0)) for name in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceSource:
    title: str
    url: str
    method_card_id: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidenceSource":
        return cls(
            title=str(_required(value, "title", str)),
            url=str(_required(value, "url", str)),
            method_card_id=str(value.get("method_card_id", "")),
        )


@dataclass(frozen=True)
class ResearchDecision:
    hypothesis_id: str
    family: str
    action: str
    hypothesis: str
    rationale: str
    parameters: dict[str, Any]
    evidence: tuple[EvidenceSource, ...]
    needs_web_search: bool = False
    parent_experiment: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ResearchDecision":
        family = str(_required(value, "family", str))
        if family not in family_names():
            raise ValueError(f"Unsupported research family: {family}")
        action = str(_required(value, "action", str))
        if action not in {"explore", "exploit", "replicate"}:
            raise ValueError(f"Unsupported research action: {action}")
        evidence_raw = _required(value, "evidence", list)
        return cls(
            hypothesis_id=str(_required(value, "hypothesis_id", str)),
            family=family,
            action=action,
            hypothesis=str(_required(value, "hypothesis", str)),
            rationale=str(_required(value, "rationale", str)),
            parameters=dict(_required(value, "parameters", dict)),
            evidence=tuple(EvidenceSource.from_dict(item) for item in evidence_raw),
            needs_web_search=bool(value.get("needs_web_search", False)),
            parent_experiment=value.get("parent_experiment"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CriticDecision:
    approved: bool
    decision: str
    rationale: str
    concerns: tuple[str, ...] = ()
    next_focus: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CriticDecision":
        return cls(
            approved=bool(_required(value, "approved", bool)),
            decision=str(_required(value, "decision", str)),
            rationale=str(_required(value, "rationale", str)),
            concerns=tuple(str(item) for item in value.get("concerns", [])),
            next_focus=str(value.get("next_focus", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateManifest:
    candidate_id: str
    hypothesis_id: str
    family: str
    code: str
    tests: str
    parameters: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CandidateManifest":
        return cls(
            candidate_id=str(_required(value, "candidate_id", str)),
            hypothesis_id=str(_required(value, "hypothesis_id", str)),
            family=str(_required(value, "family", str)),
            code=str(_required(value, "code", str)),
            tests=str(_required(value, "tests", str)),
            parameters=dict(_required(value, "parameters", dict)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DebugDecision:
    preserve_hypothesis: bool
    diagnosis: str
    replacement_code: str
    replacement_tests: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DebugDecision":
        return cls(
            preserve_hypothesis=bool(_required(value, "preserve_hypothesis", bool)),
            diagnosis=str(_required(value, "diagnosis", str)),
            replacement_code=str(_required(value, "replacement_code", str)),
            replacement_tests=str(_required(value, "replacement_tests", str)),
        )


@dataclass
class ExperimentNode:
    iteration: int
    experiment_id: str
    hypothesis_id: str
    family: str
    action: str
    parameters: dict[str, Any]
    status: str
    metrics: dict[str, float] | None = None
    artifact_path: str | None = None
    candidate_dir: str | None = None
    parent_experiment: str | None = None
    replicated_from: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExperimentNode":
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunState:
    run_id: str
    status: str
    started_at: str
    baseline_primary: float
    iteration_count: int = 0
    training_attempts: int = 0
    proposal_attempts: int = 0
    stagnant_iterations: int = 0
    meaningful_best: float | None = None
    best_experiment_id: str | None = None
    best_metrics: dict[str, float] | None = None
    best_artifact_path: str | None = None
    best_candidate_dir: str | None = None
    nodes: list[ExperimentNode] = field(default_factory=list)
    pending_replications: list[dict[str, Any]] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    manual_interventions: int = 0
    stop_reason: str | None = None
    wall_clock_seconds: float = 0.0
    data_card_path: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RunState":
        copied = dict(value)
        copied["nodes"] = [ExperimentNode.from_dict(item) for item in value.get("nodes", [])]
        copied["token_usage"] = TokenUsage.from_dict(value.get("token_usage"))
        return cls(**copied)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        return result
