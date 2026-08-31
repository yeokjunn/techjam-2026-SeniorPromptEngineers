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
    diagnostic_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    failure_class: str | None = None
    test_scores_path: str | None = None
    validation_scores_path: str | None = None
    topk_diagnostics: dict[str, Any] = field(default_factory=dict)

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
    web_searched: bool = False

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
            web_searched=bool(value.get("web_searched", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EDAResearchPlan:
    objective: str
    questions: tuple[str, ...]
    feature_hypotheses: tuple[str, ...]
    required_inputs: tuple[str, ...]
    leakage_risks: tuple[str, ...]
    expected_artifacts: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EDAResearchPlan":
        return cls(
            objective=str(_required(value, "objective", str)),
            questions=tuple(str(item) for item in _required(value, "questions", list)),
            feature_hypotheses=tuple(
                str(item) for item in _required(value, "feature_hypotheses", list)
            ),
            required_inputs=tuple(
                str(item) for item in _required(value, "required_inputs", list)
            ),
            leakage_risks=tuple(
                str(item) for item in _required(value, "leakage_risks", list)
            ),
            expected_artifacts=tuple(
                str(item) for item in _required(value, "expected_artifacts", list)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EDAFinding:
    title: str
    observation: str
    implication: str
    evidence: str
    leakage_safe: bool = True

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EDAFinding":
        return cls(
            title=str(_required(value, "title", str)),
            observation=str(_required(value, "observation", str)),
            implication=str(_required(value, "implication", str)),
            evidence=str(_required(value, "evidence", str)),
            leakage_safe=bool(value.get("leakage_safe", True)),
        )


@dataclass(frozen=True)
class FeatureCandidate:
    name: str
    description: str
    family: str
    expected_impact: str
    implementation_scope: str
    leakage_risk: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FeatureCandidate":
        return cls(
            name=str(_required(value, "name", str)),
            description=str(_required(value, "description", str)),
            family=str(_required(value, "family", str)),
            expected_impact=str(_required(value, "expected_impact", str)),
            implementation_scope=str(_required(value, "implementation_scope", str)),
            leakage_risk=str(_required(value, "leakage_risk", str)),
        )


@dataclass(frozen=True)
class EDAReport:
    summary: str
    findings: tuple[EDAFinding, ...]
    feature_candidates: tuple[FeatureCandidate, ...]
    recommended_next_focus: str
    ui_notes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EDAReport":
        def clipped(item: Any, limit: int) -> str:
            return str(item)[:limit]

        return cls(
            summary=clipped(_required(value, "summary", str), 360),
            findings=tuple(
                EDAFinding(
                    clipped(_required(item, "title", str), 80),
                    clipped(_required(item, "observation", str), 220),
                    clipped(_required(item, "implication", str), 220),
                    clipped(_required(item, "evidence", str), 180),
                    bool(item.get("leakage_safe", True)),
                ) for item in _required(value, "findings", list)[:3]
            ),
            feature_candidates=tuple(
                FeatureCandidate(
                    clipped(_required(item, "name", str), 80),
                    clipped(_required(item, "description", str), 220),
                    clipped(_required(item, "family", str), 40),
                    clipped(_required(item, "expected_impact", str), 160),
                    clipped(_required(item, "implementation_scope", str), 180),
                    clipped(_required(item, "leakage_risk", str), 180),
                ) for item in _required(value, "feature_candidates", list)[:3]
            ),
            recommended_next_focus=clipped(_required(value, "recommended_next_focus", str), 300),
            ui_notes=tuple(clipped(item, 160) for item in value.get("ui_notes", [])[:3]),
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
    admission: str = "approved"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CriticDecision":
        return cls(
            approved=bool(_required(value, "approved", bool)),
            decision=str(_required(value, "decision", str)),
            rationale=str(_required(value, "rationale", str)),
            concerns=tuple(str(item) for item in value.get("concerns", [])),
            next_focus=str(value.get("next_focus", "")),
            admission=str(value.get("admission", "approved")),
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
    diagnostic_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    artifact_path: str | None = None
    candidate_dir: str | None = None
    test_scores_path: str | None = None
    validation_scores_path: str | None = None
    parent_experiment: str | None = None
    replicated_from: str | None = None
    search: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    topk_diagnostics: dict[str, Any] = field(default_factory=dict)

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
    search_frontier: list[dict[str, Any]] = field(default_factory=list)
    closed_branches: dict[str, dict[str, Any]] = field(default_factory=dict)
    proposal_signatures: list[dict[str, Any]] = field(default_factory=list)
    branch_failures: dict[str, int] = field(default_factory=dict)
    branch_rejections: dict[str, int] = field(default_factory=dict)
    branch_stagnation: dict[str, int] = field(default_factory=dict)
    search_stats: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RunState":
        copied = dict(value)
        copied["nodes"] = [ExperimentNode.from_dict(item) for item in value.get("nodes", [])]
        copied["token_usage"] = TokenUsage.from_dict(value.get("token_usage"))
        return cls(**copied)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        return result
