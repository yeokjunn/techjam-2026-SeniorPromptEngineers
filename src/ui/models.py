from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DashboardConfig:
    run_root: Path
    eda_profile_path: Path
    data_dir: Path
    official_baseline: float = 0.6016
    active_refresh_seconds: int = 5
    stale_after_seconds: int = 300


@dataclass(frozen=True)
class ChangeSummary:
    iteration: int
    candidate_id: str
    files: tuple[dict[str, Any], ...] = ()
    lines_added: int = 0
    lines_deleted: int = 0
    patch_path: str | None = None


@dataclass(frozen=True)
class StageTransition:
    event_id: str
    iteration: int
    stage: str
    status: str
    started_at: str
    updated_at: str
    role: str | None = None
    experiment_id: str | None = None
    attempt: int = 1
    objective: str = ""
    agent_note: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] | None = None
    change_summary: ChangeSummary | None = None
    error: str | None = None
    repair: str | None = None


@dataclass(frozen=True)
class RolePass:
    sequence: int
    role: str
    model: str
    latency_seconds: float
    usage: dict[str, int] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    sources: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True)
class EDAArtifact:
    iteration: int
    path: Path
    status: str = "completed"
    summary: str = ""
    error: str | None = None
    plan: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    findings: tuple[dict[str, Any], ...] = ()
    feature_candidates: tuple[dict[str, Any], ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)



@dataclass(frozen=True)
class LLMCall:
    """One recorded model call from ``runs/<id>/passes/*.json``."""

    iteration: int
    role: str
    family: str
    sequence: int
    recorded_at: str
    model: str
    latency_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    retries: int = 0
    gist: str = ""
    prompt: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    sources: tuple[dict[str, str], ...] = ()
    path: Path | None = None


@dataclass(frozen=True)
class MemoryEvent:
    """A typed event from ``research_memory.jsonl`` (retries and failures)."""

    kind: str
    iteration: int
    label: str = ""
    error: str = ""
    error_type: str | None = None
    reprompt: int = 0


@dataclass(frozen=True)
class DebuggerEvent:
    iteration: int
    stage: str
    candidate_id: str | None = None
    error_type: str | None = None
    error: str = ""
    lesson: str = ""
    event_type: str = "debugger_memory"


@dataclass(frozen=True)
class IterationSnapshot:
    iteration: int
    experiment_id: str
    status: str
    hypothesis: str = ""
    family: str = ""
    action: str = ""
    parent_experiment: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] | None = None
    duration_seconds: float | None = None
    repairs: int = 0
    failure_class: str | None = None
    error: str | None = None
    change_summary: ChangeSummary | None = None
    agent_notes: dict[str, Any] = field(default_factory=dict)
    candidate_dir: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SubmissionCheck:
    valid: bool
    row_count: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    duplicate_pairs: int
    alignment_checked: bool = False


@dataclass(frozen=True)
class RunSnapshot:
    run_id: str
    path: Path
    status: str
    stop_reason: str | None
    started_at: str | None
    best_experiment_id: str | None
    best_metrics: dict[str, float] | None
    baseline_primary: float
    iterations: tuple[IterationSnapshot, ...] = ()
    activity: StageTransition | None = None
    transitions: tuple[StageTransition, ...] = ()
    resources: dict[str, Any] = field(default_factory=dict)
    nodes: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    gate_info: dict[str, Any] | None = None
    journal_markdown: str | None = None
    results_markdown: str | None = None
    run_config: dict[str, Any] = field(default_factory=dict)
    eda_artifacts: tuple[EDAArtifact, ...] = ()
    live_role_passes: tuple[RolePass, ...] = ()
    live_eda: EDAArtifact | None = None
    debugger_events: tuple[DebuggerEvent, ...] = ()
    # Honest-story fields recorded by the harness (summary.json is written at
    # completion, so every one of these may legitimately be None mid-run).
    converged_official: bool | None = None
    converged_official_iteration: int | None = None
    max_scored_primary: float | None = None
    best_replicated: dict[str, Any] | None = None
    interventions: tuple[dict[str, Any], ...] = ()
    interventions_recorded: bool = False
    baseline_selection: dict[str, Any] | None = None
    data_card_markdown: str | None = None
    token_cap: int | None = None
    llm_calls: tuple[LLMCall, ...] = ()
    memory_events: tuple[MemoryEvent, ...] = ()
