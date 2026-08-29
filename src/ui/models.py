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
    prompt: str
    model: str
    latency_seconds: float
    usage: dict[str, int] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    sources: tuple[dict[str, str], ...] = ()
    tool_calls: tuple[dict[str, Any], ...] = ()
    retries: int = 0


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
    change_summary: ChangeSummary | None = None
    agent_notes: dict[str, Any] = field(default_factory=dict)
    candidate_code: str | None = None
    candidate_tests: str | None = None
    role_passes: tuple[RolePass, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)


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


@dataclass(frozen=True)
class SubmissionCheck:
    valid: bool
    row_count: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    duplicate_pairs: int
    alignment_checked: bool = False
