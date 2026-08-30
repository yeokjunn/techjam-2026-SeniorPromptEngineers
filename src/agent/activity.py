from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


STAGE_ORDER = (
    "initializing",
    "eda_researcher",
    "eda_builder",
    "researcher",
    "critic_preflight",
    "builder",
    "safety_tests",
    "debugger",
    "training_evaluation",
    "critic_postflight",
    "persistence",
    "completed",
)

ROLE_OBJECTIVES = {
    "eda_researcher": "Plan leakage-safe EDA and feature-engineering analysis.",
    "eda_builder": "Summarize EDA findings and propose feature candidates for the UI.",
    "researcher": "Propose one evidence-backed ranking experiment.",
    "researcher_web": "Resolve an evidence gap using primary sources.",
    "critic_preflight": "Check novelty, leakage safety, feasibility, and scope.",
    "builder": "Generate the approved candidate and its focused tests.",
    "debugger": "Repair a bounded candidate failure without changing the hypothesis.",
    "critic_postflight": "Interpret trusted validation evidence and identify the next focus.",
}

_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|authorization|secret|token)\s*[:=]\s*\S+"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+"),
)
_JUDGE_PATTERN = re.compile(r"(?i)(?:data[\\/]+judge|judge[\\/]+)")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact_text(value: str, limit: int = 2000) -> str:
    cleaned = value
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub("[redacted]", cleaned)
    cleaned = _JUDGE_PATTERN.sub("[restricted-path]/", cleaned)
    return cleaned[:limit]


def safe_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {str(key): safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_value(item) for item in value]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return redact_text(str(value))


def summarize_role_output(role: str, data: dict[str, Any]) -> dict[str, Any]:
    """Build an inspectable decision trace without exposing prompts or hidden reasoning."""
    note: dict[str, Any] = {"objective": ROLE_OBJECTIVES.get(role, role.replace("_", " "))}
    allowed = (
        "hypothesis",
        "rationale",
        "decision",
        "action",
        "family",
        "concerns",
        "next_focus",
        "diagnosis",
        "parameters",
        "objective",
        "summary",
        "questions",
        "feature_hypotheses",
        "findings",
        "feature_candidates",
        "recommended_next_focus",
        "ui_notes",
    )
    for key in allowed:
        if key in data and data[key] not in (None, "", [], {}):
            note[key] = safe_value(data[key])
    evidence = data.get("evidence")
    if isinstance(evidence, list):
        note["evidence"] = [
            {
                key: safe_value(item[key])
                for key in ("title", "url", "method_card_id")
                if isinstance(item, dict) and item.get(key)
            }
            for item in evidence[:5]
            if isinstance(item, dict)
        ]
    return note


@dataclass(frozen=True)
class ActivityHandle:
    event_id: str
    iteration: int
    stage: str
    role: str | None
    experiment_id: str | None
    attempt: int
    started_at: str
    objective: str

    @classmethod
    def create(
        cls,
        iteration: int,
        stage: str,
        *,
        role: str | None = None,
        experiment_id: str | None = None,
        attempt: int = 1,
        objective: str = "",
    ) -> "ActivityHandle":
        return cls(
            event_id=uuid.uuid4().hex,
            iteration=int(iteration),
            stage=stage,
            role=role,
            experiment_id=experiment_id,
            attempt=int(attempt),
            started_at=utc_now(),
            objective=redact_text(objective),
        )
