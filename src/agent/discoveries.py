from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .activity import utc_now
from .audit import ResearchAudit
from .types import CriticDecision, ExperimentNode, ResearchDecision


@dataclass
class DiscoveryRecord:
    discovery_id: str
    hypothesis_id: str
    family: str
    hypothesis: str
    rationale: str
    evidence: list[dict[str, str]]
    parameters: dict[str, Any]
    status: str = "proposed"
    created_at: str = ""
    updated_at: str = ""
    first_iteration: int | None = None
    last_iteration: int | None = None
    proposed_count: int = 0
    metrics: dict[str, float] | None = None
    delta_vs_baseline: float | None = None
    experiment_id: str | None = None
    rejection: dict[str, Any] | None = None
    run_id: str | None = None
    failure: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DiscoveryRecord":
        return cls(
            discovery_id=str(value["discovery_id"]),
            hypothesis_id=str(value.get("hypothesis_id", "")),
            family=str(value.get("family", "")),
            hypothesis=str(value.get("hypothesis", "")),
            rationale=str(value.get("rationale", "")),
            evidence=[
                {
                    "title": str(item.get("title", "")),
                    "url": str(item.get("url", "")),
                    "method_card_id": str(item.get("method_card_id", "")),
                }
                for item in value.get("evidence", [])
                if isinstance(item, dict)
            ],
            parameters=dict(value.get("parameters", {})),
            status=str(value.get("status", "proposed")),
            created_at=str(value.get("created_at", "")),
            updated_at=str(value.get("updated_at", "")),
            first_iteration=value.get("first_iteration"),
            last_iteration=value.get("last_iteration"),
            proposed_count=int(value.get("proposed_count", 0)),
            metrics=dict(value["metrics"]) if isinstance(value.get("metrics"), dict) else None,
            delta_vs_baseline=value.get("delta_vs_baseline"),
            experiment_id=value.get("experiment_id"),
            rejection=dict(value["rejection"]) if isinstance(value.get("rejection"), dict) else None,
            run_id=value.get("run_id"),
            failure=value.get("failure"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discovery_id_for(decision: ResearchDecision) -> str:
    urls = sorted(source.url.strip().lower() for source in decision.evidence if source.url)
    basis = json.dumps(
        {
            "family": decision.family,
            "hypothesis": decision.hypothesis.strip().lower(),
            "parameters": decision.parameters,
            "urls": urls,
        },
        sort_keys=True,
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


class DiscoveryStore:
    """Persistent research notes that can seed later autonomous proposals."""

    def __init__(self, path: Path):
        self.path = path
        self.records: dict[str, DiscoveryRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        records = data.get("discoveries", []) if isinstance(data, dict) else []
        for item in records:
            if isinstance(item, dict) and item.get("discovery_id"):
                record = DiscoveryRecord.from_dict(item)
                self.records[record.discovery_id] = record

    def _save(self) -> None:
        payload = {
            "updated_at": utc_now(),
            "discoveries": [
                record.to_dict()
                for record in sorted(
                    self.records.values(),
                    key=lambda item: (item.updated_at, item.discovery_id),
                    reverse=True,
                )
            ],
        }
        ResearchAudit.write_json_atomic(self.path, payload)

    @staticmethod
    def _evidence(decision: ResearchDecision) -> list[dict[str, str]]:
        return [
            {
                "title": source.title,
                "url": source.url,
                "method_card_id": source.method_card_id,
            }
            for source in decision.evidence
            if source.url
        ]

    def record_proposal(
        self, iteration: int, decision: ResearchDecision, run_id: str | None = None
    ) -> str:
        return self._upsert_proposal(
            iteration, decision, increment_count=True, run_id=run_id
        )

    def _upsert_proposal(
        self,
        iteration: int,
        decision: ResearchDecision,
        *,
        increment_count: bool,
        run_id: str | None = None,
    ) -> str:
        evidence = self._evidence(decision)
        now = utc_now()
        key = discovery_id_for(decision)
        record = self.records.get(key)
        if record is None:
            record = DiscoveryRecord(
                discovery_id=key,
                hypothesis_id=decision.hypothesis_id,
                family=decision.family,
                hypothesis=decision.hypothesis,
                rationale=decision.rationale,
                evidence=evidence,
                parameters=dict(decision.parameters),
                created_at=now,
                updated_at=now,
                first_iteration=iteration,
                run_id=run_id,
            )
            self.records[key] = record
        else:
            record.hypothesis_id = decision.hypothesis_id
            record.family = decision.family
            record.hypothesis = decision.hypothesis
            record.rationale = decision.rationale
            record.evidence = evidence
            record.parameters = dict(decision.parameters)
            record.updated_at = now
            record.run_id = run_id or record.run_id
        record.status = "proposed" if record.status in {"new", "proposed"} else record.status
        record.last_iteration = iteration
        if increment_count:
            record.proposed_count += 1
        self._save()
        return key

    def record_rejection(
        self,
        iteration: int,
        decision: ResearchDecision,
        critic: CriticDecision,
        run_id: str | None = None,
    ) -> None:
        key = self._upsert_proposal(
            iteration, decision, increment_count=False, run_id=run_id
        )
        record = self.records[key]
        record.status = "critic_rejected"
        record.rejection = {
            "decision": critic.decision,
            "rationale": critic.rationale,
            "concerns": list(critic.concerns),
            "next_focus": critic.next_focus,
        }
        record.updated_at = utc_now()
        self._save()

    def record_outcome(
        self,
        iteration: int,
        decision: ResearchDecision,
        node: ExperimentNode,
        baseline_primary: float,
        run_id: str | None = None,
        failure: str | None = None,
    ) -> None:
        key = self._upsert_proposal(
            iteration, decision, increment_count=False, run_id=run_id
        )
        record = self.records[key]
        record.status = node.status
        record.metrics = dict(node.metrics) if node.metrics else None
        record.delta_vs_baseline = (
            None
            if not node.metrics
            else float(node.metrics["primary"]) - float(baseline_primary)
        )
        record.experiment_id = node.experiment_id
        record.failure = failure
        record.updated_at = utc_now()
        self._save()

    def prompt_text(self, max_items: int = 6) -> str:
        if not self.records:
            return "No persistent research outcomes have been recorded yet."
        ranked = sorted(
            self.records.values(),
            key=lambda item: (
                item.status == "success",
                item.delta_vs_baseline if item.delta_vs_baseline is not None else -999.0,
                item.updated_at,
            ),
            reverse=True,
        )
        lines = []
        for record in ranked[:max_items]:
            metrics = (
                "unscored"
                if record.metrics is None
                else f"primary={record.metrics.get('primary')}, delta={record.delta_vs_baseline}"
            )
            urls = ", ".join(item["url"] for item in record.evidence[:2])
            provenance = f"run={record.run_id or 'unknown'}, iteration={record.last_iteration}"
            failure = f"; failure={record.failure}" if record.failure else ""
            sources = f"; sources={urls}" if urls else ""
            lines.append(
                (
                    f"- {record.discovery_id} [{record.status}] family={record.family}; "
                    f"hypothesis={record.hypothesis}; parameters={record.parameters}; "
                    f"metrics={metrics}; {provenance}{failure}{sources}"
                )[:900]
            )
        return "\n".join(lines)
