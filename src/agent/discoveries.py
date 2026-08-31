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
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discovery_id_for(decision: ResearchDecision) -> str:
    urls = sorted(source.url.strip().lower() for source in decision.evidence if source.url)
    basis = json.dumps(
        {
            "family": decision.family,
            "hypothesis": decision.hypothesis.strip().lower(),
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

    def record_proposal(self, iteration: int, decision: ResearchDecision) -> str | None:
        return self._upsert_proposal(iteration, decision, increment_count=True)

    def _upsert_proposal(
        self,
        iteration: int,
        decision: ResearchDecision,
        *,
        increment_count: bool,
    ) -> str | None:
        if not decision.web_searched:
            return None
        evidence = self._evidence(decision)
        if not evidence:
            return None
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
        record.status = "proposed" if record.status in {"new", "proposed"} else record.status
        record.last_iteration = iteration
        if increment_count:
            record.proposed_count += 1
        self._save()
        return key

    def record_rejection(
        self, iteration: int, decision: ResearchDecision, critic: CriticDecision
    ) -> None:
        key = self._upsert_proposal(iteration, decision, increment_count=False)
        if key is None:
            return
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
    ) -> None:
        key = self._upsert_proposal(iteration, decision, increment_count=False)
        if key is None:
            return
        record = self.records[key]
        record.status = node.status
        record.metrics = dict(node.metrics) if node.metrics else None
        record.delta_vs_baseline = (
            None
            if not node.metrics
            else float(node.metrics["primary"]) - float(baseline_primary)
        )
        record.experiment_id = node.experiment_id
        record.updated_at = utc_now()
        self._save()

    def prompt_text(self, max_items: int = 6) -> str:
        if not self.records:
            return "No persistent web discoveries have been recorded yet."
        ranked = sorted(
            self.records.values(),
            key=lambda item: (
                item.status in {"proposed", "critic_rejected", "failed"},
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
            lines.append(
                (
                    f"- {record.discovery_id} [{record.status}] family={record.family}; "
                    f"hypothesis={record.hypothesis}; metrics={metrics}; sources={urls}"
                )[:900]
            )
        return "\n".join(lines)


# --- Cross-run campaign memory -------------------------------------------------
#
# Why this is not a `DiscoveryStore` field. The store above is *hypothesis*-grained
# and evidence-gated: `_upsert_proposal` returns None unless the decision was
# web-searched AND carries at least one URL, so a run whose proposals cited only
# curated method cards leaves no record in it at all. It also has no run
# dimension — nothing in `DiscoveryRecord` says which campaign a record came
# from, and `prompt_text` ranks records globally by delta. A cross-run digest
# needs the opposite grain: one entry per *run*, unconditional, ordered in time,
# carrying the run's verdict rather than any single hypothesis.
#
# So the campaign log is a sibling of the discovery store, not a duplicate of it:
# same module, same `research/` home, same "persistent notes that seed later
# proposals" job, and deliberately zero overlap in content — the digest carries
# families, score bands and a verdict, and never restates a hypothesis, its
# evidence, or its parameters, because `DiscoveryStore` already persists those
# and both blocks reach the same prompt.

CAMPAIGN_LOG_HEADER = """# Campaign log

One five-line digest per research run, appended at run end and read back into the
Researcher's stable prompt prefix (last 3). Written by
`src/agent/discoveries.py::append_campaign_digest`; the per-hypothesis detail
lives in the discovery store beside it. Safe to prune from the top — the reader
only ever takes the most recent entries.
"""

CAMPAIGN_ENTRY_PREFIX = "## run "

#: Per-field cap on a digest line (see ``campaign_digest._line``).
CAMPAIGN_FIELD_CHAR_LIMIT = 400


def campaign_digest(
    run_id: str,
    *,
    families: str,
    verdict: str,
    falsified: str,
    note: str,
) -> str:
    """The five lines one run contributes: id, families, verdict, falsified, note.

    Every field is flattened to a single line, so a digest is exactly five lines
    however the caller assembled it — the reader splits on the ``## run`` marker
    and the prompt block is budgeted on that shape.
    """

    def _line(value: str) -> str:
        # Capped like its two neighbours in the prompt (`prompt_text`'s 900,
        # `MEASURED_PROFILE_CHAR_LIMIT`): the verdict line embeds
        # `best_experiment_id`, an unbounded model-authored string, into a file
        # read back into every later run's stable prefix.
        return (" ".join(str(value).split()) or "none")[:CAMPAIGN_FIELD_CHAR_LIMIT]

    return "\n".join(
        [
            f"{CAMPAIGN_ENTRY_PREFIX}{_line(run_id)}",
            f"- families: {_line(families)}",
            f"- verdict: {_line(verdict)}",
            f"- falsified: {_line(falsified)}",
            f"- note: {_line(note)}",
        ]
    )


def _campaign_entries(path: Path) -> list[str]:
    """Every digest in the log, oldest first. A missing/unreadable log is empty."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return []
    entries: list[str] = []
    current: list[str] | None = None
    for line in text.splitlines():
        if line.startswith(CAMPAIGN_ENTRY_PREFIX):
            if current is not None:
                entries.append("\n".join(current).rstrip())
            current = [line]
        elif current is not None:
            current.append(line)
    if current is not None:
        entries.append("\n".join(current).rstrip())
    return entries


def campaign_log_has_run(path: Path, run_id: str) -> bool:
    marker = f"{CAMPAIGN_ENTRY_PREFIX}{run_id}"
    return any(
        entry.splitlines()[0].strip() == marker for entry in _campaign_entries(path) if entry
    )


def append_campaign_digest(path: Path, run_id: str, digest: str) -> bool:
    """Append one run's digest, creating the log with its header if absent.

    Idempotent per run id: a resumed run that reaches the end twice contributes
    one entry, not two. Returns whether anything was written.
    """
    path = Path(path)
    if campaign_log_has_run(path, run_id):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    # Append, never rewrite. This is the one file in the design that cannot be
    # regenerated from a run directory, and a read-modify-write would lose the
    # earlier of two runs finishing together and would truncate the whole
    # accumulated memory on a crash mid-write. Same convention as
    # ``ResearchAudit.append_jsonl``: concurrent writers at worst interleave
    # entries. The header is written only into a file that has no content yet.
    if not (path.is_file() and path.read_text(encoding="utf-8").strip()):
        path.write_text(CAMPAIGN_LOG_HEADER, encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{digest.rstrip()}\n")
    return True


def recent_campaign_digests(path: Path, limit: int = 3) -> list[str]:
    """The most recent ``limit`` digests, newest first."""
    entries = [entry for entry in _campaign_entries(Path(path)) if entry.strip()]
    return list(reversed(entries[-limit:])) if limit > 0 else []


def campaign_prompt_block(path: Path | None, limit: int = 3) -> str:
    """The ``PRIOR CAMPAIGNS`` prompt block, or ``""`` when there is no log.

    An empty string is the signal to omit the heading entirely: a heading with
    nothing under it is prompt bytes that teach the model nothing, and it would
    sit in the cacheable stable prefix where it is charged on every call.
    """
    if path is None:
        return ""
    digests = recent_campaign_digests(Path(path), limit)
    if not digests:
        return ""
    body = "\n".join(digests)
    return (
        "PRIOR CAMPAIGNS (most recent first; earlier runs of this same agent on this same "
        "task -- do not re-test what they already measured flat):\n"
        f"{body}"
    )
