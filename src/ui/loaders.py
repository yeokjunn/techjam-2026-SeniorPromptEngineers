from __future__ import annotations

import csv
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from .models import (
    ChangeSummary,
    DashboardConfig,
    DebuggerEvent,
    EDAArtifact,
    IterationSnapshot,
    RolePass,
    RunSnapshot,
    StageTransition,
    SubmissionCheck,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

ROLE_EXECUTION_ORDER = {
    "eda_researcher": 0,
    "eda_builder": 1,
    "researcher": 2,
    "researcher_web": 3,
    "critic_preflight": 4,
    "builder": 5,
    "debugger": 6,
    "critic_postflight": 7,
}



def _resolve(value: str | Path, base: Path = REPO_ROOT) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _reject_judge_path(path: Path) -> None:
    parts = [part.lower() for part in path.parts]
    if "judge" in parts or any("judge" in part for part in parts):
        raise ValueError("Dashboard paths must never resolve inside a judge-owned location.")


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root = root.resolve()
    _reject_judge_path(resolved)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes configured root: {resolved}") from exc
    return resolved


def load_dashboard_config(path: Path) -> DashboardConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    run_root = _resolve(raw.get("run_root", "runs"))
    eda_profile = _resolve(raw.get("eda_profile_path", "artifacts/ui/kuairand_pure_eda.json"))
    data_dir = _resolve(raw.get("data_dir", "data/KuaiRand-Pure/data"))
    for candidate in (run_root, eda_profile, data_dir):
        _reject_judge_path(candidate)
    return DashboardConfig(
        run_root=run_root,
        eda_profile_path=eda_profile,
        data_dir=data_dir,
        official_baseline=float(raw.get("official_baseline", 0.6016)),
        active_refresh_seconds=max(2, int(raw.get("active_refresh_seconds", 5))),
        stale_after_seconds=max(30, int(raw.get("stale_after_seconds", 300))),
    )


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.is_file():
        return [], []
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], [f"Could not read {path.name}: {exc}"]
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
        except json.JSONDecodeError:
            qualifier = "partial final line" if index == len(lines) else f"malformed line {index}"
            warnings.append(f"Ignored {qualifier} in {path.name}.")
    return records, warnings


def _metrics(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, float] = {}
    for key in ("GAUC", "nDCG@5", "primary"):
        try:
            number = float(value[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(number):
            result[key] = number
    return result or None


def _change(value: Any) -> ChangeSummary | None:
    if not isinstance(value, dict):
        return None
    return ChangeSummary(
        iteration=int(value.get("iteration", 0)),
        candidate_id=str(value.get("candidate_id", "")),
        files=tuple(value.get("files", [])),
        lines_added=int(value.get("lines_added", 0)),
        lines_deleted=int(value.get("lines_deleted", 0)),
        patch_path=value.get("patch_path"),
    )


def _transition(value: dict[str, Any]) -> StageTransition:
    return StageTransition(
        event_id=str(value.get("event_id", "")),
        iteration=int(value.get("iteration", 0)),
        stage=str(value.get("stage", "unknown")),
        status=str(value.get("status", "unknown")),
        started_at=str(value.get("started_at", "")),
        updated_at=str(value.get("updated_at", value.get("started_at", ""))),
        role=value.get("role"),
        experiment_id=value.get("experiment_id"),
        attempt=int(value.get("attempt", 1)),
        objective=str(value.get("objective", "")),
        agent_note=dict(value.get("agent_note") or {}),
        metrics=_metrics(value.get("metrics")),
        change_summary=_change(value.get("change_summary")),
        error=value.get("error"),
        repair=value.get("repair"),
    )


def load_role_passes(run_dir: Path | None, iteration: int) -> tuple[RolePass, ...]:
    if run_dir is None:
        return ()
    passes_dir = run_dir / "passes"
    if not passes_dir.is_dir():
        return ()
    pattern = f"{iteration:03d}_*.json"
    entries: list[tuple[Path, dict[str, Any], str, int]] = []
    for file_path in passes_dir.glob(pattern):
        data = _read_json(file_path, {}) or {}
        if not isinstance(data, dict):
            continue
        res = data.get("result") or {}
        role = str(
            res.get("role")
            or data.get("role")
            or file_path.stem.split("_", 1)[-1]
        )
        try:
            modified_ns = file_path.stat().st_mtime_ns
        except OSError:
            modified_ns = 0
        entries.append((file_path, data, role, modified_ns))

    if entries and all(item[1].get("recorded_at") for item in entries):
        entries.sort(
            key=lambda item: (
                str(item[1]["recorded_at"]),
                item[3],
                item[0].name,
            )
        )
    else:
        entries.sort(
            key=lambda item: (
                ROLE_EXECUTION_ORDER.get(item[2], len(ROLE_EXECUTION_ORDER)),
                item[3],
                item[0].name,
            )
        )

    results: list[RolePass] = []
    for index, (file_path, data, role, _modified_ns) in enumerate(entries):
        res = data.get("result") or {}
        model = str(res.get("model", "unknown"))
        latency = float(res.get("latency_seconds", 0.0))
        usage = dict(res.get("usage") or {})
        res_data = dict(res.get("data") or {})
        sources = tuple(dict(s) for s in res.get("sources", []))
        results.append(
            RolePass(
                sequence=index,
                role=role,
                model=model,
                latency_seconds=latency,
                usage=usage,
                data=res_data,
                sources=sources,
            )
        )
    return tuple(results)


def load_candidate_files(
    run_dir: Path, candidate_dir: str | Path | None
) -> tuple[str | None, str | None]:
    code: str | None = None
    tests: str | None = None
    if candidate_dir:
        raw_path = Path(candidate_dir)
        candidates = (
            (raw_path,)
            if raw_path.is_absolute()
            else (REPO_ROOT / raw_path, run_dir / raw_path)
        )
        for cand_path in dict.fromkeys(path.resolve() for path in candidates):
            _reject_judge_path(cand_path)
            if not cand_path.is_dir():
                continue
            code_file = cand_path / "candidate.py"
            test_file = cand_path / "test_candidate.py"
            if code_file.is_file():
                try:
                    code = code_file.read_text(encoding="utf-8")
                except OSError:
                    pass
            if test_file.is_file():
                try:
                    tests = test_file.read_text(encoding="utf-8")
                except OSError:
                    pass
            break
    return code, tests


def load_gate_result(
    run_dir: Path, summary: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    summary = (
        summary
        if isinstance(summary, dict)
        else (_read_json(run_dir / "summary.json") or {})
    )
    if isinstance(summary.get("gate"), dict):
        return summary["gate"]
    gate_done = _read_json(run_dir / "gate_done.json")
    if isinstance(gate_done, dict):
        return gate_done
    return None


def load_journal_reports(run_dir: Path) -> tuple[str | None, str | None]:
    journal_text: str | None = None
    results_text: str | None = None
    journal_path = run_dir / "journal.md"
    results_path = run_dir / "results.md"
    if journal_path.is_file():
        try:
            journal_text = journal_path.read_text(encoding="utf-8")
        except OSError:
            pass
    if results_path.is_file():
        try:
            results_text = results_path.read_text(encoding="utf-8")
        except OSError:
            pass
    return journal_text, results_text


def _iteration(
    value: dict[str, Any], candidate_dir: str | None = None
) -> IterationSnapshot:
    proposal = value.get("proposal") if isinstance(value.get("proposal"), dict) else {}
    outcome = value.get("outcome") if isinstance(value.get("outcome"), dict) else {}
    manifest = value.get("manifest") if isinstance(value.get("manifest"), dict) else {}
    experiment_id = (
        manifest.get("candidate_id")
        or value.get("experiment_id")
        or f"iteration_{value.get('iteration', 0)}"
    )
    parameters = proposal.get("parameters")
    if not isinstance(parameters, dict):
        parameters = value.get("configuration")
    if not isinstance(parameters, dict):
        parameters = {}
    hypothesis = proposal.get("hypothesis") or value.get("hypothesis") or ""
    family = proposal.get("family") or value.get("kind") or ""
    parent = proposal.get("parent_experiment") or value.get("parent_experiment")
    it_num = int(value.get("iteration", 0))

    return IterationSnapshot(
        iteration=it_num,
        experiment_id=str(experiment_id),
        status=str(value.get("status") or outcome.get("status") or "unknown"),
        hypothesis=str(hypothesis),
        family=str(family),
        action=str(proposal.get("action") or value.get("command_owner") or ""),
        parent_experiment=parent,
        parameters=dict(parameters),
        metrics=_metrics(outcome.get("metrics")),
        duration_seconds=outcome.get("duration_seconds"),
        repairs=int(value.get("repairs", 0)),
        change_summary=_change(value.get("change_summary")),
        agent_notes=dict(value.get("agent_notes") or {}),
        candidate_dir=candidate_dir,
        raw=value,
    )


def _candidate_dir_for_iteration(
    value: dict[str, Any], nodes: list[dict[str, Any]]
) -> str | None:
    manifest = value.get("manifest") if isinstance(value.get("manifest"), dict) else {}
    experiment_id = manifest.get("candidate_id") or value.get("experiment_id")
    if experiment_id:
        match = next(
            (node for node in nodes if str(node.get("experiment_id")) == str(experiment_id)),
            None,
        )
        if match:
            return match.get("candidate_dir")

    proposal = value.get("proposal") if isinstance(value.get("proposal"), dict) else {}
    hypothesis_id = proposal.get("hypothesis_id")
    iteration = int(value.get("iteration", 0))
    candidates: list[dict[str, Any]] = []
    for node in nodes:
        try:
            node_iteration = int(node.get("iteration", -1))
        except (TypeError, ValueError):
            continue
        if node_iteration == iteration and (
            not hypothesis_id or node.get("hypothesis_id") == hypothesis_id
        ):
            candidates.append(node)
    return candidates[0].get("candidate_dir") if len(candidates) == 1 else None


def load_current_activity(run_dir: Path) -> StageTransition | None:
    value = _read_json(_inside(run_dir / "activity.json", run_dir))
    return _transition(value) if isinstance(value, dict) else None


def load_activity_timeline(run_dir: Path) -> tuple[tuple[StageTransition, ...], tuple[str, ...]]:
    records, warnings = _read_jsonl(_inside(run_dir / "activity.jsonl", run_dir))
    return tuple(_transition(item) for item in records), tuple(warnings)


def load_change_summary(run_dir: Path, relative_path: str) -> ChangeSummary | None:
    path = _inside(run_dir / relative_path, run_dir)
    value = _read_json(path)
    return _change(value)


def load_patch_text(run_dir: Path, relative_path: str | None) -> str:
    if not relative_path:
        return ""
    path = _inside(run_dir / relative_path, run_dir)
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def load_live_eda_latest(run_dir: Path) -> EDAArtifact | None:
    eda_dir = _inside(run_dir / "eda", run_dir)
    if not eda_dir.is_dir():
        return None
    latest_path = eda_dir / "latest.json"
    if not latest_path.is_file():
        return None
    value = _read_json(latest_path, {}) or {}
    if not isinstance(value, dict):
        return None
    report = value.get("report") if isinstance(value.get("report"), dict) else {}
    plan = value.get("plan") if isinstance(value.get("plan"), dict) else {}
    try:
        iteration = int(value.get("iteration", 0))
    except (TypeError, ValueError):
        iteration = 0
    raw_findings = (
        value.get("findings")
        if isinstance(value.get("findings"), (list, tuple))
        else (report.get("findings") if isinstance(report.get("findings"), (list, tuple)) else [])
    )
    raw_features = (
        value.get("feature_candidates")
        if isinstance(value.get("feature_candidates"), (list, tuple))
        else (report.get("feature_candidates") if isinstance(report.get("feature_candidates"), (list, tuple)) else [])
    )
    summary = str(report.get("summary", "")) if isinstance(report, dict) else ""
    if not summary and plan:
        summary = str(plan.get("objective", ""))
    return EDAArtifact(
        iteration=iteration,
        path=latest_path,
        status=str(value.get("status", "completed")),
        summary=summary,
        error=value.get("error"),
        plan=plan,
        report=report,
        findings=tuple(dict(item) for item in raw_findings if isinstance(item, dict)),
        feature_candidates=tuple(
            dict(item) for item in raw_features if isinstance(item, dict)
        ),
        raw=value,
    )


def load_eda_artifacts(run_dir: Path) -> tuple[EDAArtifact, ...]:
    eda_dir = _inside(run_dir / "eda", run_dir)
    if not eda_dir.is_dir():
        return ()
    artifacts: list[EDAArtifact] = []
    for path in sorted(tuple(eda_dir.glob("*_eda.json")) + tuple(eda_dir.glob("*_eda_failed.json"))):
        value = _read_json(path, {}) or {}
        if not isinstance(value, dict):
            continue
        report = value.get("report") if isinstance(value.get("report"), dict) else {}
        plan = value.get("plan") if isinstance(value.get("plan"), dict) else {}
        try:
            iteration = int(value.get("iteration", 0))
        except (TypeError, ValueError):
            iteration = 0
        raw_findings = (
            value.get("findings")
            if isinstance(value.get("findings"), (list, tuple))
            else (report.get("findings") if isinstance(report.get("findings"), (list, tuple)) else [])
        )
        raw_features = (
            value.get("feature_candidates")
            if isinstance(value.get("feature_candidates"), (list, tuple))
            else (report.get("feature_candidates") if isinstance(report.get("feature_candidates"), (list, tuple)) else [])
        )
        summary = str(report.get("summary", "")) if isinstance(report, dict) else ""
        if not summary and plan:
            summary = str(plan.get("objective", ""))
        artifacts.append(
            EDAArtifact(
                iteration=iteration,
                path=path,
                status=str(value.get("status", "completed")),
                summary=summary,
                error=value.get("error"),
                plan=plan,
                report=report,
                findings=tuple(dict(item) for item in raw_findings if isinstance(item, dict)),
                feature_candidates=tuple(
                    dict(item) for item in raw_features if isinstance(item, dict)
                ),
                raw=value,
            )
        )
    return tuple(artifacts)



def load_debugger_events(run_dir: Path) -> tuple[DebuggerEvent, ...]:
    run_dir = run_dir.resolve()
    _reject_judge_path(run_dir)
    events: list[DebuggerEvent] = []

    dbg_path = run_dir / "debugger_memory.jsonl"
    if dbg_path.is_file():
        records, _ = _read_jsonl(dbg_path)
        for rec in records:
            events.append(
                DebuggerEvent(
                    iteration=int(rec.get("iteration", 0)),
                    stage=str(rec.get("stage", "debugger")),
                    candidate_id=rec.get("candidate_id"),
                    error_type=rec.get("error_type"),
                    error=str(rec.get("error", "")),
                    lesson=str(rec.get("lesson", "")),
                    event_type=str(rec.get("type", "debugger_memory")),
                )
            )

    rm_path = run_dir / "research_memory.jsonl"
    if rm_path.is_file():
        records, _ = _read_jsonl(rm_path)
        for rec in records:
            rec_type = str(rec.get("type", ""))
            if rec_type in {"role_retry", "eda_error", "controller_error"}:
                events.append(
                    DebuggerEvent(
                        iteration=int(rec.get("iteration", 0)),
                        stage=str(rec.get("label") or rec.get("stage") or rec_type),
                        candidate_id=rec.get("candidate_id"),
                        error_type=rec.get("error_type"),
                        error=str(rec.get("error", "")),
                        lesson=f"Re-prompt #{rec.get('reprompt', 1)}: {rec.get('error', '')}"
                        if rec_type == "role_retry"
                        else str(rec.get("error", "")),
                        event_type=rec_type,
                    )
                )
    return tuple(events)


def load_run_snapshot(
    run_dir: Path,
    official_baseline: float = 0.6016,
    *,
    include_details: bool = True,
) -> RunSnapshot:
    run_dir = run_dir.resolve()
    _reject_judge_path(run_dir)
    summary = _read_json(run_dir / "summary.json", {}) or {}
    state = _read_json(run_dir / "state.json", {}) or {}
    if include_details:
        resources = _read_json(run_dir / "resources.json", {}) or {}
        run_config = _read_json(run_dir / "run_config.json", {}) or {}
        raw_records, iteration_warnings = _read_jsonl(run_dir / "iterations.jsonl")
        deduped: dict[int, dict[str, Any]] = {}
        for item in raw_records:
            it = item.get("iteration")
            if it is not None:
                deduped[it] = item
            else:
                deduped[len(deduped)] = item
        records = list(deduped.values())
        timeline, activity_warnings = load_activity_timeline(run_dir)
    else:
        resources = {}
        run_config = {}
        records, iteration_warnings = [], []
        timeline, activity_warnings = (), ()
    current = load_current_activity(run_dir)
    best = summary.get("best") or {}
    if not best and state:
        best = {
            "experiment_id": state.get("best_experiment_id"),
            "metrics": state.get("best_metrics"),
        }
    if state:
        status = str(state.get("status", "unknown"))
    elif summary:
        status = str(summary.get("status", "completed"))
    elif current and current.status == "active":
        status = "running"
    else:
        status = "incomplete"
    nodes = (
        state.get("nodes") or _read_json(run_dir / "experiment_tree.json", []) or []
    ) if include_details else []
    gate_info = load_gate_result(run_dir, summary) if include_details else None
    journal_md, results_md = (
        load_journal_reports(run_dir) if include_details else (None, None)
    )
    eda_artifacts = load_eda_artifacts(run_dir) if include_details else ()
    live_eda = load_live_eda_latest(run_dir) if include_details else None
    debugger_events = load_debugger_events(run_dir) if include_details else ()

    current_iter = (
        current.iteration
        if current is not None
        else (max((int(item.get("iteration", 0)) for item in records), default=0) if records else 1)
    )
    live_passes = load_role_passes(run_dir, current_iter) if include_details else ()

    iterations = tuple(
        _iteration(item, _candidate_dir_for_iteration(item, nodes)) for item in records
    )
    return RunSnapshot(
        run_id=str(summary.get("run_id") or state.get("run_id") or run_dir.name),
        path=run_dir,
        status=status,
        stop_reason=summary.get("stop_reason") or state.get("stop_reason"),
        started_at=state.get("started_at"),
        best_experiment_id=best.get("experiment_id"),
        best_metrics=_metrics(best.get("metrics")),
        baseline_primary=float(state.get("baseline_primary", official_baseline)),
        iterations=iterations,
        activity=current,
        transitions=timeline,
        resources=dict(resources),
        nodes=tuple(nodes),
        warnings=tuple(iteration_warnings) + tuple(activity_warnings),
        gate_info=gate_info,
        journal_markdown=journal_md,
        results_markdown=results_md,
        run_config=dict(run_config),
        eda_artifacts=eda_artifacts,
        live_role_passes=live_passes,
        live_eda=live_eda,
        debugger_events=debugger_events,
    )



def discover_runs(run_root: Path, official_baseline: float = 0.6016) -> list[RunSnapshot]:
    run_root = run_root.resolve()
    _reject_judge_path(run_root)
    if not run_root.is_dir():
        return []
    directories: list[Path] = []
    for path in run_root.iterdir():
        if not path.is_dir():
            continue
        run_config = _read_json(path / "run_config.json", {}) or {}
        configured_mode = (
            str(run_config.get("mode", "")).lower()
            if isinstance(run_config, dict)
            else ""
        )
        is_research = (
            configured_mode == "research"
            if configured_mode
            else path.name.endswith("_research") or (path / "state.json").is_file()
        )
        if is_research:
            directories.append(path)
    directories.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [
        load_run_snapshot(path, official_baseline, include_details=False)
        for path in directories
    ]


def activity_age_seconds(activity: StageTransition | None) -> float | None:
    if activity is None or not activity.updated_at:
        return None
    try:
        updated = datetime.fromisoformat(activity.updated_at.replace("Z", "+00:00"))
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - updated).total_seconds())
    except ValueError:
        return None


def validate_submission(file_like: BinaryIO | TextIO | bytes | str) -> SubmissionCheck:
    if isinstance(file_like, bytes):
        text = file_like.decode("utf-8-sig")
    elif isinstance(file_like, str):
        text = file_like
    else:
        payload = file_like.read()
        text = payload.decode("utf-8-sig") if isinstance(payload, bytes) else payload
    reader = csv.DictReader(io.StringIO(text))
    expected = ["row_id", "user_id", "video_id", "score"]
    errors: list[str] = []
    warnings: list[str] = []
    if reader.fieldnames != expected:
        errors.append(f"Columns must be exactly {expected} in that order.")
    rows = list(reader)
    row_ids: list[int] = []
    pairs: list[tuple[str, str]] = []
    for index, row in enumerate(rows):
        try:
            row_ids.append(int(row.get("row_id", "")))
        except (TypeError, ValueError):
            errors.append(f"Row {index} has a non-integer row_id.")
        try:
            score = float(row.get("score", ""))
            if not math.isfinite(score):
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"Row {index} has a non-finite score.")
        pairs.append((str(row.get("user_id", "")), str(row.get("video_id", ""))))
    if row_ids != list(range(len(rows))):
        errors.append("row_id must start at zero and increase contiguously without duplicates.")
    duplicate_pairs = len(pairs) - len(set(pairs))
    warnings.append(
        "Judge row count, ID alignment, and duplicate preservation were not checked; this UI never opens judge data."
    )
    return SubmissionCheck(
        valid=not errors,
        row_count=len(rows),
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(warnings),
        duplicate_pairs=duplicate_pairs,
        alignment_checked=False,
    )
