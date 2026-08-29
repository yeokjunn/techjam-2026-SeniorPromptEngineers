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
    IterationSnapshot,
    RolePass,
    RunSnapshot,
    StageTransition,
    SubmissionCheck,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


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
    files = sorted(passes_dir.glob(pattern))
    results: list[RolePass] = []
    for index, file_path in enumerate(files):
        data = _read_json(file_path, {}) or {}
        prompt = str(data.get("prompt", ""))
        res = data.get("result") or {}
        role = str(res.get("role") or data.get("role") or file_path.stem.split("_", 1)[-1])
        model = str(res.get("model", "unknown"))
        latency = float(res.get("latency_seconds", 0.0))
        usage = dict(res.get("usage") or {})
        res_data = dict(res.get("data") or {})
        sources = tuple(dict(s) for s in res.get("sources", []))
        tool_calls = tuple(dict(tc) for tc in res.get("tool_calls", []))
        retries = int(res.get("retries", 0))
        results.append(
            RolePass(
                sequence=index,
                role=role,
                prompt=prompt,
                model=model,
                latency_seconds=latency,
                usage=usage,
                data=res_data,
                sources=sources,
                tool_calls=tool_calls,
                retries=retries,
            )
        )
    return tuple(results)


def load_candidate_files(
    run_dir: Path, candidate_dir: str | Path | None
) -> tuple[str | None, str | None]:
    code: str | None = None
    tests: str | None = None
    if candidate_dir:
        cand_path = Path(candidate_dir)
        if not cand_path.is_absolute():
            cand_path = REPO_ROOT / cand_path
        if cand_path.is_dir():
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
    return code, tests


def load_gate_result(run_dir: Path) -> dict[str, Any] | None:
    gate_done = _read_json(run_dir / "gate_done.json")
    if isinstance(gate_done, dict):
        return gate_done
    summary = _read_json(run_dir / "summary.json") or {}
    if isinstance(summary.get("gate"), dict):
        return summary["gate"]
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


def _iteration(value: dict[str, Any], run_dir: Path | None = None) -> IterationSnapshot:
    proposal = value.get("proposal") if isinstance(value.get("proposal"), dict) else {}
    outcome = value.get("outcome") if isinstance(value.get("outcome"), dict) else {}
    manifest = value.get("manifest") if isinstance(value.get("manifest"), dict) else {}
    experiment_id = (
        manifest.get("candidate_id")
        or value.get("experiment_id")
        or f"iteration_{value.get('iteration', 0)}"
    )
    parameters = proposal.get("parameters") or value.get("configuration") or {}
    hypothesis = proposal.get("hypothesis") or value.get("hypothesis") or ""
    family = proposal.get("family") or value.get("kind") or ""
    parent = proposal.get("parent_experiment") or value.get("parent_experiment")
    it_num = int(value.get("iteration", 0))

    role_passes = load_role_passes(run_dir, it_num)
    code = manifest.get("code")
    tests = manifest.get("tests")
    if (not code or not tests) and run_dir:
        cand_dir = value.get("candidate_dir")
        loaded_code, loaded_tests = load_candidate_files(run_dir, cand_dir)
        code = code or loaded_code
        tests = tests or loaded_tests

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
        candidate_code=code,
        candidate_tests=tests,
        role_passes=role_passes,
        raw=value,
    )


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


def load_run_snapshot(run_dir: Path, official_baseline: float = 0.6016) -> RunSnapshot:
    run_dir = run_dir.resolve()
    _reject_judge_path(run_dir)
    summary = _read_json(run_dir / "summary.json", {}) or {}
    state = _read_json(run_dir / "state.json", {}) or {}
    resources = _read_json(run_dir / "resources.json", {}) or {}
    run_config = _read_json(run_dir / "run_config.json", {}) or {}
    records, iteration_warnings = _read_jsonl(run_dir / "iterations.jsonl")
    timeline, activity_warnings = load_activity_timeline(run_dir)
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
    nodes = state.get("nodes") or _read_json(run_dir / "experiment_tree.json", []) or []
    gate_info = load_gate_result(run_dir)
    journal_md, results_md = load_journal_reports(run_dir)
    return RunSnapshot(
        run_id=str(summary.get("run_id") or state.get("run_id") or run_dir.name),
        path=run_dir,
        status=status,
        stop_reason=summary.get("stop_reason") or state.get("stop_reason"),
        started_at=state.get("started_at"),
        best_experiment_id=best.get("experiment_id"),
        best_metrics=_metrics(best.get("metrics")),
        baseline_primary=float(state.get("baseline_primary", official_baseline)),
        iterations=tuple(_iteration(item, run_dir=run_dir) for item in records),
        activity=current,
        transitions=timeline,
        resources=dict(resources),
        nodes=tuple(nodes),
        warnings=tuple(iteration_warnings) + tuple(activity_warnings),
        gate_info=gate_info,
        journal_markdown=journal_md,
        results_markdown=results_md,
        run_config=dict(run_config),
    )


def discover_runs(run_root: Path, official_baseline: float = 0.6016) -> list[RunSnapshot]:
    run_root = run_root.resolve()
    _reject_judge_path(run_root)
    if not run_root.is_dir():
        return []
    directories = [path for path in run_root.iterdir() if path.is_dir()]
    directories.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [load_run_snapshot(path, official_baseline) for path in directories]


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
