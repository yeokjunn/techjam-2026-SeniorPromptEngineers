from __future__ import annotations

import json
import os
import difflib
from pathlib import Path
from typing import Any

from .activity import ActivityHandle, redact_text, safe_value, utc_now
from .llm import LLMCallResult


class ResearchAudit:
    def __init__(self, run_dir: Path, resume: bool = False):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=resume)
        self.passes_dir = run_dir / "passes"
        self.passes_dir.mkdir(exist_ok=True)

    @staticmethod
    def write_json_atomic(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    @staticmethod
    def write_text_atomic(path: Path, value: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)

    @staticmethod
    def append_jsonl(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True) + "\n")

    def record_pass(
        self,
        iteration: int,
        role: str,
        prompt: str,
        result: LLMCallResult,
        sequence: int = 0,
    ) -> None:
        self.write_json_atomic(
            self.passes_dir / f"{iteration:03d}_{role}_{sequence}.json",
            {
                "recorded_at": utc_now(),
                "prompt": prompt,
                "result": result.to_record(),
            },
        )
        memory_record = result.to_record()
        memory_record["iteration"] = iteration
        self.append_jsonl(self.run_dir / "research_memory.jsonl", memory_record)

    def record_iteration(self, record: dict[str, Any]) -> None:
        self.append_jsonl(self.run_dir / "iterations.jsonl", record)

    def start_activity(
        self,
        iteration: int,
        stage: str,
        *,
        role: str | None = None,
        experiment_id: str | None = None,
        attempt: int = 1,
        objective: str = "",
        agent_note: dict[str, Any] | None = None,
    ) -> ActivityHandle:
        handle = ActivityHandle.create(
            iteration,
            stage,
            role=role,
            experiment_id=experiment_id,
            attempt=attempt,
            objective=objective,
        )
        event = {
            "event_id": handle.event_id,
            "iteration": handle.iteration,
            "stage": handle.stage,
            "role": handle.role,
            "experiment_id": handle.experiment_id,
            "attempt": handle.attempt,
            "status": "active",
            "started_at": handle.started_at,
            "updated_at": handle.started_at,
            "objective": handle.objective,
            "agent_note": safe_value(agent_note or {}),
        }
        self._record_activity_event(event)
        return handle

    def finish_activity(
        self,
        handle: ActivityHandle,
        *,
        status: str = "completed",
        agent_note: dict[str, Any] | None = None,
        change_summary: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        error: str | None = None,
        repair: str | None = None,
    ) -> dict[str, Any]:
        if status not in {"completed", "failed", "retrying", "interrupted"}:
            raise ValueError(f"Unsupported activity status: {status}")
        completed_at = utc_now()
        event = {
            "event_id": handle.event_id,
            "iteration": handle.iteration,
            "stage": handle.stage,
            "role": handle.role,
            "experiment_id": handle.experiment_id,
            "attempt": handle.attempt,
            "status": status,
            "started_at": handle.started_at,
            "updated_at": completed_at,
            "completed_at": completed_at,
            "objective": handle.objective,
            "agent_note": safe_value(agent_note or {}),
            "change_summary": safe_value(change_summary) if change_summary else None,
            "metrics": safe_value(metrics) if metrics else None,
            "error": redact_text(error) if error else None,
            "repair": redact_text(repair) if repair else None,
        }
        self._record_activity_event(event)
        return event

    def _record_activity_event(self, event: dict[str, Any]) -> None:
        self.write_json_atomic(self.run_dir / "activity.json", event)
        self.append_jsonl(self.run_dir / "activity.jsonl", event)

    def record_candidate_changes(
        self,
        iteration: int,
        candidate_id: str,
        current_sources: dict[str, str],
        parent_sources: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        parent_sources = parent_sources or {}
        patch_parts: list[str] = []
        files: list[dict[str, Any]] = []
        for name in sorted(current_sources):
            before = parent_sources.get(name, "")
            after = current_sources[name]
            diff = list(
                difflib.unified_diff(
                    before.splitlines(),
                    after.splitlines(),
                    fromfile=f"parent/{name}",
                    tofile=f"candidate/{name}",
                    lineterm="",
                )
            )
            added = sum(line.startswith("+") and not line.startswith("+++") for line in diff)
            deleted = sum(line.startswith("-") and not line.startswith("---") for line in diff)
            status = "added" if name not in parent_sources else ("modified" if diff else "unchanged")
            files.append(
                {
                    "path": name,
                    "status": status,
                    "lines_added": added,
                    "lines_deleted": deleted,
                }
            )
            if diff:
                patch_parts.append("\n".join(diff))
        patch_name = f"{iteration:03d}_{candidate_id}.patch"
        patch_path = self.run_dir / "changes" / patch_name
        self.write_text_atomic(patch_path, "\n\n".join(patch_parts) + ("\n" if patch_parts else ""))
        summary = {
            "iteration": int(iteration),
            "candidate_id": candidate_id,
            "files": files,
            "lines_added": sum(item["lines_added"] for item in files),
            "lines_deleted": sum(item["lines_deleted"] for item in files),
            "patch_path": f"changes/{patch_name}",
        }
        self.write_json_atomic(
            self.run_dir / "changes" / f"{iteration:03d}_{candidate_id}.json", summary
        )
        return summary

    def save_state(self, state: dict[str, Any]) -> None:
        self.write_json_atomic(self.run_dir / "state.json", state)
        self.write_json_atomic(self.run_dir / "experiment_tree.json", state.get("nodes", []))
        self.write_json_atomic(
            self.run_dir / "resources.json",
            {
                "token_usage": state.get("token_usage", {}),
                "wall_clock_seconds": state.get("wall_clock_seconds", 0.0),
                "training_attempts": state.get("training_attempts", 0),
                "iteration_count": state.get("iteration_count", 0),
                "manual_interventions": state.get("manual_interventions", 0),
                "gpu_hours": 0.0,
            },
        )
