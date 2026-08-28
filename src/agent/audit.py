from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

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
            {"prompt": prompt, "result": result.to_record()},
        )
        self.append_jsonl(self.run_dir / "research_memory.jsonl", result.to_record())

    def record_iteration(self, record: dict[str, Any]) -> None:
        self.append_jsonl(self.run_dir / "iterations.jsonl", record)

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
