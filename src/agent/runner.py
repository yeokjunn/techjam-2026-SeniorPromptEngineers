from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

from .types import ExperimentOutcome, ExperimentSpec


class ExperimentRunner:
    def __init__(self, repo_root: Path, data_dir: Path, timeout_seconds: int):
        self.repo_root = repo_root
        self.data_dir = data_dir
        self.timeout_seconds = timeout_seconds

    def run(self, iteration: int, spec: ExperimentSpec, run_dir: Path) -> ExperimentOutcome:
        work_dir = run_dir / "artifacts" / f"{iteration:03d}_{spec.experiment_id}"
        stdout_dir = run_dir / "stdout"
        work_dir.mkdir(parents=True, exist_ok=True)
        stdout_dir.mkdir(parents=True, exist_ok=True)
        spec_path = work_dir / "spec.json"
        result_path = work_dir / "result.json"
        spec_path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")

        command = [
            sys.executable,
            "-m",
            "src.experiments.run_baseline",
            "--spec",
            str(spec_path),
            "--result",
            str(result_path),
            "--data-dir",
            str(self.data_dir),
            "--artifact-dir",
            str(work_dir),
        ]
        started = time.monotonic()
        stdout_path = stdout_dir / f"{iteration:03d}_{spec.experiment_id}.stdout.log"
        stderr_path = stdout_dir / f"{iteration:03d}_{spec.experiment_id}.stderr.log"
        try:
            completed = subprocess.run(
                command,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            stdout_path.write_text(completed.stdout, encoding="utf-8")
            stderr_path.write_text(completed.stderr, encoding="utf-8")
            duration = time.monotonic() - started
            if completed.returncode != 0:
                return ExperimentOutcome(
                    status="failed",
                    metrics=None,
                    duration_seconds=duration,
                    error=f"Experiment process exited with code {completed.returncode}.",
                    recovery="Captured stdout/stderr and retained the previous best checkpoint.",
                    stdout_path=str(stdout_path.relative_to(self.repo_root)),
                    stderr_path=str(stderr_path.relative_to(self.repo_root)),
                    command=command,
                )
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            metrics = {name: float(value) for name, value in payload["metrics"].items()}
            if not metrics or any(not math.isfinite(value) for value in metrics.values()):
                raise ValueError("Worker returned missing or non-finite metrics.")
            return ExperimentOutcome(
                status="success",
                metrics=metrics,
                duration_seconds=duration,
                artifact_path=payload.get("artifact_path"),
                epoch_trace=payload.get("epoch_trace", []),
                stdout_path=str(stdout_path.relative_to(self.repo_root)),
                stderr_path=str(stderr_path.relative_to(self.repo_root)),
                command=command,
            )
        except subprocess.TimeoutExpired as exc:
            stdout_path.write_text(exc.stdout or "", encoding="utf-8")
            stderr_path.write_text(exc.stderr or "", encoding="utf-8")
            return ExperimentOutcome(
                status="failed",
                metrics=None,
                duration_seconds=time.monotonic() - started,
                error=f"Experiment timed out after {self.timeout_seconds} seconds.",
                recovery="Terminated the experiment and retained the previous best checkpoint.",
                stdout_path=str(stdout_path.relative_to(self.repo_root)),
                stderr_path=str(stderr_path.relative_to(self.repo_root)),
                command=command,
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            return ExperimentOutcome(
                status="failed",
                metrics=None,
                duration_seconds=time.monotonic() - started,
                error=f"Invalid experiment result: {exc}",
                recovery="Rejected the result and retained the previous best checkpoint.",
                stdout_path=str(stdout_path.relative_to(self.repo_root)),
                stderr_path=str(stderr_path.relative_to(self.repo_root)),
                command=command,
            )
