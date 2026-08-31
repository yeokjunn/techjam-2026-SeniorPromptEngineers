from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from .safety import (
    contained_path,
    validate_family_contract,
    validate_identifier,
    validate_source,
)
from src.evaluation.official import SANITY_CEILING, SANITY_FLOOR

from .types import CandidateManifest, ExperimentOutcome


# The candidate environment is built from scratch, never copied from
# os.environ (.env is loaded there by the LLM layer): this allowlist is what
# keeps provider keys — and everything else — out of LLM-written subprocesses.
PASSTHROUGH_KEYS = (
    "PATH",
    "LANG",
    "LC_ALL",
    "TZ",
    "SYSTEMROOT",
    "SystemRoot",
    "COMSPEC",
    "WINDIR",
)
THREAD_CAP_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _repo_relative(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


class CandidateWorkspace:
    def __init__(self, generated_root: Path, run_id: str, iteration: int, candidate_id: str):
        validate_identifier(run_id, "run ID")
        validate_identifier(candidate_id, "candidate ID")
        self.directory = contained_path(
            generated_root, run_id, f"{iteration:03d}_{candidate_id}"
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        self.code_path = self.directory / "candidate.py"
        self.test_path = self.directory / "test_candidate.py"

    def write(self, manifest: CandidateManifest) -> None:
        validate_source(manifest.code)
        validate_family_contract(manifest.code, manifest.family)
        validate_source(manifest.tests, test_file=True)
        self.code_path.write_text(manifest.code, encoding="utf-8")
        self.test_path.write_text(manifest.tests, encoding="utf-8")
        (self.directory / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8"
        )


class CandidateExecutor:
    def __init__(
        self,
        repo_root: Path,
        data_dir: Path,
        experiment_timeout_seconds: int,
        test_timeout_seconds: int,
        max_output_chars: int = 200_000,
    ):
        self.repo_root = repo_root
        self.data_dir = data_dir
        self.experiment_timeout_seconds = int(experiment_timeout_seconds)
        self.test_timeout_seconds = int(test_timeout_seconds)
        self.max_output_chars = int(max_output_chars)

    def _environment(self, workspace: CandidateWorkspace) -> dict[str, str]:
        """Minimal scratch environment for candidate subprocesses (review C3)."""
        environment = {
            name: os.environ[name]
            for name in PASSTHROUGH_KEYS
            if name in os.environ
        }
        environment.update(
            {
                "PYTHONPATH": str(self.repo_root),
                "PYTHONDONTWRITEBYTECODE": "1",
                # Fixed hash seed: set/dict iteration order otherwise varies per
                # process and makes identical candidates score differently.
                "PYTHONHASHSEED": "0",
                "HOME": str(workspace.directory),
                "TMPDIR": str(workspace.directory),
                "TEMP": str(workspace.directory),
                "TMP": str(workspace.directory),
                "KUAIRAND_DATA_DIR": str(self.data_dir),
                **{name: "1" for name in THREAD_CAP_KEYS},
            }
        )
        assert not any(
            key.startswith(("OPENAI_", "ANTHROPIC_")) for key in environment
        ), "provider keys must never reach candidate subprocesses"
        return environment

    def test(self, workspace: CandidateWorkspace) -> tuple[bool, str]:
        validate_source(workspace.code_path.read_text(encoding="utf-8"))
        validate_source(workspace.test_path.read_text(encoding="utf-8"), test_file=True)
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "unittest", "-v", "test_candidate.py"],
                cwd=workspace.directory,
                env=self._environment(workspace),
                capture_output=True,
                text=True,
                timeout=self.test_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, f"Candidate tests timed out after {self.test_timeout_seconds} seconds."
        output = (completed.stdout + "\n" + completed.stderr)[-self.max_output_chars :]
        (workspace.directory / "tests.log").write_text(output, encoding="utf-8")
        return completed.returncode == 0, output

    def train(
        self,
        iteration: int,
        manifest: CandidateManifest,
        workspace: CandidateWorkspace,
        run_dir: Path,
    ) -> ExperimentOutcome:
        artifact_dir = run_dir / "artifacts" / f"{iteration:03d}_{manifest.candidate_id}"
        stdout_dir = run_dir / "stdout"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        stdout_dir.mkdir(parents=True, exist_ok=True)
        spec_path = artifact_dir / "spec.json"
        result_path = artifact_dir / "result.json"
        spec_path.write_text(
            json.dumps({"parameters": manifest.parameters}, indent=2), encoding="utf-8"
        )
        command = [
            sys.executable,
            "-m",
            "src.experiments.run_candidate",
            "--candidate",
            str(workspace.code_path),
            "--spec",
            str(spec_path),
            "--result",
            str(result_path),
            "--data-dir",
            str(self.data_dir),
            "--artifact-dir",
            str(artifact_dir),
        ]
        stdout_path = stdout_dir / f"{iteration:03d}_{manifest.candidate_id}.stdout.log"
        stderr_path = stdout_dir / f"{iteration:03d}_{manifest.candidate_id}.stderr.log"
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=workspace.directory,
                env=self._environment(workspace),
                capture_output=True,
                text=True,
                timeout=self.experiment_timeout_seconds,
                check=False,
            )
            stdout_path.write_text(completed.stdout[-self.max_output_chars :], encoding="utf-8")
            stderr_path.write_text(completed.stderr[-self.max_output_chars :], encoding="utf-8")
            duration = time.monotonic() - started
            if completed.returncode != 0:
                return ExperimentOutcome(
                    status="failed",
                    metrics=None,
                    duration_seconds=duration,
                    error=f"Candidate exited with code {completed.returncode}: {completed.stderr[-4000:]}",
                    recovery="Eligible for bounded debugger repair.",
                    stdout_path=_repo_relative(stdout_path, self.repo_root),
                    stderr_path=_repo_relative(stderr_path, self.repo_root),
                    command=command,
                    failure_class="crash",
                )
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            metrics = {key: float(value) for key, value in payload["metrics"].items()}
            if any(not math.isfinite(value) for value in metrics.values()):
                raise ValueError("Worker returned a non-finite trusted metric.")
            sanity_class = payload.get("sanity_class")
            if sanity_class is not None:
                # A leaked 0.99 or a learned-nothing 0.40: keep the number in
                # the ledger, never promote it, and not worth a repair round.
                return ExperimentOutcome(
                    status="failed",
                    metrics=metrics,
                    duration_seconds=duration,
                    error=(
                        f"Validation primary {metrics['primary']:.6f} is outside "
                        f"the sanity band [{SANITY_FLOOR}, {SANITY_CEILING}]."
                    ),
                    recovery="Rejected without promotion; previous best retained.",
                    stdout_path=_repo_relative(stdout_path, self.repo_root),
                    stderr_path=_repo_relative(stderr_path, self.repo_root),
                    command=command,
                    failure_class=sanity_class,
                )
            test_scores_status = str(payload.get("test_scores_status", "not_required"))
            if test_scores_status not in ("ok", "not_required"):
                # Keep metrics so the ledger retains the number, but never
                # promote: observe_success only runs for status == "success".
                return ExperimentOutcome(
                    status="failed",
                    metrics=metrics,
                    duration_seconds=duration,
                    error=(
                        "CandidateOutput.test_scores must be a finite 1-D float "
                        "array of length 170588 (one score per "
                        "data.load()['test'] row, same order)."
                    ),
                    recovery="Rejected before promotion; the previous best is intact.",
                    stdout_path=_repo_relative(stdout_path, self.repo_root),
                    stderr_path=_repo_relative(stderr_path, self.repo_root),
                    command=command,
                    failure_class="missing_test_scores",
                )
            return ExperimentOutcome(
                status="success",
                metrics=metrics,
                duration_seconds=duration,
                artifact_path=payload.get("artifact_path"),
                epoch_trace=list(payload.get("training_trace", [])),
                diagnostics=dict(payload.get("diagnostics", {})),
                stdout_path=_repo_relative(stdout_path, self.repo_root),
                stderr_path=_repo_relative(stderr_path, self.repo_root),
                command=command,
                test_scores_path=payload.get("test_scores_path"),
            )
        except subprocess.TimeoutExpired:
            return ExperimentOutcome(
                status="failed",
                metrics=None,
                duration_seconds=time.monotonic() - started,
                error=f"Candidate timed out after {self.experiment_timeout_seconds} seconds.",
                recovery="Candidate process was terminated; previous best remains intact.",
                stdout_path=_repo_relative(stdout_path, self.repo_root),
                stderr_path=_repo_relative(stderr_path, self.repo_root),
                command=command,
                failure_class="timeout",
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            return ExperimentOutcome(
                status="failed",
                metrics=None,
                duration_seconds=time.monotonic() - started,
                error=f"Invalid candidate result: {exc}",
                recovery="Result was rejected; previous best remains intact.",
                command=command,
                failure_class="bad_output",
            )


def repaired_manifest(
    manifest: CandidateManifest, replacement_code: str, replacement_tests: str
) -> CandidateManifest:
    return replace(manifest, code=replacement_code, tests=replacement_tests)
