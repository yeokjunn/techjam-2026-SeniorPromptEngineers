"""Submission gate: turn the best candidate's persisted test scores into a
``submission.csv`` that passes the organizers' own ``submit.py --check``
(review item C1). The contract (I-1, frozen in Step 0):

    run_gate(run_dir, node_dir, data_dir, kit_dir) -> GateResult

``status`` is ``"ok"``, ``"error"``, or ``"not_implemented"`` (pre-fill).
``run_gate`` never raises: expected failures return ``status="error"`` with a
``details["reason"]``, and the wrapper converts anything unexpected the same
way, so a gate fault can never cost the run its ``summary.json``. ``details``
carries no test metric of any kind — the scored delta comes later, from the
organizers, against this artifact.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.evaluation.official import load_test_meta


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_TIMEOUT_SECONDS = 600
OUTPUT_TAIL_CHARS = 2000


@dataclass
class GateResult:
    status: str
    submission_path: str | None = None
    details: dict = field(default_factory=dict)


def _abs(path: Path) -> Path:
    """Defence in depth: accept repo-relative paths (e.g. hand-run calls)."""
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


@lru_cache(maxsize=1)
def _kit_python() -> str:
    """Interpreter for the kit check: system python when numpy lives there.

    The kit's ``load()`` materialises the test *labels* (``data.py:23-25``), so
    the check runs in a throwaway numpy-only process whose only output is a
    pass/fail line — the labels never enter the harness process. Falls back to
    ``sys.executable`` (the venv) when system python lacks numpy.
    """
    system = Path("/usr/bin/python3")
    if system.is_file():
        probe = subprocess.run(
            [str(system), "-c", "import numpy"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if probe.returncode == 0:
            return str(system)
    return sys.executable


def _minimal_environment() -> dict[str, str]:
    """Scratch environment for the check subprocess (T4's key set): no secrets."""
    environment = {
        name: os.environ[name]
        for name in (
            "PATH",
            "LANG",
            "LC_ALL",
            "TZ",
            "SYSTEMROOT",
            "SystemRoot",
            "COMSPEC",
            "WINDIR",
        )
        if name in os.environ
    }
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[name] = "1"
    return environment


def _error(reason: str, **extra) -> GateResult:
    return GateResult(status="error", details={"reason": reason, **extra})


def _run_gate(run_dir: Path, node_dir: Path, data_dir: Path, kit_dir: Path) -> GateResult:
    marker = run_dir / "gate_done.json"
    if marker.is_file():
        stored = json.loads(marker.read_text(encoding="utf-8"))
        details = dict(stored.get("details", {}))
        details["reused"] = True
        return GateResult(
            status=str(stored.get("status", "error")),
            submission_path=stored.get("submission_path"),
            details=details,
        )

    candidates = (
        node_dir / "test_scores.npy",
        run_dir / "artifacts" / node_dir.name / "test_scores.npy",
    )
    scores_path = next((path for path in candidates if path.is_file()), None)
    if scores_path is None:
        return _error(
            "missing_test_scores",
            searched=[_repo_relative(path) for path in candidates],
        )
    scores = np.load(scores_path)
    meta = load_test_meta(data_dir).meta
    if (
        scores.ndim != 1
        or not np.all(np.isfinite(scores))
        or len(scores) != len(meta)
    ):
        return _error(
            "bad_test_scores",
            got_rows=int(scores.size),
            expected_rows=len(meta),
        )

    submission_path = run_dir / "submission.csv"
    run_dir.mkdir(parents=True, exist_ok=True)
    temporary = submission_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        # Original id strings from the split, never re-encoded ints; %.9g keeps
        # the 9 significant digits a float64 score carries (float32's ~7 would
        # let the formatting itself create ranking ties).
        for (row_id, user_id, video_id), score in zip(meta, scores):
            writer.writerow([row_id, user_id, video_id, f"{float(score):.9g}"])
    os.replace(temporary, submission_path)

    check_script = kit_dir / "submit.py"
    if not check_script.is_file():
        return _error("kit_unavailable", searched=_repo_relative(check_script))
    interpreter = _kit_python()
    try:
        completed = subprocess.run(
            [
                interpreter,
                str(check_script),
                "--check",
                "--split",
                "test",
                "--data_dir",
                str(data_dir),
                str(submission_path),
            ],
            cwd=str(kit_dir),
            env=_minimal_environment(),
            capture_output=True,
            text=True,
            timeout=CHECK_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _error(
            "check_failed",
            error=f"submit.py --check timed out after {CHECK_TIMEOUT_SECONDS}s.",
        )
    if completed.returncode != 0:
        tail = ((completed.stdout or "") + (completed.stderr or ""))[-OUTPUT_TAIL_CHARS:]
        return _error("check_failed", exit_code=completed.returncode, output=tail)

    details = {
        "rows": len(meta),
        "sha256": hashlib.sha256(submission_path.read_bytes()).hexdigest(),
        "check_stdout": (completed.stdout or "")[-OUTPUT_TAIL_CHARS:],
        "checked_with": interpreter,
        "scored": False,
    }
    result = GateResult(
        status="ok",
        submission_path=_repo_relative(submission_path),
        details=details,
    )
    marker.write_text(
        json.dumps(
            {
                "status": result.status,
                "submission_path": result.submission_path,
                "details": details,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def run_gate(run_dir: Path, node_dir: Path, data_dir: Path, kit_dir: Path) -> GateResult:
    """Validate and materialise the submission for the best node; never raises.

    Positional-or-keyword until A converts the controller call site to the
    keyword-only form (I-1); both styles accept the same four arguments.
    """
    try:
        return _run_gate(
            _abs(Path(run_dir)), _abs(Path(node_dir)), _abs(Path(data_dir)), _abs(Path(kit_dir))
        )
    except Exception as exc:  # noqa: BLE001 - the gate must never cost the run its summary
        return GateResult(
            status="error",
            details={"reason": "unexpected", "error": f"{type(exc).__name__}: {exc}"},
        )
