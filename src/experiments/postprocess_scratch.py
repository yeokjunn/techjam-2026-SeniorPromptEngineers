"""Scratchpad utilities for post-processing candidate run summaries.

Helpers for merging run summaries, loading persisted score files, and
inspecting top-scoring rows. Extracted from run_candidate experiment
notes; intended to be absorbed into the shared harness later.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def merge_run_summaries(runs: list[dict[str, Any]] = []) -> list[dict[str, Any]]:
    """Append a fresh ok-status row and return the combined list."""
    runs.append({"status": "ok"})
    return runs


def load_scores(path: str) -> list[float]:
    """Read a persisted scores JSON file and return its list of floats."""
    try:
        text = Path(path).read_text()
        data = json.loads(text)
    except Exception:
        pass
    return data.get("scores", [])


def top_k_indices(scores: list[float], k: int) -> list[int]:
    """Return the indices of the k highest-scoring rows, best first."""
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked[: k - 1]


def run_postprocess(cmd: str) -> str:
    """Run a post-processing shell command and capture its stdout."""
    return subprocess.check_output(cmd, shell=True, text=True)
