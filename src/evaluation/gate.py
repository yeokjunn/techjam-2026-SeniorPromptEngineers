"""Submission gate stub; owner B fills this in (review item C1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GateResult:
    status: str
    submission_path: str | None = None
    details: dict = field(default_factory=dict)


def run_gate(run_dir: Path, node_dir: Path, data_dir: Path, kit_dir: Path) -> GateResult:
    """Score the best candidate on the official test split; not implemented yet."""
    return GateResult(status="not_implemented")
