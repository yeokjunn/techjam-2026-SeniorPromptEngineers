"""Promote a research run to `runs/final/` so the judges actually receive it.

Deliverables 3 and 4 of the problem statement ask for the per-iteration log (hypothesis, code
diff, metrics, error/recovery events, manual-intervention count) and the final model output plus
a results table. This harness generates all of it correctly -- and then `.gitignore:33` excludes
`runs/*_research/`, so a fresh clone of the repository contains none of it. The exclusions were
written with this in mind (`!runs/final/stdout/`, `!generated_experiments/final/`); nothing had
used the escape hatch yet.

Usage:
    python scripts/promote_final_run.py runs/<id>_research
    git add runs/final generated_experiments/final && git commit

Artifacts (`artifacts/`, the .npz checkpoints) are left behind deliberately: 22 MB of binary
weights that `.gitignore` excludes anyway, and the graded output is `submission.csv`, which is
copied. Everything a judge reads -- journal.md, results.md, DATA_CARD.md, iterations.jsonl, the
role passes, the per-iteration diffs, resources.json -- is included.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
#: Excluded from the promoted copy. `artifacts` is large binary weights that git ignores;
#: the graded artefact is submission.csv, which is kept.
SKIP = {"artifacts", "__pycache__"}


def _require_complete(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(
            f"{run_dir} has no summary.json -- it did not finish. Promote a converged run, "
            "or the results table and stop reason will be missing."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    missing = [
        name
        for name in ("journal.md", "results.md", "iterations.jsonl", "resources.json")
        if not (run_dir / name).is_file()
    ]
    if missing:
        raise SystemExit(f"{run_dir} is missing deliverable files: {', '.join(missing)}")
    return summary


def _copy(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        source, destination, ignore=lambda _, names: [n for n in names if n in SKIP]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="the run to promote, e.g. runs/<id>_research")
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else REPO_ROOT / args.run_dir
    if not run_dir.is_dir():
        raise SystemExit(f"no such run: {run_dir}")
    summary = _require_complete(run_dir)

    _copy(run_dir, REPO_ROOT / "runs" / "final")
    generated = REPO_ROOT / "generated_experiments" / run_dir.name
    if generated.is_dir():
        _copy(generated, REPO_ROOT / "generated_experiments" / "final")

    best = summary.get("best") or {}
    metrics = best.get("metrics") or {}
    print(f"promoted {run_dir.name} -> runs/final")
    print(f"  stop reason : {summary.get('stop_reason')}  ({summary.get('iterations')} iterations)")
    print(f"  best        : {best.get('experiment_id')}")
    for key in ("primary", "select_primary", "report_primary"):
        if key in metrics:
            print(f"  {key:15s}: {metrics[key]:.5f}")
    print(f"  gate        : {(summary.get('gate') or {}).get('status')}")
    submission = REPO_ROOT / "runs" / "final" / "submission.csv"
    if submission.is_file():
        rows = sum(1 for _ in submission.open(encoding="utf-8")) - 1
        print(f"  submission  : {rows:,} rows")
    else:
        print("  submission  : MISSING -- the gate did not produce one", file=sys.stderr)
    print("\nnext:\n  git add runs/final generated_experiments/final && git commit")


if __name__ == "__main__":
    main()
