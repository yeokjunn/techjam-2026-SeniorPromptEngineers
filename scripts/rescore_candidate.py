"""Re-score a run's winning candidate on several seeds, to attach error bars to the result.

Why this is separate from the loop. Scoring every candidate on N seeds triples the time per
iteration, and a run that never finishes produces no deliverable at all. The +/- figure is only
ever quoted for the candidate actually submitted, so it is enough to re-score that one -- the
loop stays single-seed and fast, and the reported number still carries a spread.

What it answers, in the form the judging criteria ask for:

  * the submitted candidate's primary as a mean over N seeds, with its sample std
  * the same on the reporting half of validation, which never informed early stopping
  * the delta against the official baseline measured on that same half, which is the only
    apples-to-apples comparison available (the halves differ in difficulty by ~0.005)

Usage:
    python scripts/rescore_candidate.py runs/final --seeds 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.holdout import split_users  # noqa: E402
from src.evaluation.official import (  # noqa: E402
    load_train_valid,
    official_evaluate,
    starter_modules,
)
from src.experiments.contracts import CandidateContext  # noqa: E402
from src.experiments.run_candidate import _load_candidate, _score_run  # noqa: E402


def _locate_candidate(run_dir: Path, experiment_id: str) -> tuple[Path, dict]:
    """Find the generated candidate for an experiment id, and its recorded parameters."""
    generated = REPO_ROOT / "generated_experiments" / run_dir.name
    matches = sorted(generated.glob(f"*_{experiment_id}/candidate.py"))
    if not matches:
        raise SystemExit(
            f"no generated candidate for {experiment_id!r} under {generated}. "
            "Promote the run first, or pass the run directory it was generated in."
        )
    candidate = matches[0]
    manifest = json.loads((candidate.parent / "manifest.json").read_text(encoding="utf-8"))
    return candidate, manifest.get("parameters", {})


def _baseline_report_primary(valid_users, valid_y, scores_by_run: dict) -> float | None:
    """The official baseline's primary on the reporting half, if a baseline run recorded it."""
    for summary_path in sorted(
        (REPO_ROOT / "runs").glob("*_baseline/summary.json"), reverse=True
    ):
        metrics = (json.loads(summary_path.read_text(encoding="utf-8")).get("best") or {}).get(
            "metrics"
        ) or {}
        if "report_primary" in metrics:
            return float(metrics["report_primary"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="a promoted run, e.g. runs/final")
    parser.add_argument("--seeds", type=int, default=3, help="how many consecutive seeds")
    parser.add_argument("--data-dir", type=Path, default=Path("data/KuaiRand-Pure/data"))
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else REPO_ROOT / args.run_dir
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    best = summary.get("best") or {}
    experiment_id = best.get("experiment_id")
    if not experiment_id:
        raise SystemExit(f"{run_dir} records no best experiment")
    candidate_path, parameters = _locate_candidate(run_dir, experiment_id)
    base_seed = int(parameters.get("seed", 0))
    seeds = [base_seed + offset for offset in range(max(1, args.seeds))]

    print(f"re-scoring {experiment_id} on seeds {seeds}")
    print(f"  candidate: {candidate_path.relative_to(REPO_ROOT)}")

    splits = load_train_valid(args.data_dir)
    data_module, _, _ = starter_modules()
    encoded, dimension = data_module.encode(splits)
    train_x, train_y, train_users = encoded["train"]
    valid_x, valid_y, valid_users = encoded["valid"]
    selection, reporting = split_users(valid_users)
    select_users = [user for user, keep in zip(valid_users, selection) if keep]
    report_users = [user for user, keep in zip(valid_users, reporting) if keep]

    def evaluate_validation(scores):
        scores = np.asarray(scores)
        return official_evaluate(select_users, valid_y[selection], scores[selection])

    context = CandidateContext(
        train_x=train_x,
        train_y=train_y,
        train_users=tuple(train_users),
        valid_x=valid_x,
        valid_users=tuple(valid_users),
        field_dimension=dimension,
        evaluate_validation=evaluate_validation,
    )
    module = _load_candidate(candidate_path)

    full: list[float] = []
    report: list[float] = []
    for seed in seeds:
        started = time.monotonic()
        output = module.run(context, {**parameters, "seed": seed})
        metrics = _score_run(output, valid_users, valid_y)
        scores = np.asarray(output.validation_scores)
        report_primary = official_evaluate(
            report_users, valid_y[reporting], scores[reporting]
        )["primary"]
        full.append(metrics["primary"])
        report.append(report_primary)
        print(
            f"  seed {seed}: full={metrics['primary']:.5f}  report={report_primary:.5f}"
            f"  ({time.monotonic() - started:.0f}s)",
            flush=True,
        )

    def summarise(values: list[float]) -> str:
        mean = statistics.mean(values)
        if len(values) < 2:
            return f"{mean:.5f} (single seed, no spread)"
        return f"{mean:.5f} +/- {statistics.stdev(values):.5f}"

    print("\n=== result ===")
    print(f"  full validation : {summarise(full)}")
    print(f"  reporting half  : {summarise(report)}")

    baseline_report = _baseline_report_primary(valid_users, valid_y, {})
    if baseline_report is None:
        print(
            "\n  no baseline run records report_primary -- re-run the baseline ladder so the "
            "delta can be measured on the same half."
        )
    else:
        delta = statistics.mean(report) - baseline_report
        print(f"\n  baseline on the same half : {baseline_report:.5f}")
        print(f"  DELTA (apples to apples)  : {delta:+.5f}")
        if len(report) > 1:
            spread = statistics.stdev(report)
            print(f"  candidate seed std        : {spread:.5f}")
            print(
                "  -> the delta is "
                f"{abs(delta) / spread:.1f}x the candidate's own seed spread"
                if spread
                else ""
            )


if __name__ == "__main__":
    main()
