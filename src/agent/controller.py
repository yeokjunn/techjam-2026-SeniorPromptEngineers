from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .audit import ResearchAudit
from .convergence import ConvergenceTracker
from .logger import RunLogger
from .proposer import ConfigProposer
from .reflector import reflect
from .runner import ExperimentRunner
from .types import RunState


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _count_interventions(run_dir: Path) -> int:
    """How many manual interventions ``<run_dir>/interventions.jsonl`` records.

    The count every writer derives rather than increments (I10): the file is the
    falsifiable evidence a judge can read, and a live loop rewriting ``state.json``
    can never clobber a count it recomputes from disk.
    """
    path = run_dir / "interventions.jsonl"
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _source_manifest() -> dict:
    files = sorted((REPO_ROOT / "src").rglob("*.py"))
    entries = []
    combined = hashlib.sha256()
    for path in files:
        relative = path.relative_to(REPO_ROOT).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append({"path": relative, "sha256": digest})
        combined.update(relative.encode("utf-8"))
        combined.update(digest.encode("ascii"))
    return {"revision": combined.hexdigest(), "files": entries}


def run_agent(config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_dir = _resolve_repo_path(config["data_dir"])
    required = (
        "video_features_basic_pure.csv",
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    )
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"KuaiRand-Pure data directory is incomplete: {data_dir}; missing {missing}"
        )

    prefix = str(config.get("run_id_prefix", ""))
    run_id = prefix + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_baseline"
    run_root = _resolve_repo_path(config.get("run_root", "runs"))
    logger = RunLogger(run_root / run_id)
    shutil.copy2(config_path, logger.run_dir / "run_config.json")
    source_manifest = _source_manifest()
    logger.write_json(logger.run_dir / "source_manifest.json", source_manifest)

    budgets = config["budgets"]
    convergence_config = config["convergence"]
    tracker = ConvergenceTracker(
        epsilon=float(convergence_config["epsilon"]),
        patience=int(convergence_config["patience"]),
    )
    proposer = ConfigProposer(config["experiments"])
    runner = ExperimentRunner(
        repo_root=REPO_ROOT,
        data_dir=data_dir,
        timeout_seconds=int(budgets["experiment_timeout_seconds"]),
    )

    history: list[dict] = []
    best_record: dict | None = None
    started = time.monotonic()
    stop_reason = "proposal_queue_exhausted"

    for iteration in range(1, int(budgets["max_iterations"]) + 1):
        elapsed = time.monotonic() - started
        if elapsed >= float(budgets["max_wall_clock_seconds"]):
            stop_reason = "wall_clock_budget_reached"
            break
        spec = proposer.propose(history)
        if spec is None:
            break
        interventions_before = _count_interventions(logger.run_dir)

        previous_best = None if best_record is None else best_record["metrics"]["primary"]
        outcome = runner.run(iteration, spec, logger.run_dir)
        reflection = reflect(
            spec,
            outcome,
            previous_best,
            float(config["official_validation_baseline"]),
        )
        record = {
            "iteration": iteration,
            "parent_experiment": None if best_record is None else best_record["experiment_id"],
            "experiment_id": spec.experiment_id,
            "kind": spec.kind,
            "hypothesis": spec.hypothesis,
            "configuration": spec.parameters,
            "code_diff": spec.code_change,
            "code_revision": source_manifest["revision"],
            "command_owner": "deterministic_config_proposer",
            "outcome": outcome.to_dict(),
            "reflection": reflection,
            "manual_intervention": _count_interventions(logger.run_dir)
            > interventions_before,
            "llm_tokens": 0,
        }
        history.append(record)
        logger.append_iteration(record)

        if outcome.status == "success" and outcome.metrics:
            if previous_best is None or outcome.metrics["primary"] > previous_best:
                best_record = {
                    "experiment_id": spec.experiment_id,
                    "iteration": iteration,
                    "metrics": outcome.metrics,
                    "artifact_path": outcome.artifact_path,
                }
                logger.write_json(logger.run_dir / "best.json", best_record)
            if tracker.observe(outcome.metrics["primary"]):
                stop_reason = "converged"
                break

        print(
            f"iteration={iteration} experiment={spec.experiment_id} "
            f"status={outcome.status} metrics={outcome.metrics}"
        )
    else:
        stop_reason = "iteration_budget_reached"

    summary = {
        "run_id": run_id,
        "config_name": config["name"],
        "stop_reason": stop_reason,
        "iterations": len(history),
        "successful_iterations": sum(
            item["outcome"]["status"] == "success" for item in history
        ),
        "manual_interventions": _count_interventions(logger.run_dir),
        "llm_tokens": 0,
        "wall_clock_seconds": time.monotonic() - started,
        "best": best_record,
    }
    logger.write_json(logger.run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return logger.run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the KuaiRand experiment agent.")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["run", "intervene"],
        default="run",
        help="Start a run (default) or record a manual intervention against one.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.json"),
        help="Path to an agent run configuration.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume an existing autonomous research run directory.",
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=None,
        help="Run directory the intervention applies to.",
    )
    parser.add_argument(
        "--reason",
        default="",
        help="Why the run was touched by hand.",
    )
    args = parser.parse_args()
    if args.command == "intervene":
        if args.run is None:
            parser.error("intervene requires --run <run_dir>")
        if not args.reason:
            parser.error('intervene requires --reason "..."')
        run_dir = args.run if args.run.is_absolute() else REPO_ROOT / args.run
        if not run_dir.is_dir():
            print(f"No such run directory: {run_dir}", file=sys.stderr)
            sys.exit(2)
        ResearchAudit.append_jsonl(
            run_dir / "interventions.jsonl",
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "run_id": run_dir.name,
                "reason": args.reason,
            },
        )
        state_path = run_dir / "state.json"
        if state_path.is_file():
            # One call refreshes state.json, experiment_tree.json and resources.json.
            state = RunState.from_dict(json.loads(state_path.read_text(encoding="utf-8")))
            state.manual_interventions = _count_interventions(run_dir)
            ResearchAudit(run_dir, resume=True).save_state(state.to_dict())
        return
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("mode") == "research":
        from .research_controller import run_research_agent

        resume_dir = None
        if args.resume is not None:
            resume_dir = args.resume if args.resume.is_absolute() else REPO_ROOT / args.resume
        run_research_agent(config_path, resume_dir=resume_dir)
    else:
        if args.resume is not None:
            raise ValueError("--resume is supported only for research-mode configs.")
        run_agent(config_path)


if __name__ == "__main__":
    main()
