from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audit import ResearchAudit
from .candidate_runner import CandidateExecutor, CandidateWorkspace, repaired_manifest
from .catalog import MethodCatalog
from .controller import REPO_ROOT, _resolve_repo_path, _source_manifest, run_agent
from .llm import LLMProvider, OpenAIResponsesProvider
from .policy import SearchPolicy, coverage_complete, required_family, sanitize_parameters
from .roles import ResearchRoles
from .types import (
    CandidateManifest,
    CriticDecision,
    ExperimentNode,
    ResearchDecision,
    RunState,
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_valid_baseline(run_root: Path, threshold: float) -> dict[str, Any] | None:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for path in run_root.glob("*/summary.json"):
        try:
            summary = _load_json(path)
            best = summary.get("best") or {}
            primary = float((best.get("metrics") or {}).get("primary", float("-inf")))
            if best.get("experiment_id") == "official_fm_seed0" and primary >= threshold:
                candidates.append((path.stat().st_mtime, {**summary, "summary_path": str(path)}))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _ensure_baseline(config: dict[str, Any]) -> dict[str, Any]:
    run_root = _resolve_repo_path(config.get("run_root", "runs"))
    official = float(config["official_validation_baseline"])
    baseline = _latest_valid_baseline(run_root, official - 0.002)
    if baseline is not None:
        return baseline
    baseline_config = _resolve_repo_path(config.get("baseline_config", "configs/baseline.json"))
    run_dir = run_agent(baseline_config)
    summary = _load_json(run_dir / "summary.json")
    primary = float(summary["best"]["metrics"]["primary"])
    if primary < official - 0.002:
        raise RuntimeError(
            f"Official FM baseline gate failed: {primary:.4f} < {official - 0.002:.4f}"
        )
    return {**summary, "summary_path": str(run_dir / "summary.json")}


class ResearchLoop:
    def __init__(
        self,
        config: dict[str, Any],
        config_path: Path,
        provider: LLMProvider | None = None,
        resume_dir: Path | None = None,
        baseline_summary: dict[str, Any] | None = None,
    ):
        self.config = config
        self.config_path = config_path
        self.data_dir = _resolve_repo_path(config["data_dir"])
        self.generated_root = _resolve_repo_path(config.get("generated_root", "generated_experiments"))
        self.run_root = _resolve_repo_path(config.get("run_root", "runs"))
        self.budgets = config["budgets"]
        self.convergence = config["convergence"]
        llm_config = config["llm"]
        self.provider = provider or OpenAIResponsesProvider(llm_config)
        self.baseline_summary = baseline_summary or _ensure_baseline(config)
        baseline_primary = float(self.baseline_summary["best"]["metrics"]["primary"])

        if resume_dir is None:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_research")
            self.run_dir = self.run_root / run_id
            self.audit = ResearchAudit(self.run_dir)
            self.state = RunState(
                run_id=run_id,
                status="running",
                started_at=datetime.now(timezone.utc).isoformat(),
                baseline_primary=baseline_primary,
                meaningful_best=baseline_primary,
                best_metrics=dict(self.baseline_summary["best"]["metrics"]),
                best_experiment_id="official_fm_seed0",
                best_artifact_path=self.baseline_summary["best"].get("artifact_path"),
            )
            shutil.copy2(config_path, self.run_dir / "run_config.json")
            self.audit.write_json_atomic(self.run_dir / "source_manifest.json", _source_manifest())
            self.audit.write_json_atomic(self.run_dir / "baseline_gate.json", self.baseline_summary)
            self.audit.write_json_atomic(self.run_dir / "interventions.json", [])
        else:
            self.run_dir = resume_dir.resolve()
            self.audit = ResearchAudit(self.run_dir, resume=True)
            frozen_config = _load_json(self.run_dir / "run_config.json")
            if frozen_config != config:
                raise ValueError("Resume config differs from the run's frozen configuration.")
            stored_manifest = _load_json(self.run_dir / "source_manifest.json")
            if stored_manifest.get("revision") != _source_manifest().get("revision"):
                raise ValueError("Project-owned source changed; start a new research run instead of resuming.")
            self.state = RunState.from_dict(_load_json(self.run_dir / "state.json"))
            self.state.status = "running"
            self.state.stop_reason = None

        catalog = MethodCatalog.load(_resolve_repo_path(config["method_catalog"]))
        self.roles = ResearchRoles(
            self.provider,
            catalog,
            self.audit,
            max_total_tokens=int(llm_config["max_total_tokens"]),
        )
        self.policy = SearchPolicy(
            epsilon=float(self.convergence["epsilon"]),
            patience=int(self.convergence["patience"]),
            replication_seeds=list(config.get("replication_seeds", [1, 2])),
        )
        self.executor = CandidateExecutor(
            REPO_ROOT,
            self.data_dir,
            experiment_timeout_seconds=int(self.budgets["experiment_timeout_seconds"]),
            test_timeout_seconds=int(self.budgets["test_timeout_seconds"]),
        )
        self.session_started = time.monotonic()

    def _elapsed(self) -> float:
        return self.state.wall_clock_seconds + (time.monotonic() - self.session_started)

    def _save(self) -> None:
        self.state.wall_clock_seconds = self._elapsed()
        self.session_started = time.monotonic()
        self.audit.save_state(self.state.to_dict())

    def _record_rejection(
        self, iteration: int, decision: ResearchDecision, critic: CriticDecision
    ) -> None:
        node = ExperimentNode(
            iteration=iteration,
            experiment_id=f"rejected_{decision.hypothesis_id}",
            hypothesis_id=decision.hypothesis_id,
            family=decision.family,
            action=decision.action,
            parameters=decision.parameters,
            status="critic_rejected",
            parent_experiment=decision.parent_experiment,
        )
        self.state.nodes.append(node)
        self.audit.record_iteration(
            {
                "iteration": iteration,
                "proposal": decision.to_dict(),
                "preflight": critic.to_dict(),
                "status": "critic_rejected",
                "manual_intervention": False,
            }
        )
        self._save()

    def _repair_until_tests_pass(
        self,
        iteration: int,
        decision: ResearchDecision,
        manifest: CandidateManifest,
        workspace: CandidateWorkspace,
        starting_error: str | None = None,
        repairs_used: int = 0,
    ) -> tuple[CandidateManifest, int, str | None]:
        current = manifest
        error = starting_error
        maximum = int(self.budgets["max_debug_repairs"])
        while True:
            if error is None:
                try:
                    workspace.write(current)
                    passed, output = self.executor.test(workspace)
                    if passed:
                        return current, repairs_used, None
                    error = output
                except Exception as exc:
                    error = str(exc)
            if repairs_used >= maximum:
                return current, repairs_used, error
            repairs_used += 1
            debug = self.roles.debug(
                self.state, iteration, decision, current, error, repairs_used
            )
            current = repaired_manifest(
                current, debug.replacement_code, debug.replacement_tests
            )
            error = None

    def _execute(
        self,
        iteration: int,
        decision: ResearchDecision,
        preflight: CriticDecision,
        manifest: CandidateManifest,
        replicated_from: str | None = None,
    ) -> ExperimentNode:
        workspace = CandidateWorkspace(
            self.generated_root,
            self.state.run_id,
            iteration,
            manifest.candidate_id,
        )
        manifest, repairs, validation_error = self._repair_until_tests_pass(
            iteration, decision, manifest, workspace
        )
        outcome = None
        while validation_error is None:
            if self.state.training_attempts >= int(self.budgets["max_iterations"]):
                validation_error = "Training-attempt budget reached before execution."
                break
            self.state.training_attempts += 1
            outcome = self.executor.train(iteration, manifest, workspace, self.run_dir)
            if outcome.status == "success":
                break
            manifest, repairs, validation_error = self._repair_until_tests_pass(
                iteration,
                decision,
                manifest,
                workspace,
                starting_error=outcome.error,
                repairs_used=repairs,
            )
            if validation_error is not None:
                break

        if outcome is not None and outcome.status == "success" and outcome.metrics:
            postflight = self.roles.critic_postflight(
                self.state,
                iteration,
                decision,
                outcome.metrics,
                outcome.diagnostics,
            )
            status = "success"
            metrics = outcome.metrics
            artifact = outcome.artifact_path
        else:
            postflight = None
            status = "failed"
            metrics = None
            artifact = None

        try:
            candidate_dir = str(workspace.directory.relative_to(REPO_ROOT))
        except ValueError:
            candidate_dir = str(workspace.directory)
        node = ExperimentNode(
            iteration=iteration,
            experiment_id=manifest.candidate_id,
            hypothesis_id=decision.hypothesis_id,
            family=decision.family,
            action=decision.action,
            parameters=manifest.parameters,
            status=status,
            metrics=metrics,
            artifact_path=artifact,
            candidate_dir=candidate_dir,
            parent_experiment=decision.parent_experiment,
            replicated_from=replicated_from,
        )
        self.state.nodes.append(node)
        if status == "success":
            self.policy.observe_success(self.state, node)

        self.audit.record_iteration(
            {
                "iteration": iteration,
                "proposal": decision.to_dict(),
                "preflight": preflight.to_dict(),
                "manifest": {
                    "candidate_id": manifest.candidate_id,
                    "hypothesis_id": manifest.hypothesis_id,
                    "family": manifest.family,
                    "parameters": manifest.parameters,
                    "code_sha256": _sha256_text(manifest.code),
                    "tests_sha256": _sha256_text(manifest.tests),
                },
                "repairs": repairs,
                "outcome": None if outcome is None else outcome.to_dict(),
                "postflight": None if postflight is None else postflight.to_dict(),
                "status": status,
                "manual_intervention": False,
            }
        )
        self._save()
        return node

    def _replication(self, task: dict[str, Any]) -> None:
        source_id = str(task["source_experiment"])
        source = next(node for node in self.state.nodes if node.experiment_id == source_id)
        source_dir = REPO_ROOT / str(source.candidate_dir)
        manifest_data = _load_json(source_dir / "manifest.json")
        seed = int(task["seed"])
        parameters = dict(manifest_data["parameters"])
        parameters["seed"] = seed
        parameters = sanitize_parameters(source.family, parameters)
        candidate_id = f"{source.experiment_id[:65]}_seed{seed}"
        if any(node.experiment_id == candidate_id for node in self.state.nodes):
            return
        manifest = CandidateManifest(
            candidate_id=candidate_id,
            hypothesis_id=source.hypothesis_id,
            family=source.family,
            code=(source_dir / "candidate.py").read_text(encoding="utf-8"),
            tests=(source_dir / "test_candidate.py").read_text(encoding="utf-8"),
            parameters=parameters,
        )
        decision = ResearchDecision(
            hypothesis_id=source.hypothesis_id,
            family=source.family,
            action="replicate",
            hypothesis=f"Exact replication of {source.experiment_id} with seed {seed}.",
            rationale="Deterministic replication required after a meaningful validation improvement.",
            parameters=parameters,
            evidence=(),
            parent_experiment=source.experiment_id,
        )
        preflight = CriticDecision(
            approved=True,
            decision="replicate",
            rationale="Replication parameters are inherited; only seed changes.",
        )
        self.state.iteration_count += 1
        self._execute(
            self.state.iteration_count,
            decision,
            preflight,
            manifest,
            replicated_from=source.experiment_id,
        )

    def run(self) -> Path:
        max_iterations = int(self.budgets["max_iterations"])
        max_wall_clock = float(self.budgets["max_wall_clock_seconds"])
        max_proposals = max_iterations * 2

        while True:
            if self._elapsed() >= max_wall_clock:
                self.state.stop_reason = "wall_clock_budget_reached"
                break
            if self.state.training_attempts >= max_iterations:
                self.state.stop_reason = "iteration_budget_reached"
                break
            if self.state.iteration_count >= max_iterations:
                self.state.stop_reason = "candidate_budget_reached"
                break
            if self.policy.should_stop(self.state):
                self.state.stop_reason = "converged"
                break
            if self.state.token_usage.total_tokens >= int(self.config["llm"]["max_total_tokens"]):
                self.state.stop_reason = "llm_token_budget_reached"
                break

            try:
                if self.state.pending_replications and coverage_complete(self.state):
                    task = self.state.pending_replications[0]
                    self._replication(task)
                    self.state.pending_replications.pop(0)
                    self._save()
                    continue
                if self.state.proposal_attempts >= max_proposals:
                    self.state.stop_reason = "proposal_budget_reached"
                    break

                iteration = self.state.iteration_count + 1
                self.state.proposal_attempts += 1
                decision = self.roles.research(
                    self.state, iteration, required_family(self.state)
                )
                preflight = self.roles.critic_preflight(
                    self.state, iteration, decision
                )
                self.state.iteration_count += 1
                if not preflight.approved:
                    self._record_rejection(iteration, decision, preflight)
                    continue
                manifest = self.roles.build(self.state, iteration, decision)
                self._execute(iteration, decision, preflight, manifest)
            except Exception as exc:
                self.audit.append_jsonl(
                    self.run_dir / "research_memory.jsonl",
                    {"type": "controller_error", "error": str(exc)},
                )
                if "token budget" in str(exc).lower():
                    self.state.stop_reason = "llm_token_budget_reached"
                    break
                self.state.stop_reason = "controller_error"
                self.audit.write_json_atomic(
                    self.run_dir / "error.json", {"error": str(exc)}
                )
                break

        self.state.status = "completed"
        self._save()
        summary = {
            "run_id": self.state.run_id,
            "status": self.state.status,
            "stop_reason": self.state.stop_reason,
            "iterations": self.state.iteration_count,
            "training_attempts": self.state.training_attempts,
            "manual_interventions": self.state.manual_interventions,
            "token_usage": self.state.token_usage.to_dict(),
            "wall_clock_seconds": self.state.wall_clock_seconds,
            "best": {
                "experiment_id": self.state.best_experiment_id,
                "metrics": self.state.best_metrics,
                "artifact_path": self.state.best_artifact_path,
                "candidate_dir": self.state.best_candidate_dir,
            },
        }
        self.audit.write_json_atomic(self.run_dir / "summary.json", summary)
        self.audit.write_json_atomic(
            self.run_dir / "best.json", summary["best"]
        )
        self.audit.write_json_atomic(
            self.run_dir / "results.json",
            [
                {
                    "iteration": node.iteration,
                    "experiment_id": node.experiment_id,
                    "family": node.family,
                    "action": node.action,
                    "status": node.status,
                    "metrics": node.metrics,
                    "delta_vs_baseline": None
                    if not node.metrics
                    else float(node.metrics["primary"]) - self.state.baseline_primary,
                }
                for node in self.state.nodes
            ],
        )
        print(json.dumps(summary, indent=2))
        return self.run_dir


def run_research_agent(
    config_path: Path,
    provider: LLMProvider | None = None,
    resume_dir: Path | None = None,
    baseline_summary: dict[str, Any] | None = None,
) -> Path:
    config = _load_json(config_path)
    return ResearchLoop(
        config,
        config_path,
        provider=provider,
        resume_dir=resume_dir,
        baseline_summary=baseline_summary,
    ).run()
