from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..evaluation.gate import run_gate
from .audit import ResearchAudit
from .candidate_runner import CandidateExecutor, CandidateWorkspace, repaired_manifest
from .catalog import MethodCatalog
from .controller import REPO_ROOT, _resolve_repo_path, _source_manifest, run_agent
from .errors import LLMError, TokenBudgetExceeded
from .llm import LLMProvider, build_provider
from .policy import SearchPolicy, coverage_complete, required_family, sanitize_parameters
from .report import render_reports
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


# Classification of every exception source the correctness review names (C4).
# ``roles.py``, ``types.py``, ``llm.py`` and ``policy.py`` are not ours to edit,
# so the loop classifies by exception *type*. The message check keeps budget
# stops correct while ``roles.py:52,63`` still raise a bare ``RuntimeError``
# (C's T2 step 7 converts them to ``TokenBudgetExceeded``, which subclasses
# ``RuntimeError``, so both spellings classify the same before and after).
# ``KeyboardInterrupt`` is a ``BaseException`` and is deliberately not caught:
# Ctrl-C still exits, and the operator records it with ``intervene``.


def _is_budget_error(exc: BaseException) -> bool:
    return isinstance(exc, TokenBudgetExceeded) or (
        isinstance(exc, RuntimeError) and "token budget" in str(exc).lower())


def _error_kind(exc: BaseException) -> str:      # 'budget' | 'proposal' | 'harness'
    if _is_budget_error(exc): return "budget"
    # ``LLMError`` first, and only after the budget check above: every non-budget
    # subclass (``RoleOutputInvalid``, ``IncompleteResponse``) marks output the
    # model can be asked to produce again, so it must classify exactly like the
    # bare ``ValueError`` the same raise site throws today. Widening this to all
    # ``RuntimeError`` would wrongly re-prompt a missing API key (``llm.py:198``).
    return (
        "proposal"
        if isinstance(exc, (LLMError, ValueError, TypeError, KeyError))
        else "harness"
    )


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
        # I5: three caps, three keys. ``max_iterations`` is the *candidate* cap
        # and nothing else; the training-attempt and proposal caps are their own
        # knobs, resolved once here because ``_execute`` enforces the training
        # cap too and must read the same number ``run()`` does. The ``.get``
        # defaults preserve the old single-knob behaviour for configs that
        # predate the split (every inline test config), so no caller has to
        # supply the new keys to keep working.
        max_iterations = int(self.budgets["max_iterations"])
        self.max_training_attempts = int(
            self.budgets.get("max_training_attempts", max_iterations)
        )
        self.max_proposals = int(self.budgets.get("max_proposals", max_iterations * 2))
        self.convergence = config["convergence"]
        llm_config = config["llm"]
        self.provider = provider or build_provider(config)
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
        self.consecutive_harness_errors = 0
        initialized = self.audit.start_activity(
            0,
            "initializing",
            objective="Load the frozen configuration, baseline, method catalog, and run state.",
        )
        self.audit.finish_activity(
            initialized,
            agent_note={
                "objective": "Prepare the autonomous research loop.",
                "decision": "Run state is ready; begin the next research iteration.",
            },
        )

    def _elapsed(self) -> float:
        return self.state.wall_clock_seconds + (time.monotonic() - self.session_started)

    def _save(self) -> None:
        self.state.wall_clock_seconds = self._elapsed()
        self.session_started = time.monotonic()
        self.audit.save_state(self.state.to_dict())

    def _role_call(self, label: str, iteration: int, call) -> Any:
        """Run one role pass, re-prompting the model while its own output is at fault.

        ``call`` takes the feedback string to hand back to the model (``None`` on
        the first attempt). A proposal-shaped failure — bad schema, off-grid
        parameters, a family the researcher was told not to pick — is the model's
        to fix, so the role is re-sampled up to ``budgets.max_role_reprompts``
        times with the rejection reason attached. Budget and harness failures are
        the run's to handle and re-raise immediately.
        """
        maximum = int(self.budgets.get("max_role_reprompts", 2))
        feedback: str | None = None
        reprompts = 0
        while True:
            try:
                return call(feedback)
            except Exception as exc:
                if reprompts >= maximum or _error_kind(exc) != "proposal":
                    raise
                reprompts += 1
                feedback = f"Your previous {label} response was rejected: {exc}"
                self.audit.append_jsonl(
                    self.run_dir / "research_memory.jsonl",
                    {
                        "type": "role_retry",
                        "label": label,
                        "iteration": iteration,
                        "reprompt": reprompts,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )

    def _record_failed_proposal(
        self, iteration: int, exc: BaseException, status: str
    ) -> None:
        """Ledger a proposal that never became an experiment.

        No ``ExperimentNode`` is appended — a failed proposal has no family — and
        ``iteration_count`` is not advanced, so the experiment tree stays a record
        of real experiments. ``proposal_attempts`` was already charged for the
        pass, so ``max_proposals`` bounds this path; the breaker is for harness
        errors only.
        """
        self._save()
        self.audit.record_iteration(
            {
                "iteration": iteration,
                "status": status,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "manual_intervention": False,
            }
        )

    def _record_rejection(
        self, iteration: int, decision: ResearchDecision, critic: CriticDecision
    ) -> None:
        persistence = self.audit.start_activity(
            iteration,
            "persistence",
            experiment_id=f"rejected_{decision.hypothesis_id}",
            objective="Persist the rejected proposal and resumable run state.",
        )
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
                "agent_notes": {
                    "researcher": {
                        "hypothesis": decision.hypothesis,
                        "rationale": decision.rationale,
                        "evidence": [asdict(item) for item in decision.evidence],
                    },
                    "critic_preflight": critic.to_dict(),
                },
                "status": "critic_rejected",
                "manual_intervention": False,
            }
        )
        self._save()
        self.audit.finish_activity(
            persistence,
            agent_note={
                "decision": "Proposal rejected before code generation or training.",
                "rationale": critic.rationale,
                "next_focus": critic.next_focus,
            },
        )

    def _parent_sources(self, parent_experiment: str | None) -> dict[str, str]:
        if not parent_experiment:
            return {}
        node = next(
            (item for item in self.state.nodes if item.experiment_id == parent_experiment),
            None,
        )
        if node is None or not node.candidate_dir:
            return {}
        directory = Path(node.candidate_dir)
        if not directory.is_absolute():
            directory = REPO_ROOT / directory
        try:
            directory.resolve().relative_to(self.generated_root.resolve())
        except ValueError:
            return {}
        sources: dict[str, str] = {}
        for name in ("candidate.py", "test_candidate.py"):
            path = directory / name
            if path.is_file():
                sources[name] = path.read_text(encoding="utf-8")
        return sources

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
                safety = self.audit.start_activity(
                    iteration,
                    "safety_tests",
                    experiment_id=current.candidate_id,
                    attempt=repairs_used + 1,
                    objective="Validate generated source and run the candidate unit tests.",
                )
                try:
                    workspace.write(current)
                    passed, output = self.executor.test(workspace)
                    if passed:
                        self.audit.finish_activity(
                            safety,
                            agent_note={
                                "decision": "Candidate passed source safety and focused unit tests.",
                                "next_focus": "Run training and trusted validation evaluation.",
                            },
                        )
                        return current, repairs_used, None
                    error = output
                except Exception as exc:
                    error = str(exc)
                self.audit.finish_activity(
                    safety,
                    status="failed",
                    error=error,
                    repair="Eligible for a bounded Debugger pass."
                    if repairs_used < maximum
                    else None,
                )
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
        change_summary = None
        parent_sources = self._parent_sources(decision.parent_experiment)
        execution_attempt = 0
        outcome = None
        while validation_error is None:
            # Refresh the authoritative patch before every training attempt because
            # a bounded Debugger pass may have changed the candidate after a failure.
            change_summary = self.audit.record_candidate_changes(
                iteration,
                manifest.candidate_id,
                {
                    "candidate.py": manifest.code,
                    "test_candidate.py": manifest.tests,
                },
                parent_sources,
            )
            if self.state.training_attempts >= self.max_training_attempts:
                validation_error = "Training-attempt budget reached before execution."
                break
            self.state.training_attempts += 1
            execution_attempt += 1
            training = self.audit.start_activity(
                iteration,
                "training_evaluation",
                experiment_id=manifest.candidate_id,
                attempt=execution_attempt,
                objective="Train the candidate and compute trusted validation GAUC and nDCG@5.",
                agent_note={
                    "hypothesis": decision.hypothesis,
                    "parameters": manifest.parameters,
                },
            )
            try:
                outcome = self.executor.train(iteration, manifest, workspace, self.run_dir)
            except Exception as exc:
                self.audit.finish_activity(training, status="failed", error=str(exc))
                raise
            self.audit.finish_activity(
                training,
                status="completed" if outcome.status == "success" else "failed",
                metrics=outcome.metrics,
                error=outcome.error,
                repair=outcome.recovery,
                change_summary=change_summary,
                agent_note={
                    "hypothesis": decision.hypothesis,
                    "decision": "Trusted result accepted."
                    if outcome.status == "success"
                    else "Trusted result rejected; consider bounded repair.",
                },
            )
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

        persistence = self.audit.start_activity(
            iteration,
            "persistence",
            experiment_id=manifest.candidate_id,
            objective="Finalize the immutable iteration record and resumable state.",
        )
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
                "change_summary": change_summary,
                "outcome": None if outcome is None else outcome.to_dict(),
                "postflight": None if postflight is None else postflight.to_dict(),
                "agent_notes": {
                    "researcher": {
                        "hypothesis": decision.hypothesis,
                        "rationale": decision.rationale,
                        "evidence": [asdict(item) for item in decision.evidence],
                    },
                    "critic_preflight": preflight.to_dict(),
                    "critic_postflight": None
                    if postflight is None
                    else postflight.to_dict(),
                },
                "status": status,
                "manual_intervention": False,
            }
        )
        self._save()
        self.audit.finish_activity(
            persistence,
            agent_note={
                "decision": "Iteration finalized.",
                "result": status,
                "next_focus": None if postflight is None else postflight.next_focus,
            },
            change_summary=change_summary,
            metrics=metrics,
            error=validation_error,
        )
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
        max_training_attempts = self.max_training_attempts
        max_proposals = self.max_proposals

        while True:
            if self._elapsed() >= max_wall_clock:
                self.state.stop_reason = "wall_clock_budget_reached"
                break
            if self.state.training_attempts >= max_training_attempts:
                self.state.stop_reason = "training_attempt_budget_reached"
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

            # Bound before the try so the error handler can always name the pass,
            # and remember whether this pass ever charged a proposal (see the
            # no-progress guard in the handler).
            iteration = self.state.iteration_count + 1
            proposals_before = self.state.proposal_attempts
            try:
                if self.state.pending_replications and coverage_complete(self.state):
                    task = self.state.pending_replications[0]
                    self._replication(task)
                    self.state.pending_replications.pop(0)
                    self.consecutive_harness_errors = 0
                    self._save()
                    continue
                if self.state.proposal_attempts >= max_proposals:
                    self.state.stop_reason = "proposal_budget_reached"
                    break

                self.state.proposal_attempts += 1
                decision = self._role_call(
                    "researcher",
                    iteration,
                    lambda fb: self.roles.research(
                        self.state, iteration, required_family(self.state)
                    ),
                )
                preflight = self._role_call(
                    "critic_preflight",
                    iteration,
                    lambda fb: self.roles.critic_preflight(
                        self.state, iteration, decision
                    ),
                )
                self.state.iteration_count += 1
                if not preflight.approved:
                    self._record_rejection(iteration, decision, preflight)
                    self.consecutive_harness_errors = 0
                    continue
                manifest = self._role_call(
                    "builder",
                    iteration,
                    lambda fb: self.roles.build(self.state, iteration, decision),
                )
                self._execute(iteration, decision, preflight, manifest)
                self.consecutive_harness_errors = 0
            except Exception as exc:
                kind = _error_kind(exc)
                if kind == "proposal" and self.state.proposal_attempts == proposals_before:
                    # No proposal was charged, so the failure came from the
                    # replication branch: there is no model output to re-prompt
                    # and `max_proposals` cannot bound it. A `continue` here would
                    # retry the identical pass at full speed until the wall clock
                    # ran out. A corrupt run directory is a harness fault however
                    # it surfaces (a truncated manifest.json raises
                    # JSONDecodeError, not OSError), so let the breaker bound it.
                    kind = "harness"
                self.audit.append_jsonl(
                    self.run_dir / "research_memory.jsonl",
                    {
                        "type": "controller_error",
                        "kind": kind,
                        "iteration": iteration,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )
                if kind == "budget":
                    self.state.stop_reason = "llm_token_budget_reached"
                    break
                self.state.stop_reason = "controller_error"
                self.audit.write_json_atomic(
                    self.run_dir / "error.json", {"error": str(exc)}
                )
                failed = self.audit.start_activity(
                    self.state.iteration_count,
                    "persistence",
                    objective="Record an unexpected controller failure without corrupting the previous best.",
                )
                self.audit.finish_activity(failed, status="failed", error=str(exc))
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
        # I-1. Two separate concerns, both the loop's and neither the gate's.
        #
        # ``best_candidate_dir`` is stored **repo-relative** (``_execute`` at
        # :363, copied onto the state by ``policy.py:87``), so the old
        # ``Path(...)`` resolved it against the *process* working directory and
        # the gate looked for ``test_scores.npy`` under wherever the run was
        # launched from. ``_resolve_repo_path`` is this module's own resolver —
        # the one ``__init__`` uses for every configured path — and it is also
        # correct for the absolute spelling ``_execute``'s ``except ValueError``
        # fallback produces for a workspace outside the repo.
        #
        # The ``try`` is containment in depth, not a fix for a fault in B's
        # module: ``gate.py:218`` is deliberately written so it cannot raise, but
        # ``summary.json`` is the one file the organizers read, and its survival
        # must not depend on another module keeping that promise. Keyword
        # arguments so a signature change cannot silently reorder the four paths.
        node_dir = (
            _resolve_repo_path(self.state.best_candidate_dir)
            if self.state.best_candidate_dir
            else self.run_dir
        )
        try:
            gate_result = run_gate(
                run_dir=self.run_dir,
                node_dir=node_dir,
                data_dir=self.data_dir,
                kit_dir=REPO_ROOT / "kuairand-starter-kit",
            )
            summary["gate"] = asdict(gate_result)
        except Exception as exc:
            # ``reason`` is not in the brief's literal dict, but B's gate sets one
            # on *every* error it returns (``gate.py:107-108``, and ``"unexpected"``
            # for a fault in its own wrapper at ``:231``), so consumers may treat
            # ``details["reason"]`` as always present. This is the one error shape
            # B cannot produce from their side; containment is more useful when it
            # is indistinguishable from the producer's real errors. ``error`` stays
            # the brief's ``str(exc)``.
            summary["gate"] = {
                "status": "error",
                "submission_path": None,
                "details": {"reason": "unexpected", "error": str(exc)},
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
        render_reports(self.run_dir)
        completed = self.audit.start_activity(
            self.state.iteration_count,
            "completed",
            experiment_id=self.state.best_experiment_id,
            objective="Mark the research run complete and expose its stop reason.",
        )
        self.audit.finish_activity(
            completed,
            agent_note={
                "decision": "Research run completed.",
                "stop_reason": self.state.stop_reason,
                "best_experiment": self.state.best_experiment_id,
            },
            metrics=self.state.best_metrics,
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
