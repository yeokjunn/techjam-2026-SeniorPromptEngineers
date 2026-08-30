from __future__ import annotations

import functools
import hashlib
import json
import shutil
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..evaluation.datacard import render_data_card
from ..evaluation.gate import run_gate
from ..evaluation.official import BASELINE_TOLERANCE, within_baseline_tolerance
from ..models.ensemble import try_blend_candidates
from .audit import ResearchAudit
from .candidate_runner import CandidateExecutor, CandidateWorkspace, repaired_manifest
from .catalog import MethodCatalog
from .convergence import official_converged
from .discoveries import DiscoveryStore
from .controller import (
    REPO_ROOT,
    _count_interventions,
    _resolve_repo_path,
    _source_manifest,
    run_agent,
)
from .errors import LLMError
from .llm import LLMProvider, build_provider
from .policy import (
    SearchPolicy,
    required_family,
    sanitize_parameters,
    scored_primaries,
)
from .report import render_reports
from .roles import ResearchRoles
from .types import (
    CandidateManifest,
    CriticDecision,
    EDAReport,
    EDAResearchPlan,
    ExperimentNode,
    ResearchDecision,
    RunState,
)


# I-3: one Debugger brief per failure class Owner B's trusted worker can tag an
# outcome with (``types.py:52``, set at ``candidate_runner.py:190``, ``:212``,
# ``:231``, ``:255``, ``:265``). The chosen brief is prepended to the worker's own
# error and reaches the Debugger verbatim on the ``ERROR:`` line of its prompt
# (``roles.py:257``), so telling the model *what kind* of failure this is costs
# Owner C's file nothing. Every one of the six keys is defined, ``"leak"``
# included even though a leak is skipped rather than repaired below: the
# dictionary is the documented contract for the classes, so a class B adds shows
# up here as a missing key rather than as a silently generic prompt. One line
# each, because the text shares a line with the error.
DEBUG_BRIEFS: dict[str, str] = {
    "timeout": (
        "The run exceeded its time budget. Reduce epochs or batch work; do not "
        "change the hypothesis."
    ),
    "crash": (
        "The candidate process raised before it produced a result. Fix the "
        "exception where it is thrown - shapes, dtypes, indexing, imports - and "
        "leave the approved hypothesis, family and parameters exactly as they are."
    ),
    "bad_output": (
        "The candidate finished but its result could not be read as a "
        "CandidateOutput. Return one finite validation score per row of "
        "context.valid_x, in that order, and do not change the hypothesis."
    ),
    "low_score": (
        "Validation ranking quality landed far below the official baseline, which "
        "usually means the loss or the negative sampler is not training the model. "
        "Repair the implementation, not the hypothesis."
    ),
    "leak": (
        "Validation ranking quality is implausibly high, so the label or a feature "
        "derived from it reached the model. Remove the leaking signal and keep the "
        "hypothesis unchanged."
    ),
    "missing_test_scores": (
        "CandidateOutput.test_scores was absent. Return test scores for "
        "context.test_x in the same row order."
    ),
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@functools.lru_cache(maxsize=None)
def _cached_data_card(data_dir: str) -> str:
    """Render the data card at most once per data directory per process (I-4).

    ``render_data_card`` scans every KuaiRand CSV: ~2-4 s on the real dataset
    against ~0 s on a directory that has none of them. Production renders once
    either way — one ``ResearchLoop`` per process — but the test suite builds
    ~25 loops against that same directory, which would cost the suite minutes
    for a string it already has. The renderer is deterministic for a given
    directory (Owner D pins that in ``test_card_is_deterministic``), so the
    cache is observationally identical to calling it every time. Keyed on the
    directory *string* rather than the ``Path``: both hash, only one prints
    unambiguously in a cache dump.
    """
    return render_data_card(Path(data_dir))


def _repo_relative(path: Path) -> str:
    """A POSIX path relative to the repo root when it lives there, else absolute.

    Same rule, and the same reason, as ``candidate_dir`` below: a path recorded
    in a run's own files is read back on another machine, and T11 accepts no
    absolute machine paths in the final run's committed artifacts.
    """
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _official_convergence_iteration(
    state: RunState, epsilon: float, patience: int
) -> int | None:
    """The iteration of the scored node whose prefix first meets the official rule.

    ``seq[0]`` is the baseline seed rather than an iteration, so the number
    reported is the node's own ``iteration`` and not the prefix length. ``None``
    when the rule never fires.
    """
    # Same filter as policy.scored_primaries; the index below depends on it.
    scored = [node for node in state.nodes if node.status == "success" and node.metrics]
    sequence = [state.baseline_primary] + scored_primaries(state)
    for length in range(patience + 1, len(sequence) + 1):
        if official_converged(sequence[:length], epsilon, patience):
            return scored[length - 2].iteration
    return None


def _is_budget_error(exc: BaseException) -> bool:
    return False


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


def _compact_error(value: str, limit: int = 1200) -> str:
    """Keep debugger memory prompt-sized while preserving the failure tail."""
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def _classify_debugger_lesson(error: str) -> str:
    """Map common failures to terse instructions future roles can act on."""
    lowered = str(error or "").lower()
    if "apply_gradients" in lowered or "inhomogeneous shape" in lowered:
        return (
            "Use FMRanker.gradients(features, score_gradients) with score gradients, "
            "unpack (grad_v, grad_w, grad_b), then call apply_gradients(grad_v, grad_w, grad_b)."
        )
    if "trusted rows" in lowered and "split" in lowered:
        return (
            "For build_features, pass explicit split-specific specs for train, valid, and test; "
            "do not let validation/test default to split='train'."
        )
    if "invalid json" in lowered or "failed schema" in lowered:
        return "Role output must be one valid JSON object matching the requested schema; no Markdown or extra prose."
    if "test_scores" in lowered:
        return "CandidateOutput must include finite test_scores for context.test_x when test_x is not None."
    return "Repair the exact failing API contract or shape issue; do not change the approved hypothesis."


def _baseline_skip_reason(
    summary: dict[str, Any], run_dir: Path, threshold: float, revision: str
) -> str | None:
    """The first admission check a candidate baseline fails, or ``None`` if it passes.

    The checks run in the review's order — identity, then score, then provenance,
    then the artifact — and only the *first* failure is reported, so the reason an
    operator reads is the one they should act on rather than a pile of
    consequences. ``revision`` is ``controller._source_manifest()["revision"]``,
    the digest over every ``src/**/*.py``: a summary produced by code that is no
    longer on disk describes an experiment nobody can re-run, and adopting its
    number silently rebases every later comparison on a phantom (C5).
    """
    best = summary.get("best") or {}
    if best.get("experiment_id") != "official_fm_seed0":
        return "experiment_id_mismatch"
    primary = float((best.get("metrics") or {}).get("primary", float("-inf")))
    # I11, B's predicate: two-sided. ``threshold`` keeps its name but now carries
    # the official *centre*, so a leaked 0.85 is as unacceptable as a 0.40 — under
    # the old one-sided lower-bound gate it was adopted as a baseline.
    if not within_baseline_tolerance(primary, threshold):
        return "outside_tolerance"
    manifest_path = run_dir / "source_manifest.json"
    if not manifest_path.is_file():
        return "no_source_manifest"
    try:
        recorded = _load_json(manifest_path).get("revision")
    except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
        # A manifest that cannot be read yields no revision to compare, which is
        # the same operational fact as having none — and is reported as such
        # rather than as an unreadable *summary*, which this is not.
        # ``AttributeError`` is the shape ``[]`` or ``"x"`` takes: the file parses
        # but is not a mapping, so ``.get`` is not a method it has.
        return "no_source_manifest"
    if recorded != revision:
        return "revision_mismatch"
    artifact = best.get("artifact_path")
    # The committed baseline's ``artifact_path`` is a Windows path from another
    # machine. It is not absolute here and not a real relative path either, so
    # ``_resolve_repo_path`` folds it under the repo root and it does not exist —
    # which is exactly the answer wanted: the file the gate would submit is absent.
    if not artifact or not _resolve_repo_path(str(artifact)).is_file():
        return "artifact_missing"
    return None


def _latest_valid_baseline(
    run_root: Path, threshold: float, revision: str
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    """Pick an adoptable baseline summary and report every one that was not.

    Returns ``(summary | None, skips)`` where each skip is
    ``{"path": str, "reason": str}``. Nothing is discarded silently: the old bare
    ``except … continue`` made a corrupt summary indistinguishable from "no
    baseline exists" (I12), so an unreadable file is now a recorded outcome.

    Ordering is by **run id** (``path.parent.name``), not by the filesystem's
    modification timestamps. Run ids are UTC timestamps, so lexicographic order
    is chronological and survives a clone, a copy or a checkout — each of which
    rewrites those timestamps and so used to reorder the candidates arbitrarily.
    """
    accepted: list[tuple[str, dict[str, Any]]] = []
    skipped: list[dict[str, str]] = []
    for path in sorted(run_root.glob("*/summary.json")):
        try:
            summary = _load_json(path)
            reason = _baseline_skip_reason(summary, path.parent, threshold, revision)
        except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
            # ``AttributeError`` covers every shape a summary can take that
            # *parses* but is not the mapping the readers assume — the whole file
            # being ``[]``, or ``best``, ``best.metrics`` being a list or a
            # string. Each of those makes a ``.get`` call raise, and every one of
            # those calls is inside ``_baseline_skip_reason``'s narrow parse of a
            # single candidate, so nothing else can be swallowed here. Without it
            # one malformed file under ``runs/`` ends every research run at
            # construction instead of costing that one candidate.
            reason = "unreadable_summary"
        if reason is not None:
            skipped.append({"path": _repo_relative(path), "reason": reason})
            continue
        accepted.append(
            (path.parent.name, {**summary, "summary_path": _repo_relative(path)})
        )
    best = max(accepted, key=lambda item: item[0])[1] if accepted else None
    return best, skipped


def _official_baseline_summary(primary: float) -> dict[str, Any]:
    """Synthetic cached baseline used when the config trusts the published score."""
    return {
        "run_id": "official_cached_baseline",
        "status": "completed",
        "best": {
            "experiment_id": "official_fm_seed0",
            "metrics": {"primary": primary},
            "artifact_path": None,
            "candidate_dir": None,
        },
        "summary_path": "official://kuairand-pure/validation-baseline",
    }


def _ensure_baseline(
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    run_root = _resolve_repo_path(config.get("run_root", "runs"))
    official = float(config["official_validation_baseline"])
    if bool(config.get("use_official_baseline_cache", False)):
        return _official_baseline_summary(official), []
    revision = str(_source_manifest()["revision"])
    baseline, skipped = _latest_valid_baseline(run_root, official, revision)
    # One line per skip, to stdout, for the operator watching the run start: the
    # JSON record is for the audit, this is for the human deciding whether a
    # six-hour run is about to be spent re-training a baseline they thought was
    # already on disk.
    for record in skipped:
        print(f"Baseline summary skipped: {record['path']} ({record['reason']})")
    if baseline is not None:
        return baseline, skipped
    baseline_config = _resolve_repo_path(config.get("baseline_config", "configs/baseline.json"))
    run_dir = run_agent(baseline_config)
    summary = _load_json(run_dir / "summary.json")
    primary = float(summary["best"]["metrics"]["primary"])
    if not within_baseline_tolerance(primary, official):
        raise RuntimeError(
            f"Official FM baseline gate failed: {primary:.4f} outside "
            f"[{official - BASELINE_TOLERANCE:.4f}, {official + BASELINE_TOLERANCE:.4f}]"
        )
    artifact = summary["best"].get("artifact_path")
    # The same admission check the adopted path applies (``_baseline_skip_reason``):
    # a baseline whose scores file is absent cannot be the run's comparison point.
    if not artifact or not _resolve_repo_path(str(artifact)).is_file():
        raise RuntimeError(f"Official FM baseline produced no artifact at {artifact}")
    return {**summary, "summary_path": _repo_relative(run_dir / "summary.json")}, skipped


class ResearchLoop:
    def __init__(
        self,
        config: dict[str, Any],
        config_path: Path,
        provider: LLMProvider | None = None,
        resume_dir: Path | None = None,
        baseline_summary: dict[str, Any] | None = None,
    ):
        # I9: before the baseline gate, which can spend minutes reproducing the
        # official run. ``resources.json`` is the file Feasibility is scored on.
        self.session_started = time.monotonic()
        self.config = config
        self.config_path = config_path
        self.data_dir = _resolve_repo_path(config["data_dir"])
        self.generated_root = _resolve_repo_path(config.get("generated_root", "generated_experiments"))
        self.run_root = _resolve_repo_path(config.get("run_root", "runs"))
        self.discovery_store = DiscoveryStore(
            _resolve_repo_path(config.get("discovery_store", "research/discoveries/discoveries.json"))
        )
        self.budgets = config["budgets"]
        max_iterations = int(self.budgets["max_iterations"])
        raw_max_training_attempts = self.budgets.get("max_training_attempts")
        self.max_training_attempts = (
            int(raw_max_training_attempts)
            if raw_max_training_attempts is not None
            else None
        )
        self.max_proposals = int(self.budgets.get("max_proposals", max_iterations * 2))
        self.convergence = config["convergence"]
        llm_config = config["llm"]
        self.provider = provider or build_provider(config)
        research_config = dict(config.get("research") or {})
        eda_config = dict(config.get("eda") or {})
        self.eda_enabled = bool(eda_config.get("enabled", False))
        self.eda_required = bool(eda_config.get("required", False))
        self.eda_max_role_reprompts = int(eda_config.get("max_role_reprompts", 0))
        self.eda_researcher_max_output_tokens = int(
            eda_config.get("researcher_max_output_tokens", 1000)
        )
        self.eda_builder_max_output_tokens = int(
            eda_config.get("builder_max_output_tokens", 1200)
        )
        self.eda_max_retries = int(eda_config.get("max_retries", 1))
        self.researcher_web_first_pass_requested = bool(
            research_config.get("allow_web_search_first_pass", False)
        )
        self.web_search_enabled = bool(getattr(self.provider, "supports_web_search", False))
        # An injected baseline is the caller's own and was never selected from
        # ``runs/``, so it has no skip list — `[]` says "nothing was rejected",
        # which is true, rather than "nothing was examined", which is not
        # something ``baseline_selection.json`` is written for on that path.
        if baseline_summary is None:
            self.baseline_summary, baseline_skips = _ensure_baseline(config)
        else:
            self.baseline_summary, baseline_skips = baseline_summary, []
        baseline_primary = float(self.baseline_summary["best"]["metrics"]["primary"])

        if resume_dir is None:
            prefix = str(config.get("run_id_prefix", ""))
            run_id = prefix + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_research")
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
            # Which summary this run is measured against, and every summary that
            # was considered and rejected, with the reason (C5 / I12). Written on
            # a new run only: a resume adopts nothing and re-selects nothing, so
            # the record the original run wrote stays the truthful one.
            self.audit.write_json_atomic(
                self.run_dir / "baseline_selection.json",
                {
                    "selected": self.baseline_summary.get("summary_path"),
                    "skipped": baseline_skips,
                },
            )
            self.audit.write_text_atomic(self.run_dir / "interventions.jsonl", "")
            # The EDA and proposal roles must be grounded in the configured
            # KuaiRand-Pure data directory. Generate the card from the same
            # directory candidates will train against, and fail early if it
            # cannot be produced.
            configured_card = config.get("data_card_path")
            if isinstance(configured_card, str) and configured_card:
                self.state.data_card_path = configured_card
            else:
                card = _cached_data_card(str(self.data_dir))
                if card.strip():
                    card_path = self.run_dir / "DATA_CARD.md"
                    self.audit.write_text_atomic(card_path, card)
                    self.state.data_card_path = _repo_relative(card_path)
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
            allow_researcher_web_first_pass=(
                self.researcher_web_first_pass_requested and self.web_search_enabled
            ),
            web_search_enabled=self.web_search_enabled,
            eda_researcher_max_output_tokens=self.eda_researcher_max_output_tokens,
            eda_builder_max_output_tokens=self.eda_builder_max_output_tokens,
            eda_max_retries=self.eda_max_retries,
            discovery_store=self.discovery_store,
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
        self.consecutive_harness_errors = 0
        # I10: how many interventions were on file when this pass began, so the
        # per-iteration flag can say whether one was recorded during it.
        self._interventions_at_iteration_start = 0
        # Logged-once flag for the mid-run convergence line, not run state: a
        # resume re-logs it, which costs a duplicate journal line and no
        # correctness (`summary.json` is recomputed from the nodes).
        self._official_converged_iteration: int | None = None
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
        # Derived, never incremented: a live loop rewrites ``state.json`` on every
        # save, so an incremented counter would clobber a concurrent ``intervene``.
        self.state.manual_interventions = _count_interventions(self.run_dir)
        self.state.wall_clock_seconds = self._elapsed()
        self.session_started = time.monotonic()
        self.audit.save_state(self.state.to_dict())

    def _debugger_memory_path(self) -> Path:
        return self.run_dir / "debugger_memory.jsonl"

    def _debugger_memory_text(self, max_entries: int = 6) -> str:
        """Render recent debugger lessons for prompts."""
        path = self._debugger_memory_path()
        if not path.is_file():
            return ""
        entries: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    entries.append(item)
        except (OSError, json.JSONDecodeError):
            return ""
        recent = entries[-max_entries:]
        lines = []
        for item in recent:
            lesson = str(item.get("lesson") or "").strip()
            if not lesson:
                continue
            prefix = (
                f"- iteration {item.get('iteration')}, "
                f"{item.get('stage', 'debugger')}, "
                f"{item.get('candidate_id', 'unknown')}: "
            )
            lines.append(prefix + lesson)
        return "\n".join(lines)

    def _record_debugger_memory(
        self,
        *,
        iteration: int,
        stage: str,
        candidate_id: str | None,
        error: str,
        error_type: str | None = None,
        lesson: str | None = None,
    ) -> None:
        """Persist a compact failure lesson for Builder/Debugger prompts."""
        self.audit.append_jsonl(
            self._debugger_memory_path(),
            {
                "type": "debugger_memory",
                "iteration": iteration,
                "stage": stage,
                "candidate_id": candidate_id,
                "error_type": error_type,
                "error": _compact_error(error),
                "lesson": lesson or _classify_debugger_lesson(error),
            },
        )

    def _note_official_convergence(self) -> None:
        """Journal the first iteration at which the organizers' rule fires (I6).

        `should_stop` is the harness agenda and is not weakened by this: the
        line exists so the journal can say "the official rule fired at iteration
        k; the harness continued for coverage".
        """
        if self._official_converged_iteration is not None:
            return
        self._official_converged_iteration = _official_convergence_iteration(
            self.state,
            float(self.convergence["epsilon"]),
            int(self.convergence["patience"]),
        )
        if self._official_converged_iteration is not None:
            self.audit.append_jsonl(
                self.run_dir / "research_memory.jsonl",
                {
                    "type": "convergence",
                    "iteration": self._official_converged_iteration,
                    "official": True,
                },
            )

    def _role_call(
        self,
        label: str,
        iteration: int,
        call,
        *,
        max_reprompts: int | None = None,
    ) -> Any:
        """Run one role pass, re-prompting the model while its own output is at fault.

        ``call`` takes the feedback string to hand back to the model (``None`` on
        the first attempt) and the attempt sequence index (``0`` on the first attempt).
        A proposal-shaped failure — bad schema, off-grid parameters, a family the
        researcher was told not to pick — is the model's to fix, so the role is
        re-sampled up to ``budgets.max_role_reprompts`` times with the rejection reason
        attached. Budget and harness failures are the run's to handle and re-raise immediately.
        """
        maximum = (
            int(self.budgets.get("max_role_reprompts", 2))
            if max_reprompts is None
            else int(max_reprompts)
        )
        feedback: str | None = None
        reprompts = 0
        while True:
            try:
                try:
                    return call(feedback, reprompts)
                except TypeError:
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
                if label in {"builder", "debugger"}:
                    self._record_debugger_memory(
                        iteration=iteration,
                        stage=f"{label}_role_retry",
                        candidate_id=None,
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )

    def _record_eda_artifact(
        self, iteration: int, plan: EDAResearchPlan, report: EDAReport
    ) -> str:
        """Persist per-iteration EDA output in a UI-readable artifact directory."""
        record = {
            "status": "completed",
            "iteration": iteration,
            "data_dir": _repo_relative(self.data_dir),
            "data_card_path": self.state.data_card_path,
            "plan": plan.to_dict(),
            "report": report.to_dict(),
            "feature_candidates": [asdict(item) for item in report.feature_candidates],
            "findings": [asdict(item) for item in report.findings],
        }
        path = self.run_dir / "eda" / f"{iteration:03d}_eda.json"
        self.audit.write_json_atomic(path, record)
        self.audit.write_json_atomic(self.run_dir / "eda" / "latest.json", record)
        return _repo_relative(path)

    def _record_eda_error_artifact(
        self,
        iteration: int,
        exc: BaseException,
        plan: EDAResearchPlan | None = None,
    ) -> str:
        record = {
            "status": "failed",
            "iteration": iteration,
            "data_dir": _repo_relative(self.data_dir),
            "data_card_path": self.state.data_card_path,
            "plan": None if plan is None else plan.to_dict(),
            "report": None,
            "feature_candidates": [],
            "findings": [],
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
        path = self.run_dir / "eda" / f"{iteration:03d}_eda_failed.json"
        self.audit.write_json_atomic(path, record)
        self.audit.write_json_atomic(self.run_dir / "eda" / "latest.json", record)
        self.audit.append_jsonl(
            self.run_dir / "research_memory.jsonl",
            {
                "type": "eda_error",
                "iteration": iteration,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "continued_without_eda": not self.eda_required,
            },
        )
        return _repo_relative(path)

    def _run_eda(
        self, iteration: int
    ) -> tuple[EDAReport | None, str | None]:
        if not self.eda_enabled:
            return None, None
        eda_plan = None
        try:
            eda_plan = self._role_call(
                "eda_researcher",
                iteration,
                lambda fb, seq=0: self.roles.eda_research(
                    self.state,
                    iteration,
                    feedback=fb,
                    sequence=seq,
                ),
                max_reprompts=self.eda_max_role_reprompts,
            )
            eda_report = self._role_call(
                "eda_builder",
                iteration,
                lambda fb, seq=0: self.roles.eda_build(
                    self.state,
                    iteration,
                    eda_plan,
                    feedback=fb,
                    sequence=seq,
                ),
                max_reprompts=self.eda_max_role_reprompts,
            )
            return eda_report, self._record_eda_artifact(iteration, eda_plan, eda_report)
        except Exception as exc:
            artifact_path = self._record_eda_error_artifact(iteration, exc, eda_plan)
            if self.eda_required:
                raise
            return None, artifact_path

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
                "manual_intervention": self.state.manual_interventions
                > self._interventions_at_iteration_start,
            }
        )

    def _record_rejection(
        self,
        iteration: int,
        decision: ResearchDecision,
        critic: CriticDecision,
        eda_report: EDAReport | None = None,
        eda_artifact_path: str | None = None,
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
        self.discovery_store.record_rejection(iteration, decision, critic)
        # I3: state first, ledger second. ``record_iteration`` appends to
        # ``iterations.jsonl``; a crash between the two used to replay the
        # iteration on resume and duplicate the line. This order loses at most
        # one ledger line instead — D de-duplicates by ``iteration`` anyway.
        self._save()
        self.audit.record_iteration(
            {
                "iteration": iteration,
                "eda_artifact_path": eda_artifact_path,
                "proposal": decision.to_dict(),
                "preflight": critic.to_dict(),
                "agent_notes": {
                    "eda": None if eda_report is None else eda_report.to_dict(),
                    "researcher": {
                        "hypothesis": decision.hypothesis,
                        "rationale": decision.rationale,
                        "evidence": [asdict(item) for item in decision.evidence],
                    },
                    "critic_preflight": critic.to_dict(),
                },
                "status": "critic_rejected",
                "manual_intervention": self.state.manual_interventions
                > self._interventions_at_iteration_start,
            }
        )
        self.audit.finish_activity(
            persistence,
            agent_note={
                "decision": "Proposal rejected before code generation or training.",
                "rationale": critic.rationale,
                "next_focus": critic.next_focus,
            },
        )

    @staticmethod
    def _critic_revision_feedback(critic: CriticDecision) -> str:
        concerns = "; ".join(critic.concerns)
        parts = [
            "PREVIOUS ATTEMPT REJECTED by critic preflight before code generation.",
            f"Rationale: {critic.rationale}",
        ]
        if concerns:
            parts.append(f"Concerns: {concerns}")
        if critic.next_focus:
            parts.append(f"Next focus: {critic.next_focus}")
        parts.append(
            "Revise the proposal directly; do not repeat the same hypothesis_id or unsupported mechanism."
        )
        return " ".join(parts)

    def _propose_until_preflight_approved(
        self,
        iteration: int,
        eda_report: EDAReport | None,
        max_proposals: int,
    ) -> tuple[ResearchDecision, CriticDecision]:
        """Use Critic preflight rejection as Researcher feedback before burning an iteration."""
        maximum = int(self.budgets.get("max_role_reprompts", 2))
        feedback: str | None = None
        revision = 0
        while True:
            if self.state.proposal_attempts >= max_proposals:
                raise RuntimeError("proposal_budget_reached")
            self.state.proposal_attempts += 1
            required = required_family(self.state, float(self.convergence["epsilon"]))
            decision = self._role_call(
                "researcher",
                iteration,
                lambda fb, seq=0: self.roles.research(
                    self.state,
                    iteration,
                    required,
                    feedback=feedback if fb is None else fb,
                    sequence=revision + seq,
                    eda_report=eda_report,
                ),
            )
            self.discovery_store.record_proposal(iteration, decision)
            preflight = self._role_call(
                "critic_preflight",
                iteration,
                lambda fb, seq=0: self.roles.critic_preflight(
                    self.state,
                    iteration,
                    decision,
                    feedback=fb,
                    sequence=revision + seq,
                    eda_report=eda_report,
                ),
            )
            if preflight.approved:
                return decision, preflight
            self.audit.append_jsonl(
                self.run_dir / "research_memory.jsonl",
                {
                    "type": "critic_preflight_rejection",
                    "iteration": iteration,
                    "proposal_attempt": self.state.proposal_attempts,
                    "hypothesis_id": decision.hypothesis_id,
                    "family": decision.family,
                    "rationale": preflight.rationale,
                    "concerns": list(preflight.concerns),
                    "next_focus": preflight.next_focus,
                    "will_revise": revision < maximum,
                },
            )
            if revision >= maximum:
                return decision, preflight
            revision += 1
            feedback = self._critic_revision_feedback(preflight)

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
        max_repairs: int | None = None,
    ) -> tuple[CandidateManifest, int, str | None]:
        """Debug the candidate until its own tests pass, or the repairs run out.

        ``max_repairs`` narrows the ``max_debug_repairs`` budget for one call, so
        a caller that knows this failure is not worth the usual number of model
        passes can say so without a second loop (I-3: a timeout gets one). It is
        a *cap*, never an extension — the caller computes it against the budget —
        and a value ``repairs_used`` has already reached returns the error
        immediately, exactly as an exhausted budget does.
        """
        current = manifest
        error = starting_error
        maximum = (
            int(self.budgets["max_debug_repairs"]) if max_repairs is None else max_repairs
        )
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
            self._record_debugger_memory(
                iteration=iteration,
                stage="pre_debugger_failure",
                candidate_id=current.candidate_id,
                error=error or "",
                lesson=_classify_debugger_lesson(error or ""),
            )
            repairs_used += 1
            try:
                debug = self.roles.debug(
                    self.state,
                    iteration,
                    decision,
                    current,
                    error,
                    repairs_used,
                    debugger_memory=self._debugger_memory_text(),
                )
            except Exception as exc:
                self._record_debugger_memory(
                    iteration=iteration,
                    stage="debugger_role_failure",
                    candidate_id=current.candidate_id,
                    error=str(exc),
                    error_type=type(exc).__name__,
                    lesson=_classify_debugger_lesson(str(exc)),
                )
                raise
            self._record_debugger_memory(
                iteration=iteration,
                stage="debugger_repair",
                candidate_id=current.candidate_id,
                error=debug.diagnosis,
                lesson=f"Debugger diagnosis: {debug.diagnosis}",
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
        eda_report: EDAReport | None = None,
        eda_artifact_path: str | None = None,
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
        # I-3: a timeout buys this candidate one Debugger pass and no more. Every
        # retry is a full training run charged to the six-hour wall clock, and a
        # candidate that did not fit its time budget usually does not fit it the
        # second time either, so the cap is measured once here rather than per
        # failure: the repairs already spent on the safety tests do not consume
        # the timeout's one pass, and a second timeout finds ``repairs_used``
        # equal to the cap and returns without calling the Debugger again.
        timeout_repair_cap = min(int(self.budgets["max_debug_repairs"]), repairs + 1)
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
            if (
                self.max_training_attempts is not None
                and self.state.training_attempts >= self.max_training_attempts
            ):
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
            if outcome.failure_class == "leak":
                # A leaked validation score is not a defect the Debugger can
                # repair: the code ran, the number is real, and a feature or the
                # label carried the answer. The node is never promotable
                # (``observe_success`` runs for successes only), so repairing it
                # would spend model passes and training attempts on a result the
                # run has already refused. Recorded as a failed node by the path
                # below, with B's error and class intact for the ledger.
                validation_error = outcome.error
                break
            # I-3: the failure class Owner B tagged the outcome with chooses the
            # Debugger's brief, prepended to B's own error on the prompt's
            # ``ERROR:`` line (``roles.py:257``). An untagged outcome — or a class
            # this file has no brief for — leaves the error exactly as it was.
            manifest, repairs, validation_error = self._repair_until_tests_pass(
                iteration,
                decision,
                manifest,
                workspace,
                starting_error=(
                    f"{DEBUG_BRIEFS.get(outcome.failure_class, '')}\n{outcome.error or ''}".strip()
                    or None
                ),
                repairs_used=repairs,
                max_repairs=(
                    timeout_repair_cap if outcome.failure_class == "timeout" else None
                ),
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
            # T11: B's worker reports the checkpoint by its absolute path. Recorded
            # repo-relative, because this string is copied onto
            # ``state.best_artifact_path`` (``policy.py:280``) and from there into
            # ``state.json``, ``best.json`` and ``summary.json`` — files that are
            # committed and read back on another machine.
            artifact = (
                _repo_relative(Path(outcome.artifact_path)) if outcome.artifact_path else None
            )
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
        self.discovery_store.record_outcome(
            iteration,
            decision,
            node,
            self.state.baseline_primary,
        )
        if status == "success":
            self.policy.observe_success(self.state, node)
            self._note_official_convergence()

        persistence = self.audit.start_activity(
            iteration,
            "persistence",
            experiment_id=manifest.candidate_id,
            objective="Finalize the immutable iteration record and resumable state.",
        )
        self._save()  # I3: see ``_record_rejection`` — state before ledger.
        # T11: the ledger is committed, and B's ``ExperimentOutcome`` already keeps
        # ``stdout_path``, ``stderr_path`` and ``test_scores_path`` repo-relative —
        # ``artifact_path`` and ``command`` are the two fields that missed that
        # convention. Rewritten on ``to_dict``'s own fresh copy, never on the
        # outcome, and only for paths under the repo: a system interpreter keeps its
        # absolute spelling, so the command stays runnable from the repo root.
        outcome_record = None if outcome is None else outcome.to_dict()
        if outcome_record is not None:
            if outcome_record.get("artifact_path"):
                outcome_record["artifact_path"] = _repo_relative(
                    Path(outcome_record["artifact_path"])
                )
            outcome_record["command"] = [
                _repo_relative(Path(part))
                if Path(part).is_absolute() and Path(part).is_relative_to(REPO_ROOT)
                else part
                for part in outcome_record.get("command") or []
            ]
        self.audit.record_iteration(
            {
                "iteration": iteration,
                "eda_artifact_path": eda_artifact_path,
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
                "outcome": outcome_record,
                "postflight": None if postflight is None else postflight.to_dict(),
                "agent_notes": {
                    "eda": None if eda_report is None else eda_report.to_dict(),
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
                "manual_intervention": self.state.manual_interventions
                > self._interventions_at_iteration_start,
            }
        )
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
        max_proposals = self.max_proposals

        while True:
            if self._elapsed() >= max_wall_clock:
                self.state.stop_reason = "wall_clock_budget_reached"
                break
            if (
                self.max_training_attempts is not None
                and self.state.training_attempts >= self.max_training_attempts
            ):
                self.state.stop_reason = "training_attempt_budget_reached"
                break
            if self.state.iteration_count >= max_iterations:
                self.state.stop_reason = "candidate_budget_reached"
                break
            if self.policy.should_stop(self.state):
                self.state.stop_reason = "converged"
                break
            # Bound before the try so the error handler can always name the pass,
            # and remember whether this pass ever charged a proposal (see the
            # no-progress guard in the handler).
            iteration = self.state.iteration_count + 1
            proposals_before = self.state.proposal_attempts
            self._interventions_at_iteration_start = self.state.manual_interventions
            try:
                if self.state.pending_replications:
                    task = self.state.pending_replications[0]
                    self._replication(task)
                    self.state.pending_replications.pop(0)
                    self.consecutive_harness_errors = 0
                    self._save()
                    continue
                if self.state.proposal_attempts >= max_proposals:
                    self.state.stop_reason = "proposal_budget_reached"
                    break

                eda_report, eda_artifact_path = self._run_eda(iteration)
                try:
                    decision, preflight = self._propose_until_preflight_approved(
                        iteration,
                        eda_report,
                        max_proposals,
                    )
                except RuntimeError as exc:
                    if str(exc) == "proposal_budget_reached":
                        self.state.stop_reason = "proposal_budget_reached"
                        break
                    raise
                if not preflight.approved:
                    self._record_rejection(
                        iteration,
                        decision,
                        preflight,
                        eda_report=eda_report,
                        eda_artifact_path=eda_artifact_path,
                    )
                    self.consecutive_harness_errors = 0
                    continue
                manifest = self._role_call(
                    "builder",
                    iteration,
                    lambda fb, seq=0: self.roles.build(
                        self.state,
                        iteration,
                        decision,
                        feedback=fb,
                        sequence=seq,
                        eda_report=eda_report,
                    ),
                )
                # A proposal does not become a candidate iteration until the
                # Builder has returned a valid manifest. Charging the iteration
                # before this call makes repeated incomplete Builder responses
                # exhaust max_iterations while producing no candidate at all.
                self.state.iteration_count += 1
                self._execute(
                    iteration,
                    decision,
                    preflight,
                    manifest,
                    eda_report=eda_report,
                    eda_artifact_path=eda_artifact_path,
                )
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
                if kind == "proposal":
                    # The model's fault and already re-prompted: drop this
                    # proposal, keep the run.
                    self.consecutive_harness_errors = 0
                    self._record_failed_proposal(iteration, exc, "proposal_failed")
                    continue
                self.consecutive_harness_errors += 1
                if self.consecutive_harness_errors >= int(
                    self.budgets.get("max_consecutive_harness_errors", 3)
                ):
                    self.audit.write_json_atomic(
                        self.run_dir / "error.json",
                        {
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                            "kind": kind,
                            "iteration": iteration,
                            "consecutive_harness_errors": self.consecutive_harness_errors,
                        },
                    )
                    failed = self.audit.start_activity(
                        self.state.iteration_count,
                        "persistence",
                        objective="Record an unexpected controller failure without corrupting the previous best.",
                    )
                    self.audit.finish_activity(failed, status="failed", error=str(exc))
                    self.state.stop_reason = "harness_error_breaker"
                    break
                self._record_failed_proposal(iteration, exc, "harness_error")
                continue

        self.state.status = "completed"
        self._save()
        # I6 / I-9: the organizers' verdict, reported beside the harness's stop
        # and never in place of it. `should_stop` is gated on unresolved
        # replications/follow-ups rather than family coverage, so a run can
        # satisfy the epsilon/N rule and still spend the next pass attributing a
        # promising lead before stopping.
        epsilon = float(self.convergence["epsilon"])
        patience = int(self.convergence["patience"])
        official_sequence = [self.state.baseline_primary] + scored_primaries(self.state)
        summary = {
            "run_id": self.state.run_id,
            "status": self.state.status,
            "stop_reason": self.state.stop_reason,
            "iterations": self.state.iteration_count,
            "training_attempts": self.state.training_attempts,
            "manual_interventions": self.state.manual_interventions,
            "token_usage": self.state.token_usage.to_dict(),
            "wall_clock_seconds": self.state.wall_clock_seconds,
            "converged_official": official_converged(official_sequence, epsilon, patience),
            "converged_official_iteration": _official_convergence_iteration(
                self.state, epsilon, patience
            ),
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
        # Automated Ensembling & Blending (MLE-STAR style)
        ensemble_result = try_blend_candidates(
            run_dir=self.run_dir,
            state=self.state,
            data_dir=self.data_dir,
            generated_root=self.generated_root,
        )
        summary["ensemble"] = ensemble_result.to_dict()

        if ensemble_result.status == "ok" and ensemble_result.ensemble_node_dir:
            node_dir = _resolve_repo_path(ensemble_result.ensemble_node_dir)
            summary["best"]["ensemble_blended"] = True
            summary["best"]["ensemble_metrics"] = ensemble_result.metrics
            summary["best"]["ensemble_weights"] = ensemble_result.weights
        else:
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
        # Reporting is the last thing the run does and the least of what it owes:
        # every artifact above is already on disk, so a rendering fault must cost
        # the reports and nothing else. It is recorded rather than swallowed, and
        # in ``research_memory.jsonl`` because ``summary.json`` is already written.
        try:
            render_reports(self.run_dir)
        except Exception as exc:
            self.audit.append_jsonl(
                self.run_dir / "research_memory.jsonl",
                {
                    "type": "report_error",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
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
