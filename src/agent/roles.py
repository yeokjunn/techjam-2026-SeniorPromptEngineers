from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .activity import ROLE_OBJECTIVES, summarize_role_output
from .audit import ResearchAudit
from .catalog import MethodCatalog
from .errors import RoleOutputInvalid, TokenBudgetExceeded
from .families import FAMILIES, builder_brief, family_names
from .llm import LLMCallResult, LLMProvider, normalize_parameters
from .policy import sanitize_parameters
from .types import (
    CandidateManifest,
    CriticDecision,
    DebugDecision,
    EvidenceSource,
    ResearchDecision,
    RunState,
)


BASE_INSTRUCTIONS = """You are one role in an autonomous ML research agent for KuaiRand-Pure.
The immutable task is within-user ranking of long_view using validation GAUC and nDCG@5.
Use train and validation only. Never request or infer hidden-test information. Do not change
the official evaluator, split, label, budgets, or reference files. Return only the requested
structured output. Evidence must be attributable to a supplied method card or primary source."""


# The organizers measured these; they are not guesses (kuairand-starter-kit/README.en.md:120-170).
# Lives in the cacheable stable prefix, so it is charged once per run, not once per call.
SEARCH_SPACE_GUIDANCE = """SEARCH-SPACE EVIDENCE (measured by the organizers; do not re-derive):

Already tried and yielding nothing -- do not spend an iteration re-testing these:
- More static feature fields: primary 0.5940 with all 13 CWM fields vs 0.5950 with the 5 -- no
  gain. The user_id x video_id cross already absorbs most of the learnable signal.
- More capacity: embedding k = 8/16/32 gives 0.5895/0.5902/0.5887 -- flat. Keep k = 16.
- First-order terms on purely user-side features contribute EXACTLY 0, because ranking is within
  a user and a constant does not reorder that user's list. User-side signal can only pay off
  through cross terms with item-side features.

Untested directions, in the organizers' own order of likelihood, mapped to registered families:
1. Change the loss to a ranking objective -- families `bpr`, `group_softmax`. Rated most likely.
2. User-behaviour sequences (DIN/SIM-style interest modelling) -- family `history_features`.
   The kit calls this "a completely blank direction": the official features use no behavioural
   history at all, and each user has hundreds to thousands of train interactions.
3. Multi-objective auxiliary signals (is_click, is_like, play_time_ms) supporting long_view --
   family `multi_task`.

The bottleneck is NOT features-as-more-columns and NOT capacity. Prefer a direction this run has
not yet tried over another point on a grid you have already sampled, unless the recorded evidence
specifically justifies repeating it."""


BASE_CANDIDATE_CONTRACT = """candidate.py must define `run(context, parameters) -> CandidateOutput`.
Use only numpy, collections, math, time, src.models.fm_core.FMRanker, src.models.sampling,
and src.experiments.contracts.CandidateOutput. The context provides train_x, train_y, train_users,
valid_x, valid_users, field_dimension, and evaluate_validation(scores).
Build the model with src.models.fm_core.FMRanker. Do NOT re-implement the factorization machine:
it gathers sparse field indices, so a dense one-hot formulation over ~40k fields overflows to NaN
and, even when it converges, breaks attribution against the official baseline. Its entire API is:

    model = FMRanker(dimension, embedding_dim=16, learning_rate=..., l2=1e-6, seed=...)
    scores, embeddings, summed = model.logits(features)   # features: int32 (n, n_fields) indices
    grad_v, grad_w, grad_b = model.gradients(features, score_gradients)  # d(loss)/d(score), (n,)
    model.apply_gradients(grad_v, grad_w, grad_b)         # Adam + L2 are applied inside
    scores = model.predict(features)                      # (n,) chunked, for validation/test
    state = model.state_dict()                            # {"V", "W", "b"} COPIES -> checkpoint_state
    model.load_state_dict(state)                          # restore, e.g. best epoch on early stop

state_dict() returns copies, not views, so writing into them does not change the model: to
restore a checkpoint call load_state_dict(state). Do not hand-roll the restore -- "b" is a
0-dimensional array, so `current["b"][:] = value` raises IndexError.

Parameters are model.V (dimension, embedding_dim), model.W (dimension,) and model.b. There is no
model.w0, model.w or model.v. Express your loss as a per-row score gradient and hand it to
gradients()/apply_gradients(); never hand-roll the optimizer or touch the arrays directly.
Do not import evaluators or perform file, network, process, or dynamic-code operations.
The trusted worker writes checkpoints and computes final metrics.
Return CandidateOutput with EXACTLY these field names -- there are no others, and a wrong name
is a TypeError that costs the iteration:

    CandidateOutput(
        validation_scores=...,   # np.ndarray (n_valid,), finite
        checkpoint_state=...,    # dict[str, np.ndarray], e.g. model.state_dict()
        training_trace=[...],    # list[dict] -- NOT "train_trace"
        diagnostics={...},       # dict; put extra numbers here, NOT as new arguments
        test_scores=...,         # np.ndarray (n_test,) or None
    )

There is no valid_primary, metrics or score argument: per-epoch numbers belong inside
training_trace, and anything else belongs in diagnostics.
Return `test_scores` — one finite score per row of `context.test_x`, same row order, from the same
trained model. Return `test_scores=None` only when `context.test_x` is None.
Tests must exercise same-user sampling/group construction without loading the real dataset.
test_candidate.py is run with `python -m unittest`, which discovers ONLY subclasses of
unittest.TestCase: put every check in a `class X(unittest.TestCase)` with `def test_*`
methods. Bare pytest-style `def test_...()` functions are collected as zero tests and the
iteration fails."""


class ResearchRoles:
    def __init__(
        self,
        provider: LLMProvider,
        catalog: MethodCatalog,
        audit: ResearchAudit,
        max_total_tokens: int,
    ):
        self.provider = provider
        self.catalog = catalog
        self.audit = audit
        self.max_total_tokens = int(max_total_tokens)
        self._data_card_cache: dict[str | None, str] = {}

    def _data_card_text(self, state: RunState) -> str:
        """Read and memoize data card text to keep the stable prefix byte-identical across calls."""
        card_path = state.data_card_path
        if card_path in self._data_card_cache:
            return self._data_card_cache[card_path]
        if card_path is None:
            result = ""
        else:
            try:
                result = Path(card_path).read_text(encoding="utf-8")
            except (OSError, ValueError):
                result = ""
        self._data_card_cache[card_path] = result
        return result

    def _stable_prefix(self, state: RunState, family: str | None) -> str:
        """Build the cacheable prompt prefix: task, contract, method cards, data card."""
        method_card_key = None
        if family is not None:
            method_card_key = Path(FAMILIES[family].method_card).stem
        method_cards = self.catalog.prompt_text(method_card_key)
        data_card = self._data_card_text(state)
        prefix = f"""{BASE_INSTRUCTIONS}

{SEARCH_SPACE_GUIDANCE}

{BASE_CANDIDATE_CONTRACT}

{method_cards}"""
        if data_card:
            prefix += f"\n\nDATA CARD:\n{data_card}"
        return prefix

    def _call(
        self,
        state: RunState,
        iteration: int,
        role: str,
        prompt: str,
        schema_name: str,
        allow_web_search: bool = False,
        sequence: int = 0,
    ) -> LLMCallResult:
        if state.token_usage.total_tokens >= self.max_total_tokens:
            raise TokenBudgetExceeded("LLM token budget reached before the next role pass.")
        stage = "researcher" if role == "researcher_web" else role
        activity = self.audit.start_activity(
            iteration,
            stage,
            role=role,
            attempt=sequence + 1,
            objective=ROLE_OBJECTIVES.get(role, f"Complete the {role} pass."),
        )
        try:
            result = self.provider.complete(
                role=role,
                instructions=BASE_INSTRUCTIONS,
                prompt=prompt,
                schema_name=schema_name,
                allow_web_search=allow_web_search,
            )
        except Exception as exc:
            self.audit.finish_activity(activity, status="failed", error=str(exc))
            raise
        state.token_usage.add(result.usage)
        self.audit.record_pass(iteration, role, prompt, result, sequence)
        self.audit.finish_activity(
            activity,
            agent_note=summarize_role_output(role, result.data),
        )
        if state.token_usage.total_tokens > self.max_total_tokens:
            raise TokenBudgetExceeded("LLM token budget exceeded by the completed role pass.")
        return result

    @staticmethod
    def _state_summary(state: RunState) -> str:
        nodes = [
            {
                "experiment_id": node.experiment_id,
                "family": node.family,
                "action": node.action,
                "parameters": node.parameters,
                "status": node.status,
                "metrics": node.metrics,
            }
            for node in state.nodes
        ]
        return json.dumps(
            {
                "baseline_primary": state.baseline_primary,
                "best_metrics": state.best_metrics,
                "stagnant_iterations": state.stagnant_iterations,
                "experiments": nodes,
            },
            indent=2,
            sort_keys=True,
        )

    def research(
        self, state: RunState, iteration: int, required_family: str | None, feedback: str | None = None
    ) -> ResearchDecision:
        family_rule = (
            f"You must choose family={required_family!r} because the other required family was already attempted."
            if required_family
            else f"Choose one registered family ({', '.join(sorted(family_names()))}) based on evidence and the experiment history."
        )
        volatile_block = f"""ROLE: Researcher
Propose one controlled experiment anywhere in the algorithmic stack -- the loss, the
feature set, or the training objective. {family_rule}
Use the curated cards first. Set needs_web_search=true only if these cards cannot support the decision.
All parameter fields in the schema must be present; use null only for parameters irrelevant to the family.

RESEARCH STATE:
{self._state_summary(state)}
"""
        if feedback:
            volatile_block += f"\nPREVIOUS ATTEMPT REJECTED: {feedback}"
        prompt = f"{self._stable_prefix(state, required_family)}\n\n{volatile_block}"
        result = self._call(state, iteration, "researcher", prompt, "research_decision")
        decision = ResearchDecision.from_dict(result.data)
        if required_family and decision.family != required_family:
            raise RoleOutputInvalid(f"Researcher violated required family {required_family!r}.")

        if decision.needs_web_search or not decision.evidence:
            web_prompt = prompt + "\nThe curated evidence was insufficient. Search primary sources, then return a final decision with URLs."
            result = self._call(
                state,
                iteration,
                "researcher_web",
                web_prompt,
                "research_decision",
                allow_web_search=True,
                sequence=1,
            )
            decision = ResearchDecision.from_dict(result.data)
            if result.sources and not decision.evidence:
                decision = ResearchDecision(
                    **{
                        **asdict(decision),
                        "evidence": tuple(
                            EvidenceSource(item["title"], item["url"], "")
                            for item in result.sources
                        ),
                    }
                )
        parameters = sanitize_parameters(decision.family, normalize_parameters(decision.parameters))
        return ResearchDecision(**{**asdict(decision), "parameters": parameters, "evidence": decision.evidence})

    def critic_preflight(
        self, state: RunState, iteration: int, decision: ResearchDecision, feedback: str | None = None
    ) -> CriticDecision:
        volatile_block = f"""ROLE: Critic preflight
Decide whether this proposal is evidence-backed, novel relative to history, leakage-safe,
computationally feasible, and isolates a ranking-loss variable. Reject unsupported evidence,
cross-user negatives, test access, evaluator changes, or unrelated architecture changes.

STATE: {self._state_summary(state)}
PROPOSAL: {json.dumps(decision.to_dict(), indent=2, sort_keys=True)}
"""
        if feedback:
            volatile_block += f"\nPREVIOUS ATTEMPT REJECTED: {feedback}"
        prompt = f"{self._stable_prefix(state, decision.family)}\n\n{volatile_block}"
        result = self._call(state, iteration, "critic_preflight", prompt, "critic_decision")
        return CriticDecision.from_dict(result.data)

    def build(
        self, state: RunState, iteration: int, decision: ResearchDecision, feedback: str | None = None
    ) -> CandidateManifest:
        sampler_brief = builder_brief(decision.family)
        volatile_block = f"""ROLE: Builder
Generate a self-contained candidate.py and test_candidate.py for the approved proposal.
{sampler_brief}

PROPOSAL:
{json.dumps(decision.to_dict(), indent=2, sort_keys=True)}
"""
        if feedback:
            volatile_block += f"\nPREVIOUS ATTEMPT REJECTED: {feedback}"
        prompt = f"{self._stable_prefix(state, decision.family)}\n\n{volatile_block}"
        result = self._call(state, iteration, "builder", prompt, "candidate_manifest")
        manifest = CandidateManifest.from_dict(result.data)
        if manifest.family != decision.family or manifest.hypothesis_id != decision.hypothesis_id:
            raise RoleOutputInvalid("Builder changed the approved family or hypothesis ID.")
        parameters = sanitize_parameters(manifest.family, normalize_parameters(manifest.parameters))
        return CandidateManifest(**{**asdict(manifest), "parameters": parameters})

    def debug(
        self,
        state: RunState,
        iteration: int,
        decision: ResearchDecision,
        manifest: CandidateManifest,
        error: str,
        repair_number: int,
    ) -> DebugDecision:
        volatile_block = f"""ROLE: Debugger
Repair the candidate code/tests for the supplied validation or execution error. Preserve the
approved hypothesis, family, parameters, and candidate contract. Do not broaden permissions.

HYPOTHESIS: {json.dumps(decision.to_dict(), indent=2, sort_keys=True)}
CODE: {manifest.code}
TESTS: {manifest.tests}
ERROR: {error}
"""
        prompt = f"{self._stable_prefix(state, decision.family)}\n\n{volatile_block}"
        result = self._call(
            state,
            iteration,
            "debugger",
            prompt,
            "debug_decision",
            sequence=repair_number,
        )
        decision_result = DebugDecision.from_dict(result.data)
        if not decision_result.preserve_hypothesis:
            raise RoleOutputInvalid("Debugger refused to preserve the approved hypothesis.")
        return decision_result

    def critic_postflight(
        self,
        state: RunState,
        iteration: int,
        decision: ResearchDecision,
        metrics: dict[str, float],
        diagnostics: dict[str, Any],
    ) -> CriticDecision:
        volatile_block = f"""ROLE: Critic postflight
Interpret the trusted validation result. State whether the hypothesis was supported and what
the next research focus should be. You cannot promote checkpoints or override stopping rules.

BASELINE PRIMARY: {state.baseline_primary}
PROPOSAL: {json.dumps(decision.to_dict(), indent=2, sort_keys=True)}
TRUSTED METRICS: {json.dumps(metrics, indent=2, sort_keys=True)}
DIAGNOSTICS: {json.dumps(diagnostics, indent=2, sort_keys=True)}
"""
        prompt = f"{self._stable_prefix(state, decision.family)}\n\n{volatile_block}"
        result = self._call(state, iteration, "critic_postflight", prompt, "critic_decision")
        return CriticDecision.from_dict(result.data)
