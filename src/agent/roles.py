from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .audit import ResearchAudit
from .catalog import MethodCatalog
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
            raise RuntimeError("LLM token budget reached before the next role pass.")
        result = self.provider.complete(
            role=role,
            instructions=BASE_INSTRUCTIONS,
            prompt=prompt,
            schema_name=schema_name,
            allow_web_search=allow_web_search,
        )
        state.token_usage.add(result.usage)
        self.audit.record_pass(iteration, role, prompt, result, sequence)
        if state.token_usage.total_tokens > self.max_total_tokens:
            raise RuntimeError("LLM token budget exceeded by the completed role pass.")
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
        )

    def research(
        self, state: RunState, iteration: int, required_family: str | None
    ) -> ResearchDecision:
        family_rule = (
            f"You must choose family={required_family!r} because the other required family was already attempted."
            if required_family
            else "Choose BPR or group-softmax based on evidence and the experiment history."
        )
        prompt = f"""ROLE: Researcher
Propose one controlled ranking-loss experiment. {family_rule}
Use the curated cards first. Set needs_web_search=true only if these cards cannot support the decision.
All parameter fields in the schema must be present; use null only for parameters irrelevant to the family.

RESEARCH STATE:
{self._state_summary(state)}

APPROVED METHOD CARDS:
{self.catalog.prompt_text(required_family)}
"""
        result = self._call(state, iteration, "researcher", prompt, "research_decision")
        decision = ResearchDecision.from_dict(result.data)
        if required_family and decision.family != required_family:
            raise ValueError(f"Researcher violated required family {required_family!r}.")

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
        self, state: RunState, iteration: int, decision: ResearchDecision
    ) -> CriticDecision:
        prompt = f"""ROLE: Critic preflight
Decide whether this proposal is evidence-backed, novel relative to history, leakage-safe,
computationally feasible, and isolates a ranking-loss variable. Reject unsupported evidence,
cross-user negatives, test access, evaluator changes, or unrelated architecture changes.

STATE: {self._state_summary(state)}
PROPOSAL: {json.dumps(decision.to_dict(), indent=2)}
"""
        result = self._call(state, iteration, "critic_preflight", prompt, "critic_decision")
        return CriticDecision.from_dict(result.data)

    def build(
        self, state: RunState, iteration: int, decision: ResearchDecision
    ) -> CandidateManifest:
        prompt = f'''ROLE: Builder
Generate a self-contained candidate.py and test_candidate.py for the approved proposal.
candidate.py must define `run(context, parameters) -> CandidateOutput`.
Use only numpy, collections, math, time, src.models.fm_core.FMRanker,
src.models.sampling, and
src.experiments.contracts.CandidateOutput. The context provides train_x, train_y,
train_users, valid_x, valid_users, field_dimension, and evaluate_validation(scores).
Do not import evaluators or perform file, network, process, or dynamic-code operations.
The trusted worker writes checkpoints and computes final metrics. Return finite validation
scores, a dict of numpy checkpoint arrays, a training trace, and diagnostics. Tests must
exercise same-user sampling/group construction without loading the real dataset.
For BPR you must call src.models.sampling.sample_bpr_pairs. For group-softmax you must
call src.models.sampling.sample_softmax_groups. These trusted samplers are mandatory.

PROPOSAL:
{json.dumps(decision.to_dict(), indent=2)}
'''
        result = self._call(state, iteration, "builder", prompt, "candidate_manifest")
        manifest = CandidateManifest.from_dict(result.data)
        if manifest.family != decision.family or manifest.hypothesis_id != decision.hypothesis_id:
            raise ValueError("Builder changed the approved family or hypothesis ID.")
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
        prompt = f"""ROLE: Debugger
Repair the candidate code/tests for the supplied validation or execution error. Preserve the
approved hypothesis, family, parameters, and candidate contract. Do not broaden permissions.

HYPOTHESIS: {json.dumps(decision.to_dict(), indent=2)}
CODE: {manifest.code}
TESTS: {manifest.tests}
ERROR: {error}
"""
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
            raise ValueError("Debugger refused to preserve the approved hypothesis.")
        return decision_result

    def critic_postflight(
        self,
        state: RunState,
        iteration: int,
        decision: ResearchDecision,
        metrics: dict[str, float],
        diagnostics: dict[str, Any],
    ) -> CriticDecision:
        prompt = f"""ROLE: Critic postflight
Interpret the trusted validation result. State whether the hypothesis was supported and what
the next research focus should be. You cannot promote checkpoints or override stopping rules.

BASELINE PRIMARY: {state.baseline_primary}
PROPOSAL: {json.dumps(decision.to_dict(), indent=2)}
TRUSTED METRICS: {json.dumps(metrics, indent=2)}
DIAGNOSTICS: {json.dumps(diagnostics, indent=2)}
"""
        result = self._call(state, iteration, "critic_postflight", prompt, "critic_decision")
        return CriticDecision.from_dict(result.data)
