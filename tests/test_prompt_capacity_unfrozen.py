"""The prompts must not re-freeze the capacity axes the registry grids just reopened (W1e).

`families.RANKING_CAPACITY_GRID` opened `k` to (8, 16, 32, 64), added an `l2` axis and widened
`learning_rate`. The Builder-facing constructor template used to be a literal
``FMRanker(dimension, embedding_dim=16, ..., l2=1e-6, ...)``, which generated candidates copied
verbatim -- so a proposal that moved along those axes still trained at k=16, l2=1e-6 and the grid
was decorative. These tests pin the template to `parameters[...]` reads and pin the Researcher
prose away from "Keep k = 16".

`l2` is read with `parameters.get("l2", 1e-6)` on purpose: `RANKING_CAPACITY_DEFAULTS` gives the
key to `bpr`/`group_softmax` only, so `sanitize_parameters` does not emit an `l2` entry for
`history_features` or `multi_task` and a subscript would be a KeyError there.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.agent.families import FAMILIES
from src.agent.llm import ScriptedProvider
from src.agent.policy import sanitize_parameters
from src.agent.roles import BASE_CANDIDATE_CONTRACT, SEARCH_SPACE_GUIDANCE, ResearchRoles
from src.agent.runtime_contracts import FM_RANKER_CONTRACT
from src.agent.types import ResearchDecision, RunState


REPO_ROOT = Path(__file__).resolve().parents[1]

FROZEN_LITERALS = ("embedding_dim=16", "l2=1e-6")

PARAMETERS = {
    "seed": 0,
    "k": 32,
    "l2": 1e-3,
    "learning_rate": 0.002,
    "epochs": 5,
    "batch_size": 2048,
    "patience": 2,
    "negatives_per_positive": 1,
    "negatives_per_group": None,
    "temperature": None,
}

DECISION_PAYLOAD = {
    "hypothesis_id": "h_bpr",
    "family": "bpr",
    "action": "explore",
    "hypothesis": "wider embeddings pay off under a pairwise loss",
    "rationale": "method card evidence",
    "parameters": dict(PARAMETERS),
    "evidence": [
        {"title": "Primary paper", "url": "https://arxiv.org/abs/1205.2618", "method_card_id": "bpr"}
    ],
    "needs_web_search": False,
    "parent_experiment": None,
}

DECISION = ResearchDecision.from_dict(DECISION_PAYLOAD)


def _roles(provider: ScriptedProvider, directory: str) -> ResearchRoles:
    from src.agent.audit import ResearchAudit
    from src.agent.catalog import MethodCatalog

    return ResearchRoles(
        provider,
        MethodCatalog.load(REPO_ROOT / "research" / "methods"),
        ResearchAudit(Path(directory) / "run"),
        max_total_tokens=10000,
    )


def _builder_prompt() -> str:
    provider = ScriptedProvider(
        [
            {
                "candidate_id": "c1",
                "hypothesis_id": "h_bpr",
                "family": "bpr",
                "code": "def run(context, parameters): return None",
                "tests": "pass",
                "parameters": dict(PARAMETERS),
            }
        ]
    )
    with tempfile.TemporaryDirectory() as directory:
        _roles(provider, directory).build(RunState("run", "running", "now", 0.6016), 1, DECISION)
    return str(provider.calls[0]["prompt"])


def _researcher_prompt() -> str:
    provider = ScriptedProvider([dict(DECISION_PAYLOAD)])
    with tempfile.TemporaryDirectory() as directory:
        _roles(provider, directory).research(RunState("run", "running", "now", 0.6016), 1, "bpr")
    return str(provider.calls[0]["prompt"])


class BuilderConstructorTemplateTests(unittest.TestCase):
    def test_builder_prompt_reads_the_constructor_arguments_from_parameters(self):
        prompt = _builder_prompt()
        for read in (
            'embedding_dim=int(parameters["k"])',
            'learning_rate=float(parameters["learning_rate"])',
            'l2=float(parameters.get("l2", 1e-6))',
            'seed=int(parameters["seed"])',
        ):
            self.assertIn(read, prompt, f"{read!r} missing from the Builder prompt")

    def test_builder_prompt_carries_no_frozen_capacity_literal(self):
        prompt = _builder_prompt()
        for literal in FROZEN_LITERALS:
            self.assertNotIn(literal, prompt, f"{literal!r} still freezes the Builder prompt")

    def test_both_constructor_templates_are_parameter_driven(self):
        # The contract lives in two places -- the always-on candidate contract and the
        # family-selected runtime API card -- and either one alone would re-teach the literal.
        for block in (BASE_CANDIDATE_CONTRACT, FM_RANKER_CONTRACT):
            self.assertIn('embedding_dim=int(parameters["k"])', block)
            self.assertIn('l2=float(parameters.get("l2", 1e-6))', block)
            for literal in FROZEN_LITERALS:
                self.assertNotIn(literal, block)


class ResearcherCapacityProseTests(unittest.TestCase):
    def test_researcher_prompt_does_not_pin_k(self):
        prompt = _researcher_prompt()
        for pinned in ("Keep k = 16", "Keep k=16", "NOT capacity"):
            self.assertNotIn(pinned, prompt, f"{pinned!r} still steers the Researcher off the axis")

    def test_search_space_guidance_calls_the_axes_searchable_and_qualifies_the_sweep(self):
        self.assertIn("POINTWISE", SEARCH_SPACE_GUIDANCE)
        # The whole phrase, not the three axis names: a bare `assertIn("k", ...)`
        # matches "kit", "blank" and "likely" and asserts nothing.
        self.assertIn(
            "k, l2 and learning_rate are all searchable", SEARCH_SPACE_GUIDANCE
        )


class LiteralMatchesRuntimeBehaviourTests(unittest.TestCase):
    """The `.get` default in the template must match what the sanitiser actually emits."""

    def test_ranking_loss_families_get_an_l2_entry_and_the_others_do_not(self):
        with_l2 = {
            name
            for name in FAMILIES
            if "l2" in sanitize_parameters(name, dict(FAMILIES[name].defaults))
        }
        self.assertEqual(with_l2, {"bpr", "group_softmax"})

    def test_every_family_supplies_the_other_three_constructor_arguments(self):
        for name in FAMILIES:
            parameters = sanitize_parameters(name, dict(FAMILIES[name].defaults))
            for key in ("k", "learning_rate", "seed"):
                self.assertIn(key, parameters, f"{name} is missing {key!r}")


if __name__ == "__main__":
    unittest.main()
