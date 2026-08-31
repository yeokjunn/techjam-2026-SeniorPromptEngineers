"""The Builder/Debugger prompts must name Owner E's safety blocklist (T-cross-owner).

Without this the Builder emits e.g. ``getattr``, ``validate_source`` rejects the candidate, and a
bounded Debugger repair burns a third of the iteration budget.
"""

from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

from src.agent.audit import ResearchAudit
from src.agent.catalog import MethodCatalog
from src.agent.families import FAMILIES
from src.agent.llm import ScriptedProvider
from src.agent.roles import (
    CANDIDATE_OUTPUT_BLOCK,
    FORBIDDEN_SOURCE_BLOCK,
    TRUSTED_SAMPLER_BLOCK,
    ResearchRoles,
)
from src.agent.safety import (
    ALLOWED_DUNDER_NAMES,
    FORBIDDEN_ATTRIBUTES,
    FORBIDDEN_CALLS,
    FORBIDDEN_TEXT,
)
from src.agent.types import CandidateManifest, ResearchDecision, RunState
from src.experiments.contracts import CandidateOutput


REPO_ROOT = Path(__file__).resolve().parents[1]

PARAMETERS = {
    "seed": 0,
    "k": 16,
    "learning_rate": 0.001,
    "epochs": 5,
    "batch_size": 2048,
    "patience": 2,
    "negatives_per_positive": 1,
    "negatives_per_group": None,
    "temperature": None,
}

DECISION = ResearchDecision.from_dict(
    {
        "hypothesis_id": "h_bpr",
        "family": "bpr",
        "action": "explore",
        "hypothesis": "ranking aligned loss",
        "rationale": "method card evidence",
        "parameters": dict(PARAMETERS),
        "evidence": [
            {"title": "Primary paper", "url": "https://arxiv.org/abs/1205.2618", "method_card_id": "bpr"}
        ],
        "needs_web_search": False,
        "parent_experiment": None,
    }
)


def _roles(provider: ScriptedProvider, directory: str) -> ResearchRoles:
    return ResearchRoles(
        provider,
        MethodCatalog.load(REPO_ROOT / "research" / "methods"),
        ResearchAudit(Path(directory) / "run"),
        max_total_tokens=10000,
    )


class BuilderBlocklistPromptTests(unittest.TestCase):
    def test_block_lists_every_member_of_the_three_safety_sets(self):
        for name in FORBIDDEN_CALLS | FORBIDDEN_ATTRIBUTES | FORBIDDEN_TEXT:
            self.assertIn(name, FORBIDDEN_SOURCE_BLOCK, f"{name!r} missing from the rendered block")

    def test_contract_block_names_every_candidate_output_field(self):
        for field in dataclasses.fields(CandidateOutput):
            self.assertIn(field.name, CANDIDATE_OUTPUT_BLOCK, f"{field.name!r} missing from the block")

    def test_builder_prompt_states_the_dunder_rule_and_the_contract_fields(self):
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
            roles = _roles(provider, directory)
            roles.build(RunState("run", "running", "now", 0.6016), 1, DECISION)
        prompt = provider.calls[0]["prompt"]
        self.assertIn(CANDIDATE_OUTPUT_BLOCK, prompt)
        self.assertIn("beginning with `__`", prompt)
        self.assertIn("__dataclass_fields__", prompt)
        for name in ALLOWED_DUNDER_NAMES:
            self.assertIn(name, prompt)

    def test_builder_prompt_carries_the_block(self):
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
            roles = _roles(provider, directory)
            roles.build(RunState("run", "running", "now", 0.6016), 1, DECISION)
        self.assertIn(FORBIDDEN_SOURCE_BLOCK, provider.calls[0]["prompt"])

    def test_debugger_prompt_carries_the_block(self):
        provider = ScriptedProvider(
            [
                {
                    "preserve_hypothesis": True,
                    "diagnosis": "removed the disallowed call",
                    "replacement_code": "def run(context, parameters): return None",
                    "replacement_tests": "pass",
                }
            ]
        )
        manifest = CandidateManifest(
            candidate_id="c1",
            hypothesis_id="h_bpr",
            family="bpr",
            code="def run(context, parameters): return None",
            tests="pass",
            parameters=dict(PARAMETERS),
        )
        with tempfile.TemporaryDirectory() as directory:
            roles = _roles(provider, directory)
            roles.debug(
                RunState("run", "running", "now", 0.6016),
                1,
                DECISION,
                manifest,
                "Call is not allowed: getattr",
                repair_number=1,
            )
        self.assertIn(FORBIDDEN_SOURCE_BLOCK, provider.calls[0]["prompt"])

    def test_sampler_block_names_every_registry_sampler_with_a_signature(self):
        samplers = {entry.trusted_sampler for entry in FAMILIES.values()}
        self.assertTrue(samplers, "registry declares no trusted samplers")
        for name in samplers:
            self.assertIn(
                f"{name}(", TRUSTED_SAMPLER_BLOCK, f"{name!r} signature missing from the block"
            )
        self.assertIn("np.random.default_rng(seed)", TRUSTED_SAMPLER_BLOCK)

    def test_builder_prompt_carries_the_sampler_block_and_the_unittest_runner_line(self):
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
            roles = _roles(provider, directory)
            roles.build(RunState("run", "running", "now", 0.6016), 1, DECISION)
        prompt = provider.calls[0]["prompt"]
        self.assertIn(TRUSTED_SAMPLER_BLOCK, prompt)
        self.assertIn("python -m unittest -v test_candidate.py", prompt)
        self.assertIn("unittest.TestCase", prompt)


if __name__ == "__main__":
    unittest.main()
