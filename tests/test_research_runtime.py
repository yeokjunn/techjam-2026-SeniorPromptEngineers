from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.agent.audit import ResearchAudit
from src.agent.catalog import MethodCatalog
from src.agent.families import FAMILIES, family_names
from src.agent.llm import FAMILY_ENUM, ScriptedProvider
from src.agent.policy import SearchPolicy, coverage_complete, required_family
from src.agent.roles import BASE_CANDIDATE_CONTRACT, ResearchRoles
from src.agent.types import ExperimentNode, ResearchDecision, RunState, TokenUsage


REPO_ROOT = Path(__file__).resolve().parents[1]


def parameters(family: str) -> dict:
    return {
        "seed": 0,
        "k": 16,
        "learning_rate": 0.001,
        "epochs": 5,
        "batch_size": 2048 if family == "bpr" else 1024,
        "patience": 2,
        "negatives_per_positive": 1 if family == "bpr" else None,
        "negatives_per_group": 4 if family == "group_softmax" else None,
        "temperature": 1.0 if family == "group_softmax" else None,
    }


def research_payload(family: str, needs_web: bool = False) -> dict:
    return {
        "hypothesis_id": f"h_{family}",
        "family": family,
        "action": "explore",
        "hypothesis": "ranking aligned loss",
        "rationale": "method card evidence",
        "parameters": parameters(family),
        "evidence": [] if needs_web else [
            {"title": "Primary paper", "url": "https://arxiv.org/abs/1205.2618", "method_card_id": family}
        ],
        "needs_web_search": needs_web,
        "parent_experiment": None,
    }


class RuntimeSchemaTests(unittest.TestCase):
    def test_missing_structured_field_is_rejected(self):
        payload = research_payload("bpr")
        del payload["hypothesis"]
        with self.assertRaises(ValueError):
            ResearchDecision.from_dict(payload)

    def test_token_usage_aggregates(self):
        total = TokenUsage(input_tokens=2, total_tokens=2)
        total.add(TokenUsage(output_tokens=3, total_tokens=3, web_search_calls=1))
        self.assertEqual(total.total_tokens, 5)
        self.assertEqual(total.output_tokens, 3)
        self.assertEqual(total.web_search_calls, 1)

    def test_curated_then_web_fallback(self):
        provider = ScriptedProvider(
            [research_payload("bpr", needs_web=True), research_payload("bpr")]
        )
        with tempfile.TemporaryDirectory() as directory:
            audit = ResearchAudit(Path(directory) / "run")
            roles = ResearchRoles(
                provider,
                MethodCatalog.load(REPO_ROOT / "research" / "methods"),
                audit,
                max_total_tokens=1000,
            )
            state = RunState("run", "running", "now", 0.6016)
            decision = roles.research(state, 1, "bpr")
            self.assertEqual(decision.family, "bpr")
            self.assertEqual(decision.evidence[0].url, "https://arxiv.org/abs/1205.2618")
            self.assertEqual(len(provider.calls), 2)
            self.assertFalse(provider.calls[0]["allow_web_search"])
            self.assertTrue(provider.calls[1]["allow_web_search"])


class PolicyTests(unittest.TestCase):
    def test_both_families_required_before_stop(self):
        state = RunState("run", "running", "now", 0.6016, meaningful_best=0.6016)
        bpr = ExperimentNode(
            1, "bpr", "h1", "bpr", "explore", {}, "success", {"primary": 0.601}
        )
        state.nodes.append(bpr)
        self.assertEqual(required_family(state), "group_softmax")
        self.assertFalse(coverage_complete(state))
        state.nodes.append(
            ExperimentNode(
                2, "list", "h2", "group_softmax", "explore", {}, "success", {"primary": 0.602}
            )
        )
        self.assertTrue(coverage_complete(state))

    def test_meaningful_improvement_enqueues_replications(self):
        state = RunState("run", "running", "now", 0.6016, meaningful_best=0.6016)
        node = ExperimentNode(
            1, "better", "h", "bpr", "explore", {}, "success", {"primary": 0.604}
        )
        policy = SearchPolicy(0.002, 3, [1, 2])
        policy.observe_success(state, node)
        self.assertEqual([item["seed"] for item in state.pending_replications], [1, 2])

    def test_state_round_trip_preserves_completed_nodes(self):
        state = RunState("run", "running", "now", 0.6016)
        state.nodes.append(
            ExperimentNode(1, "x", "h", "bpr", "explore", {}, "success", {"primary": 0.6})
        )
        restored = RunState.from_dict(json.loads(json.dumps(state.to_dict())))
        self.assertEqual(len(restored.nodes), 1)
        self.assertEqual(restored.nodes[0].experiment_id, "x")


class PromptStructureTests(unittest.TestCase):
    """Tests for T3 prompt structure, stable prefix, and registry integration."""

    def test_method_cards_precede_the_volatile_state_in_every_prompt(self):
        critic = {"approved": True, "decision": "proceed", "rationale": "ok",
                  "concerns": [], "next_focus": "continue"}
        manifest_payload = {
            "candidate_id": "c1", "hypothesis_id": "h_bpr", "family": "bpr",
            "code": "def run(context, parameters): return None", "tests": "pass",
            "parameters": parameters("bpr"),
        }
        debug = {"preserve_hypothesis": True, "diagnosis": "syntax",
                 "replacement_code": manifest_payload["code"], "replacement_tests": "pass"}
        provider = ScriptedProvider([research_payload("bpr"), critic, manifest_payload, debug, critic])
        with tempfile.TemporaryDirectory() as directory:
            audit = ResearchAudit(Path(directory) / "run")
            roles = ResearchRoles(
                provider,
                MethodCatalog.load(REPO_ROOT / "research" / "methods"),
                audit,
                max_total_tokens=1000,
            )
            state = RunState("run", "running", "now", 0.6016)
            decision = roles.research(state, 1, "bpr")
            roles.critic_preflight(state, 1, decision)
            manifest = roles.build(state, 1, decision)
            roles.debug(state, 1, decision, manifest, "bad syntax", 1)
            roles.critic_postflight(state, 1, decision, {"primary": 0.6}, {})
            for call in provider.calls:
                prompt = call["prompt"]
                self.assertLess(prompt.index("METHOD CARD"), prompt.index("ROLE:"), call["role"])

    def test_stable_prefix_is_identical_across_two_iterations(self):
        """Verify that stable prefix bytes are identical with same state/family."""
        provider = ScriptedProvider([research_payload("bpr"), research_payload("bpr")])
        with tempfile.TemporaryDirectory() as directory:
            audit = ResearchAudit(Path(directory) / "run")
            roles = ResearchRoles(
                provider,
                MethodCatalog.load(REPO_ROOT / "research" / "methods"),
                audit,
                max_total_tokens=1000,
            )
            state = RunState("run", "running", "now", 0.6016)
            roles.research(state, 1, "bpr")
            roles.research(state, 2, "bpr")
            prompt1 = provider.calls[0]["prompt"]
            prompt2 = provider.calls[1]["prompt"]
            # Extract the stable prefix (everything before ROLE: Researcher)
            prefix1 = prompt1[: prompt1.find("ROLE: Researcher")]
            prefix2 = prompt2[: prompt2.find("ROLE: Researcher")]
            self.assertEqual(prefix1, prefix2, "Stable prefix should be byte-identical across calls")

    def test_data_card_text_is_inserted_in_the_prefix(self):
        """Verify data card text is read and memoized in the stable prefix."""
        with tempfile.TemporaryDirectory() as directory:
            data_card_path = Path(directory) / "data_card.txt"
            data_card_path.write_text("DATA CARD CONTENT FOR TEST", encoding="utf-8")
            provider = ScriptedProvider([research_payload("bpr")])
            with tempfile.TemporaryDirectory() as audit_dir:
                audit = ResearchAudit(Path(audit_dir) / "run")
                roles = ResearchRoles(
                    provider,
                    MethodCatalog.load(REPO_ROOT / "research" / "methods"),
                    audit,
                    max_total_tokens=1000,
                )
                state = RunState("run", "running", "now", 0.6016, data_card_path=str(data_card_path))
                roles.research(state, 1, "bpr")
                prompt = provider.calls[0]["prompt"]
                self.assertIn("DATA CARD:", prompt)
                self.assertIn("DATA CARD CONTENT FOR TEST", prompt)

    def test_data_card_handles_missing_path_gracefully(self):
        """Verify data card gracefully handles None or missing paths."""
        provider = ScriptedProvider([research_payload("bpr")])
        with tempfile.TemporaryDirectory() as directory:
            audit = ResearchAudit(Path(directory) / "run")
            roles = ResearchRoles(
                provider,
                MethodCatalog.load(REPO_ROOT / "research" / "methods"),
                audit,
                max_total_tokens=1000,
            )
            # state with None data_card_path
            state = RunState("run", "running", "now", 0.6016, data_card_path=None)
            roles.research(state, 1, "bpr")
            prompt = provider.calls[0]["prompt"]
            # Prompt should still work, just without the DATA CARD section
            self.assertIn("ROLE: Researcher", prompt)
            self.assertNotIn("DATA CARD:", prompt)

    def test_builder_prompt_requires_test_scores(self):
        """Verify Builder prompt includes test_scores requirement."""
        # This would need a builder response payload in practice
        # For now, we verify the BASE_CANDIDATE_CONTRACT contains test_scores
        self.assertIn("test_scores", BASE_CANDIDATE_CONTRACT)
        self.assertIn("Return `test_scores`", BASE_CANDIDATE_CONTRACT)

    def test_builder_prompt_names_the_sampler_from_the_registry(self):
        for family, family_spec in FAMILIES.items():
            decision = ResearchDecision.from_dict(research_payload(family))
            payload = {"candidate_id": "c1", "hypothesis_id": decision.hypothesis_id,
                       "family": family, "code": "def run(context, parameters): return None",
                       "tests": "pass", "parameters": parameters(family)}
            provider = ScriptedProvider([payload])
            with tempfile.TemporaryDirectory() as directory:
                roles = ResearchRoles(provider, MethodCatalog.load(REPO_ROOT / "research" / "methods"),
                                      ResearchAudit(Path(directory) / "run"), 10000)
                roles.build(RunState("run", "running", "now", 0.6016), 1, decision)
            self.assertIn(family_spec.trusted_sampler, provider.calls[0]["prompt"])

    def test_schema_family_enum_follows_the_registry(self):
        """Verify schema family enums match family_names()."""
        # FAMILY_ENUM should be sorted list of family names
        expected_families = sorted(family_names())
        self.assertEqual(list(FAMILY_ENUM), expected_families)
        # Should match FAMILIES keys
        self.assertEqual(set(FAMILY_ENUM), set(FAMILIES.keys()))

    def test_feedback_keyword_appends_after_volatile_state(self):
        provider = ScriptedProvider([research_payload("bpr"), research_payload("bpr")])
        with tempfile.TemporaryDirectory() as directory:
            audit = ResearchAudit(Path(directory) / "run")
            roles = ResearchRoles(
                provider,
                MethodCatalog.load(REPO_ROOT / "research" / "methods"),
                audit,
                max_total_tokens=1000,
            )
            state = RunState("run", "running", "now", 0.6016)
            feedback_text = "Evidence was not empirical"
            roles.research(state, 1, "bpr")
            roles.research(state, 2, "bpr", feedback=feedback_text)
            plain, rejected = (call["prompt"] for call in provider.calls)
            marker = "ROLE: Researcher"
            self.assertEqual(plain[:plain.index(marker)], rejected[:rejected.index(marker)])
            self.assertTrue(rejected.rstrip().endswith(f"PREVIOUS ATTEMPT REJECTED: {feedback_text}"))


if __name__ == "__main__":
    unittest.main()
