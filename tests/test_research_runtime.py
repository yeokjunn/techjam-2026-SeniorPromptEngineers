from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.agent.audit import ResearchAudit
from src.agent.catalog import MethodCatalog
from src.agent.discoveries import DiscoveryStore
from src.agent.families import FAMILIES, coverage_families, family_names
from src.agent.llm import FAMILY_ENUM, ScriptedProvider
from src.agent.policy import SearchPolicy, coverage_complete, required_family
from src.agent.runtime_contracts import runtime_contract_prompt
from src.agent.roles import BASE_CANDIDATE_CONTRACT, ResearchRoles
from src.agent.types import CandidateManifest, ExperimentNode, ResearchDecision, RunState, TokenUsage


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


def web_research_payload(family: str) -> dict:
    payload = research_payload(family)
    payload["web_searched"] = True
    return payload


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
            self.assertTrue(decision.web_searched)

    def test_discovery_store_persists_proposal_and_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "discoveries.json"
            store = DiscoveryStore(path)
            decision = ResearchDecision.from_dict(web_research_payload("bpr"))
            discovery_id = store.record_proposal(1, decision)
            self.assertIsNotNone(discovery_id)

            node = ExperimentNode(
                1,
                "candidate_bpr",
                decision.hypothesis_id,
                decision.family,
                decision.action,
                decision.parameters,
                "success",
                {"GAUC": 0.67, "nDCG@5": 0.54, "primary": 0.605},
            )
            store.record_outcome(1, decision, node, baseline_primary=0.6016)

            reloaded = DiscoveryStore(path)
            text = reloaded.prompt_text()
            self.assertIn("family=bpr", text)
            self.assertIn("primary=0.605", text)
            self.assertIn("https://arxiv.org/abs/1205.2618", text)


class PolicyTests(unittest.TestCase):
    def test_family_coverage_is_reported_not_required(self):
        state = RunState("run", "running", "now", 0.6016, meaningful_best=0.6016)
        cov = sorted(coverage_families())
        for i, family in enumerate(cov[:-1], start=1):
            state.nodes.append(
                ExperimentNode(
                    i, family, f"h{i}", family, "explore", {}, "success", {"primary": 0.601}
                )
            )
        self.assertIsNone(required_family(state))
        self.assertFalse(coverage_complete(state))
        state.nodes.append(
            ExperimentNode(
                len(cov), cov[-1], "h_last", cov[-1], "explore", {}, "success", {"primary": 0.602}
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
        """Verify that method cards appear before volatile state (for caching)."""
        provider = ScriptedProvider([research_payload("bpr")])
        with tempfile.TemporaryDirectory() as directory:
            audit = ResearchAudit(Path(directory) / "run")
            roles = ResearchRoles(
                provider,
                MethodCatalog.load(REPO_ROOT / "research" / "methods"),
                audit,
                max_total_tokens=1000,
            )
            state = RunState("run", "running", "now", 0.6016)
            roles.research(state, 1, None)
            prompt = provider.calls[0]["prompt"]
            self.assertIn("METHOD CARD", prompt)
            self.assertIn("ROLE: Researcher", prompt)
            method_card_pos = prompt.find("METHOD CARD")
            role_pos = prompt.find("ROLE: Researcher")
            self.assertLess(method_card_pos, role_pos)

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

    def test_researcher_prompt_includes_persistent_discoveries(self):
        provider = ScriptedProvider([research_payload("bpr")])
        with tempfile.TemporaryDirectory() as directory:
            discovery_path = Path(directory) / "discoveries.json"
            store = DiscoveryStore(discovery_path)
            store.record_proposal(1, ResearchDecision.from_dict(web_research_payload("bpr")))
            audit = ResearchAudit(Path(directory) / "run")
            roles = ResearchRoles(
                provider,
                MethodCatalog.load(REPO_ROOT / "research" / "methods"),
                audit,
                max_total_tokens=1000,
                discovery_store=store,
            )
            state = RunState("run", "running", "now", 0.6016)
            roles.research(state, 2, "bpr")
            prompt = provider.calls[0]["prompt"]
            self.assertIn("PERSISTENT WEB DISCOVERIES:", prompt)
            self.assertIn("https://arxiv.org/abs/1205.2618", prompt)

    def test_builder_prompt_requires_test_scores(self):
        """Verify Builder prompt includes test_scores requirement."""
        # This would need a builder response payload in practice
        # For now, we verify the BASE_CANDIDATE_CONTRACT contains test_scores
        self.assertIn("test_scores", BASE_CANDIDATE_CONTRACT)
        self.assertIn("Return `test_scores`", BASE_CANDIDATE_CONTRACT)

    def test_builder_prompt_matches_sandbox_and_unittest_runner(self):
        self.assertIn("Never call getattr", BASE_CANDIDATE_CONTRACT)
        self.assertIn("never import from parent packages", BASE_CANDIDATE_CONTRACT)
        self.assertIn("python -m unittest -v test_candidate.py", BASE_CANDIDATE_CONTRACT)
        self.assertIn("unittest.TestCase", BASE_CANDIDATE_CONTRACT)
        self.assertIn("Do not use pytest", BASE_CANDIDATE_CONTRACT)
        self.assertIn("do not probe alternative constructors", BASE_CANDIDATE_CONTRACT)

    def test_builder_prompt_names_the_sampler_from_the_registry(self):
        """Verify Builder uses registry samplers (or falls back gracefully)."""
        provider = ScriptedProvider([
            research_payload("bpr"),
            {"candidate_id": "c1", "hypothesis_id": "h_bpr", "family": "bpr",
             "code": "def run(context, parameters): return None",
             "tests": "pass", "parameters": parameters("bpr")}
        ])
        with tempfile.TemporaryDirectory() as directory:
            audit = ResearchAudit(Path(directory) / "run")
            roles = ResearchRoles(
                provider,
                MethodCatalog.load(REPO_ROOT / "research" / "methods"),
                audit,
                max_total_tokens=10000,
            )
            state = RunState("run", "running", "now", 0.6016)
            decision = roles.research(state, 1, "bpr")
            roles.build(state, 1, decision)
            builder_prompt = provider.calls[1]["prompt"]
            self.assertIn(
                "sample_bpr_pairs(users, labels, rng, negatives_per_positive)",
                builder_prompt,
            )

    def test_runtime_contract_cards_ground_fm_api(self):
        prompt = runtime_contract_prompt("bpr")
        self.assertIn("FMRanker", prompt)
        self.assertIn("gradients(features, score_gradients)", prompt)
        self.assertIn("apply_gradients(grad_v, grad_w, grad_b=0.0)", prompt)
        self.assertIn("never call apply_gradients(grads, lr)", prompt)
        self.assertIn("grad_v_p + grad_v_n", prompt)

    def test_builder_prompt_includes_runtime_contracts_and_debugger_memory(self):
        provider = ScriptedProvider([
            {"candidate_id": "c1", "hypothesis_id": "h_bpr", "family": "bpr",
             "code": "def run(c, p): pass", "tests": "pass", "parameters": parameters("bpr")}
        ])
        with tempfile.TemporaryDirectory() as directory:
            audit = ResearchAudit(Path(directory) / "run")
            roles = ResearchRoles(
                provider,
                MethodCatalog.load(REPO_ROOT / "research" / "methods"),
                audit,
                max_total_tokens=10000,
            )
            state = RunState("run", "running", "now", 0.6016)
            decision = ResearchDecision.from_dict(research_payload("bpr"))
            roles.build(
                state,
                1,
                decision,
                debugger_memory="- iteration 1: never call apply_gradients(grads, lr)",
            )
            prompt = provider.calls[0]["prompt"]
            self.assertIn("RUNTIME CONTRACTS:", prompt)
            self.assertIn("DEBUGGER MEMORY FROM THIS RUN:", prompt)
            self.assertIn("never call apply_gradients(grads, lr)", prompt)

    def test_debugger_prompt_includes_runtime_contracts_and_memory(self):
        provider = ScriptedProvider([
            {
                "preserve_hypothesis": True,
                "diagnosis": "unpacked gradients",
                "replacement_code": "def run(c, p): pass",
                "replacement_tests": "pass",
            }
        ])
        with tempfile.TemporaryDirectory() as directory:
            audit = ResearchAudit(Path(directory) / "run")
            roles = ResearchRoles(
                provider,
                MethodCatalog.load(REPO_ROOT / "research" / "methods"),
                audit,
                max_total_tokens=10000,
            )
            state = RunState("run", "running", "now", 0.6016)
            decision = ResearchDecision.from_dict(research_payload("bpr"))
            roles.debug(
                state,
                1,
                decision,
                CandidateManifest(
                    candidate_id="c1",
                    hypothesis_id="h_bpr",
                    family="bpr",
                    code="bad code",
                    tests="bad tests",
                    parameters=parameters("bpr"),
                ),
                "ValueError from apply_gradients",
                1,
                debugger_memory="- iteration 1: unpack gradients before apply_gradients",
            )
            prompt = provider.calls[0]["prompt"]
            self.assertIn("RUNTIME CONTRACTS:", prompt)
            self.assertIn("DEBUGGER MEMORY FROM THIS RUN:", prompt)
            self.assertIn("unpack gradients before apply_gradients", prompt)

    def test_schema_family_enum_follows_the_registry(self):
        """Verify schema family enums match family_names()."""
        # FAMILY_ENUM should be sorted list of family names
        expected_families = sorted(family_names())
        self.assertEqual(list(FAMILY_ENUM), expected_families)
        # Should match FAMILIES keys
        self.assertEqual(set(FAMILY_ENUM), set(FAMILIES.keys()))

    def test_feedback_keyword_appends_after_volatile_state(self):
        """Verify feedback keyword appends as PREVIOUS ATTEMPT REJECTED block."""
        provider = ScriptedProvider([research_payload("bpr")])
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
            roles.research(state, 1, None, feedback=feedback_text)
            prompt = provider.calls[0]["prompt"]
            # Feedback should appear as PREVIOUS ATTEMPT REJECTED block
            self.assertIn("PREVIOUS ATTEMPT REJECTED:", prompt)
            self.assertIn(feedback_text, prompt)

    def test_role_sequence_writes_distinct_pass_files(self):
        """Verify sequence parameter writes distinct pass files in passes/ without overwriting."""
        provider = ScriptedProvider([
            research_payload("bpr"),
            research_payload("bpr"),
            {"approved": False, "decision": "reject", "rationale": "first", "concerns": [], "next_focus": "fix"},
            {"approved": True, "decision": "proceed", "rationale": "second", "concerns": [], "next_focus": "run"},
            {"candidate_id": "c1", "hypothesis_id": "h_bpr", "family": "bpr",
             "code": "def run(c, p): pass", "tests": "pass", "parameters": parameters("bpr")},
            {"candidate_id": "c1", "hypothesis_id": "h_bpr", "family": "bpr",
             "code": "def run(c, p): pass", "tests": "pass", "parameters": parameters("bpr")},
        ])
        with tempfile.TemporaryDirectory() as directory:
            audit = ResearchAudit(Path(directory) / "run")
            roles = ResearchRoles(
                provider,
                MethodCatalog.load(REPO_ROOT / "research" / "methods"),
                audit,
                max_total_tokens=10000,
            )
            state = RunState("run", "running", "now", 0.6016)
            d0 = roles.research(state, 1, "bpr", sequence=0)
            d1 = roles.research(state, 1, "bpr", sequence=1)
            self.assertTrue((audit.passes_dir / "001_researcher_0.json").is_file())
            self.assertTrue((audit.passes_dir / "001_researcher_1.json").is_file())

            roles.critic_preflight(state, 1, d0, sequence=0)
            roles.critic_preflight(state, 1, d1, sequence=1)
            self.assertTrue((audit.passes_dir / "001_critic_preflight_0.json").is_file())
            self.assertTrue((audit.passes_dir / "001_critic_preflight_1.json").is_file())

            roles.build(state, 1, d0, sequence=0)
            roles.build(state, 1, d1, sequence=1)
            self.assertTrue((audit.passes_dir / "001_builder_0.json").is_file())
            self.assertTrue((audit.passes_dir / "001_builder_1.json").is_file())


if __name__ == "__main__":
    unittest.main()
