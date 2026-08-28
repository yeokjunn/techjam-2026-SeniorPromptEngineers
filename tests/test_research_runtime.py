from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.agent.audit import ResearchAudit
from src.agent.catalog import MethodCatalog
from src.agent.llm import ScriptedProvider
from src.agent.policy import SearchPolicy, coverage_complete, required_family
from src.agent.roles import ResearchRoles
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


if __name__ == "__main__":
    unittest.main()
