from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import httpx
except ModuleNotFoundError:  # openai>=3 depends on the httpx2 fork instead
    import httpx2 as httpx
import numpy as np
import pytest
from openai import APIStatusError, RateLimitError

from src.agent.audit import ResearchAudit
from src.agent.catalog import MethodCatalog
from src.agent.errors import (
    IncompleteResponse,
    LLMError,
    RoleOutputInvalid,
    TokenBudgetExceeded,
)
from src.agent.llm import (
    OpenAIResponsesProvider,
    ScriptedProvider,
    _normalize_schema_output,
    _parse_structured_output,
    build_provider,
    schema_fields_note,
)
from src.agent.research_controller import ResearchLoop
from src.agent.roles import ResearchRoles
from src.agent.safety import validate_family_contract, validate_source
from src.agent.types import DebugDecision, RunState


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tests" / "fixtures" / "offline_smoke_script.json"
CONFIG_PATH = REPO_ROOT / "configs" / "offline_smoke.json"
DATA_FILE = (
    REPO_ROOT
    / "data"
    / "KuaiRand-Pure"
    / "data"
    / "log_standard_4_08_to_4_21_pure.csv"
)


def _load_script() -> list[dict]:
    return json.loads(SCRIPT_PATH.read_text(encoding="utf-8"))


def _valid_response(output_text: str | None = None, **overrides):
    payload = {
        "approved": True,
        "decision": "proceed",
        "rationale": "valid",
        "concerns": [],
        "next_focus": "run",
    }
    response = {
        "id": "response-test",
        "model": "gpt-5.5",
        "status": "completed",
        "output_text": output_text if output_text is not None else json.dumps(payload),
        "output": [],
        "usage": {
            "input_tokens": 12,
            "output_tokens": 8,
            "total_tokens": 20,
        },
    }
    response.update(overrides)
    return SimpleNamespace(**response)


def _status_error(status_code: int, headers: dict[str, str] | None = None):
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(
        status_code,
        headers=headers,
        request=request,
    )
    error_type = RateLimitError if status_code == 429 else APIStatusError
    return error_type(
        f"status {status_code}",
        response=response,
        body={"error": {"message": "retry test"}},
    )


class FakeResponses:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _openai_provider(outcomes, *, max_retries: int = 5):
    with patch("src.agent.llm.load_project_environment", return_value=False), patch.dict(
        os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False
    ):
        provider = OpenAIResponsesProvider({"max_retries": max_retries})
    responses = FakeResponses(outcomes)
    provider.client = SimpleNamespace(responses=responses)
    return provider, responses


class StructuredOutputParsingTests(unittest.TestCase):
    def test_bare_json_parses(self):
        self.assertEqual(_parse_structured_output('{"ok": true}', "r1"), {"ok": True})

    def test_json_fenced_output_is_unwrapped(self):
        text = "```json\n{\"ok\": true}\n```"
        self.assertEqual(_parse_structured_output(text, "r2"), {"ok": True})

    def test_bare_fence_is_unwrapped(self):
        self.assertEqual(_parse_structured_output("```\n{\"ok\": 1}\n```", "r3"), {"ok": 1})

    def test_invalid_json_raises_role_output_invalid(self):
        with self.assertRaises(RoleOutputInvalid):
            _parse_structured_output("not json at all", "r4")

    def test_prose_wrapped_fenced_json_is_rescued(self):
        # Live evidence (5/5 role probes, glm-5.3-flash): prose- and fence-wrapped
        # JSON is this provider's normal output shape, so the parser rescues it.
        text = "Here you go:\n```json\n{\"ok\": 1}\n```"
        self.assertEqual(_parse_structured_output(text, "r5"), {"ok": 1})


class ParserRescueTests(unittest.TestCase):
    """Live drift: GLM wraps JSON in fences, prefixes/suffixes prose, or emits YAML."""

    def test_prose_wrapped_json_is_brace_sliced(self):
        text = 'Here is my decision:\n```json\n{"approved": true}\n```\nHope this helps.'
        self.assertEqual(_parse_structured_output(text, "r1"), {"approved": True})

    def test_plain_prose_then_json_without_fence(self):
        text = 'Analysis follows. {"ok": 1} End of analysis.'
        self.assertEqual(_parse_structured_output(text, "r2"), {"ok": 1})

    def test_yaml_document_raises_with_raw_snippet(self):
        text = "```yaml\nCriticDecision:\n  decision: APPROVE\n```"
        with self.assertRaises(RoleOutputInvalid) as ctx:
            _parse_structured_output(text, "r3")
        self.assertIn("head=", str(ctx.exception))
        self.assertIn("tail=", str(ctx.exception))

    def test_truncated_json_raises_with_snippet(self):
        with self.assertRaises(RoleOutputInvalid) as ctx:
            _parse_structured_output('{"code": "def run(): pass", "tests": "trunca', "r4")
        self.assertIn("r4", str(ctx.exception))


class SchemaNoteTests(unittest.TestCase):
    def test_critic_note_lists_every_required_field(self):
        note = schema_fields_note("critic_decision")
        for field in ("approved", "decision", "rationale", "concerns", "next_focus"):
            self.assertIn(field, note)
        self.assertIn("raw JSON object", note)
        self.assertIn("no YAML", note)

    def test_research_note_lists_parameter_keys_and_evidence_keys(self):
        note = schema_fields_note("research_decision")
        for key in ("seed", "k", "learning_rate", "epochs", "batch_size", "patience",
                    "negatives_per_positive", "negatives_per_group", "temperature"):
            self.assertIn(key, note)
        for key in ("title", "url", "method_card_id"):
            self.assertIn(key, note)
        self.assertIn("hypothesis_id", note)

    def test_manifest_note_bounds_code_size(self):
        note = schema_fields_note("candidate_manifest")
        self.assertIn("120 lines", note)
        self.assertIn("60 lines", note)
        self.assertIn("from src.models.sampling import sample_bpr_pairs", note)
        self.assertIn("Never write 'from src.models import", note)
        self.assertIn("getattr", note)
        self.assertIn("FMRanker(dimension, embedding_dim=16", note)
        self.assertIn("rng is the THIRD argument", note)
        self.assertIn("unittest.TestCase", note)
        self.assertIn("checkpoint_state", note)
        self.assertIn("integer index matrices", note)


class SchemaNormalizationTests(unittest.TestCase):
    """Regressions for the live GLM drift observed in kjsmoke_20260829T1541*."""

    def test_action_synonym_mapped_and_extras_dropped(self):
        data = _normalize_schema_output(
            "research_decision",
            {
                "action": "propose_experiment",
                "control": "the FM baseline",
                "decision_basis": {"why_bpr": "..."},
                "hypothesis_id": "bpr-v1",
                "family": "bpr",
                "hypothesis": "h",
                "rationale": "r",
                "parameters": {},
                "evidence": [],
                "needs_web_search": False,
                "parent_experiment": None,
            },
        )
        self.assertEqual(data["action"], "explore")
        self.assertNotIn("control", data)
        self.assertNotIn("decision_basis", data)

    def test_missing_action_derived_from_parent_linkage(self):
        base = {
            "hypothesis_id": "bpr-v1",
            "family": "bpr",
            "hypothesis": "h",
            "rationale": "r",
            "parameters": {},
            "evidence": [],
            "needs_web_search": False,
        }
        orphan = _normalize_schema_output("research_decision", {**base, "parent_experiment": None})
        child = _normalize_schema_output("research_decision", {**base, "parent_experiment": "bpr-v1"})
        self.assertEqual(orphan["action"], "explore")
        self.assertEqual(child["action"], "exploit")

    def test_unknown_action_left_for_reprompt(self):
        data = _normalize_schema_output(
            "research_decision", {"action": "invent_new_physics"}
        )
        self.assertEqual(data["action"], "invent_new_physics")

    def test_family_case_and_spacing_normalized(self):
        data = _normalize_schema_output("research_decision", {"family": "Group Softmax"})
        self.assertEqual(data["family"], "group_softmax")
        unknown = _normalize_schema_output("research_decision", {"family": "transformers"})
        self.assertEqual(unknown["family"], "transformers")

    def test_boolean_strings_coerced(self):
        data = _normalize_schema_output("critic_decision", {"approved": "yes"})
        self.assertIs(data["approved"], True)
        data = _normalize_schema_output("debug_decision", {"preserve_hypothesis": "false"})
        self.assertIs(data["preserve_hypothesis"], False)
        untouched = _normalize_schema_output("critic_decision", {"approved": "maybe"})
        self.assertEqual(untouched["approved"], "maybe")

    def test_evidence_claim_source_keys_remapped(self):
        data = _normalize_schema_output(
            "research_decision",
            {"evidence": [{"claim": "BPR aligns loss with GAUC", "source": "https://arxiv.org/abs/1205.2618"}]},
        )
        item = data["evidence"][0]
        self.assertEqual(item["title"], "BPR aligns loss with GAUC")
        self.assertEqual(item["url"], "https://arxiv.org/abs/1205.2618")
        self.assertNotIn("claim", item)

    def test_parameter_keys_aliased_and_undeclared_dropped(self):
        data = _normalize_schema_output(
            "research_decision",
            {"parameters": {"embedding_dim": 16, "lr": 0.001, "aux_heads": ["click"], "epochs": 5}},
        )
        self.assertEqual(data["parameters"], {"k": 16, "learning_rate": 0.001, "epochs": 5})

    def test_missing_hypothesis_id_derived_deterministically(self):
        payload = {
            "family": "bpr", "action": "explore", "hypothesis": "pairwise loss beats BCE",
            "rationale": "r", "parameters": {}, "evidence": [],
        }
        first = _normalize_schema_output("research_decision", dict(payload))
        second = _normalize_schema_output("research_decision", dict(payload))
        self.assertTrue(first["hypothesis_id"].startswith("bpr-auto-"))
        self.assertEqual(first["hypothesis_id"], second["hypothesis_id"])
        manifest = _normalize_schema_output("candidate_manifest", {"code": "def run(): pass"})
        self.assertTrue(manifest["hypothesis_id"].startswith("exp-auto-"))


class OfflineSmokeTests(unittest.TestCase):
    def test_fixture_candidate_passes_the_safety_validators(self):
        manifest = _load_script()[2]
        validate_source(manifest["code"])
        validate_source(manifest["tests"], test_file=True)
        validate_family_contract(manifest["code"], "bpr")

    def test_fixture_debug_payload_parses_as_a_debug_decision(self):
        payload = {
            key: value
            for key, value in _load_script()[4].items()
            if key != "_usage"
        }
        decision = DebugDecision.from_dict(payload)
        validate_source(decision.replacement_code)

    def test_offline_smoke_config_points_at_the_committed_fixture(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        script_path = REPO_ROOT / config["llm"]["script_path"]
        self.assertTrue(script_path.is_file())
        provider = build_provider(config)
        self.assertIsInstance(provider, ScriptedProvider)
        self.assertEqual(len(provider.responses), 5)

    def test_scripted_provider_round_trips_a_non_empty_script(self):
        provider = ScriptedProvider([
            {"value": 1, "_usage": {"total_tokens": 7}},
            {"value": 2, "_usage": {"total_tokens": 11}},
        ])
        first = provider.complete(role="critic")
        second = provider.complete(role="builder")
        self.assertEqual((first.data, first.usage.total_tokens), ({"value": 1}, 7))
        self.assertEqual((second.data, second.usage.total_tokens), ({"value": 2}, 11))

    def test_relative_script_path_resolves_against_the_repo_root(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertFalse(Path(config["llm"]["script_path"]).is_absolute())
        self.assertEqual(len(build_provider(config).responses), 5)

    @pytest.mark.slow
    @unittest.skipUnless(DATA_FILE.is_file(), "KuaiRand-Pure dataset is not extracted")
    def test_scripted_loop_scores_one_real_bpr_iteration(self):
        script = copy.deepcopy(_load_script())
        # One bounded BPR batch keeps the acceptance test below 60 seconds; the
        # committed offline config retains the full three-epoch ~0.602 run.
        script[0]["parameters"]["epochs"] = 1
        script[2]["parameters"]["epochs"] = 1
        script[2]["code"] = script[2]["code"].replace(
            "range(0, len(order), batch_size)",
            "range(0, min(len(order), batch_size), batch_size)",
        )
        script[2]["code"] = script[2]["code"].replace(
            "sample_bpr_pairs(context.train_users, context.train_y, rng, per_positive)",
            "sample_bpr_pairs(context.train_users[:1000], context.train_y[:1000], rng, per_positive)",
        )
        provider = ScriptedProvider(script)
        baseline = {
            "best": {
                "experiment_id": "official_fm_seed0",
                "metrics": {
                    "GAUC": 0.6674,
                    "nDCG@5": 0.5358,
                    "primary": 0.6016,
                },
                "artifact_path": "baseline.npz",
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            config.update(
                {
                    "data_dir": str(DATA_FILE.parent),
                    "run_root": str(root / "runs"),
                    "generated_root": str(root / "generated"),
                    "method_catalog": str(REPO_ROOT / "research" / "methods"),
                }
            )
            config_path = root / "offline_smoke.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            run_dir = ResearchLoop(
                config,
                config_path,
                provider=provider,
                baseline_summary=baseline,
            ).run()

            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            error = (
                (run_dir / "error.json").read_text(encoding="utf-8")
                if (run_dir / "error.json").is_file()
                else ""
            )
            self.assertNotEqual(summary["stop_reason"], "controller_error", error)
            state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
            record = json.loads(
                (run_dir / "iterations.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(summary["training_attempts"], 1)
            self.assertEqual(state["nodes"][0]["status"], "success")
            self.assertGreaterEqual(state["nodes"][0]["metrics"]["primary"], 0.47)
            self.assertLessEqual(state["nodes"][0]["metrics"]["primary"], 0.80)
            self.assertIsNone(record["outcome"]["failure_class"])
            self.assertEqual(
                [call["role"] for call in provider.calls],
                ["researcher", "critic_preflight", "builder", "critic_postflight"],
            )

            test_scores_value = record["outcome"].get("test_scores_path")
            if test_scores_value:
                test_scores_path = Path(test_scores_value)
                if not test_scores_path.is_absolute():
                    test_scores_path = REPO_ROOT / test_scores_path
                self.assertEqual(len(np.load(test_scores_path)), 170_588)


class RetryPolicyTests(unittest.TestCase):
    def _complete(self, provider):
        return provider.complete(
            role="critic",
            instructions="trusted",
            prompt="inspect",
            schema_name="critic_decision",
        )

    def test_rate_limit_is_retried_with_exponential_backoff(self):
        provider, responses = _openai_provider(
            [_status_error(429), _status_error(429), _valid_response()]
        )
        with patch("src.agent.llm.time.sleep") as sleep:
            result = self._complete(provider)
        self.assertEqual(result.retries, 2)
        self.assertEqual(responses.calls, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2.0, 4.0])

    def test_retry_after_header_wins_over_the_backoff(self):
        provider, _ = _openai_provider(
            [_status_error(429, {"retry-after": "7"}), _valid_response()]
        )
        with patch("src.agent.llm.time.sleep") as sleep:
            self._complete(provider)
        sleep.assert_called_once_with(7.0)

    def test_five_attempts_then_the_error_propagates(self):
        provider, responses = _openai_provider(
            [_status_error(503) for _ in range(5)]
        )
        with patch("src.agent.llm.time.sleep"), self.assertRaises(APIStatusError):
            self._complete(provider)
        self.assertEqual(responses.calls, 5)

    def test_client_error_is_not_retried(self):
        provider, responses = _openai_provider([_status_error(400)])
        with patch("src.agent.llm.time.sleep") as sleep, self.assertRaises(APIStatusError):
            self._complete(provider)
        self.assertEqual(responses.calls, 1)
        sleep.assert_not_called()

    def test_incomplete_response_raises_a_typed_error(self):
        incomplete = _valid_response(
            "",
            status="incomplete",
            incomplete_details={"reason": "max_output_tokens"},
        )
        provider, responses = _openai_provider(
            [incomplete, copy.deepcopy(incomplete)], max_retries=2
        )
        with patch("src.agent.llm.time.sleep"), self.assertRaises(
            IncompleteResponse
        ) as caught:
            self._complete(provider)
        self.assertIsInstance(caught.exception, LLMError)
        self.assertEqual(responses.calls, 2)

    def test_unparseable_output_text_is_retried_then_raises(self):
        provider, responses = _openai_provider(
            [_valid_response("not-json") for _ in range(5)]
        )
        with patch("src.agent.llm.time.sleep"), self.assertRaises(RoleOutputInvalid):
            self._complete(provider)
        self.assertEqual(responses.calls, 5)

    def test_unparseable_then_valid_output_recovers(self):
        provider, responses = _openai_provider(
            [
                _valid_response("```yaml\nnope: true\n```"),
                _valid_response('{"approved": true, "decision": "approve", "rationale": "r"}'),
            ]
        )
        with patch("src.agent.llm.time.sleep"):
            result = self._complete(provider)
        self.assertTrue(result.data["approved"])
        self.assertEqual(result.retries, 1)
        self.assertEqual(responses.calls, 2)

    def test_token_budget_uses_a_typed_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            roles = ResearchRoles(
                ScriptedProvider(copy.deepcopy(_load_script())),
                MethodCatalog.load(REPO_ROOT / "research" / "methods"),
                ResearchAudit(Path(directory) / "run"),
                max_total_tokens=0,
            )
            state = RunState("run", "running", "now", 0.6016)
            with self.assertRaises(TokenBudgetExceeded):
                roles.research(state, 1, "bpr")

    def test_family_mismatch_raises_role_output_invalid(self):
        payload = copy.deepcopy(_load_script()[0])
        payload["family"] = "group_softmax"
        payload["parameters"].update(
            {
                "batch_size": 1024,
                "negatives_per_positive": None,
                "negatives_per_group": 4,
                "temperature": 1.0,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            roles = ResearchRoles(
                ScriptedProvider([payload]),
                MethodCatalog.load(REPO_ROOT / "research" / "methods"),
                ResearchAudit(Path(directory) / "run"),
                max_total_tokens=1000,
            )
            state = RunState("run", "running", "now", 0.6016)
            with self.assertRaises(RoleOutputInvalid):
                roles.research(state, 1, "bpr")


if __name__ == "__main__":
    unittest.main()
