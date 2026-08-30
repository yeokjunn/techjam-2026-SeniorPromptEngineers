from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import numpy as np
import pytest
from openai import APIStatusError, RateLimitError

from src.agent.audit import ResearchAudit
from src.agent.catalog import MethodCatalog
from src.agent.errors import (
    IncompleteResponse,
    LLMError,
    RoleOutputInvalid,
)
from src.agent.llm import (
    OpenAIResponsesProvider,
    ScriptedProvider,
    build_provider,
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
        "model": "GLM-5.3-Flash",
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

    def test_unparseable_output_text_raises_role_output_invalid(self):
        provider, responses = _openai_provider([_valid_response("not-json")])
        with self.assertRaises(RoleOutputInvalid):
            self._complete(provider)
        self.assertEqual(responses.calls, 1)

    def test_token_budget_is_reported_not_enforced_by_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            roles = ResearchRoles(
                ScriptedProvider(copy.deepcopy(_load_script())),
                MethodCatalog.load(REPO_ROOT / "research" / "methods"),
                ResearchAudit(Path(directory) / "run"),
                max_total_tokens=0,
            )
            state = RunState("run", "running", "now", 0.6016)
            roles.research(state, 1, "bpr")
            self.assertGreater(state.token_usage.total_tokens, 0)

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
