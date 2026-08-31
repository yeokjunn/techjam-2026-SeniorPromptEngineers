"""The EDA roles must survive the two replies DeepSeek actually sent.

Run `runs/kj_20260830T185458454134Z_research` logged six `role_retry` events, all
of them EDA (`eda_researcher` / `eda_builder`, iterations 1/5/6/7), in two
classes: `RoleOutputInvalid: contained invalid JSON` and `IncompleteResponse:
contained no output text`. Every non-EDA role ran under a 32,768-token cap and
never failed; the EDA roles ran under 4,000 / 5,000 and their *successful* passes
landed at 3,803 / 3,184 / 3,629 and 3,975 / 4,188 / 4,635 output tokens -- right
under the ceiling. The failures are the same calls with the reasoning a little
longer: the budget ran out, so `content` came back empty or truncated mid-object
while the reasoning field still held the JSON.

These tests pin the three defences, offline: the reply text is taken from the
reasoning field when `content` is empty or whitespace, the fenced and
preamble-wrapped shapes still parse, and both EDA prompts carry the terse
output-only line. They construct no controller and open no socket.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from src.agent.audit import ResearchAudit
from src.agent.catalog import MethodCatalog
from src.agent.errors import IncompleteResponse, RoleOutputInvalid
from src.agent.llm import OpenAICompatibleChatProvider, ScriptedProvider
from src.agent.roles import ResearchRoles
from src.agent.types import RunState


REPO_ROOT = Path(__file__).resolve().parents[1]

# The line the EDA prompts must carry, pinned verbatim: it is the only prompt-side
# defence against a reasoning preamble eating the completion budget.
TERSE_OUTPUT_LINE = "Output ONLY the JSON object — no reasoning preamble."

EDA_PLAN_PAYLOAD: dict[str, Any] = {
    "objective": "Size the pairable-user population before committing to a pairwise loss.",
    "questions": ["What share of validation users carry both classes?"],
    "feature_hypotheses": ["Impression depth interacts with item duration."],
    "required_inputs": ["MEASURED PROFILE impressions_per_user quantiles"],
    "leakage_risks": ["Per-user counts must be train-only."],
    "expected_artifacts": ["A note on pairable-user share."],
}

EDA_REPORT_PAYLOAD: dict[str, Any] = {
    "summary": "Validation lists are shallow; median user sees four impressions.",
    "findings": [
        {
            "title": "Shallow validation lists",
            "observation": "valid impressions_per_user p50=4",
            "implication": "GAUC is dominated by very short lists.",
            "evidence": "MEASURED PROFILE: valid impressions_per_user p50=4",
            "leakage_safe": True,
        }
    ],
    "feature_candidates": [
        {
            "name": "user_impression_depth",
            "description": "Bucketed train-only impression count per user.",
            "family": "bpr",
            "expected_impact": "Better negatives for shallow users.",
            "implementation_scope": "src.models.features",
            "leakage_risk": "train-only counts",
        }
    ],
    "recommended_next_focus": "bpr",
    "ui_notes": ["Show the impression-depth histogram."],
}


class FakeChatCompletions:
    """One scripted chat-completions reply per call, message fields as given."""

    def __init__(self, messages: list[dict[str, Any]]):
        self.messages = list(messages)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        fields = self.messages.pop(0) if self.messages else {}
        return SimpleNamespace(
            id=f"chatcmpl-{len(self.requests)}",
            model="deepseek-test",
            choices=[SimpleNamespace(message=SimpleNamespace(**fields))],
            usage={
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        )


def _provider(**overrides) -> OpenAICompatibleChatProvider:
    config = {
        "provider": "openai_compatible",
        "model": "deepseek-test",
        "base_url": "https://example.test/v1/chat/completions",
        "max_output_tokens": 4000,
        "max_retries": 1,
    }
    config.update(overrides)
    with patch("src.agent.llm.load_project_environment", return_value=False), patch.dict(
        os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False
    ):
        return OpenAICompatibleChatProvider(config)


def _complete(provider: OpenAICompatibleChatProvider, messages: list[dict[str, Any]], **kwargs):
    fake = FakeChatCompletions(messages)
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    result = provider.complete(
        role=kwargs.pop("role", "eda_researcher"),
        instructions="Plan the EDA pass.",
        prompt="Proceed?",
        schema_name=kwargs.pop("schema_name", "eda_research_plan"),
        **kwargs,
    )
    return result, fake


class EDAChatParsingTests(unittest.TestCase):
    """The EDA schemas go through the provider's one tolerant parse path."""

    def test_fenced_json_parses_for_both_eda_schemas(self):
        for schema_name, payload in (
            ("eda_research_plan", EDA_PLAN_PAYLOAD),
            ("eda_report", EDA_REPORT_PAYLOAD),
        ):
            with self.subTest(schema_name):
                fenced = "```json\n" + json.dumps(payload) + "\n```"
                result, fake = _complete(
                    _provider(),
                    [{"content": fenced}],
                    role="eda_builder",
                    schema_name=schema_name,
                )
                self.assertEqual(result.data, payload)
                # Same schema-note prompting every other role gets: the JSON
                # Schema is pasted into the system message, not assumed.
                system = fake.requests[0]["messages"][0]["content"]
                self.assertIn("Return only one valid JSON object", system)
                self.assertIn('"additionalProperties":false', system)

    def test_prose_wrapped_json_is_recovered_by_the_brace_scan(self):
        wrapped = (
            "Let me think about the pairable-user share first.\n"
            + json.dumps(EDA_PLAN_PAYLOAD)
            + "\nThat should be enough for one pass."
        )
        result, _ = _complete(_provider(), [{"content": wrapped}])
        self.assertEqual(result.data, EDA_PLAN_PAYLOAD)

    def test_empty_content_with_a_reasoning_field_is_harvested(self):
        # The `IncompleteResponse: contained no output text` reply: the budget
        # went to reasoning, and the object the role owes is in there.
        result, fake = _complete(
            _provider(),
            [{"content": "", "reasoning_content": json.dumps(EDA_PLAN_PAYLOAD)}],
        )
        self.assertEqual(result.data, EDA_PLAN_PAYLOAD)
        # Harvested on the first attempt: no retry, so no second call.
        self.assertEqual(len(fake.requests), 1)
        self.assertEqual(result.retries, 0)

    def test_whitespace_content_falls_through_to_the_reasoning_field(self):
        reasoning = "Weighing the options.\n```json\n" + json.dumps(EDA_REPORT_PAYLOAD) + "\n```"
        result, _ = _complete(
            _provider(),
            [{"content": "   \n  ", "reasoning": reasoning}],
            role="eda_builder",
            schema_name="eda_report",
        )
        self.assertEqual(result.data, EDA_REPORT_PAYLOAD)

    def test_content_wins_when_both_fields_carry_json(self):
        other = dict(EDA_PLAN_PAYLOAD, objective="A draft the model then discarded.")
        result, _ = _complete(
            _provider(),
            [{"content": json.dumps(EDA_PLAN_PAYLOAD), "reasoning_content": json.dumps(other)}],
        )
        self.assertEqual(result.data, EDA_PLAN_PAYLOAD)

    def test_reasoning_without_json_still_raises_a_typed_role_error(self):
        # Nothing to harvest is still a failure -- but a re-promptable typed one,
        # never a silent success or a bare exception.
        with self.assertRaises(RoleOutputInvalid):
            _complete(
                _provider(),
                [{"content": "", "reasoning_content": "I am still thinking about it."}],
            )

    def test_both_fields_empty_is_still_an_incomplete_response(self):
        with self.assertRaises(IncompleteResponse):
            _complete(_provider(), [{"content": "", "reasoning_content": ""}])

    def test_dict_shaped_message_is_read_the_same_way(self):
        provider = _provider()
        fake = FakeChatCompletions([])
        fake.create = lambda **kwargs: SimpleNamespace(  # type: ignore[method-assign]
            id="chatcmpl-dict",
            model="deepseek-test",
            choices=[
                SimpleNamespace(
                    message={"content": None, "reasoning_content": json.dumps(EDA_PLAN_PAYLOAD)}
                )
            ],
            usage={},
        )
        provider.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
        result = provider.complete(
            role="eda_researcher",
            instructions="Plan the EDA pass.",
            prompt="Proceed?",
            schema_name="eda_research_plan",
        )
        self.assertEqual(result.data, EDA_PLAN_PAYLOAD)


class EDAPromptOutputContractTests(unittest.TestCase):
    """Both EDA prompts must ask for the object and nothing else."""

    def _prompts(self) -> tuple[str, str]:
        with tempfile.TemporaryDirectory() as directory:
            provider = ScriptedProvider([dict(EDA_PLAN_PAYLOAD), dict(EDA_REPORT_PAYLOAD)])
            roles = ResearchRoles(
                provider,
                MethodCatalog.load(REPO_ROOT / "research" / "methods"),
                ResearchAudit(Path(directory) / "run"),
                max_total_tokens=10000,
            )
            state = RunState("run", "running", "now", 0.6016)
            plan = roles.eda_research(state, 1)
            roles.eda_build(state, 1, plan)
            return provider.calls[0]["prompt"], provider.calls[1]["prompt"]

    def test_terse_output_line_reaches_both_eda_prompts(self):
        researcher_prompt, builder_prompt = self._prompts()
        for label, prompt in (("eda_researcher", researcher_prompt), ("eda_builder", builder_prompt)):
            with self.subTest(label):
                self.assertIn(TERSE_OUTPUT_LINE, prompt)

    def test_eda_calls_pass_their_own_output_cap_and_retry_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = ScriptedProvider([dict(EDA_PLAN_PAYLOAD), dict(EDA_REPORT_PAYLOAD)])
            roles = ResearchRoles(
                provider,
                MethodCatalog.load(REPO_ROOT / "research" / "methods"),
                ResearchAudit(Path(directory) / "run"),
                max_total_tokens=10000,
                eda_researcher_max_output_tokens=6000,
                eda_builder_max_output_tokens=7000,
                eda_max_retries=2,
            )
            state = RunState("run", "running", "now", 0.6016)
            plan = roles.eda_research(state, 1)
            roles.eda_build(state, 1, plan)
        self.assertEqual(provider.calls[0]["max_output_tokens"], 6000)
        self.assertEqual(provider.calls[1]["max_output_tokens"], 7000)
        for call in provider.calls:
            self.assertEqual(call["max_retries"], 2)


class ShippedEDACapTests(unittest.TestCase):
    """The caps the failing run used are the ones the configs carry."""

    def test_kj_configs_leave_headroom_over_the_observed_successful_passes(self):
        # The largest successful EDA pass in the failing run spent 4,635 output
        # tokens under a 5,000 ceiling. Anything at or below that is a ceiling
        # the next slightly-longer reasoning trace walks straight into.
        for name in ("run_kj.json", "run_kj_smoke.json"):
            with self.subTest(name):
                config = json.loads((REPO_ROOT / "configs" / name).read_text(encoding="utf-8"))
                eda = config["eda"]
                self.assertGreaterEqual(eda["researcher_max_output_tokens"], 6000)
                self.assertGreaterEqual(eda["builder_max_output_tokens"], 7000)
                # Still well under the per-call ceiling the other roles run at.
                self.assertLessEqual(
                    eda["builder_max_output_tokens"], config["llm"]["max_output_tokens"]
                )


if __name__ == "__main__":
    unittest.main()
