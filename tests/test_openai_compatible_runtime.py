import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.agent.errors import RoleOutputInvalid
from src.agent.llm import OpenAICompatibleChatProvider, build_provider


VALID_CRITIC = {
    "approved": True,
    "decision": "proceed",
    "rationale": "valid",
    "concerns": [],
    "next_focus": "run",
}


class FakeChatCompletions:
    def __init__(self, content=None):
        self.content = content if content is not None else json.dumps(VALID_CRITIC)
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            id="chatcmpl-test",
            model="glm-test",
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage={
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 4},
            },
            web_search=[
                {"title": "Primary source", "link": "https://example.test/source"}
            ],
        )


def provider_config(**overrides):
    config = {
        "provider": "openai_compatible",
        "model": "glm-test",
        "base_url": "https://example.test/api/paas/v4/chat/completions",
        "max_output_tokens": 1234,
        "max_retries": 1,
    }
    config.update(overrides)
    return config


class OpenAICompatibleRuntimeTests(unittest.TestCase):
    def make_provider(self, **overrides):
        with patch("src.agent.llm.load_project_environment", return_value=False), patch.dict(
            os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False
        ):
            return OpenAICompatibleChatProvider(provider_config(**overrides))

    def test_full_chat_endpoint_is_normalized_for_the_openai_client(self):
        provider = self.make_provider()
        self.assertEqual(str(provider.client.base_url), "https://example.test/api/paas/v4/")

    def test_chat_request_uses_prompt_json_contract_and_validates_the_result(self):
        provider = self.make_provider(temperature=0.4, thinking="enabled")
        fake = FakeChatCompletions()
        provider.client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake)
        )

        result = provider.complete(
            role="critic",
            instructions="Review the proposal.",
            prompt="Proceed?",
            schema_name="critic_decision",
        )

        self.assertEqual(result.data, VALID_CRITIC)
        self.assertEqual(result.model, "glm-test")
        self.assertEqual(result.usage.total_tokens, 20)
        self.assertEqual(result.usage.cached_tokens, 4)
        self.assertEqual(fake.request["model"], "glm-test")
        self.assertEqual(fake.request["max_tokens"], 1234)
        self.assertNotIn("response_format", fake.request)
        self.assertFalse(fake.request["stream"])
        self.assertEqual(fake.request["temperature"], 0.4)
        self.assertEqual(
            fake.request["extra_body"],
            {"thinking": {"type": "enabled"}, "reasoning_effort": "low"},
        )
        self.assertIn('"additionalProperties":false', fake.request["messages"][0]["content"])

    def test_role_temperature_overrides_global_temperature(self):
        provider = self.make_provider(
            temperature=0.4,
            role_temperatures={"builder": 0.1, "debugger": 0.1},
        )
        fake = FakeChatCompletions()
        provider.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))

        provider.complete(
            role="builder",
            instructions="Build the candidate.",
            prompt="Proceed?",
            schema_name="critic_decision",
        )

        self.assertEqual(fake.request["temperature"], 0.1)

    def test_json_mode_can_be_enabled_for_compatible_chat_endpoints(self):
        provider = self.make_provider(json_mode=True)
        fake = FakeChatCompletions()
        provider.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))

        provider.complete(
            role="critic",
            instructions="Review the proposal.",
            prompt="Proceed?",
            schema_name="critic_decision",
        )

        self.assertEqual(fake.request["response_format"], {"type": "json_object"})

    def test_invalid_schema_is_a_typed_role_output_error(self):
        provider = self.make_provider()
        fake = FakeChatCompletions(content=json.dumps({"approved": True}))
        provider.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))

        with self.assertRaises(RoleOutputInvalid):
            provider.complete(
                role="critic",
                instructions="Review the proposal.",
                prompt="Proceed?",
                schema_name="critic_decision",
            )

    def test_glm_factory_alias_and_configurable_key_environment(self):
        config = provider_config(provider="glm", api_key_env="ZAI_API_KEY")
        with patch("src.agent.llm.load_project_environment", return_value=False), patch.dict(
            os.environ, {"ZAI_API_KEY": "test-key"}, clear=False
        ):
            provider = build_provider({"llm": config})
        self.assertIsInstance(provider, OpenAICompatibleChatProvider)

    def test_web_search_is_opt_in_and_sources_are_accounted(self):
        tool = {
            "type": "web_search",
            "web_search": {"enable": True, "search_result": True},
        }
        provider = self.make_provider(web_search_tool=tool)
        fake = FakeChatCompletions()
        provider.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))

        result = provider.complete(
            role="researcher_web",
            instructions="Research.",
            prompt="Find evidence.",
            schema_name="critic_decision",
            allow_web_search=True,
        )

        self.assertEqual(fake.request["tools"], [tool])
        self.assertEqual(result.usage.web_search_calls, 1)
        self.assertEqual(result.sources[0]["url"], "https://example.test/source")


if __name__ == "__main__":
    unittest.main()
