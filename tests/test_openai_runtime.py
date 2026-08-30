import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.agent.llm import OpenAIResponsesProvider, load_project_environment


class FakeResponses:
    def __init__(self):
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            id="response-test",
            model="GLM-5.3-Flash",
            output_text=json.dumps(
                {
                    "approved": True,
                    "decision": "proceed",
                    "rationale": "valid",
                    "concerns": [],
                    "next_focus": "run",
                }
            ),
            output=[],
            usage={
                "input_tokens": 12,
                "output_tokens": 8,
                "total_tokens": 20,
                "input_tokens_details": {"cached_tokens": 4},
            },
        )


class OpenAIRuntimeTests(unittest.TestCase):
    def test_dotenv_loads_api_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("OPENAI_API_KEY=dotenv-key\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                self.assertTrue(load_project_environment(env_path))
                self.assertEqual(os.environ["OPENAI_API_KEY"], "dotenv-key")

    def test_dotenv_does_not_override_existing_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("OPENAI_API_KEY=dotenv-key\n", encoding="utf-8")
            with patch.dict(os.environ, {"OPENAI_API_KEY": "shell-key"}, clear=True):
                self.assertTrue(load_project_environment(env_path))
                self.assertEqual(os.environ["OPENAI_API_KEY"], "shell-key")

    def test_missing_api_key_fails_before_a_request(self):
        with patch("src.agent.llm.load_project_environment", return_value=False), patch.dict(
            os.environ, {}, clear=False
        ):
            os.environ.pop("OPENAI_API_KEY", None)
            with self.assertRaises(RuntimeError):
                OpenAIResponsesProvider({})

    def test_structured_responses_request_and_usage_accounting(self):
        with patch("src.agent.llm.load_project_environment", return_value=False), patch.dict(
            os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False
        ):
            provider = OpenAIResponsesProvider(
                {
                    "model": "GLM-5.3-Flash",
                    "reasoning_effort": "medium",
                    "verbosity": "low",
                    "store": False,
                    "max_retries": 0,
                }
            )
        signature = inspect.signature(provider.client.responses.create)
        for parameter in ("text", "store", "reasoning", "tools", "include"):
            self.assertIn(parameter, signature.parameters)
        fake = FakeResponses()
        provider.client = SimpleNamespace(responses=fake)
        result = provider.complete(
            role="critic",
            instructions="trusted",
            prompt="inspect",
            schema_name="critic_decision",
        )
        self.assertFalse(fake.request["store"])
        self.assertEqual(fake.request["reasoning"]["effort"], "medium")
        self.assertEqual(fake.request["text"]["verbosity"], "low")
        self.assertEqual(fake.request["text"]["format"]["type"], "json_schema")
        self.assertEqual(result.usage.total_tokens, 20)
        self.assertEqual(result.usage.cached_tokens, 4)


if __name__ == "__main__":
    unittest.main()
