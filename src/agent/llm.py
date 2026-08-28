from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .types import TokenUsage


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_project_environment(env_path: str | Path | None = None) -> bool:
    """Load local configuration without replacing shell or CI variables."""
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError(
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc
    path = Path(env_path) if env_path is not None else PROJECT_ROOT / ".env"
    return bool(load_dotenv(dotenv_path=path, override=False))


PARAMETER_PROPERTIES = {
    "seed": {"type": "integer"},
    "k": {"type": "integer"},
    "learning_rate": {"type": "number"},
    "epochs": {"type": "integer"},
    "batch_size": {"type": "integer"},
    "patience": {"type": "integer"},
    "negatives_per_positive": {"type": ["integer", "null"]},
    "negatives_per_group": {"type": ["integer", "null"]},
    "temperature": {"type": ["number", "null"]},
}
PARAMETER_SCHEMA = {
    "type": "object",
    "properties": PARAMETER_PROPERTIES,
    "required": list(PARAMETER_PROPERTIES),
    "additionalProperties": False,
}


SCHEMAS: dict[str, dict[str, Any]] = {
    "research_decision": {
        "type": "object",
        "properties": {
            "hypothesis_id": {"type": "string"},
            "family": {"type": "string", "enum": ["bpr", "group_softmax"]},
            "action": {"type": "string", "enum": ["explore", "exploit", "replicate"]},
            "hypothesis": {"type": "string"},
            "rationale": {"type": "string"},
            "parameters": PARAMETER_SCHEMA,
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "method_card_id": {"type": "string"},
                    },
                    "required": ["title", "url", "method_card_id"],
                    "additionalProperties": False,
                },
            },
            "needs_web_search": {"type": "boolean"},
            "parent_experiment": {"type": ["string", "null"]},
        },
        "required": [
            "hypothesis_id",
            "family",
            "action",
            "hypothesis",
            "rationale",
            "parameters",
            "evidence",
            "needs_web_search",
            "parent_experiment",
        ],
        "additionalProperties": False,
    },
    "critic_decision": {
        "type": "object",
        "properties": {
            "approved": {"type": "boolean"},
            "decision": {"type": "string"},
            "rationale": {"type": "string"},
            "concerns": {"type": "array", "items": {"type": "string"}},
            "next_focus": {"type": "string"},
        },
        "required": ["approved", "decision", "rationale", "concerns", "next_focus"],
        "additionalProperties": False,
    },
    "candidate_manifest": {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string"},
            "hypothesis_id": {"type": "string"},
            "family": {"type": "string", "enum": ["bpr", "group_softmax"]},
            "code": {"type": "string"},
            "tests": {"type": "string"},
            "parameters": PARAMETER_SCHEMA,
        },
        "required": ["candidate_id", "hypothesis_id", "family", "code", "tests", "parameters"],
        "additionalProperties": False,
    },
    "debug_decision": {
        "type": "object",
        "properties": {
            "preserve_hypothesis": {"type": "boolean"},
            "diagnosis": {"type": "string"},
            "replacement_code": {"type": "string"},
            "replacement_tests": {"type": "string"},
        },
        "required": [
            "preserve_hypothesis",
            "diagnosis",
            "replacement_code",
            "replacement_tests",
        ],
        "additionalProperties": False,
    },
}


def normalize_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in parameters.items() if value is not None}


@dataclass
class LLMCallResult:
    data: dict[str, Any]
    response_id: str
    model: str
    role: str
    latency_seconds: float
    retries: int
    usage: TokenUsage
    tool_calls: list[str] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["usage"] = self.usage.to_dict()
        return record


class LLMProvider(Protocol):
    def complete(
        self,
        *,
        role: str,
        instructions: str,
        prompt: str,
        schema_name: str,
        allow_web_search: bool = False,
    ) -> LLMCallResult:
        ...


def _to_plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def _extract_sources(output: Any) -> tuple[list[str], list[dict[str, str]]]:
    plain = _to_plain(output)
    tool_calls: list[str] = []
    sources: list[dict[str, str]] = []
    for item in plain if isinstance(plain, list) else []:
        item_type = str(item.get("type", "")) if isinstance(item, dict) else ""
        if item_type.endswith("search_call") or "web_search" in item_type:
            tool_calls.append(item_type)
        action = item.get("action", {}) if isinstance(item, dict) else {}
        for source in action.get("sources", []) if isinstance(action, dict) else []:
            if isinstance(source, dict) and source.get("url"):
                sources.append(
                    {"title": str(source.get("title", source["url"])), "url": str(source["url"])}
                )
    unique = {(source["title"], source["url"]): source for source in sources}
    return tool_calls, list(unique.values())


class OpenAIResponsesProvider:
    def __init__(self, config: dict[str, Any]):
        load_project_environment()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for an autonomous research run.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install dependencies with: python -m pip install -r requirements.txt") from exc
        self.model = str(config.get("model", "gpt-5.5"))
        self.reasoning_effort = str(config.get("reasoning_effort", "medium"))
        self.verbosity = str(config.get("verbosity", "low"))
        self.store = bool(config.get("store", False))
        self.max_output_tokens = int(config.get("max_output_tokens", 24000))
        self.max_tool_calls = int(config.get("max_tool_calls", 2))
        self.max_retries = int(config.get("max_retries", 2))
        timeout = float(config.get("request_timeout_seconds", 180))
        self.client = OpenAI(api_key=api_key, timeout=timeout, max_retries=0)

    @staticmethod
    def _retryable(error: Exception) -> bool:
        status = getattr(error, "status_code", None)
        return status in {408, 409, 429} or (isinstance(status, int) and status >= 500) or error.__class__.__name__ in {
            "APITimeoutError",
            "APIConnectionError",
            "RateLimitError",
            "InternalServerError",
        }

    def complete(
        self,
        *,
        role: str,
        instructions: str,
        prompt: str,
        schema_name: str,
        allow_web_search: bool = False,
    ) -> LLMCallResult:
        if schema_name not in SCHEMAS:
            raise ValueError(f"Unknown structured-output schema: {schema_name}")
        tools = [{"type": "web_search"}] if allow_web_search else []
        request = {
            "model": self.model,
            "instructions": instructions,
            "input": prompt,
            "reasoning": {"effort": self.reasoning_effort},
            "text": {
                "verbosity": self.verbosity,
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": SCHEMAS[schema_name],
                    "strict": True,
                },
            },
            "store": self.store,
            "max_output_tokens": self.max_output_tokens,
            "prompt_cache_key": f"kuairand-research-{role}",
        }
        if tools:
            request.update(
                {
                    "tools": tools,
                    "max_tool_calls": self.max_tool_calls,
                    "include": ["web_search_call.action.sources"],
                }
            )

        started = time.monotonic()
        retries = 0
        while True:
            try:
                response = self.client.responses.create(**request)
                break
            except Exception as exc:
                if retries >= self.max_retries or not self._retryable(exc):
                    raise
                time.sleep(min(4.0, 0.5 * (2**retries)))
                retries += 1

        output_text = getattr(response, "output_text", "")
        if not output_text:
            raise ValueError(f"OpenAI response {getattr(response, 'id', '')} contained no output text.")
        data = json.loads(output_text)
        usage_obj = _to_plain(getattr(response, "usage", None)) or {}
        input_details = usage_obj.get("input_tokens_details", {}) or {}
        tool_calls, sources = _extract_sources(getattr(response, "output", []))
        usage = TokenUsage(
            input_tokens=int(usage_obj.get("input_tokens", 0)),
            output_tokens=int(usage_obj.get("output_tokens", 0)),
            total_tokens=int(usage_obj.get("total_tokens", 0)),
            cached_tokens=int(input_details.get("cached_tokens", 0)),
            web_search_calls=sum("web_search" in item for item in tool_calls),
        )
        return LLMCallResult(
            data=data,
            response_id=str(getattr(response, "id", "")),
            model=str(getattr(response, "model", self.model)),
            role=role,
            latency_seconds=time.monotonic() - started,
            retries=retries,
            usage=usage,
            tool_calls=tool_calls,
            sources=sources,
        )


class ScriptedProvider:
    """Deterministic provider for offline unit and integration tests."""

    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, **kwargs) -> LLMCallResult:
        if not self.responses:
            raise RuntimeError("ScriptedProvider has no response left.")
        self.calls.append(dict(kwargs))
        payload = self.responses.pop(0)
        usage = TokenUsage.from_dict(payload.pop("_usage", {"total_tokens": 10}))
        return LLMCallResult(
            data=payload,
            response_id=f"scripted-{len(self.calls)}",
            model="scripted",
            role=str(kwargs["role"]),
            latency_seconds=0.0,
            retries=0,
            usage=usage,
        )


def build_provider(config: dict[str, Any]) -> LLMProvider:
    """Build the LLM provider named by ``config["llm"]["provider"]``."""
    llm_config = config["llm"]
    provider = str(llm_config.get("provider", "openai"))
    if provider == "openai":
        return OpenAIResponsesProvider(llm_config)
    if provider == "scripted":
        script_path = Path(llm_config["script_path"])
        if not script_path.is_absolute():
            script_path = PROJECT_ROOT / script_path
        return ScriptedProvider(json.loads(script_path.read_text(encoding="utf-8")))
    raise ValueError(f"Unsupported LLM provider: {provider!r}")
