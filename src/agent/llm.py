from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .errors import IncompleteResponse, LLMError, RoleOutputInvalid, TokenBudgetExceeded
from .families import family_names
from .types import TokenUsage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RETRY_ATTEMPTS = 5
BACKOFF_INITIAL_SECONDS = 2.0
BACKOFF_MAX_SECONDS = 60.0
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})


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

# Build family enums dynamically from the registry
FAMILY_ENUM = sorted(family_names())


SCHEMAS: dict[str, dict[str, Any]] = {
    "eda_research_plan": {
        "type": "object",
        "properties": {
            "objective": {"type": "string", "maxLength": 260},
            "questions": {
                "type": "array",
                "items": {"type": "string", "maxLength": 220},
                "maxItems": 4,
            },
            "feature_hypotheses": {
                "type": "array",
                "items": {"type": "string", "maxLength": 240},
                "maxItems": 4,
            },
            "required_inputs": {
                "type": "array",
                "items": {"type": "string", "maxLength": 160},
                "maxItems": 4,
            },
            "leakage_risks": {
                "type": "array",
                "items": {"type": "string", "maxLength": 220},
                "maxItems": 4,
            },
            "expected_artifacts": {
                "type": "array",
                "items": {"type": "string", "maxLength": 180},
                "maxItems": 4,
            },
        },
        "required": [
            "objective",
            "questions",
            "feature_hypotheses",
            "required_inputs",
            "leakage_risks",
            "expected_artifacts",
        ],
        "additionalProperties": False,
    },
    "eda_report": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "maxLength": 360},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "maxLength": 80},
                        "observation": {"type": "string", "maxLength": 220},
                        "implication": {"type": "string", "maxLength": 220},
                        "evidence": {"type": "string", "maxLength": 180},
                        "leakage_safe": {"type": "boolean"},
                    },
                    "required": [
                        "title",
                        "observation",
                        "implication",
                        "evidence",
                        "leakage_safe",
                    ],
                    "additionalProperties": False,
                },
                "maxItems": 3,
            },
            "feature_candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "maxLength": 80},
                        "description": {"type": "string", "maxLength": 220},
                        "family": {"type": "string", "maxLength": 40},
                        "expected_impact": {"type": "string", "maxLength": 160},
                        "implementation_scope": {"type": "string", "maxLength": 180},
                        "leakage_risk": {"type": "string", "maxLength": 180},
                    },
                    "required": [
                        "name",
                        "description",
                        "family",
                        "expected_impact",
                        "implementation_scope",
                        "leakage_risk",
                    ],
                    "additionalProperties": False,
                },
                "maxItems": 3,
            },
            "recommended_next_focus": {"type": "string", "maxLength": 300},
            "ui_notes": {
                "type": "array",
                "items": {"type": "string", "maxLength": 160},
                "maxItems": 3,
            },
        },
        "required": [
            "summary",
            "findings",
            "feature_candidates",
            "recommended_next_focus",
            "ui_notes",
        ],
        "additionalProperties": False,
    },
    "research_decision": {
        "type": "object",
        "properties": {
            "hypothesis_id": {"type": "string"},
            "family": {"type": "string", "enum": FAMILY_ENUM},
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
            "family": {"type": "string", "enum": FAMILY_ENUM},
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
        max_output_tokens: int | None = None,
        max_retries: int | None = None,
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


def _parse_json(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
            cleaned = "\n".join(lines[1:-1]).strip()
        else:
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:].strip()
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:].strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
    
    if not (cleaned.startswith("{") or cleaned.startswith("[")):
        start = -1
        for i, char in enumerate(cleaned):
            if char in ("{", "["):
                start = i
                break
        end = -1
        for i, char in enumerate(reversed(cleaned)):
            if char in ("}", "]"):
                end = len(cleaned) - i
                break
        if start != -1 and end != -1 and start < end:
            cleaned = cleaned[start:end]
            
    return json.loads(cleaned)


class OpenAIResponsesProvider:
    def __init__(self, config: dict[str, Any]):
        load_project_environment()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for an autonomous research run.")
        try:
            from openai import (
                APIConnectionError,
                APIStatusError,
                APITimeoutError,
                OpenAI,
                RateLimitError,
            )
        except ImportError as exc:
            raise RuntimeError("Install dependencies with: python -m pip install -r requirements.txt") from exc
        self.model = str(config.get("model", "gpt-5-nano"))
        self.reasoning_effort = str(config.get("reasoning_effort", "medium"))
        self.verbosity = str(config.get("verbosity", "low"))
        self.store = bool(config.get("store", False))
        self.max_output_tokens = int(config.get("max_output_tokens", 24000))
        self.max_tool_calls = int(config.get("max_tool_calls", 2))
        self.max_retries = int(config.get("max_retries", RETRY_ATTEMPTS))
        self._retryable = (APIConnectionError, APITimeoutError, RateLimitError)
        self._status_error = APIStatusError
        timeout = float(config.get("request_timeout_seconds", 180))
        self.client = OpenAI(api_key=api_key, timeout=timeout, max_retries=0)
        self.supports_web_search = True

    def _retry_delay(self, error: Exception, attempt: int) -> float | None:
        retryable = isinstance(error, self._retryable)
        if isinstance(error, self._status_error):
            status = getattr(error, "status_code", None)
            retryable = status in RETRYABLE_STATUS_CODES or (
                isinstance(status, int) and status >= 500
            )
        if not retryable:
            return None

        delay = min(
            BACKOFF_MAX_SECONDS,
            BACKOFF_INITIAL_SECONDS * (2**attempt),
        )
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        retry_after = headers.get("retry-after") if headers is not None else None
        if retry_after is not None:
            try:
                delay = max(0.0, min(BACKOFF_MAX_SECONDS, float(retry_after)))
            except (TypeError, ValueError):
                pass
        return delay

    def complete(
        self,
        *,
        role: str,
        instructions: str,
        prompt: str,
        schema_name: str,
        allow_web_search: bool = False,
        max_output_tokens: int | None = None,
        max_retries: int | None = None,
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
            "max_output_tokens": int(max_output_tokens or self.max_output_tokens),
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
        max_attempts = max(1, int(max_retries or self.max_retries))
        response = None
        retries = 0
        for attempt in range(max_attempts):
            try:
                response = self.client.responses.create(**request)
            except Exception as exc:
                delay = self._retry_delay(exc, attempt)
                if delay is None or attempt + 1 >= max_attempts:
                    raise
                time.sleep(delay)
                retries += 1
                continue

            status = str(getattr(response, "status", "") or "")
            output_text = getattr(response, "output_text", "") or ""
            if status == "incomplete" or not output_text:
                details = getattr(response, "incomplete_details", None)
                error = IncompleteResponse(
                    f"OpenAI response {getattr(response, 'id', '')} was incomplete "
                    f"or contained no output text; details={details!r}."
                )
                if attempt + 1 >= max_attempts:
                    raise error
                time.sleep(
                    min(
                        BACKOFF_MAX_SECONDS,
                        BACKOFF_INITIAL_SECONDS * (2**attempt),
                    )
                )
                retries += 1
                continue
            break

        if response is None:
            raise IncompleteResponse("OpenAI did not return a response.")
        output_text = getattr(response, "output_text", "") or ""
        try:
            data = _parse_json(output_text)
        except json.JSONDecodeError as exc:
            raise RoleOutputInvalid(
                f"OpenAI response {getattr(response, 'id', '')} contained invalid JSON."
            ) from exc
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


def _chat_base_url(value: str) -> str:
    """Accept either an API base URL or a full Chat Completions endpoint."""
    base_url = value.strip().rstrip("/")
    suffix = "/chat/completions"
    if base_url.endswith(suffix):
        base_url = base_url[: -len(suffix)]
    return base_url


class OpenAICompatibleChatProvider:
    """Structured-output adapter for GLM and other OpenAI-compatible chat APIs."""

    def __init__(self, config: dict[str, Any]):
        load_project_environment()
        self.api_key_env = str(config.get("api_key_env", "OPENAI_API_KEY"))
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{self.api_key_env} is required for an OpenAI-compatible research run."
            )

        base_url_env = str(config.get("base_url_env", "OPENAI_BASE_URL"))
        configured_url = config.get("base_url", config.get("endpoint"))
        base_url = _chat_base_url(str(configured_url or os.environ.get(base_url_env, "")))
        if not base_url:
            raise RuntimeError(
                "Set llm.base_url (or llm.endpoint) in the config, or set "
                f"{base_url_env}, for an OpenAI-compatible research run."
            )

        try:
            from jsonschema import ValidationError, validate
            from openai import (
                APIConnectionError,
                APIStatusError,
                APITimeoutError,
                OpenAI,
                RateLimitError,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Install dependencies with: python -m pip install -r requirements.txt"
            ) from exc

        self.model = str(
            config.get("model")
            or os.environ.get("GLM_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or "glm-5.3"
        )
        self.provider_name = str(config.get("provider", "openai_compatible"))
        self.max_output_tokens = int(config.get("max_output_tokens", 24000))
        self.max_retries = int(config.get("max_retries", RETRY_ATTEMPTS))
        self.temperature = config.get("temperature")
        self.role_temperatures = dict(config.get("role_temperatures") or {})
        self.thinking = config.get("thinking")
        self.response_format = config.get("response_format")
        if self.response_format is None and bool(config.get("json_mode", False)):
            self.response_format = {"type": "json_object"}
        self.web_search_tool = config.get("web_search_tool")
        self.supports_web_search = isinstance(self.web_search_tool, dict)
        self._validate = validate
        self._validation_error = ValidationError
        self._retryable = (APIConnectionError, APITimeoutError, RateLimitError)
        self._status_error = APIStatusError
        timeout = float(config.get("request_timeout_seconds", 180))
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )

    def _retry_delay(self, error: Exception, attempt: int) -> float | None:
        retryable = isinstance(error, self._retryable)
        if isinstance(error, self._status_error):
            status = getattr(error, "status_code", None)
            retryable = status in RETRYABLE_STATUS_CODES or (
                isinstance(status, int) and status >= 500
            )
        if not retryable:
            return None

        delay = min(BACKOFF_MAX_SECONDS, BACKOFF_INITIAL_SECONDS * (2**attempt))
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        retry_after = headers.get("retry-after") if headers is not None else None
        if retry_after is not None:
            try:
                delay = max(0.0, min(BACKOFF_MAX_SECONDS, float(retry_after)))
            except (TypeError, ValueError):
                pass
        return delay

    def complete(
        self,
        *,
        role: str,
        instructions: str,
        prompt: str,
        schema_name: str,
        allow_web_search: bool = False,
        max_output_tokens: int | None = None,
        max_retries: int | None = None,
    ) -> LLMCallResult:
        if schema_name not in SCHEMAS:
            raise ValueError(f"Unknown structured-output schema: {schema_name}")
        if allow_web_search and not isinstance(self.web_search_tool, dict):
            raise LLMError(
                f"Provider {self.provider_name!r} has no portable web-search tool. "
                "Configure llm.web_search_tool if this endpoint supports one."
            )

        schema = SCHEMAS[schema_name]
        schema_instruction = (
            "Return only one valid JSON object matching this JSON Schema exactly. "
            "Do not wrap it in Markdown.\n"
            + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        )
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": f"{instructions}\n\n{schema_instruction}"},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "max_tokens": int(max_output_tokens or self.max_output_tokens),
        }
        if isinstance(self.response_format, dict):
            request["response_format"] = self.response_format
        temperature = self.role_temperatures.get(role, self.temperature)
        if temperature is not None:
            request["temperature"] = float(temperature)
        extra_body: dict[str, Any] = {}
        if self.thinking is not None:
            thinking_value = str(self.thinking).lower()
            if isinstance(self.thinking, dict):
                extra_body["thinking"] = self.thinking
            else:
                effort = thinking_value if thinking_value in {"low", "high", "max"} else "low"
                extra_body["thinking"] = {"type": "enabled"}
                extra_body["reasoning_effort"] = effort
        if extra_body:
            request["extra_body"] = extra_body
        if allow_web_search:
            request["tools"] = [self.web_search_tool]

        started = time.monotonic()
        max_attempts = max(1, int(max_retries or self.max_retries))
        response = None
        output_text = ""
        retries = 0
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(**request)
            except Exception as exc:
                delay = self._retry_delay(exc, attempt)
                if delay is None or attempt + 1 >= max_attempts:
                    raise
                time.sleep(delay)
                retries += 1
                continue

            choices = getattr(response, "choices", None) or []
            message = getattr(choices[0], "message", None) if choices else None
            output_text = str(getattr(message, "content", "") or "")
            if output_text:
                break
            error = IncompleteResponse(
                f"{self.provider_name} response {getattr(response, 'id', '')} "
                "contained no output text."
            )
            if attempt + 1 >= max_attempts:
                raise error
            time.sleep(min(BACKOFF_MAX_SECONDS, BACKOFF_INITIAL_SECONDS * (2**attempt)))
            retries += 1

        if response is None:
            raise IncompleteResponse(f"{self.provider_name} did not return a response.")
        try:
            data = _parse_json(output_text)
        except json.JSONDecodeError as exc:
            raise RoleOutputInvalid(
                f"{self.provider_name} response {getattr(response, 'id', '')} "
                "contained invalid JSON."
            ) from exc
        try:
            self._validate(instance=data, schema=schema)
        except self._validation_error as exc:
            raise RoleOutputInvalid(
                f"{self.provider_name} response {getattr(response, 'id', '')} "
                f"failed schema {schema_name!r}: {exc.message}"
            ) from exc

        usage_obj = _to_plain(getattr(response, "usage", None)) or {}
        input_details = usage_obj.get("prompt_tokens_details", {}) or {}
        web_results = _to_plain(getattr(response, "web_search", None)) or []
        sources = [
            {
                "title": str(item.get("title", item.get("link", ""))),
                "url": str(item.get("link", "")),
            }
            for item in web_results
            if isinstance(item, dict) and item.get("link")
        ]
        usage = TokenUsage(
            input_tokens=int(usage_obj.get("prompt_tokens", 0)),
            output_tokens=int(usage_obj.get("completion_tokens", 0)),
            total_tokens=int(usage_obj.get("total_tokens", 0)),
            cached_tokens=int(input_details.get("cached_tokens", 0)),
            web_search_calls=int(bool(allow_web_search and self.web_search_tool)),
        )
        return LLMCallResult(
            data=data,
            response_id=str(getattr(response, "id", "")),
            model=str(getattr(response, "model", self.model)),
            role=role,
            latency_seconds=time.monotonic() - started,
            retries=retries,
            usage=usage,
            tool_calls=["web_search"] if allow_web_search else [],
            sources=sources,
        )


class ScriptedProvider:
    """Deterministic provider for offline unit and integration tests."""

    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.supports_web_search = True

    def complete(self, **kwargs) -> LLMCallResult:
        if not self.responses:
            raise RuntimeError("ScriptedProvider has no response left.")
        self.calls.append(dict(kwargs))
        payload = self.responses.pop(0)
        usage = TokenUsage.from_dict(payload.get("_usage", {"total_tokens": 10}))
        return LLMCallResult(
            data={key: value for key, value in payload.items() if key != "_usage"},
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
    provider = str(llm_config.get("provider", "glm")).lower().replace("-", "_")
    if provider in {"openai_compatible", "glm"}:
        return OpenAICompatibleChatProvider(llm_config)
    if provider == "openai":
        return OpenAIResponsesProvider(llm_config)
    if provider == "scripted":
        script_path = Path(llm_config["script_path"])
        if not script_path.is_absolute():
            script_path = PROJECT_ROOT / script_path
        return ScriptedProvider(json.loads(script_path.read_text(encoding="utf-8")))
    raise ValueError(f"Unsupported LLM provider: {provider!r}")

