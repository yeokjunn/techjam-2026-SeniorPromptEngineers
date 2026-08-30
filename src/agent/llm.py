from __future__ import annotations

import functools
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .errors import IncompleteResponse, LLMError, RoleOutputInvalid, TokenBudgetExceeded
from . import families
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
    # DIN / deep-learning knobs. ``embedding_dim`` is the item-history embedding
    # width (distinct from ``k``, the FM-block width, which stays pinned at 16).
    # The bool toggles are non-nullable so GLM's "true"/"false" string drift is
    # coerced by ``_coerce_bool``; an omitted toggle falls back to the family
    # default inside ``sanitize_parameters``.
    "embedding_dim": {"type": ["integer", "null"]},
    "seq_len": {"type": ["integer", "null"]},
    "attention_dim": {"type": ["integer", "null"]},
    "dropout": {"type": ["number", "null"]},
    "aux_weight": {"type": ["number", "null"]},
    "use_is_click": {"type": "boolean"},
    "use_play_time": {"type": "boolean"},
    "loss_variant": {"type": ["string", "null"]},
}
# ``additionalProperties: True`` so a strict provider (OpenAI Responses) does not
# reject family-specific keys not enumerated above (e.g. ``smoothing``/``scheme``/
# ``use_user_rate`` for history_features). Hallucinated junk still cannot pass:
# ``_normalize_parameters_dict`` preserves only PARAMETER_PROPERTIES keys and
# keys declared in some family's grid (``_grid_parameter_keys``), and
# ``sanitize_parameters`` then drops anything not in the chosen family's grid.
PARAMETER_SCHEMA = {
    "type": "object",
    "properties": PARAMETER_PROPERTIES,
    "required": list(PARAMETER_PROPERTIES),
    "additionalProperties": True,
}

# Build family enums dynamically from the registry
FAMILY_ENUM = sorted(family_names())


SCHEMAS: dict[str, dict[str, Any]] = {
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


def _parse_structured_output(output_text: str, response_id: str) -> dict[str, Any]:
    """Parse a structured role output, tolerating provider wrapping drift.

    Observed live on the GLM Responses endpoint despite ``strict`` json_schema
    formatting: markdown-fenced JSON (````` ```json `````), JSON with prose
    before or after it, and outright YAML documents. Try the fence-stripped
    body, the raw text, and the outermost brace slice in order; on failure
    embed the raw head and tail in the error so the drift stays diagnosable
    (pass records are only written on success).
    """
    text = output_text.strip()
    candidates: list[str] = []
    if text.startswith("```"):
        opener_end = text.find("\n")
        if opener_end != -1:
            body = text[opener_end + 1 :]
            stripped = body.rstrip()
            if stripped.endswith("```"):
                stripped = stripped[:-3]
            candidates.append(stripped.strip())
    candidates.append(text)
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        candidates.append(text[brace_start : brace_end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    head = output_text[:300].replace("\n", "\\n")
    tail = output_text[-200:].replace("\n", "\\n")
    raise RoleOutputInvalid(
        f"OpenAI response {response_id} contained invalid JSON. head={head!r} tail={tail!r}"
    )


_ACTION_SYNONYMS = {
    "propose_experiment": "explore",
    "propose": "explore",
    "new_experiment": "explore",
    "explore_new": "explore",
    "improve_best": "exploit",
    "exploit_best": "exploit",
    "refine": "exploit",
    "replicate_best": "replicate",
}


def _coerce_bool(value: Any) -> Any:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes"}:
            return True
        if lowered in {"false", "no"}:
            return False
    return value


_EVIDENCE_KEY_MAP = {
    "claim": "title",
    "source": "url",
    "card": "method_card_id",
    "method_card": "method_card_id",
}

_PARAMETER_ALIASES = {
    # ``embedding_dim`` is NOT aliased to ``k``: it is its own DIN knob (the
    # item-history embedding width), distinct from ``k`` (the FM-block width).
    # A GLM that says ``embedding_dim`` for a bpr/group_softmax candidate no
    # longer has it coerced to ``k`` — it is preserved, and since neither family
    # names it in its grid, ``sanitize_parameters`` drops it and ``k`` falls back
    # to its default of 16 (the only grid-allowed value), so no regression.
    "num_groups_negatives": "negatives_per_group",
    "negatives": "negatives_per_positive",
    "neg_per_positive": "negatives_per_positive",
    "lr": "learning_rate",
}


@functools.lru_cache(maxsize=1)
def _grid_parameter_keys() -> frozenset[str]:
    """Union of every key any family declares in its registry ``grid``.

    Used by ``_normalize_parameters_dict`` to preserve legitimate family-specific
    knobs (``smoothing``/``scheme``/``use_*``/``aux_weight``/``seq_len``/...) through
    the schema-normalization pass, so they reach ``sanitize_parameters`` — the
    single authority that drops hallucinated junk and validates grid values.
    Without this, the fixed ``PARAMETER_PROPERTIES`` set silently stripped
    family-specific keys, so the Researcher's choice was replaced by the default
    (the bug that kept history_features/multi_task from ever running).
    """
    keys: set[str] = set()
    for entry in families.FAMILIES.values():
        keys.update(entry.grid.keys())
    return frozenset(keys)


def _remap_evidence_item(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    remapped = dict(item)
    for source_key, target_key in _EVIDENCE_KEY_MAP.items():
        if source_key in remapped and target_key not in remapped:
            remapped[target_key] = remapped.pop(source_key)
    return remapped


def _normalize_parameters_dict(parameters: dict[str, Any]) -> dict[str, Any]:
    """Preserve known + family-grid parameter keys; drop hallucinated junk.

    A key survives if its alias target is in ``PARAMETER_PROPERTIES`` (the shared
    knobs) OR is declared in some family's registry grid (family-specific
    knobs like ``smoothing``/``seq_len``/``aux_weight``). Anything else is junk
    and is dropped here, so it can never reach ``sanitize_parameters`` — defence
    in depth behind the permissive ``PARAMETER_SCHEMA``.
    """
    allowed = PARAMETER_PROPERTIES.keys() | _grid_parameter_keys()
    remapped: dict[str, Any] = {}
    for name, value in parameters.items():
        target = _PARAMETER_ALIASES.get(name, name)
        if target in allowed and target not in remapped:
            remapped[target] = value
    return remapped


def _normalize_schema_output(schema_name: str, data: dict[str, Any]) -> dict[str, Any]:
    """Best-effort conformance pass for providers with loose json_schema enforcement.

    Some OpenAI-compatible providers (observed live: the GLM Responses endpoint
    returning ``action: "propose_experiment"`` and undeclared ``decision_basis``
    keys despite ``strict`` formatting) accept the schema but do not enforce
    enums, required fields, or ``additionalProperties``. Normalize mechanically
    derivable drift here so the strict ``from_dict`` validators in ``types.py``
    see conformant data; anything substantive still missing raises there and
    takes the normal re-prompt path.
    """
    schema = SCHEMAS[schema_name]
    properties = schema.get("properties", {})
    normalized: dict[str, Any] = {}
    for key, value in data.items():
        if key not in properties:
            continue  # drop undeclared extras (e.g. "control", "decision_basis")
        if properties[key].get("type") == "boolean":
            value = _coerce_bool(value)
        if key == "action" and isinstance(value, str):
            value = _ACTION_SYNONYMS.get(value.strip().lower(), value)
        if key == "family" and isinstance(value, str):
            candidate = value.strip().lower().replace(" ", "_").replace("-", "_")
            if candidate in FAMILY_ENUM:
                value = candidate
        if key == "evidence" and isinstance(value, list):
            value = [_remap_evidence_item(item) for item in value]
        if key == "parameters" and isinstance(value, dict):
            value = _normalize_parameters_dict(value)
        normalized[key] = value
    # A missing action is mechanically derivable from parent linkage: a proposal
    # without a parent explores; one building on a parent experiment exploits.
    if schema_name == "research_decision" and "action" not in normalized:
        parent = normalized.get("parent_experiment")
        normalized["action"] = "exploit" if parent else "explore"
    # A missing hypothesis_id is derivable deterministically from the content it
    # names; it only has to stay unique and stable for builder/ledger pinning.
    if schema_name in {"research_decision", "candidate_manifest"} and "hypothesis_id" not in normalized:
        basis = str(normalized.get("hypothesis", "")) or str(normalized.get("code", ""))
        family = str(normalized.get("family", "exp"))
        digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:8]
        normalized["hypothesis_id"] = f"{family}-auto-{digest}"
    # Auto-fill required fields that the LLM may omit or return as null:
    # evidence must be a list (empty if missing/null/non-list) so from_dict sees
    # a conformant type; needs_web_search defaults to False; parent_experiment
    # defaults to None. Without these, from_dict raises and costs a re-prompt.
    if schema_name == "research_decision":
        if "evidence" not in normalized or not isinstance(normalized.get("evidence"), list):
            normalized["evidence"] = []
        if "needs_web_search" not in normalized:
            normalized["needs_web_search"] = False
        if "parent_experiment" not in normalized:
            normalized["parent_experiment"] = None
    # candidate_manifest: parameters must be a dict (empty if missing/non-dict).
    if schema_name == "candidate_manifest":
        if "parameters" not in normalized or not isinstance(normalized.get("parameters"), dict):
            normalized["parameters"] = {}
    # critic_decision: concerns must be a list; approved must be present.
    if schema_name == "critic_decision":
        if "concerns" not in normalized or not isinstance(normalized.get("concerns"), list):
            normalized["concerns"] = []
        if "next_focus" not in normalized:
            normalized["next_focus"] = ""
    return normalized


def schema_fields_note(schema_name: str) -> str:
    """Render the exact response contract from ``SCHEMAS`` for role prompts.

    Providers that do not enforce ``text.format`` (observed live: the GLM
    Responses endpoint emitting YAML or self-invented structures) never surface
    the schema to the model, so the prompt itself must carry the field names,
    types, and enums. Rendering from ``SCHEMAS`` keeps this note true by
    construction instead of a second hand-maintained copy.
    """
    schema = SCHEMAS[schema_name]
    required = schema.get("required", [])
    lines = []
    for name, prop in schema.get("properties", {}).items():
        if "enum" in prop:
            rendered = "|".join(str(option) for option in prop["enum"])
        elif isinstance(prop.get("type"), list):
            rendered = "|".join(str(option) for option in prop["type"])
        else:
            rendered = str(prop.get("type", "value"))
        marker = "" if name in required else " (optional)"
        lines.append(f"- {name}: {rendered}{marker}")
    note = (
        "RESPONSE CONTRACT — respond with one raw JSON object (no YAML, no "
        "markdown fences, no prose) containing exactly these top-level fields:\n"
        + "\n".join(lines)
    )
    if schema_name == "research_decision":
        note += (
            "\nEach evidence item is an object with exactly the keys: title, url, method_card_id."
            "\nfamily MUST be a string (one of: " + "|".join(FAMILY_ENUM) + "), NEVER null."
            "\nevidence MUST be a list (use [] if no evidence), NEVER null."
            "\nneeds_web_search MUST be a boolean (true/false), NEVER null."
            "\nparameters is an object containing only these keys: "
            + ", ".join(PARAMETER_PROPERTIES)
            + " — use null for PARAMETER keys irrelevant to the chosen family,"
            "\n  but family/evidence/needs_web_search/parent_experiment must always have"
            "\n  a non-null value of the correct type."
        )
    if schema_name == "candidate_manifest":
        note += (
            "\ncode and tests are JSON strings holding complete Python source; escape all"
            "\nnewlines and quotes correctly. Keep candidate.py under 120 lines and"
            "\ntest_candidate.py under 60 lines."
            "\nUse only these exact import spellings (a static allowlist matches them"
            "\nliterally): import numpy as np; import time; from collections import ...;"
            "\nfrom src.models.fm_core import FMRanker;"
            "\nfrom src.models.sampling import sample_bpr_pairs, sample_softmax_groups;"
            "\nfrom src.models.features import build_features, build_aux_labels;"
            "\nfrom src.models.din_trainer import run_din_trainer;"
            "\nfrom src.models.sequence import build_user_sequences;"
            "\nfrom src.experiments.contracts import CandidateOutput."
            "\nNever write 'from src.models import ...' or 'import src.models' — import"
            "\nthe specific module. Never `import torch` — torch lives only inside the"
            "\ntrusted run_din_trainer primitive and is not on the candidate allowlist."
            "\nAccess context attributes directly (context.test_x is None); getattr,"
            "\nsetattr, eval, and exec are statically rejected."
            "\nTrusted primitive signatures — call them exactly like this:"
            "\n- FMRanker(dimension, embedding_dim=16, learning_rate=0.001, l2=1e-6, seed=0)"
            "\n  where dimension is the total feature-index count (context.field_dimension);"
            "\n  the embedding-size parameter is named embedding_dim, never k."
            "\n- model.gradients(features, score_gradients) -> (grad_v, grad_w, grad_b)"
            "\n- model.apply_gradients(grad_v, grad_w, grad_b=0.0)  # learning rate is"
            "\n  set at FMRanker construction, never passed per step"
            "\n- model.predict(features) -> scores"
            "\n- sample_bpr_pairs(users, labels, rng, negatives_per_positive=1)"
            "\n  -> (positive_idx, negative_idx)"
            "\n- sample_softmax_groups(users, labels, rng, negatives_per_group=4)"
            "\n  -> (positive_idx, negative_groups)"
            "\n  In both samplers rng is the THIRD argument."
            "\n- FMRanker instance attributes (USE EXACTLY THESE NAMES — they are"
            "\n  capitalized V, W, b, never v, w, or v_/linear_weights):"
            "\n  model.V  -> (dimension, embedding_dim) float32 embedding matrix"
            "\n  model.W  -> (dimension,) float32 linear weights"
            "\n  model.b  -> float32 scalar bias"
            "\n  model.state_dict() -> {'V': ..., 'W': ..., 'b': ...}  # USE THIS for checkpoint_state"
            "\n  model.load_state_dict({'V':..., 'W':..., 'b':...})  # restore a checkpoint"
            "\n  Never invent attribute names like v_, linear_weights, or v. If you"
            "\n  need the model parameters for the checkpoint, call model.state_dict()"
            "\n  and pass its return value as checkpoint_state — do not manually access"
            "\n  individual attributes unless you use the exact names V, W, b."
            "\n- CandidateOutput(validation_scores, checkpoint_state, training_trace,"
            "\n  diagnostics, test_scores)  # the dict field is named checkpoint_state"
            "\n  training_trace is a LIST of dicts (list[dict], never a dict itself);"
            "\n  checkpoint_state is a dict[str, np.ndarray]; diagnostics is a dict."
            "\n  In tests, assert isinstance(output.training_trace, list), not dict."
            "\n- context.train_x / valid_x / test_x are integer index matrices (never"
            "\n  float), shape (rows, n_fields), values < context.field_dimension; fake"
            "\n  contexts in tests must build integer arrays (e.g. rng.integers)."
            "\n- context.evaluate_validation(scores) -> dict with EXACTLY these keys"
            "\n  (capitalized, never lowercase): {'GAUC': float, 'nDCG@5': float,"
            "\n  'primary': float}. Access them as metrics['GAUC'], metrics['nDCG@5'],"
            "\n  metrics['primary']. NEVER use metrics['gauc'] or metrics['ndcg']."
            "\n  Use this ONLY for early-stopping/patience, never as an optimization"
            "\n  objective (never gradient-ascent on the metric)."
            "\n- For the 'din' family: call run_din_trainer(context, parameters) and"
            "\n  return its (validation_scores, test_scores, checkpoint_state,"
            "\n  training_trace, diagnostics) wrapped in CandidateOutput. The trusted"
            "\n  trainer reloads the kit rows, builds leakage-safe sequences via"
            "\n  build_user_sequences, and owns all torch math — the candidate never"
            "\n  imports torch, loads data, or constructs sequences itself."
            "\ntest_candidate.py must define unittest.TestCase classes; the runner is"
            "\n`python -m unittest`, and pytest-style bare functions collect zero tests."
        )
    return note


def _compute_retry_delay(
    error: Exception,
    retryable_types: tuple[type, ...],
    status_error_cls: type,
    attempt: int,
) -> float | None:
    """Seconds to wait before retrying ``error``, or ``None`` to not retry.

    Shared by the Responses and Chat-Completions providers so their retry
    policy (transient network errors, 408/409/429, 5xx, bounded exponential
    backoff honouring ``Retry-After``) cannot drift apart.
    """
    retryable = isinstance(error, retryable_types)
    if isinstance(error, status_error_cls):
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


class OpenAIChatProvider:
    """OpenAI-compatible Chat Completions provider (OpenRouter and similar).

    OpenRouter exposes an OpenAI-compatible endpoint at ``https://openrouter.ai/api/v1``
    but only the Chat Completions API, not the Responses API. This provider routes
    role calls through ``client.chat.completions.create`` and reuses the same
    hardened JSON parser (``_parse_structured_output``) and schema-normalization
    pass (``_normalize_schema_output``) as the Responses provider, because Chat
    Completions ``response_format`` enforcement is looser than Responses'
    ``text.format.strict`` — exactly the drift those helpers already absorb.

    Web search is not supported here: the Responses ``web_search`` tool has no
    Chat-Completions equivalent on OpenRouter. The harness treats web search as
    an optional evidence fallback only (the committed live run used zero web
    searches), so this drops ``allow_web_search`` silently.
    """

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
        self.model = str(config.get("model", "z-ai/glm-5.3-flash"))
        self.reasoning_effort = str(config.get("reasoning_effort", "medium"))
        self.max_output_tokens = int(config.get("max_output_tokens", 24000))
        self.max_retries = int(config.get("max_retries", RETRY_ATTEMPTS))
        # ``object`` (response_format json_object) is the most broadly supported
        # mode across OpenRouter-backed providers; the role prompt already carries
        # the full field contract via ``schema_fields_note``, and the parser
        # rescues any remaining drift. ``schema`` requests a non-strict
        # json_schema (supported by some providers); ``none`` omits response_format.
        self.json_mode = str(config.get("json_mode", "object"))
        # Whether to send the `reasoning` param. OpenRouter's Chat Completions
        # endpoint does NOT accept it (TypeError); the Responses API does. Default
        # off for the Chat provider; the config can opt in with `send_reasoning`.
        self.send_reasoning = bool(config.get("send_reasoning", False))
        self._retryable = (APIConnectionError, APITimeoutError, RateLimitError)
        self._status_error = APIStatusError
        timeout = float(config.get("request_timeout_seconds", 180))
        base_url = config.get("base_url") or os.environ.get("OPENROUTER_BASE_URL")
        client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout, "max_retries": 0}
        if base_url:
            client_kwargs["base_url"] = base_url
        # Optional OpenRouter attribution headers (shown on your usage dashboard).
        headers: dict[str, str] = {}
        referer = config.get("http_referer") or os.environ.get("OPENROUTER_HTTP_REFERER")
        title = config.get("x_title") or os.environ.get("OPENROUTER_X_TITLE")
        if referer:
            headers["HTTP-Referer"] = str(referer)
        if title:
            headers["X-Title"] = str(title)
        if headers:
            client_kwargs["default_headers"] = headers
        self.client = OpenAI(**client_kwargs)
        self._timeout = float(config.get("request_timeout_seconds", 180))
        # Store for _call_with_deadline's fresh-client-per-call pattern.
        self._api_key = api_key
        self._base_url = base_url
        self._default_headers = headers if headers else None

    def _retry_delay(self, error: Exception, attempt: int) -> float | None:
        return _compute_retry_delay(error, self._retryable, self._status_error, attempt)

    def _build_request(self, schema_name: str, instructions: str, prompt: str, with_reasoning: bool) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_output_tokens,
        }
        # The `reasoning` parameter is OpenAI Responses-API specific.
        # OpenRouter's Chat Completions endpoint rejects it with a TypeError.
        # GLM-5.3-flash on OpenRouter supports reasoning via a different
        # mechanism (the `reasoning` field is NOT a Chat Completions param).
        # Send it only if the config explicitly opts in via `send_reasoning`.
        if with_reasoning and self.reasoning_effort and self.reasoning_effort.lower() != "none" and self.send_reasoning:
            request["reasoning"] = {"effort": self.reasoning_effort}
        if self.json_mode == "schema":
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": SCHEMAS[schema_name],
                    "strict": False,
                },
            }
        elif self.json_mode == "object":
            request["response_format"] = {"type": "json_object"}
        # json_mode == "none" -> omit response_format entirely
        return request

    def _call_with_deadline(self, request: dict[str, Any], deadline: float):
        """Call the API with a hard wall-clock deadline.

        The OpenAI SDK's built-in timeout doesn't fire when OpenRouter keeps the
        TCP connection ESTABLISHED without sending data. This creates a FRESH
        client per call (with a short SDK timeout as backup) and wraps it in a
        thread. If the deadline passes, the thread is abandoned (daemon=True)
        and APITimeoutError is raised. The fresh client ensures the abandoned
        thread's connection doesn't block subsequent retries.
        """
        import threading

        # Fresh client per call with a short SDK timeout as backup. The shared
        # ``self.client`` may have a stale connection from a previous abandoned
        # thread, so each call gets its own client to avoid connection reuse.
        from openai import OpenAI
        fresh_client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=max(30, min(600, deadline - time.monotonic())),
            max_retries=0,
            default_headers=self._default_headers or None,
        )

        result: list = []  # [exception_or_response]

        def _do_call():
            try:
                result.append(fresh_client.chat.completions.create(**request))
            except Exception as exc:
                result.append(exc)

        thread = threading.Thread(target=_do_call, daemon=True)
        thread.start()
        remaining = max(0.1, deadline - time.monotonic())
        thread.join(timeout=remaining)
        if thread.is_alive():
            from openai import APITimeoutError
            raise APITimeoutError("Request exceeded the hard wall-clock deadline.")
        if not result:
            from openai import APITimeoutError
            raise APITimeoutError("Request thread produced no result before deadline.")
        if isinstance(result[0], Exception):
            raise result[0]
        return result[0]

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
        # Chat Completions has no Responses web_search tool; allow_web_search is ignored.

        started = time.monotonic()
        max_attempts = max(1, self.max_retries)
        response = None
        retries = 0
        data: dict[str, Any] | None = None
        with_reasoning = True
        for attempt in range(max_attempts):
            # Hard wall-clock guard: the OpenAI SDK's read timeout doesn't fire
            # when OpenRouter keeps the TCP connection ESTABLISHED without
            # sending data. Enforce our own deadline and raise if exceeded.
            deadline = time.monotonic() + self._timeout
            request = self._build_request(schema_name, instructions, prompt, with_reasoning)
            try:
                response = self._call_with_deadline(request, deadline)
            except Exception as exc:
                # A 400 often means the provider rejected the ``reasoning`` param
                # shape (or json_schema). Drop reasoning once and retry without
                # sleeping before giving up — cheaper than a full role re-prompt.
                if (
                    with_reasoning
                    and isinstance(exc, self._status_error)
                    and getattr(exc, "status_code", None) == 400
                ):
                    with_reasoning = False
                    retries += 1
                    continue
                delay = self._retry_delay(exc, attempt)
                if delay is None or attempt + 1 >= max_attempts:
                    raise
                time.sleep(delay)
                retries += 1
                continue

            choices = getattr(response, "choices", None) or []
            choice = choices[0] if choices else None
            output_text = ""
            finish_reason = ""
            if choice is not None:
                message = getattr(choice, "message", None)
                if message is not None:
                    output_text = getattr(message, "content", "") or ""
                finish_reason = str(getattr(choice, "finish_reason", "") or "")

            if not output_text:
                error = IncompleteResponse(
                    f"Chat completion {getattr(response, 'id', '')} returned no content; "
                    f"finish_reason={finish_reason!r}."
                )
                if attempt + 1 >= max_attempts:
                    raise error
                time.sleep(min(BACKOFF_MAX_SECONDS, BACKOFF_INITIAL_SECONDS * (2**attempt)))
                retries += 1
                continue
            try:
                data = _parse_structured_output(output_text, str(getattr(response, "id", "")))
                data = _normalize_schema_output(schema_name, data)
            except RoleOutputInvalid:
                # A malformed payload is worth one more sample, bounded by
                # max_retries, before paying the full role re-prompt cycle.
                if attempt + 1 >= max_attempts:
                    raise
                time.sleep(min(BACKOFF_MAX_SECONDS, BACKOFF_INITIAL_SECONDS * (2**attempt)))
                retries += 1
                continue
            break

        if response is None or data is None:
            raise IncompleteResponse("Chat completion did not return a parseable response.")
        usage_obj = _to_plain(getattr(response, "usage", None)) or {}
        prompt_details = usage_obj.get("prompt_tokens_details", {}) or {}
        # Chat Completions usage uses prompt_tokens/completion_tokens; fall back to
        # the Responses-style input_tokens/output_tokens names some providers echo.
        usage = TokenUsage(
            input_tokens=int(usage_obj.get("prompt_tokens", usage_obj.get("input_tokens", 0))),
            output_tokens=int(usage_obj.get("completion_tokens", usage_obj.get("output_tokens", 0))),
            total_tokens=int(usage_obj.get("total_tokens", 0)),
            cached_tokens=int(prompt_details.get("cached_tokens", 0)),
            web_search_calls=0,
        )
        return LLMCallResult(
            data=data,
            response_id=str(getattr(response, "id", "")),
            model=str(getattr(response, "model", self.model)),
            role=role,
            latency_seconds=time.monotonic() - started,
            retries=retries,
            usage=usage,
            tool_calls=[],
            sources=[],
        )


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
        self.model = str(config.get("model", "gpt-5.5"))
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

    def _retry_delay(self, error: Exception, attempt: int) -> float | None:
        return _compute_retry_delay(error, self._retryable, self._status_error, attempt)

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
        max_attempts = max(1, self.max_retries)
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
            try:
                data = _parse_structured_output(output_text, str(getattr(response, "id", "")))
                data = _normalize_schema_output(schema_name, data)
            except RoleOutputInvalid:
                # A malformed payload (observed live: YAML or truncated JSON from
                # glm-5.3-flash) is worth one more sample, bounded by max_retries,
                # before paying the full role re-prompt cycle.
                if attempt + 1 >= max_attempts:
                    raise
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
    provider = str(llm_config.get("provider", "openai"))
    if provider == "openai":
        return OpenAIResponsesProvider(llm_config)
    if provider in ("openrouter", "chat"):
        return OpenAIChatProvider(llm_config)
    if provider == "scripted":
        script_path = Path(llm_config["script_path"])
        if not script_path.is_absolute():
            script_path = PROJECT_ROOT / script_path
        return ScriptedProvider(json.loads(script_path.read_text(encoding="utf-8")))
    raise ValueError(f"Unsupported LLM provider: {provider!r}")
