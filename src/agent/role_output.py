"""Narrow output contract layer for normalizing and adjusting LLM role payloads.

This module provides deterministic normalization and lightweight schema adjustment
without creating a redundant agent role. It prevents unnecessary role reprompts for
superficial JSON formatting/escaping/coercion issues while strictly preserving intent.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from .errors import RoleOutputInvalid
from .families import FAMILIES


PROTECTED_KEYS = frozenset({"family", "hypothesis_id", "candidate_id", "code", "tests"})


@dataclass(frozen=True)
class RoleOutputAdjustment:
    adjusted: bool
    adjustment_reason: str
    raw_response_sha256: str
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_text(text: str) -> str:
    """Compute SHA256 hex digest of raw response text."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def extract_json_like(text: str) -> str:
    """Extract candidate JSON string from prose, markdown fences, or extra whitespace.

    Handles:
    - UTF-8 BOM and leading/trailing whitespace
    - Markdown code fences (```json ... ``` or ``` ... ```)
    - Prose surrounding JSON (e.g. "Here is the JSON output: {...}")
    """
    if not text:
        return ""
    # Strip BOM and whitespace
    cleaned = text.lstrip("\ufeff").strip()

    # 1. Look for markdown fenced code blocks
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(fence_pattern, cleaned, re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()

    # 2. If it already starts with { and ends with }, return as-is
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned

    # 3. Find the outermost bracket pair '{ ... }'
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        return cleaned[first_brace : last_brace + 1].strip()

    return cleaned


def normalize_role_payload(
    schema_name: str,
    data: dict[str, Any],
    family: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Deterministically normalize LLM output payload against schema & family defaults.

    Returns (normalized_dict, list_of_adjustments_made).

    Safety constraints:
    - Never mutates `family`, `hypothesis_id`, `candidate_id`, `code`, or `tests` unless type coercion.
    - Coerces numbers passed as numeric strings ("2048" -> 2048).
    - Coerces boolean strings ("true" / "false" -> True / False).
    - Injects missing default family parameters for parameter dictionaries.
    - Strips additional properties if additionalProperties is False in schema.
    """
    if not isinstance(data, dict):
        return data, []

    adjustments: list[str] = []
    result = dict(data)

    from .llm import SCHEMAS

    schema = SCHEMAS.get(schema_name)
    schema_props = schema.get("properties", {}) if schema else {}
    allow_additional = schema.get("additionalProperties", True) if schema else True

    # 1. Remove unrecognized additional properties if additionalProperties is False
    if not allow_additional and schema_props:
        extra_keys = [k for k in result if k not in schema_props]
        for k in extra_keys:
            del result[k]
            adjustments.append(f"Removed unrecognized property '{k}'")

    # 2. Coerce types based on schema property expectations
    for key, prop_def in schema_props.items():
        if key not in result or result[key] is None:
            continue
        val = result[key]
        expected_type = prop_def.get("type")

        # Handle numeric string coercion to int
        if expected_type == "integer" and isinstance(val, str) and not isinstance(val, bool):
            try:
                coerced = int(val.strip())
                result[key] = coerced
                adjustments.append(f"Coerced '{key}' string '{val}' to int {coerced}")
            except ValueError:
                pass
        # Handle numeric string coercion to float
        elif expected_type == "number" and isinstance(val, str) and not isinstance(val, bool):
            try:
                coerced = float(val.strip())
                result[key] = coerced
                adjustments.append(f"Coerced '{key}' string '{val}' to float {coerced}")
            except ValueError:
                pass
        # Handle boolean string coercion
        elif expected_type == "boolean" and isinstance(val, str):
            lowered = val.strip().lower()
            if lowered in {"true", "1"}:
                result[key] = True
                adjustments.append(f"Coerced '{key}' string '{val}' to True")
            elif lowered in {"false", "0"}:
                result[key] = False
                adjustments.append(f"Coerced '{key}' string '{val}' to False")

    # 3. Handle parameter dictionary defaults (e.g. inside research_decision or candidate_manifest)
    if "parameters" in result and isinstance(result["parameters"], dict):
        params = dict(result["parameters"])
        target_family = family or result.get("family")
        if target_family and target_family in FAMILIES:
            family_obj = FAMILIES[target_family]
            defaults = family_obj.defaults
            for p_name, default_val in defaults.items():
                if p_name not in params or params[p_name] is None:
                    params[p_name] = default_val
                    adjustments.append(
                        f"Injected default parameter '{p_name}'={default_val!r} for family {target_family!r}"
                    )
            # Type coerce parameters dict entries
            for p_name, p_val in list(params.items()):
                if isinstance(p_val, str) and not isinstance(p_val, bool):
                    lowered = p_val.strip().lower()
                    if lowered in {"true", "false"}:
                        params[p_name] = lowered == "true"
                        adjustments.append(
                            f"Coerced parameter '{p_name}' string '{p_val}' to boolean {params[p_name]}"
                        )
                    else:
                        try:
                            params[p_name] = int(p_val.strip())
                            adjustments.append(f"Coerced parameter '{p_name}' string '{p_val}' to int")
                            continue
                        except ValueError:
                            pass
                        try:
                            params[p_name] = float(p_val.strip())
                            adjustments.append(f"Coerced parameter '{p_name}' string '{p_val}' to float")
                        except ValueError:
                            pass
        result["parameters"] = params

    return result, adjustments


def enforce_protected_keys(
    original_data: dict[str, Any] | None,
    adjusted_data: dict[str, Any],
) -> None:
    """Verify that protected keys (family, hypothesis_id, candidate_id, code, tests) were not modified arbitrarily."""
    if not original_data:
        return
    for key in PROTECTED_KEYS:
        if key in original_data and key in adjusted_data:
            if original_data[key] != adjusted_data[key]:
                raise RoleOutputInvalid(
                    f"Adjustment mutated protected field {key!r} from {original_data[key]!r} to {adjusted_data[key]!r}"
                )
