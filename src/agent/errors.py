"""Typed harness exceptions shared across owners (frozen in Step 0b)."""

from __future__ import annotations


class LLMError(RuntimeError):
    """Base class for LLM-layer failures the controller can classify."""


class TokenBudgetExceeded(LLMError):
    """The run's total token budget is exhausted; the controller must stop."""


class RoleOutputInvalid(LLMError):
    """A role returned output that fails schema or consistency checks; re-prompt or skip."""


class IncompleteResponse(LLMError):
    """The model returned an incomplete/empty response (e.g. reasoning consumed max_output_tokens); retry."""
