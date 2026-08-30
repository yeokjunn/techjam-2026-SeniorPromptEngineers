"""Registry of the approved research families (shared by types, policy, prompts and safety).

Each entry carries everything the rest of the harness needs to know about a family, so that
adding one is a single edit here rather than a literal list in four modules:

* ``grid`` / ``defaults`` -- the approved search space. ``policy.sanitize_parameters`` fills from
  ``defaults`` and rejects any value ``not in`` the matching grid entry; ``in`` is exact and O(1)
  for both a ``tuple`` of allowed values and a ``range`` integer bound.
* ``required_calls`` -- the trusted helpers a candidate must call, as a tuple of *one-of groups*:
  the candidate must call at least one name from each group. ``safety.validate_family_contract``
  enforces it against the AST. Empty means "just the trusted sampler".
* ``method_card`` -- the card ``MethodCatalog`` serves to the Researcher; the filename stem must
  equal the family name, because the catalog globs ``*.md`` and keys by stem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Family:
    name: str
    method_card: str  # repo-relative path, e.g. "research/methods/bpr.md"
    trusted_sampler: str  # function name in src.models.sampling
    # ``compare=False`` keeps the frozen dataclass hashable despite the dict fields.
    grid: dict[str, Any] = field(default_factory=dict, compare=False)
    defaults: dict[str, Any] = field(default_factory=dict, compare=False)
    required_calls: tuple[tuple[str, ...], ...] = ()


# Exactly today's values from policy.py:27-64, so pointing the sanitiser at the registry
# changes no behaviour for the two existing families.
SHARED_GRID: dict[str, Any] = {
    "seed": range(0, 1000),
    "k": (16,),
    "learning_rate": (0.0003, 0.0005, 0.001),
    "epochs": range(1, 41),
    "patience": range(1, 7),
}
SHARED_DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "k": 16,
    "learning_rate": 0.001,
    "epochs": 40,
    "patience": 4,
}

# Where each trusted helper lives, for prompt rendering only.
TRUSTED_CALL_MODULES = {
    "sample_bpr_pairs": "src.models.sampling",
    "sample_softmax_groups": "src.models.sampling",
    "build_features": "src.models.features",
    "build_aux_labels": "src.models.features",
}

TRUSTED_CALL_SIGNATURES = {
    "sample_bpr_pairs": "(users, labels, rng, negatives_per_positive)",
    "sample_softmax_groups": "(users, labels, rng, negatives_per_group)",
    "build_features": "(rows, spec)",
    "build_aux_labels": "(rows, spec)",
}

TRUSTED_CALL_RETURNS = {
    "sample_bpr_pairs": (
        "returns (positives, negatives), both int64 row indices of shape (n_pairs,), parallel "
        "and 1-D"
    ),
    "sample_softmax_groups": (
        "returns (positives, negatives) row indices: positives (n_groups,), negatives "
        "(n_groups, negatives_per_group) -- already 2-D, do NOT reshape"
    ),
    "build_features": (
        "returns (len(rows), enabled_groups) int32, already offset by field_offset. Call once "
        "per split with spec['split'] matching the rows you pass: the row count is checked "
        "against the trusted split"
    ),
    "build_aux_labels": "returns (len(rows), enabled_heads) float32 in [0, 1], train split only",
    "FMRanker": (
        "the trusted FM: sparse field-index gather plus Adam. Do NOT re-implement it with a "
        "dense one-hot matrix -- that overflows to NaN and breaks attribution against the "
        "official baseline"
    ),
}

# Literals, deliberately not imported from ``src.models.features``: ``types.py`` imports this
# module, so this import must stay light (no numpy). ``tests/test_features.py`` pins them equal
# to the feature module's own tuples, so the duplication cannot drift silently.
HISTORY_GROUPS = ("user_rate", "user_author", "user_tab", "recency", "video_age", "tab_cross")
AUX_HEADS = ("is_click", "is_like", "is_follow", "is_comment", "is_forward", "play_time")

#: Either trusted sampler satisfies the loss requirement for the feature-side families -- the
#: loss is not what they vary, so both are legitimate.
_EITHER_SAMPLER = ("sample_bpr_pairs", "sample_softmax_groups")

FAMILIES: dict[str, Family] = {
    "bpr": Family(
        name="bpr",
        method_card="research/methods/bpr.md",
        trusted_sampler="sample_bpr_pairs",
        grid={**SHARED_GRID, "batch_size": (2048, 4096), "negatives_per_positive": (1, 2)},
        defaults={**SHARED_DEFAULTS, "batch_size": 2048, "negatives_per_positive": 1},
    ),
    "group_softmax": Family(
        name="group_softmax",
        method_card="research/methods/group_softmax.md",
        trusted_sampler="sample_softmax_groups",
        grid={
            **SHARED_GRID,
            "batch_size": (512, 1024, 2048),
            "negatives_per_group": (4, 8),
            "temperature": (0.5, 1.0, 2.0),
        },
        defaults={
            **SHARED_DEFAULTS,
            "batch_size": 2048,
            "negatives_per_group": 4,
            "temperature": 1.0,
        },
    ),
    # The loss is unchanged; the *field set* is the axis under test. epochs caps at 20 because
    # six extra fields roughly double the gather/scatter cost (one FM epoch ~12s) against
    # experiment_timeout_seconds: 900. k stays 16 -- capacity is a measured dead end.
    "history_features": Family(
        name="history_features",
        method_card="research/methods/history_features.md",
        trusted_sampler="sample_bpr_pairs",
        grid={
            **SHARED_GRID,
            "epochs": range(1, 21),
            "batch_size": (2048, 4096),
            "negatives_per_positive": (1, 2),
            "smoothing": (5.0, 20.0, 100.0),
            "scheme": ("prior_days", "leave_one_out"),
            **{f"use_{group}": (True, False) for group in HISTORY_GROUPS},
        },
        defaults={
            **SHARED_DEFAULTS,
            "epochs": 20,
            "batch_size": 2048,
            "negatives_per_positive": 1,
            "smoothing": 20.0,
            "scheme": "prior_days",
            **{f"use_{group}": True for group in HISTORY_GROUPS},
        },
        required_calls=(_EITHER_SAMPLER, ("build_features",)),
    ),
    # Auxiliary targets add a loss term, not FM fields, so the epoch budget is bpr's.
    # The conservative default is click-only at low weight: prior runs showed that enabling every
    # sparse/noisy head at 0.3 can swamp the ranking objective before the family gets a fair test.
    "multi_task": Family(
        name="multi_task",
        method_card="research/methods/multi_task.md",
        trusted_sampler="sample_bpr_pairs",
        grid={
            **SHARED_GRID,
            "batch_size": (2048, 4096),
            "negatives_per_positive": (1, 2),
            "aux_weight": (0.05, 0.1, 0.3, 1.0),
            **{f"use_{head}": (True, False) for head in AUX_HEADS},
        },
        defaults={
            **SHARED_DEFAULTS,
            "batch_size": 2048,
            "negatives_per_positive": 1,
            "aux_weight": 0.05,
            **{f"use_{head}": head == "is_click" for head in AUX_HEADS},
        },
        required_calls=(_EITHER_SAMPLER, ("build_aux_labels",)),
    ),
}

# The *minimum* coverage set the harness stop rule must satisfy.
COVERAGE_FAMILIES = frozenset({"bpr", "group_softmax", "history_features", "multi_task"})


def family_names() -> frozenset[str]:
    return frozenset(FAMILIES)


def coverage_families() -> frozenset[str]:
    """Families a run must cover before it may stop (see policy.coverage_complete)."""
    return COVERAGE_FAMILIES


def required_call_groups(family: str) -> tuple[tuple[str, ...], ...]:
    """One-of groups of trusted calls a ``family`` candidate must make.

    Raises ``KeyError`` for an unregistered family; callers translate that to their own error.
    """
    entry = FAMILIES[family]
    return entry.required_calls or ((entry.trusted_sampler,),)


def _qualified(call: str) -> str:
    module = TRUSTED_CALL_MODULES.get(call)
    return f"{module}.{call}" if module else call


def _render_grid_value(value: Any) -> str:
    if isinstance(value, range):
        return f"{value.start}-{value.stop - 1}"
    return ", ".join(repr(item) if isinstance(item, str) else str(item) for item in value)


def _signature(call: str) -> str:
    qualified = _qualified(call)
    sig = TRUSTED_CALL_SIGNATURES.get(call, "()")
    rendered = f"{qualified}{sig}"
    returns = TRUSTED_CALL_RETURNS.get(call)
    return f"{rendered} -- {returns}" if returns else rendered


def builder_brief(name: str) -> str:
    """Prompt text for the Builder: the mandatory trusted calls plus the approved grid.

    Consumed by ``roles.py`` so no role prompt has to track the family list by hand.
    """
    entry = FAMILIES[name]
    lines = [
        "candidate_id must match [A-Za-z0-9_-]{1,80} -- letters, digits, underscore and hyphen "
        "only. No dots, spaces or other punctuation (it becomes a directory name).",
    ]
    for group in required_call_groups(name):
        rendered = ", ".join(_signature(call) for call in group)
        if len(group) == 1:
            lines.append(f"You must call {rendered}.")
        else:
            lines.append(f"You must call at least one of: {rendered}.")
    if entry.grid:
        lines.append(f"Approved search space for {name} (values outside it are rejected):")
        lines.extend(
            f"  {key}: {_render_grid_value(entry.grid[key])}" for key in sorted(entry.grid)
        )
    return "\n".join(lines)
