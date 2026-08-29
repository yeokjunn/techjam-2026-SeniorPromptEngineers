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
}

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
}

# The *minimum* coverage set the harness stop rule must satisfy. Deliberately not
# ``family_names()``: every family added later would otherwise make the rule unsatisfiable.
COVERAGE_FAMILIES = frozenset({"bpr", "group_softmax"})


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


def builder_brief(name: str) -> str:
    """Prompt text for the Builder: the mandatory trusted calls plus the approved grid.

    Consumed by ``roles.py`` so no role prompt has to track the family list by hand.
    """
    entry = FAMILIES[name]
    lines = []
    for group in required_call_groups(name):
        rendered = ", ".join(f"{_qualified(call)}()" for call in group)
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
