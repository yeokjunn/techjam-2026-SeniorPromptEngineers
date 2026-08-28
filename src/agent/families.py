"""Registry of the approved research families (shared by types, policy and prompts)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Family:
    name: str
    method_card: str  # repo-relative path, e.g. "research/methods/bpr.md"
    trusted_sampler: str  # function name in src.models.sampling


FAMILIES: dict[str, Family] = {
    "bpr": Family("bpr", "research/methods/bpr.md", "sample_bpr_pairs"),
    "group_softmax": Family(
        "group_softmax", "research/methods/group_softmax.md", "sample_softmax_groups"
    ),
}


def family_names() -> frozenset[str]:
    return frozenset(FAMILIES)
