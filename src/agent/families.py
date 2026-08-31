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


# Exactly today's values from policy.py:51-110 (`_SHARED`'s fallbacks and the
# `_SHARED_BOUNDS`/`_FAMILY_BOUNDS` checks), so pointing the sanitiser at the registry
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

# The capacity axes for the two *loss* families. They were frozen: `k` was the single value
# 16, `l2` appeared in no grid at all (so the only value the trainer ever saw was FMRanker's
# own 1e-6 default) and `learning_rate` capped at 0.001.
#
# The pin on `k` cited the starter kit's k-sweep (8/16/32 -> 0.5895/0.5902/0.5887, flat,
# `kuairand-starter-kit/README.en.md:133-139`). That sweep was measured under **pointwise
# logloss**: it says capacity does not help a *pointwise* model, which is not evidence about
# where a pairwise or listwise objective saturates -- a ranking loss uses the embedding space
# differently, and the whole point of these two families is that the loss changed. So the
# axis is reopened, cheaply: 64 is 4x the pinned width on an FM whose cost is a gather over
# n_fields, not a dense matmul.
#
# `l2` is the regularisation knob that pairs with the extra capacity: `FMRanker.__init__`
# already accepts it (`src/models/fm_core.py:18`) and applies it in `apply_gradients`, so it
# needs no new trusted code -- only a grid entry, so that it can be proposed and sanitised at
# all. 0.0 is included deliberately: at k=8 the extra shrinkage is not obviously wanted, and
# "off" has to be expressible for the axis to be attributable.
#
# The *magnitudes* are set by W1b's decoupled (AdamW-style) decay: the per-step shrink is now
# `learning_rate * l2`, applied to the rows a batch touched, instead of `l2` folded into the
# gradient. Under the old coupling Adam's scale normalisation turned any l2 into an ~lr-sized
# pull to zero, which made 1e-6..1e-4 look like a live axis; decoupled, that whole range is
# 1e-9..1e-7 per step -- regularisation off at every point. So the axis climbs to 1e-2, and
# `1e-6` is kept only because it is today's effective value and `Family.defaults` must name a
# member of its own grid (`tests/test_features.py:315-321`); it is deliberately the "off" end.
#
# `learning_rate` keeps all three of today's values -- widening must not *remove* a point the
# method cards already advertise -- and adds two faster ones, because a larger k with a rate
# tuned for k=16 tends to look like a capacity dead end when it is really an optimisation one.
RANKING_CAPACITY_GRID: dict[str, Any] = {
    "k": (8, 16, 32, 64),
    "l2": (0.0, 1e-6, 1e-4, 1e-3, 1e-2),
    "learning_rate": (0.0003, 0.0005, 0.001, 0.002, 0.005),
}
#: Exactly today's values, so an unchanged proposal trains byte-identically: `k=16` and
#: `learning_rate=0.001` are `SHARED_DEFAULTS`', and `l2=1e-6` is `FMRanker`'s own default --
#: which, under decoupled decay, is regularisation *off*. That is the point: the default
#: preserves current behaviour, and the grid is what lets the agent turn it genuinely on.
RANKING_CAPACITY_DEFAULTS: dict[str, Any] = {
    "k": 16,
    "l2": 1e-6,
    "learning_rate": 0.001,
}

# Where each trusted helper lives, for prompt rendering only.
TRUSTED_CALL_MODULES = {
    "sample_bpr_pairs": "src.models.sampling",
    "sample_softmax_groups": "src.models.sampling",
    "build_features": "src.models.features",
    "build_aux_labels": "src.models.features",
    "FMRanker": "src.models.fm_core",
}

# Return shapes for the mandatory helpers. The method cards carry the full contract, but a
# family is only ever shown its *own* card -- history_features and multi_task call the samplers
# without seeing bpr.md or group_softmax.md -- so the shape that actually gets mis-guessed
# travels with the signature instead. An observed run called .reshape(-1, K) on the already-2-D
# negatives array and lost the iteration to "cannot reshape array of size 184".
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

TRUSTED_CALL_SIGNATURES = {
    "sample_bpr_pairs": "(users, labels, rng, negatives_per_positive)",
    "sample_softmax_groups": "(users, labels, rng, negatives_per_group)",
    "build_features": "(rows, spec)",
    "build_aux_labels": "(rows, spec)",
}

# Literals, deliberately not imported from ``src.models.features``: ``types.py`` imports this
# module, so this import must stay light (no numpy). ``tests/test_features.py`` pins them equal
# to the feature module's own tuples, so the duplication cannot drift silently.
HISTORY_GROUPS = (
    "user_rate",
    "user_author",
    "user_tab",
    "recency",
    "video_age",
    "tab_cross",
    # The per-video train-window long_view rate: measured as the strongest single feature on
    # KuaiRand-Pure (primary 0.5807 alone vs 0.4827 random; +0.0021 blended with the FM), and the
    # leakage-clean replacement for the kit's `video_features_statistic_pure.csv`, whose counting
    # window spans the test dates. Registered so the agent can toggle it like any other group.
    "video_rate",
)
AUX_HEADS = ("is_click", "is_like", "is_follow", "is_comment", "is_forward", "play_time")

#: Either trusted sampler satisfies the loss requirement for the feature-side families -- the
#: loss is not what they vary, so both are legitimate.
_EITHER_SAMPLER = ("sample_bpr_pairs", "sample_softmax_groups")


FAMILIES: dict[str, Family] = {
    "bpr": Family(
        name="bpr",
        method_card="research/methods/bpr.md",
        trusted_sampler="sample_bpr_pairs",
        grid={
            **SHARED_GRID,
            **RANKING_CAPACITY_GRID,
            "batch_size": (2048, 4096),
            "negatives_per_positive": (1, 2),
            # LambdaRank-style pair weighting (report 6, C6c). `none` is today's plain BPR
            # gradient; `delta_ndcg` scales each sampled pair's loss by the |dnDCG@5| its swap
            # would cause in that user's current score order. It is the one loss-space lever
            # report 5's gradient-vanishing argument does not already exclude, because it aims
            # at the nDCG@5 half of the primary rather than at the pairwise/pointwise question.
            # The trusted sampler is unchanged: the weight is a per-pair multiplier the
            # candidate computes, so this is a grid key only, no trusted-code change.
            "pair_weighting": ("none", "delta_ndcg"),
        },
        defaults={
            **SHARED_DEFAULTS,
            **RANKING_CAPACITY_DEFAULTS,
            "batch_size": 2048,
            "negatives_per_positive": 1,
            # `none` is today's behaviour, so an unchanged proposal trains identically.
            "pair_weighting": "none",
        },
    ),
    "group_softmax": Family(
        name="group_softmax",
        method_card="research/methods/group_softmax.md",
        trusted_sampler="sample_softmax_groups",
        grid={
            **SHARED_GRID,
            **RANKING_CAPACITY_GRID,
            "batch_size": (512, 1024, 2048),
            "negatives_per_group": (4, 8),
            "temperature": (0.5, 1.0, 2.0),
        },
        defaults={
            **SHARED_DEFAULTS,
            **RANKING_CAPACITY_DEFAULTS,
            "batch_size": 2048,
            "negatives_per_group": 4,
            "temperature": 1.0,
        },
    ),
    # The loss is unchanged; the *field set* is the axis under test. epochs caps at 20 because
    # seven extra fields roughly double the gather/scatter cost (one FM epoch ~12s) against
    # experiment_timeout_seconds: 900. This family therefore keeps SHARED_GRID's k=16 and does
    # not take RANKING_CAPACITY_GRID: not because capacity is settled, but because it is
    # already the family closest to the timeout, and a second free axis would make the field
    # set -- the thing under test -- unattributable.
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
            # `leave_one_out` was dropped from the proposable axis (report 6, C6c-bis). It stays
            # implemented in `features.py` -- `SCHEMES` is unchanged, so it remains available to
            # a direct `build_features` call and to its tests -- but the agent may no longer
            # spend an iteration on it: CatBoost (NeurIPS 2018, arXiv:1706.09516) measures these
            # exact target-statistic schemes on click prediction and reports leave-one-out at
            # +2.7 % relative logloss against holdout's +1.5 %, i.e. the repo's time-respecting
            # `prior_days` is the better estimator and LOO is the measurably worse one. Half of
            # this family's `scheme` axis was being spent on a known-inferior option; the
            # default was already `prior_days`, so nothing about today's behaviour changes.
            "scheme": ("prior_days",),
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

# The *minimum* coverage set the harness stop rule must satisfy. `multi_task` stays
# explorable but is deliberately not a convergence precondition: it has never produced a
# successful node in any recorded run, so demanding it would make `converged` unreachable
# and every run could only ever end on a budget.
COVERAGE_FAMILIES = frozenset({"bpr", "group_softmax", "history_features"})


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


def _signature(call: str) -> str:
    """Render ``module.call(args)`` with the *real* signature, read at prompt time.

    Naming a mandatory helper without its signature makes the Builder guess the argument
    order, and a wrong guess costs the whole iteration: an observed run called
    ``sample_bpr_pairs(X, y, users, 1)`` instead of ``(users, labels, rng, n)``, which fails
    deep inside trusted code with ``unhashable type: 'numpy.ndarray'`` and burned both
    Debugger repairs. Reading the signature from the function keeps the prompt correct by
    construction rather than by a hand-copied string that can drift.

    The import is deliberately lazy: ``types.py`` imports this module, so module-level
    imports here must stay light. ``builder_brief`` only runs while building a prompt, by
    which time numpy is loaded anyway.

    ``TRUSTED_CALL_SIGNATURES`` is the fallback when the function cannot be imported (a
    missing optional dependency, say): a hand-maintained argument list can drift from the
    source, so it is only consulted once introspection has failed.
    """
    qualified = _qualified(call)
    fallback = f"{qualified}{TRUSTED_CALL_SIGNATURES.get(call, '()')}"
    module_name = TRUSTED_CALL_MODULES.get(call)
    if not module_name:
        return fallback
    try:
        import importlib
        import inspect

        function = getattr(importlib.import_module(module_name), call)
        rendered = f"{qualified}{inspect.signature(function)}"
    except Exception:
        return fallback
    returns = TRUSTED_CALL_RETURNS.get(call)
    return f"{rendered} -- {returns}" if returns else rendered


def _render_grid_value(value: Any) -> str:
    if isinstance(value, range):
        return f"{value.start}-{value.stop - 1}"
    return ", ".join(repr(item) if isinstance(item, str) else str(item) for item in value)


def builder_brief(name: str) -> str:
    """Prompt text for the Builder: the mandatory trusted calls plus the approved grid.

    Consumed by ``roles.py`` so no role prompt has to track the family list by hand.
    """
    entry = FAMILIES[name]
    lines = [
        # safety.validate_identifier turns candidate_id into a directory name and allows only
        # [A-Za-z0-9_-]. An observed run proposed 'gsm_1_seed42_k16_lr5e-4_neg4_T1.0' and lost
        # the whole iteration to SafetyViolation, because nothing in the prompt said so.
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
