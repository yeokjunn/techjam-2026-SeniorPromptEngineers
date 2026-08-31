from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import replace
from statistics import mean, pstdev
from typing import Any, Callable

from . import families
from .convergence import official_converged, stagnation
from .types import ExperimentNode, ResearchDecision, RunState


# The registry is the single source of truth for which families exist (review
# I-7): Owner E registers one in `families.py` and nothing here changes. Read at
# import, so E's static additions are picked up; the *behavioural* reads below
# (`families.FAMILIES`, `_coverage_families()`) are live on every call.
FAMILIES = families.family_names()

# The pair the harness stop rule was written against, and the fallback for
# `families.coverage_families()` until Owner E's T3 adds it. Deliberately NOT
# `family_names()`: a third registered family would then make `should_stop`
# unsatisfiable and every run could only end on a budget.
_MINIMUM_COVERAGE = frozenset({"bpr", "group_softmax"})

# Deterministic, resume-safe 70/30 exploration schedule. Failed candidates count
# as attempts here but never enter the successful-score convergence sequence.
EXPLORATION_SLOTS = frozenset(range(7))
EXPLORATION_CYCLE = 10

# The parameters every family takes: name -> (coercion, fallback). The fallbacks
# are today's, so an entry that carries no `defaults` behaves exactly as before.
_SHARED: dict[str, tuple[Callable[[Any], Any], Any]] = {
    "seed": (int, 0),
    "k": (int, 16),
    "learning_rate": (float, 0.001),
    "epochs": (int, 40),
    "batch_size": (int, 2048),
    "patience": (int, 4),
}

# The per-family parameters `policy.py` knows about on its own. Once E's grids
# land these are also grid keys and the grid supersedes the bounds below; until
# then this is the live path, and it is what keeps the switch-over a no-op.
_FAMILY_KEYS: dict[str, dict[str, tuple[Callable[[Any], Any], Any]]] = {
    "bpr": {"negatives_per_positive": (int, 1)},
    "group_softmax": {"negatives_per_group": (int, 4), "temperature": (float, 1.0)},
}

# Today's hard-coded bounds, in today's evaluation order and with today's exact
# messages (`tests/test_controller_robustness.py:370` asserts one of them, and
# the messages are fed back to the model as the re-prompt reason). A key the
# registry grid names is checked against the grid instead and skipped here.
_Check = tuple[str, Callable[[Any], bool], str]

_SHARED_BOUNDS: tuple[_Check, ...] = (
    # `k` stays pinned at 16: the kit already measured the k-sweep (8/16/32 →
    # 0.5895/0.5902/0.5887) as a dead end, `kuairand-starter-kit/README.en.md:133-139`.
    ("k", lambda value: value == 16,
     "Ranking-loss attribution requires k=16 in the first research run."),
    ("learning_rate", lambda value: value in {0.0003, 0.0005, 0.001},
     "learning_rate is outside the approved method-card search space."),
    ("epochs", lambda value: 1 <= value <= 40, "epochs must be between 1 and 40."),
    ("patience", lambda value: 1 <= value <= 6, "patience must be between 1 and 6."),
    ("seed", lambda value: value >= 0, "seed must be non-negative."),
)

_FAMILY_BOUNDS: dict[str, tuple[_Check, ...]] = {
    "bpr": (
        ("negatives_per_positive", lambda value: value in {1, 2},
         "BPR negatives_per_positive must be 1 or 2."),
        ("batch_size", lambda value: value in {2048, 4096},
         "BPR batch_size must be 2048 or 4096."),
    ),
    "group_softmax": (
        ("negatives_per_group", lambda value: value in {4, 8},
         "Group-softmax negatives_per_group must be 4 or 8."),
        ("temperature", lambda value: value in {0.5, 1.0, 2.0},
         "Group-softmax temperature must be 0.5, 1.0, or 2.0."),
        ("batch_size", lambda value: value in {512, 1024, 2048},
         "Group-softmax batch_size must be 512, 1024, or 2048."),
    ),
}

# `batch_size` is absent from `_SHARED_BOUNDS` because each shipped family pins
# its own exact set above. A family Owner E registers has neither, and under the
# old code that combination was unreachable — an unregistered family raised
# `Unsupported family` before any bound ran. Making the family set
# registry-driven opened it: without the limit below, a new family accepted
# `batch_size=9_999_999` (an OOM or a timeout charged to the 6-hour wall clock
# Feasibility is scored on) and equally `0` or `-1` (an immediate crash).
#
# 65_536 is 16x the largest value either shipped family uses, so it blocks no
# plausible variation, and it is a *sanity* limit rather than a search space —
# E should still pin an exact grid, which supersedes this entirely.
_BATCH_SIZE_CEILING = 65_536
_BATCH_SIZE_SANITY: _Check = (
    "batch_size",
    lambda value: 1 <= value <= _BATCH_SIZE_CEILING,
    f"batch_size must be between 1 and {_BATCH_SIZE_CEILING}.",
)


def _bounds_for(family: str) -> tuple[_Check, ...]:
    """The hard-coded bounds for `family`, in today's evaluation order.

    Gated on *absence*: a family that contributes no `batch_size` bound of its
    own gets the shared sanity limit appended last, exactly where the shipped
    families carry theirs. `bpr` and `group_softmax` therefore see byte-identical
    checks in byte-identical order — which matters because the first failure is
    the one reported, and T1 feeds that message back as the re-prompt reason.

    Note that the gate is **defence in depth, not the load-bearing part**:
    appending last already makes the limit unreachable for both shipped
    families, whose own allowed sets are subsets of `[1, _BATCH_SIZE_CEILING]`,
    so removing the gate changes nothing measurable today (verified: 426-case
    parity sweep, 0 diffs either way). The gate is what keeps that true if these
    tuples are ever reordered — e.g. if someone groups all shared bounds
    together — which would otherwise silently change `bpr`'s message.
    """
    family_bounds = _FAMILY_BOUNDS.get(family, ())
    if any(name == "batch_size" for name, _, _ in family_bounds):
        return _SHARED_BOUNDS + family_bounds
    return _SHARED_BOUNDS + family_bounds + (_BATCH_SIZE_SANITY,)


def _coverage_families() -> frozenset[str]:
    """The minimum family coverage `should_stop` requires.

    Prefers Owner E's `families.coverage_families()` the moment it exists; until
    then, the literal pair. This is separate from `family_names()` on purpose —
    see `_MINIMUM_COVERAGE`.
    """
    registry_coverage = getattr(families, "coverage_families", None)
    if registry_coverage is None:
        return _MINIMUM_COVERAGE
    return frozenset(registry_coverage())


def family_experiment_score(state: RunState, family: str) -> float:
    """Prioritize families that are underexplored or underperforming relative to baseline.

    Higher score means the loop should prefer this family for the next proposal.
    """
    if family not in FAMILIES:
        raise ValueError(f"Unsupported family: {family}")

    family_nodes = [node for node in state.nodes if node.family == family and node.status == "success"]
    if not family_nodes:
        return 1.0

    best_primary = max(float(node.metrics["primary"]) for node in family_nodes if node.metrics)
    baseline = float(state.baseline_primary)
    score_gap = baseline - best_primary
    coverage_penalty = 0.0
    if len(family_nodes) == 1:
        coverage_penalty = 0.15
    if state.best_metrics is not None and family != state.best_experiment_id.split("_")[0] if False else False:
        pass
    # More observations reduce uncertainty and therefore the exploration bonus.
    # The previous positive node-count term rewarded saturated families.
    return max(0.0, score_gap + coverage_penalty + 0.05 / math.sqrt(len(family_nodes)))


_MECHANISM_STOPWORDS = frozenset({
    "the", "a", "an", "and", "with", "for", "to", "of", "v1", "v2",
    "seed", "probe", "experiment", "test",
})


def mechanism_key(decision_or_node: Any) -> str:
    """Stable, conservative mechanism identity used for branch accounting."""
    text = str(
        getattr(decision_or_node, "hypothesis", "")
        or getattr(decision_or_node, "hypothesis_id", "")
    ).lower()
    words = [
        word for word in re.findall(r"[a-z][a-z0-9]+", text)
        if word not in _MECHANISM_STOPWORDS and not word.isdigit()
    ]
    return f"{decision_or_node.family}:" + "_".join(words[:6])


def proposal_signature(decision: Any) -> dict[str, Any]:
    payload = {
        "family": decision.family,
        "mechanism": mechanism_key(decision),
        "parameters": {key: decision.parameters[key] for key in sorted(decision.parameters)},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return {**payload, "digest": hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]}


def _near_duplicate(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.get("family") != right.get("family") or left.get("mechanism") != right.get("mechanism"):
        return False
    a = dict(left.get("parameters") or {})
    b = dict(right.get("parameters") or {})
    keys = set(a) | set(b)
    # Seed-only changes are replications, not novel research proposals.
    material = [key for key in keys if key != "seed" and a.get(key) != b.get(key)]
    return len(material) <= 1


def _topk_mechanism(item: ResearchDecision | ExperimentNode) -> bool:
    parameters = getattr(item, "parameters", {}) or {}
    text = " ".join(
        str(value).lower()
        for value in (
            getattr(item, "hypothesis", ""),
            getattr(item, "rationale", ""),
            getattr(item, "hypothesis_id", ""),
            getattr(item, "family", ""),
            parameters.get("negative_sampler", ""),
            parameters.get("hard_negative_strategy", ""),
            parameters.get("sampler", ""),
            parameters.get("blend_mode", ""),
            parameters.get("blend_components", ""),
            parameters.get("loss", ""),
        )
    )
    markers = (
        "hard_negative",
        "hard negative",
        "same_tab",
        "same-author",
        "same_author",
        "top-weight",
        "top_weight",
        "lambda",
        "listwise",
        "group_softmax",
        "rank-normal",
        "rank_normal",
        "score_blend",
        "blend",
        "tab_cross",
        "user_author",
        "recency",
    )
    return any(marker in text for marker in markers)


def non_replication_attempts(state: RunState) -> list[ExperimentNode]:
    """Real research attempts, excluding exact seed replications and rejections."""
    return [
        node
        for node in state.nodes
        if node.action != "replicate" and node.status != "critic_rejected"
    ]


def exploration_slot(state: RunState) -> bool:
    """Whether the next attempt belongs to the deterministic 70% explore window."""
    return len(non_replication_attempts(state)) % EXPLORATION_CYCLE in EXPLORATION_SLOTS


def retryable_failed_family(state: RunState) -> str | None:
    """Give one implementation failure a same-family recovery opportunity."""
    attempts = non_replication_attempts(state)
    if not attempts or attempts[-1].status != "failed":
        return None
    family = attempts[-1].family
    family_attempts = [node for node in attempts if node.family == family]
    if len(family_attempts) != 1 or family in successful_families(state):
        return None
    return family


def next_family_hint(state: RunState) -> str | None:
    """Return the least-attempted family, preferring one without a valid score."""
    attempts = non_replication_attempts(state)
    completed = successful_families(state)
    counts = {family: sum(node.family == family for node in attempts) for family in FAMILIES}
    missing = FAMILIES - completed
    if missing:
        return min(missing, key=lambda family: (counts[family], family))

    scored = [
        (family_experiment_score(state, family), family)
        for family in sorted(FAMILIES)
    ]
    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1]


def successful_families(state: RunState) -> set[str]:
    return {node.family for node in state.nodes if node.status == "success"}


def _best_node(state: RunState) -> ExperimentNode | None:
    if not state.best_experiment_id:
        return None
    for node in state.nodes:
        if node.experiment_id == state.best_experiment_id:
            return node
    return None


def _best_family(state: RunState) -> str | None:
    node = _best_node(state)
    return None if node is None else node.family


def _best_primary(state: RunState) -> float | None:
    if not state.best_metrics:
        return None
    return float(state.best_metrics["primary"])


def exploit_family(state: RunState, epsilon: float = 0.002) -> str | None:
    """Return a meaningful best family only during a scheduled exploit slot."""
    if exploration_slot(state):
        return None
    best_family = _best_family(state)
    best_primary = _best_primary(state)
    if best_family is None or best_primary is None:
        return None
    if best_primary - float(state.baseline_primary) <= float(epsilon):
        return None
    return best_family


def required_family(state: RunState, epsilon: float = 0.002) -> str | None:
    retry = retryable_failed_family(state)
    if retry is not None:
        return retry
    lead = exploit_family(state, epsilon)
    if lead is not None:
        return lead
    completed = successful_families(state)
    missing = _coverage_families() - completed
    if not missing:
        return None
    attempts = non_replication_attempts(state)
    counts = {family: sum(node.family == family for node in attempts) for family in missing}
    return min(missing, key=lambda family: (counts[family], family))


def coverage_complete(state: RunState) -> bool:
    return _coverage_families().issubset(successful_families(state))


def scored_primaries(state: RunState) -> list[float]:
    """Every successful scored experiment iteration, in run order.

    The organizer baseline is an external reference target, not a run iteration;
    it must not consume one of the convergence patience slots.
    """
    return [
        float(node.metrics["primary"])
        for node in state.nodes
        if node.status == "success" and node.metrics
    ]


def research_primaries(state: RunState) -> list[float]:
    """Successful non-replication probes used by the harness stagnation agenda.

    Exact seed replications measure variance; they do not test a new mechanism and
    therefore must not consume the harness's no-progress research window. The
    official convergence sequence still uses :func:`scored_primaries` and reports
    every successful iteration.
    """
    return [
        float(node.metrics["primary"])
        for node in state.nodes
        if node.status == "success" and node.metrics and node.action != "replicate"
    ]


def sanitize_parameters(family: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce and validate a proposal's parameters against the family registry.

    The family's `grid` is the authority for every key it names; keys it does not
    name keep the bounds hard-coded above. `Family` carries no `grid`/`defaults`
    yet (`families.py:8-12`), so both `getattr`s return `{}` and the hard-coded
    path is what actually runs until Owner E's T3 lands — at which point the
    grid takes over key by key, with no edit here.
    """
    entry = families.FAMILIES.get(family)
    if entry is None:
        # First, so the re-prompt is told about the family rather than about
        # whichever shared bound happened to fail on the way past.
        raise ValueError(f"Unsupported family: {family}")
    defaults = dict(getattr(entry, "defaults", {}) or {})
    grid = dict(getattr(entry, "grid", {}) or {})
    if family == "history_features" and any(
        name.startswith("use_") for name in raw
    ):
        for name in grid:
            if name.startswith("use_") and name not in raw:
                defaults[name] = False

    parameters: dict[str, Any] = {}
    for name, (coerce, fallback) in _SHARED.items():
        parameters[name] = coerce(raw.get(name, defaults.get(name, fallback)))
    for name, (coerce, fallback) in _FAMILY_KEYS.get(family, {}).items():
        # `or` rather than a `.get` default: today's semantics, under which an
        # explicit null or zero from the model falls back to the default.
        parameters[name] = coerce(raw.get(name) or defaults.get(name, fallback))
    for name in grid:
        # Keys only the registry knows about. Everything else in `raw` is
        # dropped rather than fatal — a hallucinated knob must not cost an
        # iteration — which is automatic, since this dict is built from scratch.
        if name not in parameters and (name in raw or name in defaults):
            parameters[name] = raw.get(name, defaults.get(name))

    for name, allowed in grid.items():
        if name not in parameters:
            raise ValueError(
                f"{family} parameter {name!r} is in the registry grid but neither the "
                f"proposal nor the registry defaults supply a value."
            )
        # `in` is exact and O(1) for both shapes E ships: a tuple of allowed
        # values and a `range` of allowed integers.
        if parameters[name] not in allowed:
            legacy_message = next(
                (
                    message
                    for bound_name, permitted, message in _bounds_for(family)
                    if bound_name == name and not permitted(parameters[name])
                ),
                None,
            )
            if legacy_message is not None:
                raise ValueError(legacy_message)
            raise ValueError(
                f"{family} {name}={parameters[name]!r} is outside the registry grid {allowed!r}."
            )
    for name, permitted, message in _bounds_for(family):
        if name in grid or name not in parameters:
            continue
        if not permitted(parameters[name]):
            raise ValueError(message)
    history_keys = tuple(f"use_{group}" for group in families.HISTORY_GROUPS)
    if (
        family == "history_features"
        and any(key in grid for key in history_keys)
        and not any(bool(parameters.get(key)) for key in history_keys)
    ):
        raise ValueError(
            "history_features must enable at least one candidate-dependent history group; "
            "an all-false configuration is only the baseline FM under a misleading family name."
        )
    auxiliary_keys = tuple(f"use_{head}" for head in families.AUX_HEADS)
    if (
        family == "multi_task"
        and any(key in grid for key in auxiliary_keys)
        and not any(bool(parameters.get(key)) for key in auxiliary_keys)
    ):
        raise ValueError("multi_task must enable at least one auxiliary target head.")
    return parameters


class SearchPolicy:
    def __init__(
        self,
        epsilon: float,
        patience: int,
        replication_seeds: list[int],
        beam_width: int = 3,
        max_lineage_depth: int = 3,
        ndcg_focus: dict[str, Any] | None = None,
    ):
        self.epsilon = float(epsilon)
        self.patience = int(patience)
        self.replication_seeds = [int(seed) for seed in replication_seeds]
        self.beam_width = int(beam_width)
        self.max_lineage_depth = int(max_lineage_depth)
        self.ndcg_focus = dict(ndcg_focus or {})
        self.ndcg_enabled = bool(self.ndcg_focus.get("enabled", False))
        self.ndcg_lag_trigger = float(self.ndcg_focus.get("ndcg_lag_trigger", 0.003))
        self.min_gauc_lead = float(self.ndcg_focus.get("min_gauc_lead", 0.002))

    @staticmethod
    def _node(state: RunState, experiment_id: str | None) -> ExperimentNode | None:
        return next(
            (node for node in state.nodes if node.experiment_id == experiment_id),
            None,
        )

    def _lineage_depth(self, state: RunState, node: ExperimentNode) -> int:
        depth = 0
        seen: set[str] = set()
        current = node
        while current.parent_experiment and current.parent_experiment not in seen:
            seen.add(current.parent_experiment)
            parent = self._node(state, current.parent_experiment)
            if parent is None or parent.status != "success":
                break
            depth += 1
            current = parent
        return depth

    def estimated_seed_noise(self, state: RunState) -> float:
        groups: dict[str, list[float]] = {}
        for node in state.nodes:
            if node.status != "success" or not node.metrics:
                continue
            source = node.replicated_from or node.experiment_id
            groups.setdefault(source, []).append(float(node.metrics["primary"]))
        deviations = [pstdev(values) for values in groups.values() if len(values) >= 2]
        return max(deviations, default=0.0)

    def improvement_margin(self, state: RunState) -> float:
        return max(self.epsilon, self.estimated_seed_noise(state))

    @staticmethod
    def _normal_pdf(value: float) -> float:
        return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)

    @staticmethod
    def _normal_cdf(value: float) -> float:
        return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))

    def _acquisition(self, state: RunState, node: ExperimentNode) -> dict[str, float]:
        family_scores = [
            float(item.metrics["primary"])
            for item in state.nodes
            if item.family == node.family and item.status == "success" and item.metrics
        ]
        mu = float(node.metrics["primary"]) if node.metrics else state.baseline_primary
        sigma = max(pstdev(family_scores) if len(family_scores) >= 2 else self.epsilon, 1e-6)
        incumbent = max(
            [state.baseline_primary]
            + [
                float(item.metrics["primary"])
                for item in state.nodes
                if item.status == "success" and item.metrics
            ]
        )
        target = incumbent + self.improvement_margin(state)
        z = (mu - target) / sigma
        expected_improvement = max(
            0.0,
            (mu - target) * self._normal_cdf(z) + sigma * self._normal_pdf(z),
        )
        family_attempts = [item for item in state.nodes if item.family == node.family]
        family_failures = sum(item.status == "failed" for item in family_attempts)
        failure_risk = family_failures / max(1, len(family_attempts))
        observed_costs = [
            item.duration_seconds
            for item in family_attempts
            if item.duration_seconds > 0
        ]
        expected_cost = max(mean(observed_costs) if observed_costs else 1.0, 1e-6)
        uncertainty = sigma / math.sqrt(max(1, len(family_scores)))
        novelty = 1.0 / math.sqrt(max(1, len(family_attempts)))
        ndcg_bonus = 0.0
        if self.ndcg_enabled and node.metrics:
            baseline_gauc = 0.6674
            baseline_ndcg = 0.5357
            best = state.best_metrics or {}
            best_gauc_lead = float(best.get("GAUC", baseline_gauc)) - baseline_gauc
            best_ndcg_lead = float(best.get("nDCG@5", baseline_ndcg)) - baseline_ndcg
            ndcg_lagging = best_gauc_lead - best_ndcg_lead >= self.ndcg_lag_trigger
            direct_topk = _topk_mechanism(node)
            if ndcg_lagging or direct_topk:
                ndcg_delta = float(node.metrics.get("nDCG@5", baseline_ndcg)) - baseline_ndcg
                gauc_delta = float(node.metrics.get("GAUC", baseline_gauc)) - baseline_gauc
                comparable_gauc = gauc_delta >= -self.min_gauc_lead
                if comparable_gauc:
                    ndcg_bonus = max(0.0, ndcg_delta) * (1.5 if direct_topk else 1.0)
        priority = (
            expected_improvement + 0.25 * uncertainty + 0.10 * self.epsilon * novelty
        ) * (1.0 - failure_risk) / expected_cost
        priority += ndcg_bonus / expected_cost
        return {
            "expected_improvement": expected_improvement,
            "uncertainty": uncertainty,
            "novelty": novelty,
            "failure_risk": failure_risk,
            "expected_cost": expected_cost,
            "ndcg_bonus": ndcg_bonus,
            "priority": priority,
        }

    def parameter_sensitivity(self, state: RunState, family: str | None) -> list[dict[str, Any]]:
        """Finite-difference fallback when model Fisher gradients are unavailable."""
        if family is None:
            return []
        nodes = [
            node for node in state.nodes
            if node.family == family and node.status == "success" and node.metrics
        ]
        sensitivities: list[dict[str, Any]] = []
        numeric_keys = sorted({
            key for node in nodes for key, value in node.parameters.items()
            if key != "seed" and isinstance(value, (int, float)) and not isinstance(value, bool)
        })
        for key in numeric_keys:
            observations = sorted({
                (float(node.parameters[key]), float(node.metrics["primary"]))
                for node in nodes if key in node.parameters
            })
            slopes = [
                abs((right_score - left_score) / (right_value - left_value))
                for (left_value, left_score), (right_value, right_score) in zip(observations, observations[1:])
                if right_value != left_value
            ]
            if slopes:
                sensitivities.append({"parameter": key, "finite_difference": mean(slopes)})
        return sorted(
            sensitivities,
            key=lambda item: (-float(item["finite_difference"]), str(item["parameter"])),
        )[:5]

    def refresh_frontier(self, state: RunState) -> list[dict[str, Any]]:
        candidates: list[tuple[ExperimentNode, dict[str, float]]] = []
        for node in state.nodes:
            branch = mechanism_key(node)
            if (
                node.status != "success"
                or node.action == "replicate"
                or branch in state.closed_branches
                or f"family:{node.family}" in state.closed_branches
            ):
                continue
            candidates.append((node, self._acquisition(state, node)))
        ranked = sorted(
            candidates,
            key=lambda item: (
                -item[1]["priority"],
                -float(item[0].metrics["primary"] if item[0].metrics else -math.inf),
                item[0].experiment_id,
            ),
        )
        chosen: list[tuple[ExperimentNode, dict[str, float]]] = []
        if ranked:
            chosen.append(ranked[0])
            distinct = next((item for item in ranked if item[0].family != ranked[0][0].family), None)
            if distinct is not None:
                chosen.append(distinct)
            novel = max(
                (item for item in ranked if item not in chosen),
                key=lambda item: (item[1]["novelty"], item[1]["priority"]),
                default=None,
            )
            if novel is not None:
                chosen.append(novel)
        state.search_frontier = [
            {
                "rank": rank,
                "experiment_id": node.experiment_id,
                "family": node.family,
                "mechanism": mechanism_key(node),
                "lineage_depth": self._lineage_depth(state, node),
                **acquisition,
            }
            for rank, (node, acquisition) in enumerate(chosen[: self.beam_width], start=1)
        ]
        return state.search_frontier

    def search_context(self, state: RunState) -> dict[str, Any]:
        frontier = self.refresh_frontier(state)
        slot = state.proposal_attempts % 10
        # Give most proposal slots to family diversity. With an official
        # patience of three, waiting until late in a run to try another family
        # means that family will never be observed before convergence.
        allocation = "exploit" if slot in {0, 5, 9} else "family_explore"
        open_families = sorted(
            family for family in FAMILIES if f"family:{family}" not in state.closed_branches
        )
        family_hint = required_family(state, self.epsilon)
        if allocation == "family_explore" or family_hint is None:
            eligible = [family for family in open_families if family != _best_family(state)]
            family_hint = next_family_hint(state) if eligible else family_hint
            if family_hint not in eligible and eligible:
                attempts = non_replication_attempts(state)
                family_hint = min(
                    eligible,
                    key=lambda family: (
                        sum(node.family == family for node in attempts), family
                    ),
                )
        if family_hint and f"family:{family_hint}" in state.closed_branches:
            family_hint = None

        family_frontier = [item for item in frontier if item["family"] == family_hint]
        if not frontier:
            parent = None
        elif family_frontier:
            parent = max(
                family_frontier,
                key=lambda item: (item["priority"], -item["rank"]),
            )
        else:
            parent = frontier[0]
        if family_hint is None and parent is not None:
            family_hint = parent["family"]
        if family_hint is None:
            family_hint = open_families[0] if open_families else None
        return {
            "allocation": allocation,
            "parent_experiment": None if parent is None else parent["experiment_id"],
            "family": family_hint,
            "objective": "maximize expected validation improvement per unit runtime",
            "ndcg_focus": self._ndcg_context(state),
            "frontier": frontier,
            "closed_branches": state.closed_branches,
            "tabu_signatures": [item.get("digest") for item in state.proposal_signatures[-20:]],
            "local_parameter_sensitivity": self.parameter_sensitivity(state, family_hint),
        }

    def _ndcg_context(self, state: RunState) -> dict[str, Any]:
        if not self.ndcg_enabled:
            return {"enabled": False}
        best = state.best_metrics or {}
        baseline_gauc = 0.6674
        baseline_ndcg = 0.5357
        gauc_lead = float(best.get("GAUC", baseline_gauc)) - baseline_gauc
        ndcg_lead = float(best.get("nDCG@5", baseline_ndcg)) - baseline_ndcg
        active = (gauc_lead - ndcg_lead) >= self.ndcg_lag_trigger
        return {
            "enabled": True,
            "active": active,
            "topk": int(self.ndcg_focus.get("topk", 5)),
            "objective": "raise validation nDCG@5 through top-heavy within-user ranking while keeping official primary as the selection metric",
            "hard_negative_sources": list(self.ndcg_focus.get("hard_negative_sources", [])),
            "blend_grid": list(self.ndcg_focus.get("blend_grid", [])),
            "preferred_mechanisms": [
                "same-user hard-negative BPR",
                "same-user group-softmax with hard negatives",
                "top-weighted BPR",
                "validation-safe raw or per-user rank-normalized score blending",
                "within-user-varying history features: user_author, user_tab, tab_cross, recency",
            ],
            "discouraged_mechanisms": [
                "user-only/static features that do not vary within user",
                "random negatives when hard same-user negatives are available",
            ],
            "gauc_lead": gauc_lead,
            "ndcg5_lead": ndcg_lead,
        }

    def admit_decision(
        self, state: RunState, decision: ResearchDecision, context: dict[str, Any]
    ) -> ResearchDecision:
        signature = proposal_signature(decision)
        if decision.action != "replicate" and any(
            _near_duplicate(signature, previous) for previous in state.proposal_signatures
        ):
            state.search_stats["duplicates_avoided"] = int(
                state.search_stats.get("duplicates_avoided", 0)
            ) + 1
            raise ValueError("Duplicate or near-duplicate proposal is tabu for this run.")
        if signature["mechanism"] in state.closed_branches:
            raise ValueError("Proposal attempts to reopen a closed mechanism branch.")
        if f"family:{decision.family}" in state.closed_branches:
            raise ValueError("Proposal attempts to reopen a family closed after repeated failures.")
        parent_id = context.get("parent_experiment")
        parent = self._node(state, parent_id)
        if parent is not None and parent.status != "success":
            parent_id = None
        if parent is not None and self._lineage_depth(state, parent) >= self.max_lineage_depth:
            best = self._node(state, state.best_experiment_id)
            parent_id = best.experiment_id if best is not None and best.status == "success" else None
        return replace(decision, parent_experiment=parent_id)

    def commit_decision(self, state: RunState, decision: ResearchDecision) -> None:
        signature = proposal_signature(decision)
        if not any(item.get("digest") == signature["digest"] for item in state.proposal_signatures):
            state.proposal_signatures.append(signature)
        state.search_stats["proposals_admitted"] = int(
            state.search_stats.get("proposals_admitted", 0)
        ) + 1

    def observe_rejection(self, state: RunState, decision: ResearchDecision, rationale: str) -> None:
        signature = proposal_signature(decision)
        if not any(item.get("digest") == signature["digest"] for item in state.proposal_signatures):
            state.proposal_signatures.append(signature)
        branch = mechanism_key(decision)
        count = int(state.branch_rejections.get(branch, 0)) + 1
        state.branch_rejections[branch] = count
        if count >= 2:
            state.closed_branches[branch] = {"reason": "critic_rejections", "count": count, "detail": rationale}
            state.search_stats["branches_pruned"] = int(state.search_stats.get("branches_pruned", 0)) + 1
        self.refresh_frontier(state)

    def observe_outcome(self, state: RunState, node: ExperimentNode) -> None:
        branch = mechanism_key(node)
        if node.status == "failed":
            failure_branch = f"family:{node.family}"
            count = int(state.branch_failures.get(failure_branch, 0)) + 1
            state.branch_failures[failure_branch] = count
            if count >= 2:
                state.closed_branches[failure_branch] = {
                    "reason": "consecutive_failures",
                    "count": count,
                    "scope": "family",
                }
                state.search_stats["branches_pruned"] = int(state.search_stats.get("branches_pruned", 0)) + 1
            node.search = {
                **node.search,
                "mechanism": branch,
                "failure_count": count,
                "closed": failure_branch in state.closed_branches,
            }
        elif node.status == "success" and node.metrics and node.action != "replicate":
            parent = self._node(state, node.parent_experiment)
            improved = (
                parent is None
                or not parent.metrics
                or float(node.metrics["primary"]) - float(parent.metrics["primary"]) > self.improvement_margin(state)
            )
            state.branch_failures[branch] = 0
            state.branch_failures[f"family:{node.family}"] = 0
            state.branch_stagnation[branch] = 0 if improved else int(state.branch_stagnation.get(branch, 0)) + 1
            if state.branch_stagnation[branch] >= 2:
                state.closed_branches[branch] = {"reason": "stagnant_children", "count": state.branch_stagnation[branch]}
                state.search_stats["branches_pruned"] = int(state.search_stats.get("branches_pruned", 0)) + 1
            node.search = {
                **node.search,
                "mechanism": branch,
                **self._acquisition(state, node),
                "closed": branch in state.closed_branches,
            }
        self.refresh_frontier(state)

    def observe_success(self, state: RunState, node: ExperimentNode) -> None:
        assert node.metrics is not None
        score = float(node.metrics["primary"])
        # One ratchet, in `convergence.py` (I7). `node` is already on
        # `state.nodes` when the loop calls this. Convergence patience is counted
        # over this run's successful non-replication research probes; the official
        # baseline remains a reporting target and does not consume patience.
        state.meaningful_best, state.stagnant_iterations = stagnation(
            research_primaries(state), self.epsilon
        )

        if state.best_metrics is None or score > float(state.best_metrics["primary"]):
            state.best_metrics = dict(node.metrics)
            state.best_experiment_id = node.experiment_id
            state.best_artifact_path = node.artifact_path
            state.best_candidate_dir = node.candidate_dir

        previous_best = max(
            [state.baseline_primary]
            + [
                float(item.metrics["primary"])
                for item in state.nodes
                if item is not node and item.status == "success" and item.metrics
            ]
        )
        improvement = score - previous_best
        if improvement > self.improvement_margin(state) and node.action != "replicate":
            existing_sources = {item.get("source_experiment") for item in state.pending_replications}
            if node.experiment_id not in existing_sources:
                for seed in self.replication_seeds:
                    state.pending_replications.append(
                        {"source_experiment": node.experiment_id, "seed": seed}
                    )
        self.observe_outcome(state, node)

    def should_stop(self, state: RunState) -> bool:
        return official_converged(
            scored_primaries(state),
            self.epsilon,
            self.patience,
        )
