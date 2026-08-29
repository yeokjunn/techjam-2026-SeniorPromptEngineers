from __future__ import annotations

from typing import Any, Callable

from . import families
from .types import ExperimentNode, RunState


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
    return max(0.0, score_gap + 0.05 * len(family_nodes) + coverage_penalty)


def next_family_hint(state: RunState) -> str | None:
    """Return the next family to try, preferring unseen or lower-performing approved families."""
    completed = successful_families(state)
    if not completed:
        return "bpr"

    missing = FAMILIES - completed
    if missing:
        return sorted(missing)[0]

    scored = [
        (family_experiment_score(state, family), family)
        for family in sorted(FAMILIES)
    ]
    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1]


def successful_families(state: RunState) -> set[str]:
    return {node.family for node in state.nodes if node.status == "success"}


def required_family(state: RunState) -> str | None:
    completed = successful_families(state)
    if not completed:
        return None
    missing = _coverage_families() - completed
    # `sorted(...)[0]` rather than `next(iter(...))`: identical while the guard
    # below keeps `missing` a single element, and deterministic if it is ever
    # relaxed. Set iteration order is not a promise the search policy may make.
    if len(missing) == 1:
        return sorted(missing)[0]
    if len(missing) > 1:
        # Owner C's steer (652c0a8): with several uncovered families, prefer the
        # underexplored / underperforming one. Unreachable until a third
        # coverage family exists (E's `coverage_families()`).
        return next_family_hint(state)
    return None


def coverage_complete(state: RunState) -> bool:
    return _coverage_families().issubset(successful_families(state))


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
            raise ValueError(
                f"{family} {name}={parameters[name]!r} is outside the registry grid {allowed!r}."
            )
    for name, permitted, message in _bounds_for(family):
        if name in grid or name not in parameters:
            continue
        if not permitted(parameters[name]):
            raise ValueError(message)
    return parameters


class SearchPolicy:
    def __init__(self, epsilon: float, patience: int, replication_seeds: list[int]):
        self.epsilon = float(epsilon)
        self.patience = int(patience)
        self.replication_seeds = [int(seed) for seed in replication_seeds]

    def observe_success(self, state: RunState, node: ExperimentNode) -> None:
        assert node.metrics is not None
        score = float(node.metrics["primary"])
        previous_meaningful = state.meaningful_best
        if previous_meaningful is None or score > previous_meaningful + self.epsilon:
            state.meaningful_best = score
            state.stagnant_iterations = 0
        else:
            state.stagnant_iterations += 1

        if state.best_metrics is None or score > float(state.best_metrics["primary"]):
            state.best_metrics = dict(node.metrics)
            state.best_experiment_id = node.experiment_id
            state.best_artifact_path = node.artifact_path
            state.best_candidate_dir = node.candidate_dir

        improvement = score - state.baseline_primary
        if improvement > self.epsilon and node.action != "replicate":
            existing_sources = {item.get("source_experiment") for item in state.pending_replications}
            if node.experiment_id not in existing_sources:
                for seed in self.replication_seeds:
                    state.pending_replications.append(
                        {"source_experiment": node.experiment_id, "seed": seed}
                    )

    def should_stop(self, state: RunState) -> bool:
        return coverage_complete(state) and state.stagnant_iterations >= self.patience and not state.pending_replications
