"""The end-of-run wiring the research loop owes its collaborators (review I-1).

``ResearchLoop.run()`` ends with cross-cutting calls into modules other owners
own. Two properties of the gate call are pinned here, and both are the loop's
responsibility rather than the gate's:

* **A gate fault costs the run its gate result, never its ``summary.json``.**
  ``src/evaluation/gate.py:218`` is deliberately written so it cannot raise, so
  the ``try``/``except`` on the call site is *containment in depth*, not a fix
  for a bug in B's module: the guarantee for the one file the organizers read
  lives next to the code that writes it, instead of being borrowed from another
  owner's implementation detail.
* **The four arguments go by keyword, and ``node_dir`` is absolute.**
  ``policy.py:220`` stores ``best_candidate_dir`` **repo-relative**, so building
  the path with ``Path(...)`` resolved it against the *process* working
  directory — wrong for every process not launched from the repo root. The
  recorder below therefore accepts keyword arguments only, and the loop runs
  from a scratch directory so that only repo-root resolution can pass.

T10 ships as four sibling PRs; this file carries step 1's two tests and step 4's
one test (`RegistryDrivenPolicyTests`, review I-7 — `policy.py` reading Owner E's
family registry instead of its own literals). Steps 2 and 3 append theirs here.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

from src.agent import families, policy, research_controller
from src.agent.research_controller import ResearchLoop
from src.agent.types import ExperimentNode, RunState
from src.evaluation.gate import GateResult


REPO_ROOT = Path(__file__).resolve().parents[1]

# The shape ``_execute`` stores (``research_controller.py:363``) and ``policy.py:220``
# copies onto the state: relative to the repo root, no leading slash. Nothing has
# to exist on disk — the gate is a double in both tests.
CANDIDATE_DIR = "generated_experiments/20260829T000000Z_research/1/candidate_bpr"

BASELINE_SUMMARY = {
    "best": {
        "experiment_id": "official_fm_seed0",
        "metrics": {"GAUC": 0.6671, "nDCG@5": 0.5358, "primary": 0.6015},
        "artifact_path": "baseline.npz",
    }
}


class UnusedProvider:
    """Provider for a loop that must not reach the model: any call is a bug."""

    def complete(self, **kwargs: Any):
        raise AssertionError(f"unexpected {kwargs.get('role')} call to the provider")


@contextlib.contextmanager
def wired_loop():
    """A ``ResearchLoop`` parked at the end of ``run()``, from a foreign cwd.

    ``max_wall_clock_seconds: 0`` trips the first loop-top budget check
    (``research_controller.py:458``), so the loop body never executes and no
    model call is made: what the test observes is only what ``run()`` does
    *after* the loop. The working directory is moved out of the repo for the
    duration of ``run()`` because a path resolved against the cwd is the I-1 bug.
    """
    provider = UnusedProvider()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = {
            "mode": "research",
            "name": "wiring",
            "data_dir": str(REPO_ROOT / "data" / "KuaiRand-Pure" / "data"),
            "run_root": str(root / "runs"),
            "generated_root": str(root / "generated"),
            "method_catalog": str(REPO_ROOT / "research" / "methods"),
            "official_validation_baseline": 0.6016,
            "llm": {"max_total_tokens": 1000},
            "budgets": {
                "max_iterations": 1,
                "max_wall_clock_seconds": 0,
                "experiment_timeout_seconds": 10,
                "test_timeout_seconds": 10,
                "max_debug_repairs": 2,
            },
            "convergence": {"epsilon": 0.002, "patience": 3},
            "replication_seeds": [1, 2],
        }
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        loop = ResearchLoop(
            config,
            config_path,
            provider=provider,
            baseline_summary=BASELINE_SUMMARY,
        )
        elsewhere = root / "not_the_repo_root"
        elsewhere.mkdir()
        with contextlib.chdir(elsewhere):
            yield loop


class GateWiringTests(unittest.TestCase):
    def gate_call(
        self, gate, best_candidate_dir: str | None
    ) -> tuple[ResearchLoop, dict[str, Any], list[str]]:
        """Run one loop to completion against ``gate``; snapshot what it wrote.

        The snapshot is taken before the temporary run root is removed, so the
        caller asserts on data rather than on paths that no longer exist.
        """
        with wired_loop() as loop:
            loop.state.best_candidate_dir = best_candidate_dir
            with patch.object(research_controller, "run_gate", gate):
                run_dir = loop.run()
            self.assertTrue((run_dir / "summary.json").is_file(), "run() lost summary.json")
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            written = sorted(path.name for path in run_dir.iterdir() if path.is_file())
        # A precondition, not the subject: the run ended at the first loop-top
        # budget check, so the gate call is the only thing that happened.
        self.assertEqual(summary["stop_reason"], "wall_clock_budget_reached")
        self.assertEqual(summary["iterations"], 0)
        return loop, summary, written

    def test_gate_failure_does_not_lose_the_summary(self):
        """An exception from the gate becomes ``summary["gate"]``, not a dead run.

        Six hours of experiments must not hinge on another module staying
        exception-free: whatever the gate does, the loop still writes the
        summary, and the failure is recorded *inside* it.

        The double accepts either call style on purpose, so that this test is
        about containment alone — the keyword conversion is the next test's
        subject, and each of the two must be able to fail on its own.
        """

        def exploding_gate(*args: Any, **kwargs: Any) -> GateResult:
            raise RuntimeError("submit.py --check is not executable")

        loop, summary, written = self.gate_call(exploding_gate, CANDIDATE_DIR)
        # ``reason`` is shape parity with the producer: B's gate sets one on every
        # error it returns (``gate.py:107-108``), using ``"unexpected"`` for a fault
        # in its own wrapper (``:231``, pinned by their ``test_gate.py:164``), so a
        # consumer may read ``details["reason"]`` unconditionally.
        self.assertEqual(
            summary["gate"],
            {
                "status": "error",
                "submission_path": None,
                "details": {
                    "reason": "unexpected",
                    "error": "submit.py --check is not executable",
                },
            },
        )
        # Everything the summary exists for survived the gate fault ...
        self.assertEqual(summary["run_id"], loop.state.run_id)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["best"]["experiment_id"], "official_fm_seed0")
        self.assertEqual(summary["best"]["candidate_dir"], CANDIDATE_DIR)
        # ... and so did the artifacts ``run()`` writes after the gate.
        for name in ("summary.json", "best.json", "results.json", "state.json"):
            self.assertIn(name, written)

    def test_gate_is_called_with_keyword_arguments(self):
        """The four arguments go by name, and ``node_dir`` is absolute.

        The recorder takes ``**kwargs`` and nothing else, so a positional call
        raises ``TypeError``: that is what lets this test *fail* against the old
        call site instead of merely describing the new one. It is also the shape
        B's keyword-only ``run_gate(*, run_dir, node_dir, data_dir, kit_dir)``
        has once the signature is tightened, so this pins the call site against
        that follow-up too.
        """
        calls: list[dict[str, Any]] = []

        def recorder(**kwargs: Any) -> GateResult:
            calls.append(dict(kwargs))
            return GateResult(
                status="ok",
                submission_path="runs/wiring/submission.csv",
                details={"rows": 0},
            )

        loop, summary, _ = self.gate_call(recorder, CANDIDATE_DIR)
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(sorted(call), ["data_dir", "kit_dir", "node_dir", "run_dir"])
        self.assertEqual(call["run_dir"], loop.run_dir)
        self.assertEqual(call["data_dir"], loop.data_dir)
        self.assertEqual(call["kit_dir"], REPO_ROOT / "kuairand-starter-kit")
        # The live bug this closes: ``best_candidate_dir`` is repo-relative, so
        # ``Path(...)`` pointed at <cwd>/generated_experiments/... The loop ran
        # from a scratch directory, so neither the raw path nor a cwd-based
        # ``resolve()`` can satisfy this.
        self.assertEqual(call["node_dir"], REPO_ROOT / CANDIDATE_DIR)
        for name, value in call.items():
            with self.subTest(argument=name):
                self.assertIsInstance(value, Path)
                self.assertTrue(value.is_absolute(), value)
        # The gate's own result still reaches the summary unchanged.
        self.assertEqual(
            summary["gate"],
            {
                "status": "ok",
                "submission_path": "runs/wiring/submission.csv",
                "details": {"rows": 0},
            },
        )

        # No best candidate yet — a run that stopped before any experiment
        # succeeded. The gate still gets an absolute directory: the run's own.
        fallback_loop, _, _ = self.gate_call(recorder, None)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["node_dir"], fallback_loop.run_dir)
        self.assertTrue(calls[1]["node_dir"].is_absolute())


# --------------------------------------------------------------------------- #
# T10 step 4 · I-7 — the family registry, not `policy.py`, owns the search space
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GriddedFamily:
    """A registry entry that already carries ``grid`` and ``defaults``.

    ``families.Family`` is a **frozen** dataclass whose only fields today are
    ``name``, ``method_card`` and ``trusted_sampler`` (``families.py:8-12``), so
    a ``Family`` carrying a grid can be neither constructed nor mutated:
    ``Family(..., grid=...)`` raises ``TypeError`` and ``entry.grid = ...``
    raises ``FrozenInstanceError``. Those two fields are Owner E's T3 step 1 and
    ``families.py`` is not A's file, so the test brings its own entry, shaped
    exactly as E has committed to (``plans/E-search-surface-safety.md:113-120``,
    ``compare=False`` on both dicts so the frozen entry stays hashable).

    ``policy.py`` reads both fields through ``getattr``, so it accepts this
    stand-in and E's real ``Family`` identically — which is the point: the day
    E's fields land, this class becomes redundant rather than wrong, and the
    assertions below keep holding against the real registry.
    """

    name: str
    method_card: str
    trusted_sampler: str
    grid: dict[str, Any] = field(default_factory=dict, compare=False)
    defaults: dict[str, Any] = field(default_factory=dict, compare=False)
    required_calls: tuple[tuple[str, ...], ...] = ()


# A proposal that is valid against the *shipped* registry, so every rejection
# below is attributable to the one key under test.
BPR_RAW: dict[str, Any] = {
    "seed": 0,
    "k": 16,
    "learning_rate": 0.001,
    "epochs": 3,
    "batch_size": 2048,
    "patience": 2,
    "negatives_per_positive": 1,
}


def successful_state(*done: str) -> RunState:
    """A ``RunState`` whose only history is one successful node per family."""
    state = RunState("run", "running", "now", 0.6016)
    for index, name in enumerate(done, start=1):
        state.nodes.append(
            ExperimentNode(
                index, f"e{index}", f"h{index}", name, "explore", {}, "success", {"primary": 0.601}
            )
        )
    return state


class RegistryDrivenPolicyTests(unittest.TestCase):
    """`policy.py` must gain nothing to edit when Owner E registers a family."""

    def test_sanitize_parameters_uses_the_registry_grid(self):
        # --- the family set is derived from the registry, not restated here ---
        self.assertEqual(policy.FAMILIES, families.family_names())
        # Equality alone is vacuous: `family_names()` returns a `frozenset` and
        # `{"bpr", "group_softmax"} == frozenset({...})` is True, so the old
        # literal would satisfy the line above. The type is the discriminator —
        # the literal was a mutable `set`, a value read off `family_names()` is
        # not.
        self.assertIsInstance(policy.FAMILIES, frozenset)

        # --- an unregistered family is the brief's ValueError, verbatim -------
        with self.assertRaises(ValueError) as unknown:
            policy.sanitize_parameters("history_features", BPR_RAW)
        self.assertEqual(str(unknown.exception), "Unsupported family: history_features")
        # The lookup comes first, so the family — not an incidental bound — is
        # what the re-prompt is told about. The old `if/elif/else` chain reached
        # its `else` only after the shared checks, so this said "epochs must be
        # between 1 and 40." for a family that does not exist.
        with self.assertRaises(ValueError) as unknown_first:
            policy.sanitize_parameters("history_features", {**BPR_RAW, "epochs": 99})
        self.assertEqual(str(unknown_first.exception), "Unsupported family: history_features")

        # --- with no grid on the entry, today's bounds are the live path ------
        # `Family` has no `grid` field yet, so this is what actually runs until
        # E ships: the hard-coded checks, unchanged, messages included.
        shipped = policy.sanitize_parameters("bpr", BPR_RAW)
        self.assertEqual(shipped["batch_size"], 2048)
        self.assertEqual(shipped["negatives_per_positive"], 1)
        for override, message in (
            ({"batch_size": 256}, "BPR batch_size must be 2048 or 4096."),
            ({"k": 8}, "Ranking-loss attribution requires k=16 in the first research run."),
            (
                {"learning_rate": 0.002},
                "learning_rate is outside the approved method-card search space.",
            ),
            ({"epochs": 99}, "epochs must be between 1 and 40."),
        ):
            with self.subTest(shipped_bound=sorted(override)[0]):
                with self.assertRaises(ValueError) as rejected:
                    policy.sanitize_parameters("bpr", {**BPR_RAW, **override})
                self.assertEqual(str(rejected.exception), message)

        # --- and with a grid on the entry, the grid is the authority ----------
        # `batch_size` is a `tuple`, `epochs` a `range`; both are membership-
        # tested with `in`, which is exact and O(1) for either. `history_window`
        # is a key `policy.py` has never heard of, which is the whole finding:
        # E adds a parameter, A's file does not change.
        gridded = GriddedFamily(
            "bpr",
            "research/methods/bpr.md",
            "sample_bpr_pairs",
            grid={
                "batch_size": (256,),
                "epochs": range(1, 4),
                "history_window": (7, 14),
            },
            defaults={"negatives_per_positive": 2, "history_window": 7},
        )
        with patch.dict(families.FAMILIES, {"bpr": gridded}, clear=False):
            accepted = policy.sanitize_parameters("bpr", {**BPR_RAW, "batch_size": 256})
            # 256 is off today's hard-coded {2048, 4096} and on the grid ...
            self.assertEqual(accepted["batch_size"], 256)
            # ... `epochs: 3` sits inside the `range` ...
            self.assertEqual(accepted["epochs"], 3)
            # ... a key absent from the proposal is filled from `defaults`, not
            # from the `raw.get(...) or 1` fallback baked into `policy.py` ...
            without_negatives = {
                name: value
                for name, value in BPR_RAW.items()
                if name != "negatives_per_positive"
            }
            filled = policy.sanitize_parameters(
                "bpr", {**without_negatives, "batch_size": 256}
            )
            self.assertEqual(filled["negatives_per_positive"], 2)
            # ... and a grid key `policy.py` does not know reaches the output.
            self.assertEqual(accepted["history_window"], 7)
            self.assertEqual(
                policy.sanitize_parameters(
                    "bpr", {**BPR_RAW, "batch_size": 256, "history_window": 14}
                )["history_window"],
                14,
            )

            # Off-grid values are rejected on every one of those three keys.
            for override in (
                {"batch_size": 512},  # off the tuple (and off today's set too)
                {"epochs": 4},  # inside today's 1..40, outside the range
                {"history_window": 30},  # a key only the registry knows
            ):
                with self.subTest(off_grid=sorted(override)[0]):
                    with self.assertRaises(ValueError) as off_grid:
                        policy.sanitize_parameters(
                            "bpr", {**BPR_RAW, "batch_size": 256, **override}
                        )
                    self.assertIn(sorted(override)[0], str(off_grid.exception))

            # A key the grid does *not* name keeps today's bound and today's
            # message — `k` in particular stays pinned at 16, because the kit
            # already measured the k-sweep as a dead end
            # (`kuairand-starter-kit/README.en.md:133-139`).
            with self.assertRaises(ValueError) as pinned:
                policy.sanitize_parameters("bpr", {**BPR_RAW, "batch_size": 256, "k": 32})
            self.assertEqual(
                str(pinned.exception),
                "Ranking-loss attribution requires k=16 in the first research run.",
            )

            # A raw key that is neither shared nor in the grid is dropped, not
            # fatal: a hallucinated parameter must not cost an iteration.
            surplus = policy.sanitize_parameters(
                "bpr", {**BPR_RAW, "batch_size": 256, "hallucinated_knob": 12345}
            )
            self.assertNotIn("hallucinated_knob", surplus)
            self.assertEqual(surplus["batch_size"], 256)

        third = GriddedFamily(
            "history_features", "research/methods/history_features.md", "build_history"
        )

        # --- a new family inherits the shared bounds, including the one no
        # --- family-specific entry covers ------------------------------------
        # `_FAMILY_BOUNDS` has no entry for a family E registers, so every
        # `batch_size` bound in the file was keyed to a family that is not this
        # one: 9_999_999 was accepted (an OOM or a timeout charged to the 6-hour
        # wall clock) and so were 0 and -1 (an immediate crash). Unreachable
        # under the old code, which raised `Unsupported family` first; opened by
        # making the family set registry-driven, so it is closed here.
        shared_only = {"seed": 0, "k": 16, "learning_rate": 0.001, "epochs": 3, "patience": 2}
        with patch.dict(families.FAMILIES, {"history_features": third}, clear=False):
            for batch_size in (9_999_999, 0, -1):
                with self.subTest(unbounded_batch_size=batch_size):
                    with self.assertRaises(ValueError) as absurd:
                        policy.sanitize_parameters(
                            "history_features", {**shared_only, "batch_size": batch_size}
                        )
                    self.assertEqual(
                        str(absurd.exception), "batch_size must be between 1 and 65536."
                    )
            # A plausible value still passes — the limit is a sanity floor and
            # ceiling, not a search space; E's grid supersedes it entirely.
            self.assertEqual(
                policy.sanitize_parameters(
                    "history_features", {**shared_only, "batch_size": 4096}
                )["batch_size"],
                4096,
            )
            # And the rest of the shared bounds reach the new family unchanged.
            with self.assertRaises(ValueError) as still_pinned:
                policy.sanitize_parameters(
                    "history_features", {**shared_only, "batch_size": 4096, "k": 32}
                )
            self.assertEqual(
                str(still_pinned.exception),
                "Ranking-loss attribution requires k=16 in the first research run.",
            )

        # Both shipped families pin their own `batch_size`, so the shared limit
        # is never consulted for them and their messages are untouched.
        for family, off_bound, message in (
            ("bpr", 100, "BPR batch_size must be 2048 or 4096."),
            ("group_softmax", 100, "Group-softmax batch_size must be 512, 1024, or 2048."),
        ):
            with self.subTest(shared_limit_not_consulted=family):
                raw = {**BPR_RAW, "batch_size": off_bound}
                if family == "group_softmax":
                    raw = {**shared_only, "batch_size": off_bound,
                           "negatives_per_group": 4, "temperature": 1.0}
                with self.assertRaises(ValueError) as own_bound:
                    policy.sanitize_parameters(family, raw)
                self.assertEqual(str(own_bound.exception), message)

        # --- coverage: the stop rule reads the coverage set, not the registry -

        def third_family_does_not_break_the_stop_rule(source: str) -> None:
            """A family E registers must leave `should_stop` satisfiable."""
            with self.subTest(coverage_source=source):
                with patch.dict(families.FAMILIES, {"history_features": third}, clear=False):
                    self.assertEqual(
                        sorted(families.family_names()),
                        ["bpr", "group_softmax", "history_features"],
                    )
                    covered = successful_state("bpr", "group_softmax")
                    self.assertTrue(policy.coverage_complete(covered))
                    self.assertIsNone(policy.required_family(covered))
                    self.assertEqual(
                        policy.required_family(successful_state("bpr")), "group_softmax"
                    )
                    # The stop rule itself, which is what the run hangs on: with
                    # coverage read off `family_names()` this is False forever and
                    # the run can only end on a budget, never `converged`.
                    covered.stagnant_iterations = 3
                    self.assertTrue(policy.SearchPolicy(0.002, 3, [1, 2]).should_stop(covered))

        # E's `coverage_families()` does not exist yet, so `policy.py` falls back
        # to the pair the stop rule was written against. Guarded on `hasattr`
        # rather than asserted absent, so E shipping the function does not turn
        # A's test red on E's own PR — the block below covers that side.
        if not hasattr(families, "coverage_families"):
            third_family_does_not_break_the_stop_rule("A's fallback")

        # The same holds through E's function, which is preferred the moment it
        # appears — so this half of the property is asserted either way.
        with patch.object(
            families,
            "coverage_families",
            lambda: frozenset({"bpr", "group_softmax"}),
            create=True,
        ):
            third_family_does_not_break_the_stop_rule("families.coverage_families()")

        # Preferred, not merely consulted: a *narrower* coverage set really is
        # narrower. This is the assertion that fails while `policy.py` keeps its
        # own family literal.
        with patch.object(
            families, "coverage_families", lambda: frozenset({"bpr"}), create=True
        ):
            only_bpr = successful_state("bpr")
            self.assertTrue(policy.coverage_complete(only_bpr))
            self.assertIsNone(policy.required_family(only_bpr))

        # No history, no steer — unchanged, and the reason `required_family`
        # cannot simply return `sorted(missing)[0]` unconditionally.
        self.assertIsNone(policy.required_family(successful_state()))


if __name__ == "__main__":
    unittest.main()
