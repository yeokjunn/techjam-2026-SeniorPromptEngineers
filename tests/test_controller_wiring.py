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
  ``policy.py:87`` stores ``best_candidate_dir`` **repo-relative**, so building
  the path with ``Path(...)`` resolved it against the *process* working
  directory — wrong for every process not launched from the repo root. The
  recorder below therefore accepts keyword arguments only, and the loop runs
  from a scratch directory so that only repo-root resolution can pass.

T10 ships as four sibling PRs; this file carries step 1's two tests only.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from src.agent import research_controller
from src.agent.research_controller import ResearchLoop
from src.evaluation.gate import GateResult


REPO_ROOT = Path(__file__).resolve().parents[1]

# The shape ``_execute`` stores (``research_controller.py:363``) and ``policy.py:87``
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


if __name__ == "__main__":
    unittest.main()
