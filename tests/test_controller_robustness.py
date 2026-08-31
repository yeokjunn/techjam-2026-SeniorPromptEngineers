"""The research loop survives everything a role pass can raise (review C4 + I13).

Every exception the correctness report names is classified into one of three
kinds and handled without ending the run: a **proposal** error is the model's
fault, so the role is re-prompted and then ledgered as a failed proposal; a
**budget** error stops the run cleanly; only a run of consecutive **harness**
errors trips the circuit breaker. ``stop_reason`` is never ``controller_error``.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from src.agent import controller
from src.agent.audit import ResearchAudit
from src.agent.controller import _source_manifest
from src.agent.errors import (
    IncompleteResponse,
    LLMError,
    RoleOutputInvalid,
)
from src.agent.llm import LLMCallResult
from src.agent.research_controller import (
    ResearchLoop,
    _ensure_baseline,
    _error_kind,
    _is_budget_error,
    _latest_valid_baseline,
)
from src.agent.types import (
    CriticDecision,
    ExperimentNode,
    ExperimentOutcome,
    ResearchDecision,
    TokenUsage,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Offline doubles
# --------------------------------------------------------------------------- #


def failure(exception_type: type[BaseException], message: str):
    """Script entry whose provider call raises a fresh exception of that type."""
    return lambda: exception_type(message)


class ProgrammedProvider:
    """Offline provider whose script mixes response payloads with failures.

    Each entry is either a payload ``dict`` or a zero-argument factory returning
    the exception to raise on that call (see :func:`failure`). ``repeat`` pins the
    final entry in place forever, which is how the breaker test fails every call.
    """

    def __init__(self, script: list[Any], repeat: bool = False):
        self.script = list(script)
        self.repeat = repeat
        self.calls: list[dict[str, Any]] = []

    def complete(self, **kwargs) -> LLMCallResult:
        self.calls.append(dict(kwargs))
        if not self.script:
            raise AssertionError(
                "ProgrammedProvider script exhausted; the loop made an unscripted call."
            )
        entry = self.script[0] if self.repeat and len(self.script) == 1 else self.script.pop(0)
        if not isinstance(entry, dict):
            raise entry()
        payload = dict(entry)
        usage = TokenUsage.from_dict(payload.pop("_usage", {"total_tokens": 10}))
        return LLMCallResult(
            data=payload,
            response_id=f"programmed-{len(self.calls)}",
            model="programmed",
            role=str(kwargs["role"]),
            latency_seconds=0.0,
            retries=0,
            usage=usage,
        )


class FakeExecutor:
    """Trusted-worker stand-in: tests always pass, training always succeeds."""

    def test(self, workspace):
        return True, "ok"

    def train(self, iteration, manifest, workspace, run_dir):
        primary = 0.601 if manifest.family == "bpr" else 0.602
        return ExperimentOutcome(
            status="success",
            metrics={"GAUC": primary, "nDCG@5": primary, "primary": primary},
            duration_seconds=0.01,
            artifact_path=f"artifact-{manifest.family}.npz",
            diagnostics={"eligible_users": 10},
        )


# --------------------------------------------------------------------------- #
# Payload builders
# --------------------------------------------------------------------------- #


def parameters(family: str, **overrides) -> dict[str, Any]:
    values = {
        "seed": 0,
        "k": 16,
        "learning_rate": 0.001,
        "epochs": 5,
        "batch_size": 2048 if family == "bpr" else 1024,
        "patience": 2,
        "negatives_per_positive": 1 if family == "bpr" else None,
        "negatives_per_group": 4 if family == "group_softmax" else None,
        "temperature": 1.0 if family == "group_softmax" else None,
    }
    values.update(overrides)
    return values


def research(family: str = "bpr") -> dict[str, Any]:
    return {
        "hypothesis_id": f"h_{family}",
        "family": family,
        "action": "explore",
        "hypothesis": f"controlled {family} ranking loss",
        "rationale": "approved method card",
        "parameters": parameters(family),
        "evidence": [
            {
                "title": "Primary paper",
                "url": "https://arxiv.org/abs/1205.2618",
                "method_card_id": family,
            }
        ],
        "needs_web_search": False,
        "parent_experiment": None,
    }


def critic() -> dict[str, Any]:
    return {
        "approved": True,
        "decision": "proceed",
        "rationale": "safe controlled experiment",
        "concerns": [],
        "next_focus": "compare trusted metrics",
    }


CANDIDATE_TESTS = """import unittest
import candidate

class CandidateTests(unittest.TestCase):
    def test_contract(self):
        self.assertTrue(callable(candidate.run))
"""


def code(family: str) -> str:
    sampler = "sample_bpr_pairs" if family == "bpr" else "sample_softmax_groups"
    final_argument = "1" if family == "bpr" else "4"
    return f'''import numpy as np
from src.experiments.contracts import CandidateOutput
from src.models.sampling import {sampler}

def run(context, parameters):
    {sampler}(context.train_users, context.train_y, np.random.default_rng(0), {final_argument})
    return CandidateOutput(np.zeros(len(context.valid_x)), {{"weights": np.zeros(1)}}, [], {{"pairs": 1}})
'''


def manifest(family: str = "bpr") -> dict[str, Any]:
    return {
        "candidate_id": f"candidate_{family}",
        "hypothesis_id": f"h_{family}",
        "family": family,
        "code": code(family),
        "tests": CANDIDATE_TESTS,
        "parameters": parameters(family),
    }


def good_iteration(family: str = "bpr") -> list[Any]:
    """The four provider calls one clean research iteration consumes."""
    return [research(family), critic(), manifest(family), critic()]


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


BASELINE_SUMMARY = {
    "best": {
        "experiment_id": "official_fm_seed0",
        "metrics": {"GAUC": 0.6671, "nDCG@5": 0.5358, "primary": 0.6015},
        "artifact_path": "baseline.npz",
    }
}


@contextlib.contextmanager
def research_loop(script: list[Any], max_iterations: int = 1, repeat: bool = False, **budgets):
    """A ``ResearchLoop`` on a programmed provider, a stub executor, and a temp root."""
    provider = ProgrammedProvider(script, repeat=repeat)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = {
            "mode": "research",
            "name": "robustness",
            "data_dir": str(REPO_ROOT / "data" / "KuaiRand-Pure" / "data"),
            "run_root": str(root / "runs"),
            "generated_root": str(root / "generated"),
            "method_catalog": str(REPO_ROOT / "research" / "methods"),
            "discovery_store": str(root / "discoveries.json"),
            "official_validation_baseline": 0.6016,
            "llm": {"max_total_tokens": 1000},
            "budgets": {
                "max_iterations": max_iterations,
                "max_wall_clock_seconds": 60,
                "experiment_timeout_seconds": 10,
                "test_timeout_seconds": 10,
                "max_debug_repairs": 2,
                **budgets,
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
        loop.executor = FakeExecutor()
        yield loop, provider


def jsonl(run_dir: Path, name: str) -> list[dict[str, Any]]:
    path = run_dir / name
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def memory(run_dir: Path, record_type: str) -> list[dict[str, Any]]:
    return [
        record
        for record in jsonl(run_dir, "research_memory.jsonl")
        if record.get("type") == record_type
    ]


def summary_of(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def error_report(run_dir: Path) -> str:
    """``error.json`` if the run wrote one, for use as an assertion message."""
    path = run_dir / "error.json"
    return path.read_text(encoding="utf-8") if path.is_file() else "<no error.json>"


# --------------------------------------------------------------------------- #
# Classification (T1 step 2 / T8)
# --------------------------------------------------------------------------- #


class ErrorClassificationTests(unittest.TestCase):
    def test_every_exception_source_in_the_review_is_classified(self):
        cases = [
            # source in the correctness report -> kind
            (ValueError("epochs must be between 1 and 40."), "proposal"),
            (ValueError("Researcher violated required family 'bpr'."), "proposal"),
            (ValueError("Unsupported research family: nope"), "proposal"),
            (ValueError("'hypothesis' is required and must be <class 'str'>"), "proposal"),
            (ValueError("OpenAI response resp_1 contained no output text."), "proposal"),
            (json.JSONDecodeError("Expecting value", "{", 0), "proposal"),
            (TypeError("unsupported operand"), "proposal"),
            (KeyError("parameters"), "proposal"),
            # Owner C's typed LLM-layer errors. These are the same raise sites as
            # the ValueError rows above (roles.py:111/175/207, llm.py:276) once C's
            # T2 step 7 lands, so they must classify identically -- otherwise
            # malformed model output silently starts tripping the harness breaker.
            (RoleOutputInvalid("Builder changed the approved family."), "proposal"),
            (IncompleteResponse("Reasoning consumed max_output_tokens."), "proposal"),
            (LLMError("Unspecified LLM-layer fault."), "proposal"),
            (RuntimeError("LLM token budget reached before the next role pass."), "harness"),
            (RuntimeError("LLM token budget exceeded by the completed role pass."), "harness"),
            # Bare RuntimeError stays harness: llm.py:198,202,310 are the run's
            # problem (no API key, SDK missing, script exhausted), not the model's.
            (RuntimeError("OPENAI_API_KEY is required for an autonomous research run."), "harness"),
            (RuntimeError("ScriptedProvider has no response left."), "harness"),
            (RuntimeError("boom"), "harness"),
            (OSError("No space left on device"), "harness"),
        ]
        for exc, kind in cases:
            with self.subTest(exception=type(exc).__name__, message=str(exc)):
                self.assertEqual(_error_kind(exc), kind)
                self.assertEqual(_is_budget_error(exc), kind == "budget")

# --------------------------------------------------------------------------- #
# Proposal errors: re-prompt, then ledger, never break
# --------------------------------------------------------------------------- #


class ProposalErrorTests(unittest.TestCase):
    def test_builder_identity_drift_is_canonicalized_without_proposal_failure(self):
        drifted = manifest("bpr")
        drifted["family"] = "group_softmax"
        drifted["hypothesis_id"] = "stale_hypothesis"
        with research_loop([research("bpr"), critic(), drifted, critic()]) as (
            loop,
            provider,
        ):
            run_dir = loop.run()
            # Read run artifacts before TemporaryDirectory cleanup; Windows
            # removes the directory as the context manager exits.
            self.assertEqual(len(provider.calls), 4)
            self.assertEqual([node.status for node in loop.state.nodes], ["success"])
            self.assertEqual(loop.state.nodes[0].family, "bpr")
            self.assertEqual(loop.state.nodes[0].hypothesis_id, "h_bpr")
            self.assertEqual(
                [record["status"] for record in jsonl(run_dir, "iterations.jsonl")],
                ["success"],
            )

    def test_malformed_research_response_is_reprompted_and_the_run_survives(self):
        broken = research("bpr")
        broken["family"] = "nope"  # types.py:113 -> ValueError
        with research_loop([broken, *good_iteration("bpr")]) as (loop, provider):
            run_dir = loop.run()
            summary = summary_of(run_dir)
            self.assertNotEqual(summary["stop_reason"], "controller_error", error_report(run_dir))
            self.assertEqual(summary["stop_reason"], "candidate_budget_reached")
            self.assertEqual(len(provider.calls), 5)
            retries = memory(run_dir, "role_retry")
            self.assertEqual(len(retries), 1)
            self.assertEqual(retries[0]["label"], "researcher")
            self.assertIn("Unsupported research family: nope", retries[0]["error"])
            self.assertEqual(memory(run_dir, "controller_error"), [])
            self.assertEqual([node.status for node in loop.state.nodes], ["success"])
            self.assertFalse((run_dir / "error.json").is_file())

    def test_three_bad_responses_record_proposal_failed_and_continue(self):
        broken = research("bpr")
        broken["family"] = "nope"
        script = [dict(broken), dict(broken), dict(broken), *good_iteration("bpr")]
        with research_loop(script) as (loop, provider):
            run_dir = loop.run()
            summary = summary_of(run_dir)
            self.assertNotEqual(summary["stop_reason"], "controller_error", error_report(run_dir))
            # One initial attempt plus max_role_reprompts=2 re-prompts, then the
            # proposal is abandoned and the next pass starts from scratch.
            self.assertEqual(len(provider.calls), 7)
            self.assertEqual(len(memory(run_dir, "role_retry")), 2)
            iterations = jsonl(run_dir, "iterations.jsonl")
            self.assertEqual([record["status"] for record in iterations], ["proposal_failed", "success"])
            # The abandoned proposal did not consume an iteration number.
            self.assertEqual([record["iteration"] for record in iterations], [1, 1])
            self.assertEqual([node.status for node in loop.state.nodes], ["success"])
            self.assertEqual(loop.state.proposal_attempts, 2)
            self.assertEqual(loop.state.iteration_count, 1)
            failed = memory(run_dir, "controller_error")
            self.assertEqual([record["kind"] for record in failed], ["proposal"])
            self.assertFalse((run_dir / "error.json").is_file())

    def test_off_grid_parameters_do_not_kill_the_run(self):
        off_grid = manifest("bpr")
        off_grid["parameters"] = parameters("bpr", epochs=99)  # policy.py:41
        script = [research("bpr"), critic(), off_grid, manifest("bpr"), critic()]
        with research_loop(script) as (loop, provider):
            run_dir = loop.run()
            summary = summary_of(run_dir)
            self.assertNotEqual(summary["stop_reason"], "controller_error", error_report(run_dir))
            self.assertEqual(len(provider.calls), 5)
            retries = memory(run_dir, "role_retry")
            self.assertEqual(len(retries), 1)
            self.assertEqual(retries[0]["label"], "builder")
            self.assertIn("epochs", retries[0]["error"])
            self.assertEqual([node.status for node in loop.state.nodes], ["success"])
            self.assertEqual(loop.state.nodes[0].parameters["epochs"], 5)

    def test_empty_output_text_is_a_proposal_failure(self):
        empty = failure(ValueError, "OpenAI response resp_1 contained no output text.")
        script = [empty, empty, empty, *good_iteration("bpr")]
        with research_loop(script) as (loop, provider):
            run_dir = loop.run()
            summary = summary_of(run_dir)
            self.assertNotEqual(summary["stop_reason"], "controller_error", error_report(run_dir))
            self.assertEqual(len(provider.calls), 7)
            self.assertEqual(len(memory(run_dir, "role_retry")), 2)
            failed = memory(run_dir, "controller_error")
            self.assertEqual([record["kind"] for record in failed], ["proposal"])
            ledger = jsonl(run_dir, "iterations.jsonl")[0]
            self.assertEqual(ledger["status"], "proposal_failed")
            self.assertEqual(ledger["error_type"], "ValueError")
            self.assertIn("contained no output text.", ledger["error"])
            self.assertEqual([node.status for node in loop.state.nodes], ["success"])
            self.assertEqual(loop.consecutive_harness_errors, 0)
            self.assertFalse((run_dir / "error.json").is_file())

    def test_typed_role_output_invalid_is_reprompted_not_a_harness_error(self):
        # The same rejection Owner C's T2 step 7 will raise from roles.py:175
        # instead of a bare ValueError. It must be re-prompted exactly as the
        # untyped spelling is, and must not touch the harness breaker.
        invalid = failure(RoleOutputInvalid, "Builder changed the approved family or hypothesis ID.")
        script = [research("bpr"), critic(), invalid, manifest("bpr"), critic()]
        with research_loop(script) as (loop, provider):
            run_dir = loop.run()
            summary = summary_of(run_dir)
            self.assertNotEqual(summary["stop_reason"], "harness_error_breaker", error_report(run_dir))
            self.assertEqual(summary["stop_reason"], "candidate_budget_reached")
            self.assertEqual(len(provider.calls), 5)
            retries = memory(run_dir, "role_retry")
            self.assertEqual([record["label"] for record in retries], ["builder"])
            self.assertEqual(retries[0]["error_type"], "RoleOutputInvalid")
            self.assertEqual(loop.consecutive_harness_errors, 0)
            self.assertEqual([node.status for node in loop.state.nodes], ["success"])

    def test_incomplete_response_is_reprompted_not_a_harness_error(self):
        # llm.py:276's empty-output-text case after C types it (I13).
        empty = failure(IncompleteResponse, "Reasoning consumed max_output_tokens.")
        with research_loop([empty, *good_iteration("bpr")]) as (loop, provider):
            run_dir = loop.run()
            self.assertEqual(summary_of(run_dir)["stop_reason"], "candidate_budget_reached")
            self.assertEqual(len(provider.calls), 5)
            self.assertEqual(len(memory(run_dir, "role_retry")), 1)
            self.assertEqual(loop.consecutive_harness_errors, 0)
            self.assertEqual([node.status for node in loop.state.nodes], ["success"])

    def test_incomplete_builder_does_not_consume_a_candidate_iteration(self):
        empty = failure(IncompleteResponse, "Reasoning consumed max_output_tokens.")
        script = [research("bpr"), critic(), empty, empty, empty, *good_iteration("bpr")]
        with research_loop(script, max_iterations=1) as (loop, provider):
            run_dir = loop.run()
            summary = summary_of(run_dir)

            self.assertEqual(summary["stop_reason"], "candidate_budget_reached")
            self.assertEqual(len(provider.calls), 9)
            self.assertEqual(loop.state.iteration_count, 1)
            self.assertEqual([node.status for node in loop.state.nodes], ["success"])
            self.assertEqual(
                [record["status"] for record in jsonl(run_dir, "iterations.jsonl")],
                ["proposal_failed", "success"],
            )
            self.assertEqual(
                [record["iteration"] for record in jsonl(run_dir, "iterations.jsonl")],
                [1, 1],
            )


# --------------------------------------------------------------------------- #
# Harness errors: counted, and only the breaker stops the run
# --------------------------------------------------------------------------- #


class HarnessErrorTests(unittest.TestCase):
    def test_consecutive_harness_errors_trip_the_breaker(self):
        disk_full = failure(OSError, "No space left on device")
        with research_loop([disk_full], max_iterations=4, repeat=True) as (loop, provider):
            run_dir = loop.run()
            summary = summary_of(run_dir)
            self.assertEqual(summary["stop_reason"], "harness_error_breaker")
            self.assertEqual(len(provider.calls), 3)
            self.assertEqual(loop.consecutive_harness_errors, 3)
            self.assertTrue((run_dir / "error.json").is_file())
            recorded = json.loads((run_dir / "error.json").read_text(encoding="utf-8"))
            self.assertIn("No space left on device", recorded["error"])
            # A harness error is never re-prompted: it is not the model's fault.
            self.assertEqual(memory(run_dir, "role_retry"), [])
            failed = memory(run_dir, "controller_error")
            self.assertEqual([record["kind"] for record in failed], ["harness"] * 3)
            # The pass that trips the breaker goes to error.json, not the ledger.
            self.assertEqual(
                [record["status"] for record in jsonl(run_dir, "iterations.jsonl")],
                ["harness_error", "harness_error"],
            )
            self.assertEqual(loop.state.nodes, [])

    def test_harness_error_counter_resets_after_a_good_iteration(self):
        disk_full = failure(OSError, "No space left on device")
        script = [disk_full, *good_iteration("bpr"), disk_full, disk_full]
        with research_loop(script, max_iterations=2) as (loop, provider):
            run_dir = loop.run()
            summary = summary_of(run_dir)
            self.assertNotEqual(summary["stop_reason"], "harness_error_breaker")
            self.assertEqual(summary["stop_reason"], "proposal_budget_reached")
            self.assertEqual(len(provider.calls), 7)
            self.assertEqual(loop.consecutive_harness_errors, 2)
            self.assertFalse((run_dir / "error.json").is_file())
            self.assertEqual(
                [record["status"] for record in jsonl(run_dir, "iterations.jsonl")],
                ["harness_error", "success", "harness_error", "harness_error"],
            )
            self.assertEqual([node.status for node in loop.state.nodes], ["success"])

    def test_a_failing_replication_terminates_instead_of_spinning(self):
        """A corrupt manifest in the replication branch must not busy-loop.

        ``_replication`` raises ``json.JSONDecodeError`` (a ``ValueError``) before
        ``pending_replications.pop(0)`` and before ``proposal_attempts`` is
        charged. Treated as an ordinary proposal error the pass would ``continue``
        with nothing advanced and repeat at full speed until ``max_wall_clock``,
        flooding ``iterations.jsonl``. It must terminate instead.
        """
        with research_loop([], max_iterations=4) as (loop, provider):
            # Both families succeeded, so coverage is complete and the loop takes
            # the replication branch before proposing anything.
            broken = loop.run_dir / "broken_candidate"
            broken.mkdir(parents=True, exist_ok=True)
            (broken / "manifest.json").write_text('{"parameters": {', encoding="utf-8")
            loop.state.nodes = [
                ExperimentNode(
                    1, "source_bpr", "h_bpr", "bpr", "explore", parameters("bpr"),
                    "success", {"primary": 0.61}, candidate_dir=str(broken),
                ),
                ExperimentNode(
                    2, "source_gs", "h_gs", "group_softmax", "explore",
                    parameters("group_softmax"), "success", {"primary": 0.61},
                ),
            ]
            loop.state.pending_replications = [{"source_experiment": "source_bpr", "seed": 1}]
            run_dir = loop.run()

            summary = summary_of(run_dir)
            # Terminated, and not by exhausting the six-hour wall clock.
            self.assertEqual(summary["stop_reason"], "harness_error_breaker")
            self.assertLess(summary["wall_clock_seconds"], 30.0)
            # A pass that never reached the model is bounded by the breaker, so
            # the ledger stays short instead of growing without limit.
            self.assertEqual(
                [record["status"] for record in jsonl(run_dir, "iterations.jsonl")],
                ["harness_error", "harness_error"],
            )
            self.assertEqual(len(memory(run_dir, "controller_error")), 3)
            self.assertTrue((run_dir / "error.json").is_file())
            # The queue was never popped and no proposal was ever made.
            self.assertEqual(len(loop.state.pending_replications), 1)
            self.assertEqual(loop.state.proposal_attempts, 0)
            self.assertEqual(provider.calls, [])


# --------------------------------------------------------------------------- #
# Token usage is reported for feasibility, not enforced as a stop condition
# --------------------------------------------------------------------------- #


class TokenUsageReportingTests(unittest.TestCase):
    def test_token_budget_message_is_not_a_clean_stop_condition(self):
        message = "LLM token budget reached before the next role pass."
        with research_loop([failure(RuntimeError, message)]) as (loop, provider):
            run_dir = loop.run()
            self.assertEqual(summary_of(run_dir)["stop_reason"], "proposal_budget_reached")
            self.assertEqual(len(provider.calls), 2)
            self.assertNotIn(
                "budget",
                [record["kind"] for record in memory(run_dir, "controller_error")],
            )
            self.assertFalse((run_dir / "error.json").is_file())


class OperatorInterruptTests(unittest.TestCase):
    def test_keyboard_interrupt_is_not_a_catchable_loop_error(self):
        # The loop catches ``Exception``, never ``BaseException``: Ctrl-C during a
        # role pass must leave run() entirely so the operator can record it with
        # ``intervene`` (T7), not be laundered into a stop_reason.
        with research_loop([failure(KeyboardInterrupt, "operator pressed ctrl-c")]) as (
            loop,
            provider,
        ):
            with self.assertRaises(KeyboardInterrupt):
                loop.run()
            self.assertEqual(len(provider.calls), 1)
            self.assertIsNone(loop.state.stop_reason)
            self.assertEqual(loop.state.status, "running")
            self.assertEqual(loop.consecutive_harness_errors, 0)
            self.assertEqual(memory(loop.run_dir, "controller_error"), [])
            self.assertEqual(memory(loop.run_dir, "role_retry"), [])
            self.assertFalse((loop.run_dir / "error.json").is_file())
            self.assertFalse((loop.run_dir / "summary.json").is_file())


# --------------------------------------------------------------------------- #
# Baseline adoption: recorded revision, real artifact, every skip logged
# (T2 -> C5 + I12, carrying B's I11 two-sided tolerance gate)
# --------------------------------------------------------------------------- #


# The artifact path the stale committed baseline actually carries (review C5):
# a Windows absolute path from another machine. It is not absolute on this OS,
# so ``_resolve_repo_path`` folds it under the repo root and it does not exist.
WINDOWS_ARTIFACT = (
    r"C:\Users\Admin\OneDrive - Nanyang Technological University"
    r"\runs\20260828T141646Z_baseline\test_scores.npy"
)
STALE_REVISION = "0" * 64


def write_baseline_run(
    run_root: Path,
    run_id: str,
    *,
    primary: float = 0.6015,
    experiment_id: str = "official_fm_seed0",
    revision: str | None = None,
    with_manifest: bool = True,
    artifact_path: str | None = None,
) -> Path:
    """Build one ``<run_root>/<run_id>/`` baseline fixture; return its summary path.

    Defaults describe an *adoptable* baseline: the official experiment id, a
    primary inside the two-sided tolerance of 0.6016, a ``source_manifest.json``
    recording the current source revision, and an artifact that exists. Each
    keyword turns exactly one of those admission checks off.
    """
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if artifact_path is None:
        artifact = run_dir / "test_scores.npy"
        artifact.write_bytes(b"\x00")
        artifact_path = str(artifact)
    if with_manifest:
        (run_dir / "source_manifest.json").write_text(
            json.dumps({"revision": revision or _source_manifest()["revision"]}),
            encoding="utf-8",
        )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "completed",
                "best": {
                    "experiment_id": experiment_id,
                    "metrics": {"GAUC": 0.6671, "nDCG@5": 0.5358, "primary": primary},
                    "artifact_path": artifact_path,
                },
            }
        ),
        encoding="utf-8",
    )
    return run_dir / "summary.json"


def write_raw_summary(run_root: Path, run_id: str, text: str) -> Path:
    """A ``runs/<id>/summary.json`` holding exactly ``text``, well-formed or not."""
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "summary.json"
    path.write_text(text, encoding="utf-8")
    return path


def baseline_config(root: Path) -> dict[str, Any]:
    """A research config whose ``run_root`` is the fixture tree, nothing else."""
    return {
        "mode": "research",
        "name": "baseline-selection",
        "data_dir": str(REPO_ROOT / "data" / "KuaiRand-Pure" / "data"),
        "run_root": str(root / "runs"),
        "generated_root": str(root / "generated"),
        "method_catalog": str(REPO_ROOT / "research" / "methods"),
        "discovery_store": str(root / "discoveries.json"),
        "official_validation_baseline": 0.6016,
        "llm": {"max_total_tokens": 1000},
        "budgets": {
            "max_iterations": 1,
            "max_wall_clock_seconds": 60,
            "experiment_timeout_seconds": 10,
            "test_timeout_seconds": 10,
            "max_debug_repairs": 2,
        },
        "convergence": {"epsilon": 0.002, "patience": 3},
        "replication_seeds": [1, 2],
    }


def adopting_loop(root: Path, baseline_summary: dict[str, Any] | None = None) -> ResearchLoop:
    """Construct a real ``ResearchLoop``, by default with **no** injected baseline.

    Every other test in the repo passes ``baseline_summary=``; these must not, or
    ``_ensure_baseline`` never runs and the selection is not exercised. Passing
    one takes the other branch of the same call site, which is how the injected
    path's empty skip list is pinned.
    """
    config = baseline_config(root)
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return ResearchLoop(
        config,
        config_path,
        provider=ProgrammedProvider([]),
        baseline_summary=baseline_summary,
    )


def selection_of(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "baseline_selection.json").read_text(encoding="utf-8"))


def skip_reasons(skipped: list[dict[str, str]]) -> dict[str, str]:
    return {record["path"]: record["reason"] for record in skipped}


def _posix(path: Path | str) -> str:
    return Path(path).as_posix()


class BaselineSelectionTests(unittest.TestCase):
    def test_official_baseline_cache_skips_baseline_reproduction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = baseline_config(root)
            config["use_official_baseline_cache"] = True

            with patch("src.agent.research_controller.run_agent") as run_agent_mock:
                baseline, skipped = _ensure_baseline(config)

            run_agent_mock.assert_not_called()
            self.assertEqual(skipped, [])
            self.assertEqual(baseline["run_id"], "official_cached_baseline")
            self.assertEqual(baseline["summary_path"], "official://kuairand-pure/validation-baseline")
            self.assertEqual(baseline["best"]["experiment_id"], "official_fm_seed0")
            self.assertEqual(baseline["best"]["metrics"], {"primary": 0.6016})
            self.assertIsNone(baseline["best"]["artifact_path"])

    def test_baseline_is_rejected_when_the_source_revision_differs(self):
        """A summary produced by code that is no longer on disk is not adoptable.

        This is C5's core: ``runs/20260828T141646Z_baseline/`` was written by code
        in no commit in this range, yet its metrics were adopted wholesale as the
        number every later experiment is compared against.
        """
        revision = _source_manifest()["revision"]
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "runs"
            stale = write_baseline_run(
                run_root, "20260828T141646Z_baseline", revision=STALE_REVISION
            )

            selected, skipped = _latest_valid_baseline(run_root, 0.6016, revision)
            self.assertIsNone(selected)
            self.assertEqual(skipped, [{"path": _posix(stale), "reason": "revision_mismatch"}])

            # A run that recorded no manifest at all is its own named reason, not
            # the same one: "built by other code" and "we cannot tell" differ.
            unrecorded = write_baseline_run(
                run_root, "20260828T141647Z_baseline", with_manifest=False
            )
            selected, skipped = _latest_valid_baseline(run_root, 0.6016, revision)
            self.assertIsNone(selected)
            self.assertEqual(
                skip_reasons(skipped),
                {_posix(stale): "revision_mismatch", _posix(unrecorded): "no_source_manifest"},
            )

            # Non-vacuity: the recorded revision is the only discriminator here.
            # Rewrite it to the current one and the same directory is adopted.
            (stale.parent / "source_manifest.json").write_text(
                json.dumps({"revision": revision}), encoding="utf-8"
            )
            selected, skipped = _latest_valid_baseline(run_root, 0.6016, revision)
            self.assertIsNotNone(selected)
            self.assertEqual(selected["summary_path"], _posix(stale))
            self.assertEqual(
                skip_reasons(skipped), {_posix(unrecorded): "no_source_manifest"}
            )

    def test_baseline_is_rejected_when_the_artifact_is_missing(self):
        """``best.artifact_path`` must resolve to a file that exists on this box.

        The committed baseline points at ``C:\\Users\\Admin\\OneDrive - …``; the
        loop copied it straight onto ``state.best_artifact_path``, so the gate's
        submission would have been built from a path that cannot be opened.
        """
        revision = _source_manifest()["revision"]
        with self.subTest("artifact_path must resolve to a file that exists"), \
                tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "runs"
            windows = write_baseline_run(
                run_root, "20260828T141646Z_baseline", artifact_path=WINDOWS_ARTIFACT
            )
            selected, skipped = _latest_valid_baseline(run_root, 0.6016, revision)
            self.assertIsNone(selected)
            self.assertEqual(skipped, [{"path": _posix(windows), "reason": "artifact_missing"}])
            # The Windows string is neither absolute here nor a real relative
            # path, so it resolves under the repo root to something absent.
            self.assertFalse((REPO_ROOT / WINDOWS_ARTIFACT).is_file())

            # A summary carrying no artifact_path key at all is the same reason.
            absent = write_baseline_run(run_root, "20260828T141647Z_baseline")
            payload = json.loads(absent.read_text(encoding="utf-8"))
            payload["best"].pop("artifact_path")
            absent.write_text(json.dumps(payload), encoding="utf-8")
            selected, skipped = _latest_valid_baseline(run_root, 0.6016, revision)
            self.assertIsNone(selected)
            self.assertEqual(
                skip_reasons(skipped),
                {_posix(windows): "artifact_missing", _posix(absent): "artifact_missing"},
            )

        # T2's acceptance clause, on a tree holding *only* the stale run:
        # constructing a ResearchLoop re-runs the baseline instead of adopting
        # it, and baseline_selection.json names the reason. ``run_agent`` is
        # patched (the real ladder needs the dataset) but the decision to call
        # it, and the record of why, are the loop's own.
        with self.subTest("T2 acceptance: an unadoptable tree regenerates the baseline"), \
                tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "runs"
            stale = write_baseline_run(
                run_root, "20260828T141646Z_baseline", artifact_path=WINDOWS_ARTIFACT
            )
            regenerated = run_root / "20260829T120000000000Z_baseline"
            calls: list[Path] = []

            def fake_run_agent(config_path: Path) -> Path:
                calls.append(config_path)
                write_baseline_run(run_root, regenerated.name)
                return regenerated

            with patch("src.agent.research_controller.run_agent", fake_run_agent):
                with contextlib.redirect_stdout(io.StringIO()):
                    loop = adopting_loop(root)

            self.assertEqual(calls, [REPO_ROOT / "configs" / "baseline.json"])
            self.assertEqual(
                loop.baseline_summary["summary_path"], _posix(regenerated / "summary.json")
            )
            selection = selection_of(loop.run_dir)
            self.assertEqual(selection["selected"], _posix(regenerated / "summary.json"))
            self.assertEqual(
                selection["skipped"], [{"path": _posix(stale), "reason": "artifact_missing"}]
            )

        # I11's *second* site, on the same regeneration path: the re-run is gated
        # two-sided as well, so a fresh run scoring 0.85 is a leak rather than a
        # reproduction. The old one-sided lower-bound gate accepted it.
        with self.subTest("I11: a regenerated baseline is gated two-sided too"), \
                tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "runs"
            regenerated = run_root / "20260829T120000000000Z_baseline"

            def leaking_run_agent(config_path: Path) -> Path:
                write_baseline_run(run_root, regenerated.name, primary=0.85)
                return regenerated

            with patch("src.agent.research_controller.run_agent", leaking_run_agent):
                with self.assertRaises(RuntimeError) as raised:
                    _ensure_baseline(baseline_config(root))
            message = str(raised.exception)
            self.assertIn("0.8500", message)
            self.assertIn("outside", message)
            self.assertIn("0.5986", message)
            self.assertIn("0.6046", message)

            # M5, same gate: a regenerated run is artifact-checked too, or the
            # loop adopts a baseline whose submission file cannot be opened.
            def artifactless_run_agent(config_path: Path) -> Path:
                write_baseline_run(
                    run_root, regenerated.name, artifact_path=WINDOWS_ARTIFACT
                )
                return regenerated

            with patch("src.agent.research_controller.run_agent", artifactless_run_agent):
                with self.assertRaisesRegex(RuntimeError, "produced no artifact"):
                    _ensure_baseline(baseline_config(root))

    def test_baseline_selection_logs_every_skipped_summary(self):
        """I12: no summary is skipped silently — every rejection is named on disk.

        The old ``except … continue`` made a corrupt summary indistinguishable
        from "no baseline exists". One fully adoptable run is present so the
        selection succeeds without regenerating anything, and every other
        directory covers a reason the selector can record — including each shape
        of malformed ``summary.json`` that used to raise out of ``__init__``.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "runs"
            adoptable = write_baseline_run(run_root, "20260829T000000000001Z_baseline")
            stale = write_baseline_run(
                run_root, "20260829T000000000002Z_baseline", revision=STALE_REVISION
            )
            windows = write_baseline_run(
                run_root, "20260829T000000000003Z_baseline", artifact_path=WINDOWS_ARTIFACT
            )
            # A leaked 0.85 is *further* from the official 0.6016 than the
            # tolerance allows. The one-sided ``primary >= official - 0.002``
            # accepted it; B's two-sided predicate (I11) does not.
            leaked = write_baseline_run(
                run_root, "20260829T000000000004Z_baseline", primary=0.85
            )
            other_experiment = write_baseline_run(
                run_root, "20260829T000000000005Z_baseline", experiment_id="candidate_bpr"
            )
            # Every shape a summary.json can take that the readers' ``.get``
            # calls cannot survive. Only the first is a JSONDecodeError; the
            # other four parse and then raise ``AttributeError`` from
            # ``summary.get``, ``best.get`` or ``metrics.get`` — none of which
            # the original handler caught, so any one of them ended *every*
            # research run at construction instead of costing one candidate.
            malformed = {
                _posix(write_raw_summary(run_root, run_id, text)): "unreadable_summary"
                for run_id, text in (
                    ("20260829T000000000006Z_baseline", '{"best": {'),
                    ("20260829T000000000007Z_baseline", "[]"),
                    ("20260829T000000000008Z_baseline", '{"best": ["official_fm_seed0"]}'),
                    ("20260829T000000000009Z_baseline", '{"best": "official_fm_seed0"}'),
                    (
                        "20260829T000000000010Z_baseline",
                        '{"best": {"experiment_id": "official_fm_seed0",'
                        ' "metrics": "0.6015"}}',
                    ),
                )
            }
            # A source_manifest.json that parses but is not a mapping has no
            # revision to compare, so it is `no_source_manifest` — the narrow
            # handler's reason, not the outer one's.
            unusable_manifest = write_baseline_run(run_root, "20260829T000000000011Z_baseline")
            (unusable_manifest.parent / "source_manifest.json").write_text(
                "[]", encoding="utf-8"
            )

            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                loop = adopting_loop(root)

            selection = selection_of(loop.run_dir)
            self.assertEqual(selection["selected"], _posix(adoptable))
            self.assertEqual(loop.baseline_summary["summary_path"], _posix(adoptable))
            self.assertEqual(
                skip_reasons(selection["skipped"]),
                {
                    _posix(stale): "revision_mismatch",
                    _posix(windows): "artifact_missing",
                    _posix(leaked): "outside_tolerance",
                    _posix(other_experiment): "experiment_id_mismatch",
                    _posix(unusable_manifest): "no_source_manifest",
                    **malformed,
                },
            )
            # One printed line per skip, so a live operator sees them too.
            lines = printed.getvalue().splitlines()
            self.assertEqual(len(lines), len(selection["skipped"]))
            self.assertEqual(len(lines), 10)
            for record in selection["skipped"]:
                self.assertIn(
                    f"{record['path']} ({record['reason']})",
                    printed.getvalue(),
                )

            # Amendment 3, first half: an injected baseline was never selected
            # from ``runs/``, so nothing was examined and nothing is reported as
            # rejected — on this very tree, where ten runs *would* have been.
            with self.subTest("an injected baseline records no skips"):
                injected = adopting_loop(root, baseline_summary=BASELINE_SUMMARY)
                self.assertEqual(
                    selection_of(injected.run_dir), {"selected": None, "skipped": []}
                )

            # Amendment 3, second half: a resume adopts nothing and re-selects
            # nothing, so it must not write the file at all — the record the
            # original run made is the truthful one and stays untouched.
            with self.subTest("a resume writes no selection record"):
                loop._save()
                (loop.run_dir / "baseline_selection.json").unlink()
                resumed = ResearchLoop(
                    baseline_config(root),
                    root / "config.json",
                    provider=ProgrammedProvider([]),
                    resume_dir=loop.run_dir,
                    baseline_summary=BASELINE_SUMMARY,
                )
                self.assertFalse(
                    (resumed.run_dir / "baseline_selection.json").is_file()
                )

    def test_baseline_prefers_the_newest_matching_run_id(self):
        """Ordering is by run id, not by mtime: ids are UTC stamps, mtimes are not.

        A ``git clone`` or a ``cp -r`` rewrites every mtime, so the old
        ``max(st_mtime)`` picked whichever file the filesystem touched last.
        """
        revision = _source_manifest()["revision"]
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "runs"
            older_id = "20260101T000000000000Z_baseline"
            newer_id = "20260102T000000000000Z_baseline"
            older = write_baseline_run(run_root, older_id, primary=0.6010)
            newer = write_baseline_run(run_root, newer_id, primary=0.6015)
            # Invert the filesystem's opinion: the newer *id* is the older file.
            os.utime(newer, (1_000_000, 1_000_000))
            os.utime(older, (2_000_000, 2_000_000))
            self.assertLess(newer.stat().st_mtime, older.stat().st_mtime)

            selected, skipped = _latest_valid_baseline(run_root, 0.6016, revision)
            self.assertEqual(skipped, [])
            self.assertEqual(selected["run_id"], newer_id)
            self.assertEqual(selected["summary_path"], _posix(newer))
            self.assertEqual(selected["best"]["metrics"]["primary"], 0.6015)


class SaveOrderTests(unittest.TestCase):
    def test_rejection_state_is_saved_before_the_audit_event(self):
        """Critic rejections persist pruning state but never become DAG nodes."""
        with research_loop([]) as (loop, _provider):
            decision = ResearchDecision.from_dict(research())
            preflight = CriticDecision.from_dict({**critic(), "approved": False})
            original = loop.audit.append_jsonl
            def fail_research_memory(path, value):
                if path.name == "research_memory.jsonl":
                    raise RuntimeError("disk full")
                return original(path, value)
            with patch.object(loop.audit, "append_jsonl", side_effect=fail_research_memory):
                with self.assertRaises(RuntimeError):
                    loop._record_rejection(1, decision, preflight)
            state = json.loads((loop.run_dir / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["nodes"], [])
            self.assertEqual(sum(state["branch_rejections"].values()), 1)


class WallClockTests(unittest.TestCase):
    def test_wall_clock_includes_the_baseline_gate(self):
        """I9: the clock starts before the baseline gate, not after it.

        ``_ensure_baseline`` can spend minutes reproducing the official run, and
        ``resources.json`` is the file Feasibility is scored on.
        """
        def slow_baseline(config):
            time.sleep(0.05)
            return BASELINE_SUMMARY, []

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("src.agent.research_controller._ensure_baseline", slow_baseline):
                loop = adopting_loop(root)
            loop._save()
            self.assertGreaterEqual(loop.state.wall_clock_seconds, 0.05)


class InterventionTests(unittest.TestCase):
    def test_intervene_appends_and_increments(self):
        """I10 / I-8: one command, one line, and the count follows it."""
        with research_loop([]) as (loop, _provider):
            loop._save()
            argv = [
                "controller",
                "intervene",
                "--run",
                str(loop.run_dir),
                "--reason",
                "restarted after API outage",
            ]
            with patch("sys.argv", argv):
                controller.main()

            lines = (loop.run_dir / "interventions.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["run_id"], loop.run_dir.name)
            self.assertEqual(record["reason"], "restarted after API outage")
            self.assertIn("+00:00", record["ts"])
            state = json.loads((loop.run_dir / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["manual_interventions"], 1)
            resources = json.loads(
                (loop.run_dir / "resources.json").read_text(encoding="utf-8")
            )
            self.assertEqual(resources["manual_interventions"], 1)

    def test_intervention_count_is_derived_from_the_file(self):
        """The count is read off the file on every save, never incremented.

        A live loop holds ``self.state`` in memory and rewrites ``state.json`` on
        every save, so an incremented counter would be clobbered by a concurrent
        ``intervene``; a derived one cannot be.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = baseline_config(root)
            config["run_id_prefix"] = "kj_"
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            loop = ResearchLoop(
                config,
                config_path,
                provider=ProgrammedProvider([]),
                baseline_summary=BASELINE_SUMMARY,
            )
            # T11 step 1: the optional prefix personalises the directory while the
            # id still ends in ``_research``, which is what D's .gitignore matches.
            self.assertTrue(loop.run_dir.name.startswith("kj_"))
            self.assertTrue(loop.run_dir.name.endswith("_research"))

            (loop.run_dir / "interventions.jsonl").write_text(
                "".join(
                    json.dumps({"ts": "t", "run_id": loop.run_dir.name, "reason": str(n)})
                    + "\n"
                    for n in range(3)
                ),
                encoding="utf-8",
            )
            loop._save()
            self.assertEqual(loop.state.manual_interventions, 3)
            resources = json.loads(
                (loop.run_dir / "resources.json").read_text(encoding="utf-8")
            )
            self.assertEqual(resources["manual_interventions"], 3)

    def test_intervene_on_a_missing_run_dir_exits_nonzero(self):
        """A typo in ``--run`` must fail loudly rather than create a stray file."""
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "runs" / "not_a_run"
            argv = ["controller", "intervene", "--run", str(missing), "--reason", "x"]
            errors = io.StringIO()
            with patch("sys.argv", argv), contextlib.redirect_stderr(errors):
                with self.assertRaises(SystemExit) as raised:
                    controller.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn(str(missing), errors.getvalue())

    def test_baseline_cli_still_accepts_the_documented_flags(self):
        """The README's and B's invocation keeps parsing and keeps dispatching."""
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "baseline.json"
            config_path.write_text(json.dumps({"name": "baseline"}), encoding="utf-8")
            calls: list[Path] = []
            with patch("sys.argv", ["controller", "--config", str(config_path)]), \
                    patch("src.agent.controller.run_agent", calls.append):
                controller.main()
            self.assertEqual(calls, [config_path])


if __name__ == "__main__":
    unittest.main()
