"""The research loop survives everything a role pass can raise (review C4 + I13).

Every exception the correctness report names is classified into one of three
kinds and handled without ending the run: a **proposal** error is the model's
fault, so the role is re-prompted and then ledgered as a failed proposal; a
**budget** error stops the run cleanly; only a run of consecutive **harness**
errors trips the circuit breaker. ``stop_reason`` is never ``controller_error``.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.agent.errors import (
    IncompleteResponse,
    LLMError,
    RoleOutputInvalid,
    TokenBudgetExceeded,
)
from src.agent.llm import LLMCallResult
from src.agent.research_controller import ResearchLoop, _error_kind, _is_budget_error
from src.agent.types import ExperimentNode, ExperimentOutcome, TokenUsage


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
            (TokenBudgetExceeded("Run exhausted its LLM allowance."), "budget"),
            (RuntimeError("LLM token budget reached before the next role pass."), "budget"),
            (RuntimeError("LLM token budget exceeded by the completed role pass."), "budget"),
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

    def test_budget_is_checked_before_the_llm_error_arm(self):
        # TokenBudgetExceeded is itself an LLMError, so a reordering that put the
        # proposal arm first would re-prompt an exhausted budget twice and then
        # abandon the proposal instead of stopping the run.
        self.assertTrue(issubclass(TokenBudgetExceeded, LLMError))
        self.assertEqual(_error_kind(TokenBudgetExceeded("out of tokens")), "budget")


# --------------------------------------------------------------------------- #
# Proposal errors: re-prompt, then ledger, never break
# --------------------------------------------------------------------------- #


class ProposalErrorTests(unittest.TestCase):
    def test_malformed_research_response_is_reprompted_and_the_run_survives(self):
        broken = research("bpr")
        broken["family"] = "nope"  # types.py:113 -> ValueError
        with research_loop([broken, *good_iteration("bpr")]) as (loop, provider):
            run_dir = loop.run()
            summary = summary_of(run_dir)
            self.assertNotEqual(summary["stop_reason"], "controller_error", error_report(run_dir))
            self.assertEqual(summary["stop_reason"], "iteration_budget_reached")
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
            self.assertIn("epochs must be between 1 and 40.", retries[0]["error"])
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
            self.assertEqual(summary["stop_reason"], "iteration_budget_reached")
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
            self.assertEqual(summary_of(run_dir)["stop_reason"], "iteration_budget_reached")
            self.assertEqual(len(provider.calls), 5)
            self.assertEqual(len(memory(run_dir, "role_retry")), 1)
            self.assertEqual(loop.consecutive_harness_errors, 0)
            self.assertEqual([node.status for node in loop.state.nodes], ["success"])


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
# Budget errors: stop cleanly, before and after C's typed exception (T8 / I13)
# --------------------------------------------------------------------------- #


class TokenBudgetStopTests(unittest.TestCase):
    def test_token_budget_message_stops_the_run_cleanly(self):
        # roles.py:52 as it stands today: a bare RuntimeError whose message
        # carries the phrase. The message check in _is_budget_error covers it.
        message = "LLM token budget reached before the next role pass."
        with research_loop([failure(RuntimeError, message)]) as (loop, provider):
            run_dir = loop.run()
            self.assertEqual(summary_of(run_dir)["stop_reason"], "llm_token_budget_reached")
            self.assertEqual(len(provider.calls), 1)
            self.assertEqual(memory(run_dir, "role_retry"), [])
            self.assertEqual(
                [record["kind"] for record in memory(run_dir, "controller_error")], ["budget"]
            )
            self.assertEqual(jsonl(run_dir, "iterations.jsonl"), [])
            self.assertFalse((run_dir / "error.json").is_file())

    def test_token_budget_exceeded_type_stops_the_run(self):
        # roles.py:52 after C's T2 step 7. The message deliberately omits the
        # phrase, so only the isinstance branch can classify this as a budget stop.
        message = "Run exhausted its LLM allowance."
        self.assertNotIn("token budget", message.lower())
        with research_loop([failure(TokenBudgetExceeded, message)]) as (loop, provider):
            run_dir = loop.run()
            self.assertEqual(summary_of(run_dir)["stop_reason"], "llm_token_budget_reached")
            self.assertEqual(len(provider.calls), 1)
            self.assertEqual(memory(run_dir, "role_retry"), [])
            self.assertEqual(
                [record["kind"] for record in memory(run_dir, "controller_error")], ["budget"]
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


if __name__ == "__main__":
    unittest.main()
