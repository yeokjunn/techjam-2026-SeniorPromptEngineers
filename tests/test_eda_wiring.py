"""EDA is only wired if the report reaches the roles that act on it.

`configs/run_kj*.json` now carry the `eda` block, so `_run_eda` runs first in
every iteration. Enabling it is worth nothing unless three things hold at once:
the EDA roles see the measurements the harness already took, the report they
produce lands in the Researcher / Critic / Builder prompts instead of the
"no EDA" sentinel, and a bad EDA pass costs the iteration nothing while
`required` is false. Those are the four properties pinned here, plus the
re-prompt bound that keeps a stubborn EDA role from eating the run.

Everything is offline: a role-keyed scripted provider, a fake trainer, an empty
data directory (so `render_data_card` returns "" and no CSV is read) and a
hand-written profile fixture. Every run directory is a `TemporaryDirectory`.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.agent.llm import LLMCallResult
from src.agent.research_controller import ResearchLoop
from src.agent.types import ExperimentOutcome, TokenUsage


REPO_ROOT = Path(__file__).resolve().parents[1]

# `roles.py` `_eda_context`: what every downstream prompt says when no report
# exists. Its presence is the negative control for the whole file.
NO_EDA_SENTINEL = "No EDA report has been produced for this iteration."

# Same shape as `artifacts/ui/kuairand_pure_eda.json`, cut down to what the
# digest renders. `p95=127` below is the assertion target: a number that can
# only have come through the profile file.
PROFILE_FIXTURE = {
    "schema_version": 1,
    "provenance": "Aggregates from the trusted train split and validation split.",
    "splits": {
        "train": {
            "rows": 1141112,
            "users": 26210,
            "positives": 384121,
            "positive_rate": 0.33661989357749283,
            "impressions_per_user": {"min": 1, "p25": 13, "p50": 31, "p75": 59, "p95": 127, "max": 809},
            "positives_per_user": {"min": 0, "p25": 4, "p50": 10, "p75": 21, "p95": 45, "max": 155},
        },
        "valid": {
            "rows": 124909,
            "users": 22377,
            "positives": 39132,
            "positive_rate": 0.313284070803545,
            "impressions_per_user": {"min": 1, "p25": 2, "p50": 4, "p75": 7, "p95": 16, "max": 74},
            "positives_per_user": {"min": 0, "p25": 0, "p50": 1, "p75": 2, "p95": 6, "max": 24},
        },
    },
    "duration_histogram": [{"seconds": "60–120", "rows": 339346}],
    "activity_by_date": [
        {"split": "train", "date": 20220409, "rows": 52736, "positives": 17728, "positive_rate": 0.336}
    ],
}

BASELINE_SUMMARY = {
    "best": {
        "experiment_id": "official_fm_seed0",
        "metrics": {"GAUC": 0.6671, "nDCG@5": 0.5358, "primary": 0.6015},
        "artifact_path": "baseline.npz",
    }
}

CANDIDATE_CODE = '''import numpy as np
from src.experiments.contracts import CandidateOutput
from src.models.sampling import sample_bpr_pairs

def run(context, parameters):
    sample_bpr_pairs(context.train_users, context.train_y, np.random.default_rng(0), 1)
    return CandidateOutput(np.zeros(len(context.valid_x)), {"weights": np.zeros(1)}, [], {"pairs": 1})
'''

CANDIDATE_TESTS = """import unittest
import candidate

class CandidateTests(unittest.TestCase):
    def test_contract(self):
        self.assertTrue(callable(candidate.run))
"""

BPR_PARAMETERS: dict[str, Any] = {
    "seed": 0,
    "k": 16,
    "learning_rate": 0.001,
    "epochs": 5,
    "batch_size": 2048,
    "patience": 2,
    "negatives_per_positive": 1,
    "negatives_per_group": None,
    "temperature": None,
}

CRITIC_PAYLOAD: dict[str, Any] = {
    "approved": True,
    "decision": "proceed",
    "rationale": "safe controlled experiment",
    "concerns": [],
    "next_focus": "compare trusted metrics",
}

EDA_PLAN_PAYLOAD: dict[str, Any] = {
    "objective": "Size the pairable-user population before committing to a pairwise loss.",
    "questions": ["What share of validation users carry both classes?"],
    "feature_hypotheses": ["Impression depth interacts with item duration."],
    "required_inputs": ["MEASURED PROFILE impressions_per_user quantiles"],
    "leakage_risks": ["Per-user counts must be train-only."],
    "expected_artifacts": ["A note on pairable-user share."],
}

EDA_REPORT_PAYLOAD: dict[str, Any] = {
    "summary": "Validation lists are shallow; median user sees four impressions.",
    "findings": [
        {
            "title": "Shallow validation lists",
            "observation": "valid impressions_per_user p50=4",
            "implication": "GAUC is dominated by very short lists.",
            "evidence": "MEASURED PROFILE: valid impressions_per_user p50=4",
            "leakage_safe": True,
        }
    ],
    "feature_candidates": [
        {
            "name": "user_impression_depth",
            "description": "Bucketed train-only impression count per user.",
            "family": "bpr",
            "expected_impact": "Better negatives for shallow users.",
            "implementation_scope": "src.models.features",
            "leakage_risk": "train-only counts",
        }
    ],
    "recommended_next_focus": "bpr",
    "ui_notes": ["Show the impression-depth histogram."],
}

# Answered by role rather than in call order, as in `tests/test_controller_wiring.py`:
# how many passes the loop makes is part of what these tests measure.
ROLE_PAYLOADS: dict[str, dict[str, Any]] = {
    "eda_researcher": EDA_PLAN_PAYLOAD,
    "eda_builder": EDA_REPORT_PAYLOAD,
    "researcher": {
        "hypothesis_id": "h_bpr",
        "family": "bpr",
        "action": "explore",
        "hypothesis": "controlled bpr ranking loss",
        "rationale": "approved method card",
        "parameters": BPR_PARAMETERS,
        "evidence": [
            {
                "title": "Primary paper",
                "url": "https://arxiv.org/abs/1205.2618",
                "method_card_id": "bpr",
            }
        ],
        "needs_web_search": False,
        "parent_experiment": None,
    },
    "critic_preflight": CRITIC_PAYLOAD,
    "builder": {
        "candidate_id": "candidate_bpr",
        "hypothesis_id": "h_bpr",
        "family": "bpr",
        "code": CANDIDATE_CODE,
        "tests": CANDIDATE_TESTS,
        "parameters": BPR_PARAMETERS,
    },
    "critic_postflight": CRITIC_PAYLOAD,
}

DOWNSTREAM_ROLES = ("researcher", "critic_preflight", "builder")


class RoleScriptedProvider:
    """Offline provider answering by role, recording every prompt it is sent.

    ``overrides`` supplies a per-role queue that is consumed before the default
    payload, which is how a case scripts one bad EDA response followed by a good
    one without having to predict the rest of the pass order.
    """

    def __init__(self, overrides: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self.prompts: list[tuple[str, str]] = []
        self.overrides = {role: list(items) for role, items in (overrides or {}).items()}

    def complete(self, **kwargs: Any) -> LLMCallResult:
        role = str(kwargs["role"])
        self.prompts.append((role, str(kwargs["prompt"])))
        queue = self.overrides.get(role)
        payload = queue.pop(0) if queue else ROLE_PAYLOADS.get(role)
        if payload is None:
            raise AssertionError(f"unscripted {role} call")
        return LLMCallResult(
            data=dict(payload),
            response_id=f"scripted-{len(self.prompts)}",
            model="scripted",
            role=role,
            latency_seconds=0.0,
            retries=0,
            usage=TokenUsage(total_tokens=10),
        )

    def prompts_for(self, role: str) -> list[str]:
        return [prompt for name, prompt in self.prompts if name == role]


class FakeExecutor:
    """Trusted-worker double: safety tests pass, training is canned."""

    def test(self, workspace: Any) -> tuple[bool, str]:
        return True, "ok"

    def train(self, iteration: int, manifest: Any, workspace: Any, run_dir: Path):
        return ExperimentOutcome(
            status="success",
            metrics={"GAUC": 0.601, "nDCG@5": 0.601, "primary": 0.601},
            duration_seconds=0.01,
            artifact_path="artifact-bpr.npz",
            diagnostics={"eligible_users": 10},
        )


@contextlib.contextmanager
def eda_loop(
    eda: dict[str, Any] | None,
    overrides: dict[str, list[dict[str, Any]]] | None = None,
    max_proposals: int = 4,
):
    """One research iteration with the given ``eda`` config block (``None`` omits it).

    ``data_dir`` is an empty directory on purpose: Owner D's renderer returns ""
    for it, so the run start reads no CSV and leaves no data card behind. The
    measured profile is a fixture in the same temp tree, so nothing under
    ``artifacts/`` is touched either.
    """
    provider = RoleScriptedProvider(overrides)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        data_dir = root / "data"
        data_dir.mkdir()
        profile_path = root / "profile.json"
        profile_path.write_text(json.dumps(PROFILE_FIXTURE), encoding="utf-8")
        config: dict[str, Any] = {
            "mode": "research",
            "name": "eda-wiring",
            "data_dir": str(data_dir),
            "run_root": str(root / "runs"),
            "generated_root": str(root / "generated"),
            "method_catalog": str(REPO_ROOT / "research" / "methods"),
            "discovery_store": str(root / "discoveries.json"),
            "campaign_log": str(root / "campaign_log.md"),
            "official_validation_baseline": 0.6016,
            "llm": {"max_total_tokens": 100000},
            "budgets": {
                "max_iterations": 1,
                "max_training_attempts": 2,
                "max_proposals": max_proposals,
                "max_wall_clock_seconds": 60,
                "experiment_timeout_seconds": 10,
                "test_timeout_seconds": 10,
                "max_debug_repairs": 1,
            },
            "convergence": {"epsilon": 0.002, "patience": 3},
            "replication_seeds": [1, 2],
        }
        if eda is not None:
            config["eda"] = dict(eda, profile_path=str(profile_path))
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        loop = ResearchLoop(
            config,
            config_path,
            provider=provider,
            baseline_summary=BASELINE_SUMMARY,
        )
        loop.executor = FakeExecutor()
        run_dir = loop.run()
        yield loop, provider, run_dir


def _memory(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "research_memory.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _iterations(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "iterations.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _summary(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


class EDAWiringTests(unittest.TestCase):
    def test_eda_enabled_writes_artifacts_and_reaches_downstream_prompts(self):
        eda = {
            "enabled": True,
            "required": False,
            "researcher_max_output_tokens": 1200,
            "builder_max_output_tokens": 1500,
            "max_retries": 1,
            "max_role_reprompts": 1,
        }
        with eda_loop(eda) as (_loop, provider, run_dir):
            record = json.loads((run_dir / "eda" / "001_eda.json").read_text(encoding="utf-8"))
            latest = json.loads((run_dir / "eda" / "latest.json").read_text(encoding="utf-8"))
            iterations = _iterations(run_dir)
            eda_prompts = provider.prompts_for("eda_researcher")
            builder_eda_prompts = provider.prompts_for("eda_builder")
            downstream = {role: provider.prompts_for(role) for role in DOWNSTREAM_ROLES}

        self.assertEqual(record["status"], "completed")
        self.assertEqual(latest, record)
        self.assertEqual(record["plan"]["objective"], EDA_PLAN_PAYLOAD["objective"])
        self.assertEqual(record["report"]["summary"], EDA_REPORT_PAYLOAD["summary"])
        self.assertEqual(record["feature_candidates"][0]["name"], "user_impression_depth")
        self.assertEqual(len(record["findings"]), 1)

        # The measured profile reached both EDA roles: `p95=127` exists nowhere
        # but the digest of the profile file.
        self.assertEqual(len(eda_prompts), 1)
        self.assertEqual(len(builder_eda_prompts), 1)
        for prompt in eda_prompts + builder_eda_prompts:
            self.assertIn("MEASURED PROFILE", prompt)
            self.assertIn("p95=127", prompt)
            self.assertIn("1,141,112", prompt)
        self.assertIn("EDA PLAN:", builder_eda_prompts[0])
        self.assertIn(EDA_PLAN_PAYLOAD["objective"], builder_eda_prompts[0])
        # Empty data directory: no card is rendered, and no heading is left over.
        self.assertNotIn("DATA CARD:", eda_prompts[0])

        # ...and the report reached everyone downstream who takes one.
        for role, prompts in downstream.items():
            self.assertTrue(prompts, f"{role} was never called")
            for prompt in prompts:
                self.assertIn("EDA EVIDENCE:", prompt)
                self.assertNotIn(NO_EDA_SENTINEL, prompt)
                self.assertIn(EDA_REPORT_PAYLOAD["summary"], prompt)
                self.assertIn("user_impression_depth", prompt)

        self.assertEqual(len(iterations), 1)
        ledger = iterations[0]
        self.assertTrue(
            str(ledger["eda_artifact_path"]).endswith("eda/001_eda.json"),
            ledger["eda_artifact_path"],
        )
        self.assertIsNotNone(ledger["agent_notes"]["eda"])
        self.assertEqual(
            ledger["agent_notes"]["eda"]["summary"], EDA_REPORT_PAYLOAD["summary"]
        )

    def test_eda_absent_config_leaves_no_artifacts_and_keeps_the_sentinel(self):
        with eda_loop(None) as (_loop, provider, run_dir):
            eda_dir_exists = (run_dir / "eda").exists()
            roles_called = {role for role, _ in provider.prompts}
            downstream = {role: provider.prompts_for(role) for role in DOWNSTREAM_ROLES}
            iterations = _iterations(run_dir)

        self.assertFalse(eda_dir_exists)
        self.assertNotIn("eda_researcher", roles_called)
        self.assertNotIn("eda_builder", roles_called)
        for role, prompts in downstream.items():
            self.assertTrue(prompts, f"{role} was never called")
            for prompt in prompts:
                self.assertIn(NO_EDA_SENTINEL, prompt)
        self.assertIsNone(iterations[0]["eda_artifact_path"])
        self.assertIsNone(iterations[0]["agent_notes"]["eda"])

    def test_eda_failure_is_survivable_when_not_required(self):
        broken_report = {key: value for key, value in EDA_REPORT_PAYLOAD.items() if key != "findings"}
        eda = {"enabled": True, "required": False, "max_role_reprompts": 0}
        with eda_loop(eda, overrides={"eda_builder": [broken_report]}) as (
            _loop,
            provider,
            run_dir,
        ):
            failed = json.loads(
                (run_dir / "eda" / "001_eda_failed.json").read_text(encoding="utf-8")
            )
            latest = json.loads((run_dir / "eda" / "latest.json").read_text(encoding="utf-8"))
            completed_exists = (run_dir / "eda" / "001_eda.json").exists()
            memory = _memory(run_dir)
            summary = _summary(run_dir)
            iterations = _iterations(run_dir)
            downstream = {role: provider.prompts_for(role) for role in DOWNSTREAM_ROLES}

        self.assertEqual(failed["status"], "failed")
        self.assertIsNone(failed["report"])
        # The plan survived the builder's failure and is kept for the dashboard.
        self.assertEqual(failed["plan"]["objective"], EDA_PLAN_PAYLOAD["objective"])
        self.assertIn("findings", failed["error"])
        self.assertEqual(latest, failed)
        self.assertFalse(completed_exists)

        errors = [line for line in memory if line.get("type") == "eda_error"]
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0]["continued_without_eda"])
        self.assertEqual(errors[0]["iteration"], 1)

        # The iteration ran to completion anyway, on the sentinel.
        self.assertEqual(summary["training_attempts"], 1)
        self.assertEqual(iterations[0]["status"], "success")
        self.assertTrue(
            str(iterations[0]["eda_artifact_path"]).endswith("eda/001_eda_failed.json")
        )
        self.assertIsNone(iterations[0]["agent_notes"]["eda"])
        for role, prompts in downstream.items():
            self.assertTrue(prompts, f"{role} was never called")
            for prompt in prompts:
                self.assertIn(NO_EDA_SENTINEL, prompt)

    def test_eda_required_stops_the_proposal_before_any_training(self):
        broken_report = {key: value for key, value in EDA_REPORT_PAYLOAD.items() if key != "findings"}
        eda = {"enabled": True, "required": True, "max_role_reprompts": 0}
        with eda_loop(
            eda,
            overrides={"eda_builder": [broken_report, dict(broken_report)]},
            max_proposals=2,
        ) as (_loop, provider, run_dir):
            summary = _summary(run_dir)
            memory = _memory(run_dir)
            iterations = _iterations(run_dir)
            roles_called = {role for role, _ in provider.prompts}
            failed_exists = (run_dir / "eda" / "001_eda_failed.json").exists()

        # `required: true` means the proposal dies with the EDA pass: no
        # Researcher, no Builder, no candidate, and the proposal budget is what
        # stops the run rather than a trained experiment.
        self.assertTrue(failed_exists)
        self.assertEqual(summary["training_attempts"], 0)
        self.assertEqual(summary["stop_reason"], "proposal_budget_reached")
        self.assertNotIn("researcher", roles_called)
        self.assertNotIn("builder", roles_called)
        self.assertEqual([line["status"] for line in iterations], ["proposal_failed"] * 2)
        errors = [line for line in memory if line.get("type") == "eda_error"]
        self.assertEqual(len(errors), 2)
        self.assertFalse(any(line["continued_without_eda"] for line in errors))

    def test_eda_role_reprompt_is_bounded_by_its_own_setting(self):
        broken_plan = {key: value for key, value in EDA_PLAN_PAYLOAD.items() if key != "questions"}
        eda = {"enabled": True, "required": False, "max_role_reprompts": 1}
        with eda_loop(eda, overrides={"eda_researcher": [broken_plan]}) as (
            _loop,
            provider,
            run_dir,
        ):
            researcher_prompts = provider.prompts_for("eda_researcher")
            pass_names = sorted(
                path.name for path in (run_dir / "passes").glob("*_eda_researcher_*.json")
            )
            memory = _memory(run_dir)
            record = json.loads((run_dir / "eda" / "001_eda.json").read_text(encoding="utf-8"))

        # Exactly one re-prompt: the bound is `eda.max_role_reprompts`, not the
        # loop-wide `budgets.max_role_reprompts`.
        self.assertEqual(len(researcher_prompts), 2)
        self.assertEqual(pass_names, ["001_eda_researcher_0.json", "001_eda_researcher_1.json"])
        self.assertNotIn("PREVIOUS ATTEMPT REJECTED", researcher_prompts[0])
        self.assertIn("PREVIOUS ATTEMPT REJECTED", researcher_prompts[1])
        retries = [
            line
            for line in memory
            if line.get("type") == "role_retry" and line.get("label") == "eda_researcher"
        ]
        self.assertEqual(len(retries), 1)
        self.assertEqual(retries[0]["reprompt"], 1)
        # The second attempt succeeded, so the iteration still gets its report.
        self.assertEqual(record["status"], "completed")


if __name__ == "__main__":
    unittest.main()
