"""Additions A, B and C: pre-registration, the negative-result artifact, campaign memory.

A. The Researcher pre-registers a signed `predicted_delta`; the loop pairs it with the
   realized delta of every scored candidate and scores the run on calibration.
B. Every run writes `falsified.md` — what it measured flat, against the shipped noise
   constants and published effect sizes — inside the same containment as the reports.
C. Every run appends a five-line digest to `research/campaign_log.md`, and the last three
   digests ride in the Researcher's cacheable stable prefix.

Spec: docs/superpowers/specs/2026-08-31-additions-plan.md items A/B/C.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError, validate

from src.agent import discoveries, llm, research_controller
from src.agent.catalog import MethodCatalog
from src.agent.falsified import flat_families, render_falsified
from src.agent.llm import ScriptedProvider
from src.agent.research_controller import ResearchLoop, _calibration_summary
from src.agent.roles import ResearchRoles
from src.agent.types import ExperimentNode, ExperimentOutcome, ResearchDecision, RunState


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PRIMARY = 0.6015


# --------------------------------------------------------------------------- helpers


def parameters(family: str) -> dict:
    return {
        "seed": 0,
        "k": 16,
        "learning_rate": 0.001,
        "epochs": 5,
        "batch_size": 2048 if family == "bpr" else 1024,
        "patience": 2,
        "negatives_per_positive": 1 if family == "bpr" else None,
        "negatives_per_group": None if family == "bpr" else 4,
        "temperature": None if family == "bpr" else 1.0,
    }


def research(family: str, predicted_delta: float | None = None) -> dict:
    payload = {
        "hypothesis_id": f"h_{family}",
        "family": family,
        "action": "explore",
        "hypothesis": f"test {family}",
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
    if predicted_delta is not None:
        payload["predicted_delta"] = predicted_delta
    return payload


def critic() -> dict:
    return {
        "approved": True,
        "decision": "proceed",
        "rationale": "safe controlled experiment",
        "concerns": [],
        "next_focus": "compare trusted metrics",
    }


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


TESTS = """import unittest
import candidate

class CandidateTests(unittest.TestCase):
    def test_contract(self):
        self.assertTrue(callable(candidate.run))
"""


def manifest(family: str) -> dict:
    return {
        "candidate_id": f"candidate_{family}",
        "hypothesis_id": f"h_{family}",
        "family": family,
        "code": code(family),
        "tests": TESTS,
        "parameters": parameters(family),
    }


class FakeExecutor:
    """Two scored candidates: one below the promotion margin, one above the baseline."""

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


def build_loop(root: Path, responses: list[dict]) -> ResearchLoop:
    config = {
        "mode": "research",
        "name": "additions",
        "data_dir": str(REPO_ROOT / "data" / "KuaiRand-Pure" / "data"),
        "run_root": str(root / "runs"),
        "generated_root": str(root / "generated"),
        "method_catalog": str(REPO_ROOT / "research" / "methods"),
        "discovery_store": str(root / "discoveries.json"),
        "campaign_log": str(root / "campaign_log.md"),
        "official_validation_baseline": 0.6016,
        "llm": {"max_total_tokens": 1000},
        "budgets": {
            "max_iterations": 2,
            "max_wall_clock_seconds": 60,
            "experiment_timeout_seconds": 10,
            "test_timeout_seconds": 10,
            "max_debug_repairs": 2,
        },
        "convergence": {"epsilon": 0.002, "patience": 3},
        "replication_seeds": [],
    }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    loop = ResearchLoop(
        config,
        config_path,
        provider=ScriptedProvider(responses),
        baseline_summary={
            "best": {
                "experiment_id": "official_fm_seed0",
                "metrics": {"GAUC": 0.6671, "nDCG@5": 0.5358, "primary": BASELINE_PRIMARY},
                "artifact_path": "baseline.npz",
            }
        },
    )
    loop.executor = FakeExecutor()
    return loop


#: One pre-registered proposal (`bpr`) and one that is not (`group_softmax`), so a single
#: run exercises both branches of "collect only when a prediction was made".
TWO_ITERATION_SCRIPT = [
    research("bpr", predicted_delta=0.002),
    critic(),
    manifest("bpr"),
    critic(),
    research("group_softmax"),
    critic(),
    manifest("group_softmax"),
    critic(),
]


def node(
    family: str,
    primary: float | None,
    *,
    experiment_id: str = "",
    status: str = "success",
    params: dict | None = None,
) -> ExperimentNode:
    return ExperimentNode(
        iteration=1,
        experiment_id=experiment_id or f"{family}_{primary}",
        hypothesis_id=f"h_{family}",
        family=family,
        action="explore",
        parameters=params or parameters(family),
        status=status,
        metrics=None if primary is None else {"primary": primary, "GAUC": primary},
    )


def synthetic_state(nodes: list[ExperimentNode]) -> RunState:
    return RunState(
        run_id="20260831T000000000000Z_research",
        status="completed",
        started_at="2026-08-31T00:00:00+00:00",
        baseline_primary=BASELINE_PRIMARY,
        iteration_count=len(nodes),
        nodes=list(nodes),
    )


# ------------------------------------------------------------------- A: calibration


class CalibrationLedgerTests(unittest.TestCase):
    def test_pairs_are_collected_only_for_pre_registered_scored_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            loop = build_loop(Path(directory), list(TWO_ITERATION_SCRIPT))
            run_dir = loop.run()
            state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))

        self.assertEqual(len(state["nodes"]), 2, "both candidates must have scored")
        records = state["calibration"]
        self.assertEqual(
            [record["experiment_id"] for record in records],
            ["candidate_bpr"],
            "only the proposal that carried a prediction may produce a pair",
        )
        record = records[0]
        self.assertAlmostEqual(record["predicted_delta"], 0.002)
        # Predicted against the CURRENT best, which is still the baseline when the
        # first candidate lands -- not against the baseline by coincidence.
        self.assertAlmostEqual(record["reference_primary"], BASELINE_PRIMARY)
        self.assertAlmostEqual(record["realized_delta"], 0.601 - BASELINE_PRIMARY)

    def test_the_reference_is_the_incumbent_once_one_exists(self):
        """The one semantic that would silently corrupt every pair.

        `TWO_ITERATION_SCRIPT` never clears the promotion margin, so in that run
        the incumbent *is* the baseline and the two readings are
        indistinguishable. Here iteration 1 clears it and becomes the new best,
        so a `reference` of `baseline_primary` -- or a `_record_calibration`
        moved after `observe_success` -- would fail. Own executor and own script:
        three other tests pin the shared ones' exact scores.
        """

        class LeadThenPredictExecutor:
            """0.6030 clears baseline+margin; 0.6020 is then a regression on it."""

            def test(self, workspace):
                return True, "ok"

            def train(self, iteration, manifest, workspace, run_dir):
                primary = 0.6030 if manifest.candidate_id.endswith("_lead") else 0.6020
                return ExperimentOutcome(
                    status="success",
                    metrics={"GAUC": primary, "nDCG@5": primary, "primary": primary},
                    duration_seconds=0.01,
                    artifact_path=f"artifact-{manifest.candidate_id}.npz",
                    diagnostics={"eligible_users": 10},
                )

        def named(family: str, suffix: str) -> dict:
            return {
                **manifest(family),
                "candidate_id": f"candidate_{family}_{suffix}",
                "hypothesis_id": f"h_{family}_{suffix}",
            }

        # Both iterations are `bpr`: once iteration 1 takes a lead over
        # baseline+margin, `policy.required_family` locks the next proposal to the
        # best family, and a different one would be rejected as a contract breach.
        script = [
            {**research("bpr"), "hypothesis_id": "h_bpr_lead"},
            critic(),
            named("bpr", "lead"),
            critic(),
            {**research("bpr", predicted_delta=0.004), "hypothesis_id": "h_bpr_follow"},
            critic(),
            named("bpr", "follow"),
            critic(),
        ]
        with tempfile.TemporaryDirectory() as directory:
            loop = build_loop(Path(directory), script)
            loop.executor = LeadThenPredictExecutor()
            run_dir = loop.run()
            state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))

        self.assertEqual(state["best_metrics"]["primary"], 0.6030, "iteration 1 must promote")
        records = state["calibration"]
        self.assertEqual(len(records), 1, "only iteration 2 pre-registered")
        record = records[0]
        self.assertAlmostEqual(record["reference_primary"], 0.6030)
        self.assertNotAlmostEqual(record["reference_primary"], BASELINE_PRIMARY)
        # Realized against the incumbent is a regression, not the +0.0005 that
        # the same score would have shown against the baseline.
        self.assertAlmostEqual(record["realized_delta"], 0.6020 - 0.6030)
        self.assertLess(record["realized_delta"], 0.0)

    def test_summary_reports_the_calibration_of_that_run(self):
        with tempfile.TemporaryDirectory() as directory:
            loop = build_loop(Path(directory), list(TWO_ITERATION_SCRIPT))
            run_dir = loop.run()
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

        calibration = summary["calibration"]
        self.assertEqual(calibration["n"], 1)
        # predicted +0.0020, realized -0.0005 => signed error +0.0025, outside epsilon.
        self.assertAlmostEqual(calibration["mean_signed_error"], 0.002 - (0.601 - BASELINE_PRIMARY))
        self.assertAlmostEqual(calibration["mean_abs_error"], abs(calibration["mean_signed_error"]))
        self.assertEqual(calibration["within_epsilon_rate"], 0.0)

    def test_the_math_on_a_synthetic_set(self):
        records = [
            {"predicted_delta": 0.002, "realized_delta": 0.001},  # error +0.001
            {"predicted_delta": -0.001, "realized_delta": 0.002},  # error -0.003
            {"predicted_delta": 0.000, "realized_delta": 0.000},  # error  0.000
        ]
        result = _calibration_summary(records, epsilon=0.002)
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["mean_abs_error"], (0.001 + 0.003 + 0.0) / 3)
        self.assertAlmostEqual(result["mean_signed_error"], (0.001 - 0.003 + 0.0) / 3)
        # |error| <= 0.002 for the first and third only.
        self.assertAlmostEqual(result["within_epsilon_rate"], 2 / 3)
        self.assertAlmostEqual(result["epsilon"], 0.002)

    def test_no_predictions_reports_null_rather_than_zeros(self):
        self.assertIsNone(_calibration_summary([], epsilon=0.002))
        self.assertIsNone(
            _calibration_summary(
                [{"predicted_delta": None, "realized_delta": 0.001}], epsilon=0.002
            ),
            "a record with no prediction is not a perfectly calibrated one",
        )

    def test_a_run_with_no_predictions_writes_null_into_the_summary(self):
        script = [
            research("bpr"),
            critic(),
            manifest("bpr"),
            critic(),
        ]
        with tempfile.TemporaryDirectory() as directory:
            loop = build_loop(Path(directory), script)
            loop.config["budgets"]["max_iterations"] = 1
            loop.budgets["max_iterations"] = 1
            run_dir = loop.run()
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertIsNone(summary["calibration"])


class PredictedDeltaContractTests(unittest.TestCase):
    """`predicted_delta` is optional on both validator paths and never fatal."""

    DECISION = {
        "hypothesis_id": "h_1",
        "family": "bpr",
        "action": "explore",
        "hypothesis": "pairwise loss",
        "rationale": "method card",
        "parameters": parameters("bpr"),
        "evidence": [],
        "needs_web_search": False,
        "parent_experiment": None,
    }

    def test_the_chat_path_accepts_a_decision_with_and_without_it(self):
        schema = llm.SCHEMAS["research_decision"]
        Draft202012Validator.check_schema(schema)
        self.assertNotIn("predicted_delta", schema["required"])
        validate(instance=self.DECISION, schema=schema)
        validate(instance={**self.DECISION, "predicted_delta": -0.0004}, schema=schema)
        validate(instance={**self.DECISION, "predicted_delta": None}, schema=schema)
        with self.assertRaises(ValidationError):
            validate(instance={**self.DECISION, "predicted_delta": "a lot"}, schema=schema)

    def test_the_strict_path_picked_the_new_property_up(self):
        """Strict structured outputs require `required` to name every property."""
        for name, schema in llm.STRICT_SCHEMAS.items():
            with self.subTest(schema=name):
                Draft202012Validator.check_schema(schema)
                self.assertEqual(list(schema["required"]), list(schema["properties"]))
                self.assertFalse(schema["additionalProperties"])
        strict = llm.STRICT_SCHEMAS["research_decision"]
        self.assertIn("predicted_delta", strict["required"])
        # Optional is spelled the only way strict mode allows it: required, nullable.
        self.assertIn("null", strict["properties"]["predicted_delta"]["type"])
        # A strict payload names every key, so `null` is how "I have no prediction"
        # is said -- exactly as it already is for every derived parameter key.
        strict_parameters = {
            key: parameters("bpr").get(key)
            for key in llm.STRICT_PARAMETER_SCHEMA["properties"]
        }
        strict_decision = {**self.DECISION, "parameters": strict_parameters}
        validate(instance={**strict_decision, "predicted_delta": None}, schema=strict)
        validate(instance={**strict_decision, "predicted_delta": 0.0012}, schema=strict)
        with self.assertRaises(ValidationError):
            validate(instance=strict_decision, schema=strict)

    def test_the_transform_left_every_other_schema_untouched(self):
        for name in ("critic_decision", "debug_decision", "eda_report", "eda_research_plan"):
            with self.subTest(schema=name):
                self.assertEqual(llm.STRICT_SCHEMAS[name], llm.SCHEMAS[name])
        # `candidate_manifest` carries `parameters`, so it changes in exactly one
        # property and in none of the others -- the same claim, stated precisely.
        strict = llm.STRICT_SCHEMAS["candidate_manifest"]
        chat = llm.SCHEMAS["candidate_manifest"]
        self.assertEqual(list(strict["required"]), list(chat["required"]))
        self.assertIs(strict["properties"]["parameters"], llm.STRICT_PARAMETER_SCHEMA)
        self.assertEqual(
            {key: value for key, value in strict["properties"].items() if key != "parameters"},
            {key: value for key, value in chat["properties"].items() if key != "parameters"},
        )

    def test_the_decision_object_tolerates_absent_and_malformed_predictions(self):
        self.assertIsNone(ResearchDecision.from_dict(self.DECISION).predicted_delta)
        self.assertAlmostEqual(
            ResearchDecision.from_dict({**self.DECISION, "predicted_delta": 0.003}).predicted_delta,
            0.003,
        )
        for junk in ("optimistic", float("nan"), float("inf"), None, True, [0.1]):
            with self.subTest(value=repr(junk)):
                self.assertIsNone(
                    ResearchDecision.from_dict(
                        {**self.DECISION, "predicted_delta": junk}
                    ).predicted_delta
                )


# ------------------------------------------------------ B: the negative-result artifact


class FalsifiedArtifactTests(unittest.TestCase):
    def test_a_flat_family_is_named_with_its_score_band(self):
        state = synthetic_state(
            [
                node("bpr", 0.6010, experiment_id="bpr_a"),
                node("bpr", 0.6018, experiment_id="bpr_b"),
                node("group_softmax", 0.6031, experiment_id="gs_a"),
                node("history_features", None, experiment_id="hf_a", status="failed"),
            ]
        )
        self.assertEqual(flat_families(state, promotion_margin=0.001), ["bpr"])
        text = render_falsified(state, promotion_margin=0.001, epsilon=0.002)

        self.assertIn("## 1. Families measured flat", text)
        self.assertIn("Falsified this run: `bpr`", text)
        self.assertIn("0.601000 – 0.601800", text)
        # `group_softmax` cleared the margin, so it must not be reported as falsified.
        self.assertNotIn("`group_softmax`.", text)
        # The failed node contributes no band at all rather than a zero one.
        self.assertNotIn("history_features", text)
        # The noise context cites the shipped constants, not fresh guesses.
        self.assertIn("0.00091", text)
        self.assertIn("policy.MEASURED_SEED_SIGMA", text)
        self.assertIn("policy.DEFAULT_PROMOTION_MARGIN", text)
        self.assertIn("E[best Δ] under the null", text)
        self.assertIn("research/reference_effect_sizes.md", text)
        self.assertIn("+0.001", text)

    def test_a_varied_axis_that_moved_nothing_is_reported_flat(self):
        state = synthetic_state(
            [
                node("bpr", 0.6010, experiment_id="a", params={**parameters("bpr"), "k": 16}),
                node("bpr", 0.6012, experiment_id="b", params={**parameters("bpr"), "k": 64}),
            ]
        )
        text = render_falsified(state, promotion_margin=0.001)
        self.assertIn("## 2. Parameter axes measured flat", text)
        self.assertIn("| `bpr` | `k` | 16, 64 | 0.000200 | **flat** (< 1σ) |", text)

    def test_a_run_that_scored_nothing_claims_nothing(self):
        text = render_falsified(synthetic_state([]), promotion_margin=0.001)
        self.assertIn("nothing was measured and nothing is", text)
        self.assertIn("No parameter took two distinct values", text)

    def test_the_reference_table_is_checked_in(self):
        reference = REPO_ROOT / "research" / "reference_effect_sizes.md"
        self.assertTrue(reference.is_file())
        body = reference.read_text(encoding="utf-8")
        self.assertIn("arxiv.org/abs/2403.14144", body)  # Lin et al., KDD '24
        self.assertIn("arxiv.org/abs/1706.06978", body)  # DIN, KDD '18

    def test_the_run_writes_it_and_records_the_path(self):
        with tempfile.TemporaryDirectory() as directory:
            loop = build_loop(Path(directory), list(TWO_ITERATION_SCRIPT))
            run_dir = loop.run()
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue((run_dir / "falsified.md").is_file())
            self.assertTrue(summary["falsified_path"].endswith("falsified.md"))
            self.assertIn("# What this run falsified", (run_dir / "falsified.md").read_text())

    def test_a_raising_renderer_costs_the_artifact_and_nothing_else(self):
        def explode(*args, **kwargs):
            raise RuntimeError("renderer exploded")

        original = research_controller.write_falsified
        research_controller.write_falsified = explode
        try:
            with tempfile.TemporaryDirectory() as directory:
                loop = build_loop(Path(directory), list(TWO_ITERATION_SCRIPT))
                run_dir = loop.run()
                summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
                memory = [
                    json.loads(line)
                    for line in (run_dir / "research_memory.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line.strip()
                ]
                self.assertFalse((run_dir / "falsified.md").exists())
                self.assertIsNone(summary["falsified_path"])
                # Everything the organizers actually read survived.
                self.assertEqual(summary["status"], "completed")
                self.assertTrue((run_dir / "best.json").is_file())
                self.assertTrue((run_dir / "results.json").is_file())
                self.assertTrue((run_dir / "results.md").is_file())
        finally:
            research_controller.write_falsified = original

        errors = [entry for entry in memory if entry.get("type") == "falsified_error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["error"], "renderer exploded")


# ------------------------------------------------------------ C: cross-run campaign memory


class CampaignLogTests(unittest.TestCase):
    def test_a_run_appends_exactly_one_digest_and_never_a_second(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loop = build_loop(root, list(TWO_ITERATION_SCRIPT))
            loop.run()
            log = root / "campaign_log.md"
            self.assertTrue(log.is_file())
            first = log.read_text(encoding="utf-8")

            entry = discoveries.recent_campaign_digests(log)[0]
            lines = entry.splitlines()
            self.assertEqual(len(lines), 5, "the digest contract is five lines")
            self.assertEqual(lines[0], f"## run {loop.state.run_id}")
            self.assertIn("bpr 0.601000-0.601000", lines[1])
            self.assertIn("group_softmax 0.602000-0.602000", lines[1])
            self.assertIn("margin 0.0010", lines[2])
            # Both families are inside the 0.001 margin (0.6010 and 0.6020 over a
            # 0.6015 baseline), so both are falsified -- assert the whole line.
            self.assertEqual(lines[3], "- falsified: bpr, group_softmax")
            self.assertIn("stop_reason=", lines[4])

            # Idempotent per run id: a resumed run reaching the end again adds nothing.
            self.assertFalse(loop._append_campaign_digest({}))
            self.assertEqual(log.read_text(encoding="utf-8"), first)

    def test_the_log_is_created_with_a_header_and_keeps_prior_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "nested" / "campaign_log.md"
            self.assertTrue(
                discoveries.append_campaign_digest(
                    log,
                    "run_a",
                    discoveries.campaign_digest(
                        "run_a", families="bpr", verdict="flat", falsified="bpr", note="n=1"
                    ),
                )
            )
            self.assertIn("# Campaign log", log.read_text(encoding="utf-8"))
            self.assertTrue(
                discoveries.append_campaign_digest(
                    log,
                    "run_b",
                    discoveries.campaign_digest(
                        "run_b", families="gs", verdict="flat", falsified="gs", note="n=1"
                    ),
                )
            )
            self.assertFalse(
                discoveries.append_campaign_digest(
                    log,
                    "run_a",
                    discoveries.campaign_digest(
                        "run_a", families="x", verdict="y", falsified="z", note="w"
                    ),
                )
            )
            self.assertEqual(
                [entry.splitlines()[0] for entry in discoveries.recent_campaign_digests(log)],
                ["## run run_b", "## run run_a"],
                "newest first",
            )

    def test_a_multi_line_field_is_flattened_so_a_digest_stays_five_lines(self):
        digest = discoveries.campaign_digest(
            "run_x", families="a\nb", verdict="c\n\nd", falsified="", note="  e  "
        )
        self.assertEqual(len(digest.splitlines()), 5)
        self.assertEqual(digest.splitlines()[1], "- families: a b")
        self.assertEqual(digest.splitlines()[3], "- falsified: none")


class CampaignPromptTests(unittest.TestCase):
    """The last three digests ride in the cacheable stable prefix."""

    @staticmethod
    def _roles(campaign_log_path):
        return ResearchRoles(
            provider=None,
            catalog=MethodCatalog.load(REPO_ROOT / "research" / "methods"),
            audit=None,
            max_total_tokens=1000,
            campaign_log_path=campaign_log_path,
        )

    @staticmethod
    def _state() -> RunState:
        return RunState(
            run_id="r",
            status="running",
            started_at="2026-08-31T00:00:00+00:00",
            baseline_primary=BASELINE_PRIMARY,
        )

    def test_only_the_last_three_campaigns_reach_the_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "campaign_log.md"
            for index in range(5):
                discoveries.append_campaign_digest(
                    log,
                    f"run_{index}",
                    discoveries.campaign_digest(
                        f"run_{index}",
                        families=f"bpr band_{index}",
                        verdict="margin NOT cleared",
                        falsified="bpr",
                        note=f"note {index}",
                    ),
                )
            prefix = self._roles(log)._stable_prefix(self._state(), None, campaigns=True)

        self.assertIn("PRIOR CAMPAIGNS", prefix)
        for index in (4, 3, 2):
            self.assertIn(f"## run run_{index}", prefix)
        for index in (1, 0):
            self.assertNotIn(f"## run run_{index}", prefix)
        # Newest first, so the most recent campaign is read before the older ones.
        self.assertLess(prefix.index("## run run_4"), prefix.index("## run run_2"))

    def test_an_absent_log_omits_the_block_entirely(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = self._roles(Path(directory) / "missing.md")._stable_prefix(
                self._state(), None, campaigns=True
            )
        self.assertNotIn("PRIOR CAMPAIGNS", prefix)

    def test_a_role_configured_with_no_log_omits_the_block(self):
        self.assertNotIn(
            "PRIOR CAMPAIGNS",
            self._roles(None)._stable_prefix(self._state(), None, campaigns=True),
        )

    def test_the_prefix_is_byte_identical_across_calls(self):
        """It is prompt-cached: a prefix that changes mid-run charges full price."""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "campaign_log.md"
            discoveries.append_campaign_digest(
                log,
                "run_a",
                discoveries.campaign_digest(
                    "run_a", families="bpr", verdict="flat", falsified="bpr", note="n=1"
                ),
            )
            roles = self._roles(log)
            first = roles._stable_prefix(self._state(), None, campaigns=True)
            log.unlink()
            self.assertEqual(
                roles._stable_prefix(self._state(), None, campaigns=True), first
            )

    def test_the_researcher_prompt_asks_for_a_pre_registered_prediction(self):
        """The prompt the provider was actually handed -- not the source text.

        A source grep passes if the sentence has drifted into a comment, a
        docstring or a dead branch; the recorded pass is what the model saw.
        """
        with tempfile.TemporaryDirectory() as directory:
            run_dir = build_loop(Path(directory), list(TWO_ITERATION_SCRIPT)).run()
            prompt = json.loads(
                (run_dir / "passes" / "001_researcher_0.json").read_text(encoding="utf-8")
            )["prompt"]
        self.assertIn("Pre-register the outcome: emit `predicted_delta`", prompt)
        self.assertIn("you will be scored on calibration", prompt)

    def test_only_the_researcher_pays_for_the_campaign_block(self):
        """Spec C scopes cross-run memory to the proposing role."""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "campaign_log.md"
            discoveries.append_campaign_digest(
                log,
                "run_a",
                discoveries.campaign_digest(
                    "run_a", families="bpr", verdict="flat", falsified="bpr", note="n=1"
                ),
            )
            roles = self._roles(log)
            state = self._state()
            self.assertIn("PRIOR CAMPAIGNS", roles._stable_prefix(state, None, campaigns=True))
            # Every other role keeps the prefix -- and the prompt cache -- it had.
            self.assertNotIn("PRIOR CAMPAIGNS", roles._stable_prefix(state, None))


if __name__ == "__main__":
    unittest.main()
