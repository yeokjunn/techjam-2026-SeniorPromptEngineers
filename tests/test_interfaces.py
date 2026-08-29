from __future__ import annotations

import importlib
import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.agent.families import FAMILIES, family_names
from src.agent.llm import ScriptedProvider, build_provider
from src.agent.report import render_reports
from src.agent.types import ExperimentOutcome, ResearchDecision, RunState
from src.evaluation.datacard import render_data_card
from src.evaluation.gate import run_gate
from src.experiments.contracts import CandidateContext, CandidateOutput
from src.models.features import build_features


REPO_ROOT = Path(__file__).resolve().parents[1]


def decision_payload(family: str) -> dict:
    return {
        "hypothesis_id": "h_1",
        "family": family,
        "action": "explore",
        "hypothesis": "test hypothesis",
        "rationale": "approved method card",
        "parameters": {"seed": 0},
        "evidence": [],
    }


class FamilyRegistryTests(unittest.TestCase):
    def test_family_names_expose_the_registered_families(self):
        expected = {"bpr", "group_softmax", "history_features", "multi_task"}
        self.assertEqual(family_names(), frozenset(expected))
        self.assertEqual(family_names(), frozenset(FAMILIES))

    def test_each_family_points_at_a_real_method_card_and_sampler(self):
        sampling = importlib.import_module("src.models.sampling")
        for name, family in FAMILIES.items():
            with self.subTest(family=name):
                self.assertEqual(family.name, name)
                self.assertTrue((REPO_ROOT / family.method_card).is_file())
                self.assertTrue(callable(getattr(sampling, family.trusted_sampler)))

    def test_research_decisions_accept_registered_families_only(self):
        for name in family_names():
            with self.subTest(family=name):
                self.assertEqual(
                    ResearchDecision.from_dict(decision_payload(name)).family, name
                )
        with self.assertRaises(ValueError):
            ResearchDecision.from_dict(decision_payload("listwise_magic"))


class CandidateContractTests(unittest.TestCase):
    def context(self, **overrides) -> CandidateContext:
        fields = {
            "train_x": np.zeros((2, 5)),
            "train_y": np.zeros(2),
            "train_users": ("u1", "u2"),
            "valid_x": np.zeros((2, 5)),
            "valid_users": ("u1", "u2"),
            "field_dimension": 5,
            "evaluate_validation": lambda scores: {"primary": 0.0},
        }
        fields.update(overrides)
        return CandidateContext(**fields)

    def test_candidate_output_test_scores_default_to_none(self):
        output = CandidateOutput(validation_scores=np.zeros(3), checkpoint_state={})
        self.assertIsNone(output.test_scores)

    def test_candidate_context_takes_optional_test_features(self):
        self.assertIsNone(self.context().test_x)
        with_test = self.context(test_x=np.zeros((2, 5)))
        self.assertEqual(with_test.test_x.shape, (2, 5))


class StateFieldTests(unittest.TestCase):
    def test_experiment_outcome_failure_class_defaults_to_none(self):
        outcome = ExperimentOutcome(status="failed", metrics=None, duration_seconds=0.5)
        self.assertIsNone(outcome.failure_class)
        self.assertIsNone(outcome.to_dict()["failure_class"])

    def test_run_state_written_without_data_card_path_still_loads(self):
        legacy = RunState(
            run_id="run-1",
            status="running",
            started_at="2026-01-01T00:00:00+00:00",
            baseline_primary=0.6,
        ).to_dict()
        legacy.pop("data_card_path")
        restored = RunState.from_dict(json.loads(json.dumps(legacy)))
        self.assertIsNone(restored.data_card_path)
        self.assertIn("data_card_path", restored.to_dict())


class ProviderFactoryTests(unittest.TestCase):
    def test_scripted_provider_is_loaded_from_a_script_file(self):
        with tempfile.TemporaryDirectory() as directory:
            script_path = Path(directory) / "script.json"
            script_path.write_text("[]", encoding="utf-8")
            provider = build_provider(
                {"llm": {"provider": "scripted", "script_path": str(script_path)}}
            )
        self.assertIsInstance(provider, ScriptedProvider)
        self.assertEqual(provider.responses, [])

    def test_unknown_provider_is_rejected_by_name(self):
        with self.assertRaises(ValueError) as raised:
            build_provider({"llm": {"provider": "nope"}})
        self.assertIn("nope", str(raised.exception))

    def test_openai_provider_still_requires_an_api_key(self):
        with patch("src.agent.llm.load_project_environment", return_value=False), patch.dict(
            os.environ, {}, clear=False
        ):
            os.environ.pop("OPENAI_API_KEY", None)
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                build_provider({"llm": {"provider": "openai"}})


class StubTests(unittest.TestCase):
    def test_gate_reports_error_when_scores_are_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            result = run_gate(path, path, path, path)
        self.assertEqual(result.status, "error")
        self.assertIsNone(result.submission_path)
        self.assertEqual(asdict(result)["details"]["reason"], "missing_test_scores")

    def test_data_card_stub_renders_nothing_yet(self):
        self.assertEqual(render_data_card(Path(".")), "")

    def test_report_stub_returns_none(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(render_reports(Path(directory)))

    def test_history_features_return_one_column_per_enabled_group(self):
        train = [(20220408, "u", "v", "a", "1", 1.0, 1)]
        spec = {"split": "train", "history_rows": {"train": train}}
        self.assertEqual(build_features(np.zeros((1, 5)), spec).shape, (1, 6))


class ResearchConfigTests(unittest.TestCase):
    def test_config_exposes_the_frozen_budget_and_data_card_keys(self):
        config = json.loads(
            (REPO_ROOT / "configs" / "ranking_losses.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["budgets"]["max_iterations"], 50)
        self.assertEqual(config["budgets"]["max_training_attempts"], 50)
        self.assertIn("data_card_path", config)


if __name__ == "__main__":
    unittest.main()
