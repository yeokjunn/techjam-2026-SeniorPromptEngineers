"""The EDA roles must be handed the measurements the harness already took.

The two EDA roles execute nothing: `research_controller._run_eda` is two model
calls with no analysis step between them, and `EDAResearchPlan` has no field a
plan could be executed from. So every number an EDA finding can honestly cite
has to arrive inside the prompt. The DATA CARD and RESEARCH STATE already do;
`artifacts/ui/kuairand_pure_eda.json` -- the per-user impression/positive
quantiles and the duration histogram the recorded plans keep asking for -- did
not reach any prompt at all before this wiring.

These tests pin the reader, not the model: the digest renders real measured
numbers, stays bounded, is read once, degrades to nothing (never to a dangling
heading) when the file is missing or malformed, and lands in both EDA prompts.
No controller, no network, no dataset scan.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.agent.audit import ResearchAudit
from src.agent.catalog import MethodCatalog
from src.agent.llm import ScriptedProvider
from src.agent.roles import (
    DEFAULT_MEASURED_PROFILE_PATH,
    MEASURED_PROFILE_CHAR_LIMIT,
    ResearchRoles,
)
from src.agent.types import EDAReport, RunState


REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_PROFILE = REPO_ROOT / DEFAULT_MEASURED_PROFILE_PATH

# The sentinel `_eda_context` returns when no report exists (`roles.py`). Pinned
# verbatim: the downstream tests in `tests/test_eda_wiring.py` assert on its
# absence, so a silent reword here would quietly disarm them.
NO_EDA_SENTINEL = "No EDA report has been produced for this iteration."

# Same shape as the real artifact, small enough to read: two splits, quantile
# blocks, a two-bucket histogram and two logged days.
FIXTURE_PROFILE = {
    "schema_version": 1,
    "generated_at": "2026-08-29T14:21:34.332456+00:00",
    "provenance": "Aggregates from the trusted train split and validation split.",
    "splits": {
        "train": {
            "rows": 1141112,
            "users": 26210,
            "positives": 384121,
            "positive_rate": 0.33661989357749283,
            "impressions_per_user": {
                "min": 1.0,
                "p25": 13.0,
                "p50": 31.0,
                "p75": 59.0,
                "p95": 127.0,
                "max": 809.0,
            },
            "positives_per_user": {
                "min": 0.0,
                "p25": 4.0,
                "p50": 10.0,
                "p75": 21.0,
                "p95": 45.0,
                "max": 155.0,
            },
        },
        "valid": {
            "rows": 124909,
            "users": 22377,
            "positives": 39132,
            "positive_rate": 0.313284070803545,
            "impressions_per_user": {
                "min": 1.0,
                "p25": 2.0,
                "p50": 4.0,
                "p75": 7.0,
                "p95": 16.0,
                "max": 74.0,
            },
            "positives_per_user": {
                "min": 0.0,
                "p25": 0.0,
                "p50": 1.0,
                "p75": 2.0,
                "p95": 6.0,
                "max": 24.0,
            },
        },
    },
    "duration_histogram": [
        {"seconds": "0–5", "rows": 26183},
        {"seconds": "60–120", "rows": 339346},
    ],
    "activity_by_date": [
        {"split": "train", "date": 20220409, "rows": 52736, "positives": 17728, "positive_rate": 0.33616504854368934},
        {"split": "valid", "date": 20220422, "rows": 17000, "positives": 5100, "positive_rate": 0.3},
    ],
}

PLAN_PAYLOAD = {
    "objective": "Characterise per-user impression depth before choosing a pairwise loss.",
    "questions": ["How many validation users carry both a positive and a negative?"],
    "feature_hypotheses": ["User impression depth interacts with item duration."],
    "required_inputs": ["MEASURED PROFILE impressions_per_user quantiles"],
    "leakage_risks": ["Any per-user statistic must be computed on train rows only."],
    "expected_artifacts": ["A short note on pairable-user share."],
}

REPORT_PAYLOAD = {
    "summary": "Validation users are shallow: p50 impressions_per_user is 4.",
    "findings": [
        {
            "title": "Half the validation users see four impressions or fewer",
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
            "expected_impact": "Better negative sampling for shallow users.",
            "implementation_scope": "src.models.features",
            "leakage_risk": "train-only counts",
        }
    ],
    "recommended_next_focus": "bpr",
    "ui_notes": ["Show the impression-depth histogram."],
}


def _roles(
    provider: ScriptedProvider,
    directory: str,
    measured_profile_path: str | Path | None = None,
) -> ResearchRoles:
    """The role harness of `tests/test_builder_blocklist_prompt.py`, plus the profile."""
    return ResearchRoles(
        provider,
        MethodCatalog.load(REPO_ROOT / "research" / "methods"),
        ResearchAudit(Path(directory) / "run"),
        max_total_tokens=10000,
        measured_profile_path=measured_profile_path,
    )


def _state(data_card_path: str | None = None) -> RunState:
    return RunState("run", "running", "now", 0.6016, data_card_path=data_card_path)


class MeasuredProfileDigestTests(unittest.TestCase):
    @unittest.skipUnless(REAL_PROFILE.is_file(), f"{REAL_PROFILE} is not present")
    def test_profile_digest_renders_the_real_artifact_within_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            roles = _roles(ScriptedProvider([]), directory, REAL_PROFILE)
            text = roles._measured_profile_text()
        # Measured values, not placeholders: rows/users off `splits`, the p95 the
        # recorded EDA plans asked for, and a duration bucket label.
        for token in ("1,141,112", "26,210", "124,909", "p95=127", "p50=1", "60–120"):
            self.assertIn(token, text, f"{token!r} missing from the rendered digest")
        self.assertLessEqual(len(text), MEASURED_PROFILE_CHAR_LIMIT)
        self.assertNotIn("(truncated)", text)

    def test_digest_renders_every_section_of_a_fixture_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(FIXTURE_PROFILE), encoding="utf-8")
            roles = _roles(ScriptedProvider([]), directory, path)
            text = roles._measured_profile_text()
        self.assertIn("| train | 1,141,112 | 26,210 | 384,121 | 0.3366 |", text)
        self.assertIn("| valid | 124,909 | 22,377 | 39,132 | 0.3133 |", text)
        self.assertIn(
            "- train impressions_per_user: min=1, p25=13, p50=31, p75=59, p95=127, max=809",
            text,
        )
        self.assertIn("- valid positives_per_user: min=0, p25=0, p50=1, p75=2, p95=6, max=24", text)
        self.assertIn("60–120: 339,346", text)
        # 20 rows of daily activity are summarised, never dumped.
        self.assertIn("Daily activity: 2 logged days; per-day positive rate 0.3000 to 0.3362.", text)
        self.assertNotIn("20220409", text)

    def test_missing_and_malformed_profile_render_empty_and_add_no_heading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truncated = root / "truncated.json"
            truncated.write_text("{", encoding="utf-8")
            not_a_mapping = root / "list.json"
            not_a_mapping.write_text("[1, 2, 3]", encoding="utf-8")
            wrong_types = root / "wrong_types.json"
            wrong_types.write_text(
                json.dumps({"provenance": "x", "splits": {"train": {"rows": "many"}}}),
                encoding="utf-8",
            )
            a_directory = root / "adir"
            a_directory.mkdir()
            candidates = {
                "unset": None,
                "missing": root / "nope.json",
                "directory": a_directory,
                "truncated": truncated,
                "not a mapping": not_a_mapping,
                "wrong types": wrong_types,
            }
            for label, profile_path in candidates.items():
                with self.subTest(profile=label):
                    # A fresh audit root per case: ``ResearchAudit`` creates the
                    # directory and refuses an existing one outside a resume.
                    roles = _roles(
                        ScriptedProvider([]),
                        str(root / f"audit_{label.replace(' ', '_')}"),
                        profile_path,
                    )
                    self.assertEqual(roles._measured_profile_text(), "")
                    # Not merely empty: the prefix must gain no heading with
                    # nothing under it, exactly as a missing DATA CARD adds none.
                    self.assertNotIn("MEASURED PROFILE", roles._eda_prefix(_state()))

    def test_oversized_profile_is_truncated(self):
        oversized = dict(FIXTURE_PROFILE, provenance="x" * (MEASURED_PROFILE_CHAR_LIMIT + 500))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(oversized), encoding="utf-8")
            roles = _roles(ScriptedProvider([]), directory, path)
            text = roles._measured_profile_text()
        self.assertTrue(text.endswith("… (truncated)"), text[-40:])
        # The guard bounds the prompt cost: the body is capped, and only the
        # marker is added on top.
        self.assertLessEqual(len(text), MEASURED_PROFILE_CHAR_LIMIT + len("\n… (truncated)"))

    def test_profile_is_read_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(FIXTURE_PROFILE), encoding="utf-8")
            roles = _roles(ScriptedProvider([]), directory, path)
            first = roles._measured_profile_text()
            path.unlink()
            self.assertEqual(roles._measured_profile_text(), first)
            # The memoized text is what the prefix keeps using, so the EDA prefix
            # stays byte-identical across the two calls of one iteration.
            self.assertIn(first, roles._eda_prefix(_state()))


class EDAPromptContentTests(unittest.TestCase):
    def test_eda_researcher_and_builder_prompts_carry_both_measured_blocks(self):
        provider = ScriptedProvider([dict(PLAN_PAYLOAD), dict(REPORT_PAYLOAD)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "profile.json"
            profile_path.write_text(json.dumps(FIXTURE_PROFILE), encoding="utf-8")
            card_path = root / "DATA_CARD.md"
            card_path.write_text("# Data card\n\ntrain | 1,141,112 | 26,210 | 7,538\n", encoding="utf-8")
            roles = _roles(provider, directory, profile_path)
            state = _state(str(card_path))
            plan = roles.eda_research(state, 1)
            report = roles.eda_build(state, 1, plan)

        self.assertEqual(plan.objective, PLAN_PAYLOAD["objective"])
        self.assertEqual(report.summary, REPORT_PAYLOAD["summary"])
        researcher_prompt = provider.calls[0]["prompt"]
        builder_prompt = provider.calls[1]["prompt"]
        for prompt in (researcher_prompt, builder_prompt):
            self.assertIn("DATA CARD:", prompt)
            self.assertIn("MEASURED PROFILE", prompt)
            self.assertIn("impressions_per_user", prompt)
            self.assertIn("p95=127", prompt)
            self.assertIn("RESEARCH STATE:", prompt)
            # The EDA prefix deliberately skips the candidate contract and the
            # method cards; adding the profile must not turn it into the big one.
            self.assertLess(len(prompt), 25_000)
        # Each role is told to stay inside the measured blocks.
        self.assertIn("do not request a statistic no one will compute", researcher_prompt)
        self.assertIn("quote a number that appears verbatim", builder_prompt)
        self.assertIn("EDA PLAN:", builder_prompt)
        self.assertIn(PLAN_PAYLOAD["objective"], builder_prompt)

    def test_eda_context_renders_a_report_and_the_sentinel(self):
        self.assertEqual(ResearchRoles._eda_context(None), NO_EDA_SENTINEL)
        empty = EDAReport.from_dict(
            {
                "summary": "nothing conclusive",
                "findings": [],
                "feature_candidates": [],
                "recommended_next_focus": "bpr",
            }
        )
        rendered = json.loads(ResearchRoles._eda_context(empty))
        # An empty report is still a report: there is no empty-report branch, so
        # downstream prompts get JSON rather than the sentinel.
        self.assertEqual(rendered["summary"], "nothing conclusive")
        self.assertEqual(rendered["feature_candidates"], [])
        self.assertEqual(rendered["findings"], [])


if __name__ == "__main__":
    unittest.main()
