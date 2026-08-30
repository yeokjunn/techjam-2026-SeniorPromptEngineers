"""Probe one research role against the live LLM provider.

Exercises the exact call path the controller uses (prompt construction ->
provider call -> fence/enum normalization -> strict from_dict validation) for a
single role, with realistic canned state, and prints a JSON verdict.

Usage (from the repository root):

    env -u OPENAI_API_KEY PYTHONDONTWRITEBYTECODE=1 \\
        .venv/bin/python scripts/role_probe.py <role>

Roles: researcher | critic_preflight | builder | debugger | critic_postflight

Artifacts land in /tmp/role_probes/<role>/ (prompts, parsed output, pass
records) for inspection. Each run makes real LLM calls and spends real tokens.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.agent.audit import ResearchAudit  # noqa: E402
from src.agent.catalog import MethodCatalog  # noqa: E402
from src.agent.errors import LLMError  # noqa: E402
from src.agent.families import family_names  # noqa: E402
from src.agent.llm import OpenAIResponsesProvider  # noqa: E402
from src.agent.roles import ResearchRoles  # noqa: E402
from src.agent.safety import validate_family_contract, validate_source  # noqa: E402
from src.agent.types import (  # noqa: E402
    CandidateManifest,
    ExperimentNode,
    ResearchDecision,
    RunState,
)

FM_METRICS = {
    "GAUC": 0.66713,
    "nDCG@5": 0.53581,
    "primary": 0.60147,
    "rows": 124909.0,
    "users": 22377.0,
}

BPR_PROPOSAL = {
    "hypothesis_id": "bpr-pairwise-v1",
    "family": "bpr",
    "action": "explore",
    "hypothesis": (
        "Replacing the pointwise BCE objective with a same-user pairwise BPR "
        "softplus loss on the shared k=16 FM improves within-user ordering, "
        "because GAUC and nDCG@5 reward relative order within a user rather "
        "than absolute probability calibration."
    ),
    "rationale": (
        "The official FM optimizes P(long_view | impression) pointwise while "
        "GAUC measures P(score_pos > score_neg | same user). BPR optimizes "
        "exactly that margin on same-user sampled pairs (Rendle et al., UAI "
        "2009). Single-variable change: loss and sampler, architecture and "
        "input fields fixed."
    ),
    "parameters": {
        "seed": 0,
        "k": 16,
        "learning_rate": 0.001,
        "epochs": 5,
        "batch_size": 4096,
        "negatives_per_positive": 1,
    },
    "evidence": [
        {
            "title": "BPR: Bayesian Personalized Ranking from Implicit Feedback",
            "url": "https://arxiv.org/abs/1205.2618",
            "method_card_id": "bpr",
        }
    ],
    "needs_web_search": False,
    "parent_experiment": None,
}

BROKEN_CODE = '''"""BPR candidate with a deliberate bug for the debugger probe."""
import numpy as np

from src.experiments.contracts import CandidateOutput
from src.models.fm_core import FMRanker
from src.models.sampling import sample_bpr_pairs


def run(context, parameters):
    rng = np.random.default_rng(parameters["seed"])
    model = FMRanker(field_dimension=context.field_dimension, k=parameters["k"])
    batch = parameters["batch_size"]
    for epoch in range(parameters["epochs"]):
        pos, negs = sample_bpr_pairs(
            context.train_users, context.train_y, batch, rng
        )
        for start in range(0, len(pos), batch):
            p = pos[start : start + batch]
            n = negs[start : start + batch]
            scores_p = model.predict(context.train_x[p])
            scores_n = model.predict(context.train_x[n])
            grads = model.gradients(context.train_x[p], scores_p - scores_n)
            model.apply_gradients(grads, parameters["learning_rate"])
    validation_scores = model.predict(context.valid_x)
    test_scores = model.predict(context.test_x)
    return CandidateOutput(
        validation_scores=validation_scores,
        test_scores=test_scores,
        checkpoint_state={},
        training_trace=[],
        diagnostics={},
    )
'''

BROKEN_TESTS = '''import unittest

import numpy as np

import candidate


class SameUserPairingTests(unittest.TestCase):
    def test_pairs_come_from_the_same_user(self):
        from src.models.sampling import eligible_user_indices, sample_bpr_pairs

        users = np.array([0, 0, 0, 1, 1, 2])
        labels = np.array([1, 0, 1, 1, 0, 1])
        rng = np.random.default_rng(0)
        pos, neg = sample_bpr_pairs(users, labels, 8, rng)
        self.assertEqual(len(pos), len(neg))
        self.assertTrue((users[pos] == users[neg]).all())
        self.assertTrue((labels[pos] == 1).all())
        self.assertTrue((labels[neg] == 0).all())


if __name__ == "__main__":
    unittest.main()
'''

DEBUGGER_ERROR = (
    "Training attempt failed (unit tests). Traceback (most recent call last):\n"
    '  File "generated_experiments/probe/001/test_candidate.py", line 12, in '
    "test_pairs_come_from_the_same_user\n"
    "    pos, neg = sample_bpr_pairs(users, labels, 8, rng)\n"
    '  File "src/models/sampling.py", line 33, in sample_bpr_pairs\n'
    "    rng.integers(0, len(pool))\n"
    "AttributeError: 'int' object has no attribute 'integers'. Hint: "
    "sample_bpr_pairs(users, labels, rng, negatives_per_positive=1) — the "
    "third argument is the rng, not a batch size."
)


def build_state() -> RunState:
    return RunState(
        run_id="role_probe",
        status="running",
        started_at="2026-08-29T00:00:00+00:00",
        baseline_primary=0.6016,
        iteration_count=1,
        training_attempts=1,
        proposal_attempts=1,
        stagnant_iterations=0,
        meaningful_best=0.60147,
        best_experiment_id="official_fm_seed0",
        best_metrics=dict(FM_METRICS),
        nodes=[
            ExperimentNode(
                iteration=1,
                experiment_id="official_fm_seed0",
                hypothesis_id="official-fm-baseline",
                family="official_fm",
                action="explore",
                parameters={"k": 16, "lr": 0.001},
                status="success",
                metrics=dict(FM_METRICS),
            )
        ],
    )


def probe(role: str) -> dict[str, Any]:
    scratch = Path("/tmp/role_probes") / role
    shutil.rmtree(scratch, ignore_errors=True)
    config = json.loads((REPO_ROOT / "configs/run_kj.json").read_text(encoding="utf-8"))
    provider = OpenAIResponsesProvider(config["llm"])
    catalog = MethodCatalog.load(REPO_ROOT / "research/methods")
    roles = ResearchRoles(provider, catalog, ResearchAudit(scratch), config["llm"]["max_total_tokens"])
    state = build_state()
    decision = ResearchDecision.from_dict(BPR_PROPOSAL)

    checks: list[dict[str, Any]] = []
    started = time.monotonic()
    error: str | None = None
    summary: dict[str, Any] = {}

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    try:
        if role == "researcher":
            result = roles.research(state, iteration=1, required_family=None)
            check("family_registered", result.family in family_names(), result.family)
            check("action_valid", result.action in {"explore", "exploit", "replicate"}, result.action)
            check("evidence_present", len(result.evidence) > 0, f"{len(result.evidence)} sources")
            check("parameters_nonempty", bool(result.parameters), json.dumps(result.parameters)[:200])
            summary = {
                "hypothesis_id": result.hypothesis_id,
                "family": result.family,
                "action": result.action,
                "hypothesis": result.hypothesis[:300],
                "parameters": result.parameters,
                "evidence": [f"{e.title[:80]} ({e.url})" for e in result.evidence],
            }
        elif role == "critic_preflight":
            result = roles.critic_preflight(state, iteration=1, decision=decision)
            check("approved_is_bool", isinstance(result.approved, bool), repr(result.approved))
            check("rationale_present", bool(result.rationale.strip()), result.rationale[:200])
            check("verdict_stated", bool(result.decision.strip()), result.decision)
            summary = {
                "approved": result.approved,
                "decision": result.decision,
                "concerns": list(result.concerns),
                "next_focus": result.next_focus[:300],
            }
        elif role == "builder":
            result = roles.build(state, iteration=1, decision=decision)
            check("family_pinned", result.family == decision.family, result.family)
            check("hypothesis_pinned", result.hypothesis_id == decision.hypothesis_id, result.hypothesis_id)
            try:
                validate_source(result.code)
                check("code_passes_safety", True, f"{len(result.code.splitlines())} lines")
            except Exception as exc:
                check("code_passes_safety", False, str(exc)[:300])
            try:
                validate_source(result.tests, test_file=True)
                check("tests_pass_safety", True, f"{len(result.tests.splitlines())} lines")
            except Exception as exc:
                check("tests_pass_safety", False, str(exc)[:300])
            try:
                validate_family_contract(result.code, decision.family)
                check("family_contract", True, "trusted sampler used")
            except Exception as exc:
                check("family_contract", False, str(exc)[:300])
            check("defines_run", "def run(" in result.code, "")
            check("returns_test_scores", "test_scores" in result.code, "")
            summary = {
                "candidate_id": result.candidate_id,
                "code_lines": len(result.code.splitlines()),
                "test_lines": len(result.tests.splitlines()),
                "parameters": result.parameters,
            }
            (scratch / "candidate.py").write_text(result.code, encoding="utf-8")
            (scratch / "test_candidate.py").write_text(result.tests, encoding="utf-8")
            # Execute the generated tests exactly the way the harness does — the
            # gate that rejects candidates in the real loop (smoke evidence: both
            # iterations died here on guessed API signatures, not static checks).
            environment = {
                name: value
                for name, value in os.environ.items()
                if not name.startswith(("OPENAI_", "ANTHROPIC_"))
            }
            environment.update(
                {
                    "PYTHONPATH": str(REPO_ROOT),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "KUAIRAND_DATA_DIR": str(REPO_ROOT / "data" / "KuaiRand-Pure" / "data"),
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                }
            )
            try:
                completed = subprocess.run(
                    [sys.executable, "-m", "unittest", "-v", "test_candidate.py"],
                    cwd=scratch,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=180,
                    check=False,
                )
                tail = (completed.stdout + completed.stderr).strip().splitlines()
                check(
                    "tests_execute_and_pass",
                    completed.returncode == 0,
                    "; ".join(tail[-3:])[:300],
                )
            except subprocess.TimeoutExpired:
                check("tests_execute_and_pass", False, "tests timed out after 180s")
        elif role == "debugger":
            manifest = CandidateManifest(
                candidate_id="probe-broken-001",
                hypothesis_id=decision.hypothesis_id,
                family=decision.family,
                code=BROKEN_CODE,
                tests=BROKEN_TESTS,
                parameters=decision.parameters,
            )
            result = roles.debug(
                state, iteration=1, decision=decision, manifest=manifest,
                error=DEBUGGER_ERROR, repair_number=1,
            )
            check("preserves_hypothesis", result.preserve_hypothesis is True, repr(result.preserve_hypothesis))
            check("diagnosis_present", bool(result.diagnosis.strip()), result.diagnosis[:300])
            try:
                validate_source(result.replacement_code)
                check("replacement_passes_safety", True, f"{len(result.replacement_code.splitlines())} lines")
            except Exception as exc:
                check("replacement_passes_safety", False, str(exc)[:300])
            try:
                validate_family_contract(result.replacement_code, decision.family)
                check("replacement_family_contract", True, "")
            except Exception as exc:
                check("replacement_family_contract", False, str(exc)[:300])
            check("code_actually_changed", result.replacement_code != BROKEN_CODE, "")
            fixed = "sample_bpr_pairs(" in result.replacement_code
            check("keeps_sampler_call", fixed, "")
            summary = {
                "diagnosis": result.diagnosis[:400],
                "replacement_lines": len(result.replacement_code.splitlines()),
            }
            (scratch / "replacement_code.py").write_text(result.replacement_code, encoding="utf-8")
        elif role == "critic_postflight":
            metrics = {
                "GAUC": 0.66788,
                "nDCG@5": 0.53598,
                "primary": 0.60193,
                "rows": 124909.0,
                "users": 22377.0,
            }
            diagnostics = {
                "delta_vs_baseline": 0.00033,
                "family": "bpr",
                "note": "trusted worker metrics; candidate-reported values ignored",
            }
            result = roles.critic_postflight(state, iteration=1, decision=decision, metrics=metrics, diagnostics=diagnostics)
            check("approved_is_bool", isinstance(result.approved, bool), repr(result.approved))
            check("rationale_present", bool(result.rationale.strip()), result.rationale[:200])
            check("next_focus_present", bool(result.next_focus.strip()), result.next_focus[:200])
            summary = {"decision": result.decision, "next_focus": result.next_focus[:300]}
        else:
            raise ValueError(f"Unknown role: {role}")
    except LLMError as exc:
        error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001 - probe must report, not crash
        error = f"{type(exc).__name__}: {exc}"

    usage = state.token_usage.to_dict()
    verdict = {
        "role": role,
        "ok": error is None and all(c["ok"] for c in checks),
        "error": error,
        "latency_seconds": round(time.monotonic() - started, 1),
        "usage": usage,
        "checks": checks,
        "summary": summary,
        "scratch": str(scratch),
    }
    (scratch / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return verdict


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {
        "researcher", "critic_preflight", "builder", "debugger", "critic_postflight",
    }:
        raise SystemExit(__doc__)
    print(json.dumps(probe(sys.argv[1]), indent=2))
