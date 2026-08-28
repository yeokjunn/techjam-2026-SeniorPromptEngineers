from __future__ import annotations

from .types import ExperimentOutcome, ExperimentSpec


def reflect(
    spec: ExperimentSpec,
    outcome: ExperimentOutcome,
    previous_best: float | None,
    official_baseline: float,
) -> dict:
    if outcome.status != "success" or not outcome.metrics:
        return {
            "decision": "recover_and_continue",
            "summary": f"{spec.experiment_id} failed; retain the previous best state.",
            "next_focus": "Inspect the captured error and choose a lower-risk experiment.",
        }

    score = outcome.metrics["primary"]
    improved = previous_best is None or score > previous_best
    baseline_delta = score - official_baseline
    return {
        "decision": "promote_to_best" if improved else "retain_previous_best",
        "summary": (
            f"Validation primary={score:.4f}; delta versus the official validation "
            f"baseline={baseline_delta:+.4f}."
        ),
        "next_focus": (
            "Establish the next baseline rung."
            if spec.kind != "fm"
            else "After baseline reproduction, test a ranking-aligned pairwise objective."
        ),
    }

