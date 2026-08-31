"""Prompt-sized API contracts for code-generating roles.

These cards ground Builder and Debugger in the repository's actual trusted
runtime APIs. They are intentionally concise: Researcher decides what to try;
these contracts tell implementation roles how the selected components behave.
"""

from __future__ import annotations

from .families import required_call_groups


FM_RANKER_CONTRACT = """RUNTIME API CARD: src.models.fm_core.FMRanker
- Instantiate: FMRanker(dimension, embedding_dim=16, learning_rate=lr, l2=1e-6, seed=seed).
- logits(features) returns (scores, embeddings, summed); use logits(...)[0] for scores.
- gradients(features, score_gradients) expects dLoss/dScore, not binary labels, and returns (grad_v, grad_w, grad_b).
- apply_gradients(grad_v, grad_w, grad_b=0.0) takes three unpacked gradient values; never call apply_gradients(grads, lr).
- predict(features) returns one score per row. state_dict()/load_state_dict() preserve checkpoints.
- BPR pattern: gradient = (sigmoid(pos_score - neg_score) - 1.0) / batch_size; call gradients(pos_x, gradient) and gradients(neg_x, -gradient), then apply grad_v_p + grad_v_n and grad_w_p + grad_w_n."""


SAMPLING_CONTRACTS = {
    "sample_bpr_pairs": """RUNTIME API CARD: src.models.sampling.sample_bpr_pairs
- Call sample_bpr_pairs(users, labels, rng, negatives_per_positive).
- Returns (positive_rows, negative_rows) arrays with matched length.
- Each pair must be same-user, positive label for positive_rows, non-positive label for negative_rows.
- negatives_per_positive may duplicate positive row indices; do not assert len(negatives) == positives_count.""",
    "sample_softmax_groups": """RUNTIME API CARD: src.models.sampling.sample_softmax_groups
- Call sample_softmax_groups(users, labels, rng, negatives_per_group).
- Use same-user candidate groups only.
- Optimize one positive against its same-user negatives; do not use cross-user negatives.""",
}


FEATURE_CONTRACTS = {
    "build_features": """RUNTIME API CARD: src.models.features.build_features
- Call build_features(rows, spec); rows must be the exact split array passed by context.
- For history features, pass explicit split and field_offset in spec.
- You can compose or transform the trusted feature columns inside candidate.py, but do not read raw CSVs, filesystem paths, or dataset files directly.
- Use feature_dimension(spec) for the added index-space width, not the number of generated columns.
- Concatenate returned int feature columns to the original id fields; preserve row order.""",
    "build_aux_labels": """RUNTIME API CARD: src.models.features.build_aux_labels
- Call build_aux_labels(rows, spec) for train-time multi-task auxiliary targets (is_click, is_like, is_follow, is_comment, is_forward, play_time).
- You can combine multiple auxiliary heads or create weighted multi-task loss terms from build_aux_labels only.
- Do not derive validation/test labels, read hidden labels, or read raw CSVs/dataset files directly.
- Keep the primary CandidateOutput validation scores for long_view ranking.""",
}


CANDIDATE_OUTPUT_CONTRACT = """RUNTIME API CARD: src.experiments.contracts.CandidateOutput
- Return CandidateOutput(validation_scores=..., checkpoint_state=..., training_trace=..., diagnostics=..., test_scores=..., random_validation_scores=...).
- validation_scores length must equal len(context.valid_x), finite, same row order.
- test_scores must be finite scores for context.test_x in the same row order when context.test_x is not None; otherwise None.
- random_validation_scores should be model scores for context.random_valid_x when present. This is a diagnostic random-exposure split; never use it for training, early stopping, checkpoint selection, or hyperparameter selection.
- checkpoint_state must be a dict of numpy arrays from the selected model state."""


TEST_CONTRACT = """RUNTIME API CARD: generated test_candidate.py
- Prefer real trusted components on tiny synthetic arrays over mocks.
- If a mock is necessary, its public method signatures must exactly match the real API.
- In particular, fake FMRanker.apply_gradients must accept (grad_v, grad_w, grad_b=0.0), not (grads, lr)."""


def runtime_contract_prompt(family: str) -> str:
    """Return implementation contracts relevant to a selected family."""
    cards = [FM_RANKER_CONTRACT, CANDIDATE_OUTPUT_CONTRACT, TEST_CONTRACT]
    seen: set[str] = set()
    for group in required_call_groups(family):
        for call in group:
            if call in seen:
                continue
            seen.add(call)
            card = SAMPLING_CONTRACTS.get(call) or FEATURE_CONTRACTS.get(call)
            if card:
                cards.append(card)
    return "\n\n".join(cards)
