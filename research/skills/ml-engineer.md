# Role Skill: ML Engineer (Numerical Optimization & Production Candidate Implementation)

Expert guidelines for implementing robust, vectorized candidate models in the KuaiRand execution sandbox:

1. Sandbox Isolation & Allowed Libraries:
   - ONLY numpy, math, collections, time, and src.models.fm_core.FMRanker are permitted.
   - Do NOT import pandas, scipy, sklearn, torch, or lightgbm. External imports trigger an immediate SafetyViolation and terminate the run.

2. Numerical Stability & Vectorization:
   - Vectorize all operations across batch rows using NumPy; avoid Python for-loops over impressions.
   - For sigmoid/BPR loss: clip score difference d to [-30.0, 30.0] before computing sigmoid to prevent overflow/NaN.
   - For group softmax: subtract np.max(logits, axis=-1, keepdims=True) before exponentiating (log-sum-exp trick).
   - Verify that all outputs (validation_scores, test_scores) contain only finite float numbers (no NaN or Inf).

3. FMRanker API & Checkpoint Discipline:
   - Do not re-implement the FM with dense matrices (~40,000 fields overflow memory). Use model.logits(features).
   - Feed score gradients d(loss)/d(score) directly to model.gradients(features, score_gradients) and call model.apply_gradients().
   - model.state_dict() returns detached copies. Always restore the best epoch weights via model.load_state_dict(best_state).

4. Candidate Contract & Unittest Requirements:
   - Return CandidateOutput with exact fields: validation_scores, checkpoint_state, training_trace, diagnostics, test_scores.
   - test_candidate.py MUST define test classes subclassing unittest.TestCase with def test_* methods (bare pytest functions are ignored by python -m unittest).
