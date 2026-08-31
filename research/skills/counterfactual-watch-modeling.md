# Role Skill: Counterfactual Watch Modeling

Consider Counterfactual Watch Model (CWM) experiments when duration bias may distort video
interest. CWM treats observed watch time as right-censored by video duration: a play ending before
the video ends is an observed stopping time, while a play reaching the duration supplies only a
lower bound on the user's latent desired watch time. The reference method is "Counteracting
Duration Bias in Video Recommendation via Counterfactual Watch Time" (KDD 2024):
https://github.com/hyz20/CWM

Use CWM as a bounded research direction, not as a replacement task. The immutable target remains
`long_view`; validation GAUC and nDCG@5 remain the selection metrics. Prefer, in order:

- a `multi_task` probe adding duration-aware censored-watch supervision at low auxiliary weight;
- an ablation comparing ordinary scaled play time with a censored-duration correction while the
  FM backbone, ranking loss, seed, and training budget stay fixed;
- validation-selected score blending only if the registered runtime explicitly supports both
  component scores and records the blend weight.

Before proposing a faithful CWM probe, verify that trusted train-only code supplies aligned
`play_time_ms` and `duration_ms` (or an equivalent censoring indicator). The current
`build_aux_labels` play-time head is log-scaled play time alone; do not describe it as CWM and do
not reconstruct duration or read raw logs from generated `candidate.py`. If the registered family
or approved parameter grid cannot express the probe, identify the missing trusted capability as a
future method extension instead of inventing parameters, imports, or file access.

For an admissible CWM experiment, state:

- the censoring rule, including treatment of replay watch time above duration;
- the counterfactual-to-interest transform and numerically safe parameter bounds;
- the auxiliary weight and exactly one primary contrast against the parent;
- a cheap runtime/epoch budget and the expected duration-bias mechanism;
- diagnostics split by duration bucket, alongside the official aggregate metrics.

Keep the implementation numerically safe. Compute censored and uncensored terms only when their
subsets are non-empty, clamp probabilities before inverse-logit or logarithms, use finite epsilons,
and reject NaN/Inf scores. Treat CWM hyperparameters such as the cost coefficient and observation
noise as validation-selected parameters within an explicitly approved bounded search space.

Do not copy the reference repository's preprocessing, ground-truth generation, data splits,
evaluation code, or test-scoring workflow. Never use validation/test auxiliary labels, future
interactions, or hidden-test outcomes. A better CWM surrogate loss is useful only if the normal
trusted validation path improves within-user `long_view` ranking.
