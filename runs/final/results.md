# Results Summary

## Validation Performance

| Metric | Official baseline | Best | Δ |
|---|---|---|---|
| GAUC | 0.6674 | 0.6707 | +0.0033 |
| nDCG@5 | 0.5357 | 0.5382 | +0.0025 |
| primary | 0.6016 | 0.6045 | +0.0029 |

**Validation score_dataset (mean of GAUC and nDCG@5 deltas): +0.0029**

This applies the judging formula to validation. The ranked score uses the same formula on the hidden test, which is scored once and is not computable here; the official test baseline is primary 0.5946 (GAUC 0.6610 / nDCG@5 0.5282).

- Headroom context: the attainable ceiling is primary 0.8645 (random 0.4753); this run's validation gain covers 1.1% of the baseline-to-ceiling span.

## Test Submission

- Gate status: ok
- Submission path: runs/20260831T141845874517Z_research/submission.csv

## Token Usage

| Role | Tokens |
|---|---|
| builder | 95,606 |
| critic_postflight | 60,313 |
| critic_preflight | 61,515 |
| debugger | 86,665 |
| eda_builder | 22,545 |
| eda_researcher | 20,392 |
| researcher | 99,174 |
| unknown | 0 |
| **total** | **446,210** |

## Compute

- Wall-clock: 1772 s (0.49 h)
- GPU-hours: 0.0

## Iterations

- Iterations used: 6 of 50
- Failed: 0
- Rejected before code: 0

## Convergence

- Stop reason: converged
- Converged (official rule): True

## Interventions

- Count: 0
- Reasons: none recorded

