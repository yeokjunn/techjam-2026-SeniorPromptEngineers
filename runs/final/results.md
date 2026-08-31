# Results Summary

## Validation Performance

| Metric | Official baseline | Best | Δ |
|---|---|---|---|
| GAUC | 0.6674 | 0.6710 | +0.0036 |
| nDCG@5 | 0.5357 | 0.5374 | +0.0017 |
| primary | 0.6016 | 0.6042 | +0.0026 |

**Validation score_dataset (mean of GAUC and nDCG@5 deltas): +0.0026**

This applies the judging formula to validation. The ranked score uses the same formula on the hidden test, which is scored once and is not computable here; the official test baseline is primary 0.5946 (GAUC 0.6610 / nDCG@5 0.5282).

- Headroom context: the attainable ceiling is primary 0.8645 (random 0.4753); this run's validation gain covers 1.0% of the baseline-to-ceiling span.

## Test Submission

- Gate status: ok
- Submission path: runs/20260831T115602777469Z_research/submission.csv

## Token Usage

| Role | Tokens |
|---|---|
| builder | 92,994 |
| critic_postflight | 35,939 |
| critic_preflight | 110,991 |
| debugger | 128,157 |
| eda_builder | 23,006 |
| eda_researcher | 17,868 |
| preflight_adjudicator | 18,578 |
| researcher | 151,498 |
| unknown | 0 |
| **total** | **579,031** |

## Compute

- Wall-clock: 1906 s (0.53 h)
- GPU-hours: 0.0

## Iterations

- Iterations used: 7 of 50
- Failed: 3
- Rejected before code: 0

## Convergence

- Stop reason: converged
- Converged (official rule): True

## Interventions

- Count: 0
- Reasons: none recorded

