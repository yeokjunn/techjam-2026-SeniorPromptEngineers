# Results Summary

## Validation Performance

| Metric | Official baseline | Best | Δ |
|---|---|---|---|
| GAUC | 0.6674 | 0.6705 | +0.0031 |
| nDCG@5 | 0.5357 | 0.5375 | +0.0018 |
| primary | 0.6016 | 0.6040 | +0.0024 |

**Validation score_dataset (mean of GAUC and nDCG@5 deltas): +0.0025**

This applies the judging formula to validation. The ranked score uses the same formula on the hidden test, which is scored once and is not computable here; the official test baseline is primary 0.5946 (GAUC 0.6610 / nDCG@5 0.5282).

- Headroom context: the attainable ceiling is primary 0.8645 (random 0.4753); this run's validation gain covers 0.9% of the baseline-to-ceiling span.

## Test Submission

- Gate status: ok
- Submission path: runs/20260830T141756693797Z_research/submission.csv

## Token Usage

| Role | Tokens |
|---|---|
| builder | 83,285 |
| critic_postflight | 28,351 |
| critic_preflight | 42,523 |
| debugger | 90,788 |
| eda_builder | 25,706 |
| eda_researcher | 22,210 |
| researcher | 61,427 |
| unknown | 0 |
| **total** | **354,290** |

## Compute

- Wall-clock: 2965 s (0.82 h)
- GPU-hours: 0.0

## Iterations

- Iterations used: 9 of 50
- Failed: 1
- Rejected before code: 2

## Convergence

- Stop reason: converged
- Converged (official rule): True

## Interventions

- Count: 0
- Reasons: none recorded

