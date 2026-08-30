# Dataset Profile

## Splits

| Split | Rows | Users | Videos |
|---|---|---|---|
| train | 1,141,112 | 26,210 | 7,538 |
| valid | 124,909 | 22,377 | 5,951 |
| test | 170,588 | — | — |

## Label Rates

| Label | Train Rate | Valid Rate |
|---|---|---|
| long_view | 33.6620 % | 31.3284 % |
| is_click | 46.3447 % | 44.3827 % |
| is_like | 1.8677 % | 1.7973 % |
| is_follow | 0.1007 % | 0.1305 % |
| is_comment | 0.2568 % | 0.2330 % |
| is_forward | 0.0996 % | 0.0777 % |
| is_hate | 0.0421 % | 0.0624 % |
| is_profile_enter | 2.5391 % | 1.9454 % |

## Tab Breakdown (train)

| Tab | Rows | Share | Click Rate | Long View Rate |
|---|---|---|---|---|
| 1 | 834,876 | 73.16 % | 52.97 % | 38.61 % |
| 0 | 150,013 | 13.15 % | 9.20 % | 4.22 % |
| 4 | 75,524 | 6.62 % | 61.96 % | 48.93 % |
| 2 | 39,291 | 3.44 % | 48.00 % | 38.05 % |
| 6 | 29,671 | 2.60 % | 18.08 % | 8.70 % |
| 3 | 3,574 | 0.31 % | 2.24 % | 0.42 % |
| 5 | 3,402 | 0.30 % | 28.57 % | 16.99 % |
| 8 | 2,551 | 0.22 % | 4.16 % | 1.76 % |
| 12 | 834 | 0.07 % | 14.27 % | 9.59 % |
| 11 | 417 | 0.04 % | 0.00 % | 11.51 % |
| 7 | 333 | 0.03 % | 40.84 % | 21.32 % |
| 13 | 283 | 0.02 % | 22.61 % | 18.02 % |
| 9 | 252 | 0.02 % | 99.60 % | 16.67 % |
| 10 | 80 | 0.01 % | 100.00 % | 61.25 % |
| 14 | 11 | 0.00 % | 0.00 % | 0.00 % |

## Rows per User (train + valid)

| Percentile | Count |
|---|---|
| p25 | 15 |
| p50 | 34 |
| p75 | 65 |
| p90 | 105 |
| p95 | 137 |
| p99 | 219 |
| max | 853 |

## Data Quality

- 239 videos with zero duration (24,076 train rows affected)
- 15,609 exact duplicate rows in the training-period log
- Sentinel: is_live_streamer = -124 on 21,127 of 27,285 user rows
- UNKNOWN values: user_active_degree (6), video_type (1), upload_type (80)
- Constant columns: is_lowactive_period, visible_status, is_rand
- hourmin encodes hour times 100 in UTC+8
- The training-period log begins on 20220409, not 20220408

## Feature Coverage

- Video features: 100.000 % of train+valid video IDs found in the feature table
- User features: 100.000 % of train+valid user IDs found in the profile table

## Leakage Flag

The video statistics table is a period aggregate over full-platform traffic.
Implied total shows: 12.83 B vs 2.62 M logged
exposures (ratio approximately 4,892 to one). Observation periods per video
range 45 to 181, exceeding the interaction log span.
Legal for use, but any feature derived from it must be caveated as leaky.

## Metric Conventions

- Task: within-user ranking (each user's impressions ranked against each other)
- nDCG@5: sorts by score with a stable sort (ties fall back to row order)
- AUC: averages ranks over ties (Mann-Whitney U)
- GAUC: per-user AUC, only users with 0 < positives < impressions,
  weighted by positive count
- Zero-positive users: nDCG recorded as 0.0 and included in the average
- Primary score = mean(GAUC, nDCG@5)

## Measured Dead Ends

- Adding all 13 static feature fields: primary 0.5940 vs 0.5950 for the default 5
- Embedding dimension k = 8 / 16 / 32: primary 0.5895 / 0.5902 / 0.5887 (flat)
- The bottleneck is not features or capacity
- First-order user-only terms contribute exactly 0 to within-user ranking
  (any term constant within a user does not change intra-group ordering)
