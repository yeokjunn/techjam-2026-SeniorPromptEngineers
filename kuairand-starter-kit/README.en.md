# KuaiRand-Pure Starter Kit

> English translation of `README.md`. The Chinese original is authoritative; this
> is a faithful rendering for convenience.

## Dependencies

Python 3.9+ and numpy. **Nothing else.** No torch, no pandas, no sklearn.

## Data

Download from https://kuairand.com (direct Zenodo link, no registration needed):

```bash
# run inside the Starter Kit directory; unpacking yields ./KuaiRand-Pure/
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

## Running

```bash
python3 baseline.py --model fm
```

`--data_dir` defaults to `./KuaiRand-Pure/data`; specify it explicitly if your data
lives elsewhere.

`--model` accepts `fm` (the official baseline) / `pop` (trivial baseline) /
`random` (lower bound, for sanity-checking your evaluation code).
FM takes about 40 seconds end to end (CPU, single core).

## Task definition (conventions are pinned — do not change them)

| | |
|---|---|
| Task | **Within-user ranking** — each user's impressions in the evaluation set are ranked only against each other; no full-catalog retrieval |
| Relevance label | `long_view` (native column, 0/1) |
| Metrics | `GAUC`, `nDCG@5`; **primary score = the mean of the two** |
| Splits | train `20220408–20220421` / valid `20220422–20220428` / test `20220429–20220508` |
| Zero-positive users | nDCG recorded as 0.0 and **included** in the average; GAUC counts only users with `0 < positives < impressions`, weighted by positive count |
| nDCG gain | `2^rel − 1` (equivalent to identity under binary labels) |

The implementation is in `evaluate.py`; all conventions are documented in the
file's header comment.

## Baseline ladder

Scores on the test set. **The row you have to beat is FM.**

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (lower bound, sanity check) | 0.4996 | 0.4511 | 0.4753 |
| item popularity (trivial) | 0.6308 | 0.5121 | 0.5715 |
| **FM (official baseline)** | **0.6610** | **0.5282** | **0.5946** |

### ⚠️ The metric's real range: nDCG@5 tops out at 0.729, not 1.0

Of the 23,875 users in the test set:

| | Share | Effect on the metric |
|---|---|---|
| All-negative users (none of their impressions is a `long_view`) | **27.1%** | nDCG is permanently **0** — no model can fix this; excluded from GAUC |
| All-positive users | **9.2%** | nDCG is permanently **1**; excluded from GAUC |
| Discriminative users | **63.7%** | the actual sample GAUC is computed over |

So even using the true labels as prediction scores (the oracle, i.e. a perfect
ranking) only gets you:

| | random | FM baseline | **oracle ceiling** | Range already captured by FM |
|---|---|---|---|---|
| GAUC | 0.4996 | 0.6610 | **1.0000** | 32.3% |
| nDCG@5 | 0.4511 | 0.5282 | **0.7289** | 27.8% |
| **primary** | 0.4753 | **0.5946** | **0.8645** | **30.7%** |

**Measure your progress against the oracle, not against 1.0.** Seeing 0.5946 and
concluding "we're still a long way from a perfect 1.0" is a misreading — the
baseline has already captured 30% of the usable range, and the remaining headroom
is 0.27, not 0.41.

FM's standard deviation across 5 random seeds is **0.0008** on every metric. The
convergence criterion is set from that: **ε = 0.002 (≈2.5σ), N = 3** — a run is
declared converged once the validation primary score has improved by no more than
0.002 for 3 consecutive iterations.

> Sanity check: if your evaluation code doesn't produce primary ≈ 0.475 (±0.001)
> on `--model random`, your harness is broken — fix that first.

## Submission format

CSV with a header row, one line per row of the evaluation set:

```
row_id,user_id,video_id,score
0,0,7531,-3.34176
1,0,4214,-1.4955
...
```

| Field | Description |
|---|---|
| `row_id` | Consecutive and increasing from 0, matching the row order of `data.load()[split]` (deterministic: `log_standard_4_08_to_4_21_pure.csv` is read first, then `log_standard_4_22_to_5_08_pure.csv`, and after filtering by date the original file order is preserved) |
| `user_id` / `video_id` | Redundant fields, used only to verify alignment |
| `score` | Your model's score for that row; any real number, only relative magnitude matters; NaN / Inf are not allowed |

> **Why `row_id` is mandatory:** `(user_id, video_id)` is **not unique** in the
> evaluation set — the test set contains 3.06% duplicate pairs, repeated up to 12
> times. So it cannot serve as a primary key.

Generating and validating:

```bash
python3 submit.py --make  --split test  submission.csv    # generate an example submission using the official FM baseline
python3 submit.py --check --split test  submission.csv    # validate format and alignment
python3 submit.py --score --split valid submission.csv    # validate and score (available locally for valid)
```

`--check` will reject: a wrong header, a row-count mismatch, gaps in `row_id`,
`user_id`/`video_id` misaligned against the evaluation set, and `score` values
that are non-numeric or NaN/Inf. **Please run `--check` yourself before
submitting.**

## Where to start making changes

The ordering below is **empirically measured, not guessed.** Dead ends the
organizing committee has already tried are marked explicitly — don't walk into
them again.

### Already measured: these two yield nothing, don't waste iterations on them

| What was tried | Result |
|---|---|
| **Adding static features** — wiring in all 13 of CWM's feature fields (+`music_id`/`video_type`/`upload_type` + 6 coarse user-side buckets) | primary **0.5940** vs **0.5950** for the 5 fields — no difference within noise, if anything slightly worse |
| **Adding model capacity** — embedding dimension k = 8 / 16 / 32 | 0.5895 / 0.5902 / 0.5887 — essentially flat |

The reason: the `user_id × video_id` cross already absorbs most of the learnable
signal. Coarse buckets like `follow_user_num_range` are redundant once you have
`user_id`; and 1.14M rows can't support greater capacity anyway. **The bottleneck
is not features or capacity.**

⚠️ Also note: **first-order terms on purely user-side features contribute exactly
0 to the score.** Because ranking happens *within* a user, any term that is
constant within a user does not change the intra-group ordering (measured:
`item_pop × user bias` and plain `item_pop` score identically to the last digit).
User-side features can only take effect through **cross terms with item-side
features.**

### Unexplored: the headroom should be here

Ordered by our estimate of likelihood (**the committee has not tested these — they
are left for you**):

1. **Change the loss function.** It is currently pointwise logloss, but the metrics
   (GAUC / nDCG) are **ranking metrics**. Switching to pairwise (BPR) or listwise
   (softmax over that user's impressions) aligns the objective with the evaluation
   convention — **we believe this is the most likely to work.**
2. **User behaviour sequences.** The current features make **no use of behavioural
   sequences at all.** Each user has hundreds to thousands of interactions in
   train; interest modelling of the DIN / SIM variety is a completely blank
   direction.
3. **Multi-objective.** The logs also contain `is_click`, `is_like`, `is_follow`,
   `is_comment`, `is_forward`, `play_time_ms`, which can serve as auxiliary tasks
   supporting the primary `long_view` task.
4. **Modelling watch time.** This is precisely [CWM](https://github.com/hyz20/CWM)'s
   contribution: it treats watch time as a **censored regression** (when a video
   plays to completion the true watch time is truncated, so a one-sided loss is
   used instead of squared error). A direction with real research depth.
5. **Change the model.** DeepFM / DCN / xDeepFM. Given that capacity is measurably
   not the bottleneck, **rank this below items 1–4.**
6. **Temporal features and distribution drift.** `hourmin`, `date`, and the drift
   between train and test.
7. **Unbiased validation (advanced).** `log_random_4_22_to_5_08_pure.csv` is a
   random-exposure log (1.18M rows) and can serve as an additional unbiased
   validation set, to check whether a model is only overfitting biased traffic.

## Using your own model (including CWM)

`evaluate.py` is completely decoupled from the model — it only needs three
equal-length arrays:

```python
from evaluate import evaluate
print(evaluate(user_ids, labels, scores))   # scores can come from any model
```

- `user_ids`: the user_id of each row in the evaluation set
- `labels`: that row's `long_view` (0/1)
- `scores`: your model's score for that row (any real number, only relative
  magnitude is used)

So you can skip `baseline.py` entirely and use PyTorch, LightGBM, or
[CWM](https://github.com/hyz20/CWM)'s xDeepFM instead — all that matters is
handing the final `scores` to `evaluate()`. **The scoring convention is determined
solely by `evaluate.py`.**

> Caveats for using CWM: it depends on `torch==1.6.0` (a 2020 release, probably
> uninstallable on a modern GPU), its loss optimizes counterfactual watch time,
> and its evaluation label is a self-reconstructed `long_view2`. It is the
> research code for a duration-debiasing paper — useful as an **advanced
> reference**, not recommended as a starting point.

## Files

| | |
|---|---|
| `evaluate.py` | Metric implementations + all convention decisions. **Do not modify.** |
| `data.py` | Data loading, official splits, feature encoding. Add features here. |
| `baseline.py` | The three baselines. FM is the one to beat. |
| `baseline_scores.json` | Officially published scores + seed variance + convergence parameters. |
| `submit.py` | Generate / validate a submission file. |
| `ablation_features.py` | Feature ablation experiment; reproduces the "adding features yields nothing" numbers. |
