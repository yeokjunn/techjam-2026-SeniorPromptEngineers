# User-History Features

## Primary source

- Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan,
  Junqi Jin, Han Li, Kun Gai, "Deep Interest Network for Click-Through Rate
  Prediction," KDD 2018. https://arxiv.org/abs/1706.06978
- The starter kit's own ranking of untested directions puts user-behaviour
  sequences second and calls it "a completely blank direction"
  (`kuairand-starter-kit/README.en.md:150-160`).

## Hypothesis

The official FM sees five id fields per row: user, video, author, tab and a
duration bucket. It therefore has no notion of what a user has already done —
whether they usually finish videos, whether they have an affinity for this
author, or whether they were active yesterday. DIN's central claim is that a
user's history is the strongest available signal for their next interaction.

Encoding that history as extra FM fields should help most where the id fields are
weakest: users and videos with few interactions, where the embeddings are barely
trained but the aggregate rate is already informative.

## Objective

The loss is unchanged — this family may use either trusted sampler. What changes
is the field set. `src.models.features.build_features(rows, spec)` returns one
`int32` column per enabled group, already offset by `spec["field_offset"]`:

Call it **once per split**, and `spec["split"]` must name the split whose rows you are
passing. The builder cross-checks the row count against the trusted split and raises
`ValueError: Split 'train' has 1141112 trusted rows but 124909 were passed` if they disagree —
that is the error you get from scoring validation rows with `split="train"` still set.

```python
base = {"field_offset": context.field_dimension, **parameters}

train_x = np.concatenate(
    [context.train_x, build_features(context.train_x, {**base, "split": "train"})], axis=1
)
valid_x = np.concatenate(
    [context.valid_x, build_features(context.valid_x, {**base, "split": "valid"})], axis=1
)
# only when context.test_x is not None:
test_x = np.concatenate(
    [context.test_x, build_features(context.test_x, {**base, "split": "test"})], axis=1
)

model = FMRanker(context.field_dimension + feature_dimension(spec), embedding_dim=16, ...)
```

Row width goes from 5 to `5 + g`. Train rows are scored from strictly earlier days; valid and
test rows are scored from all of train (see the time-respecting rule below), so the split label
changes the statistics, not just the length check.

Six groups are available, each occupying 9 slots (8 quantile buckets computed on
train values only, plus a reserved slot for "no history / unknown"):

| group | value |
|---|---|
| `user_rate` | smoothed long_view rate of the user |
| `user_author` | smoothed rate of (user, author) — the DIN affinity signal |
| `user_tab` | smoothed rate of (user, tab) |
| `recency` | days since the user's last long_view, capped at 14 |
| `video_age` | days between the row's date and the video's upload date |
| `tab_cross` | smoothed rate of the (tab, duration-bucket) cell |

The new columns are ordinary FM fields: they interact with `user_id`, `video_id`,
`author_id`, `tab` and `dur_bucket` through the existing second-order term, and
`fm_core.py` needs no change (verified field-count agnostic at 5 and 11 fields).

Smoothing is `(positives + m·prior) / (count + m)` with `prior` the global train
long_view rate and `m = spec["smoothing"]`. All tables are built from train rows
only, and `spec["scheme"]` controls how a train row sees them: `"prior_days"`
(the default) counts only strictly earlier days, `"leave_one_out"` uses the full
tables minus the row's own contribution.

Keep `k == 16`. Capacity is a measured dead end: k=8/16/32 gives
0.5895/0.5902/0.5887 (`kuairand-starter-kit/README.en.md:133-139`).

## Safe initial search space

- Either trusted same-user sampler; `build_features` is mandatory
- FM embedding dimension fixed at 16 for attribution
- Learning rate: 0.0003, 0.0005, or 0.001
- Batch size: 2048 or 4096; one or two negatives per positive
- Smoothing `m`: 5.0, 20.0, or 100.0
- Scheme: `prior_days` (preferred) or `leave_one_out` (the contrast)
- Any subset of the six groups via `use_<group>`
- Epochs at most 20 — six extra fields roughly double the gather/scatter cost
  (one FM epoch is about 12 s) against a 900 s experiment timeout

## Known failure modes

- User-side-only features are constant within a user. Because GAUC ranks within a
  user, they contribute exactly nothing through the first-order term and act only
  through crosses with video-side fields
  (`kuairand-starter-kit/README.en.md:141-148`). Enabling `user_rate` alone is
  expected to do very little.
- `leave_one_out` lets a train row see the same user's *later* days. The feature
  looks stronger during training and the gain does not transfer to validation,
  because every validation row is scored from strictly earlier data. Prefer
  `prior_days`; use `leave_one_out` only to measure the gap.
- Low counts without smoothing memorise the label: a key seen once with a single
  positive has rate 1.0. That is what `m` is for; do not set it near zero.
- Unknown keys belong in the reserved slot, not the prior bucket. A key with no
  history and a key whose history happens to sit at the global rate are different
  statements, and collapsing them costs the model a distinction it can learn.
- More fields is not more capacity. `k` stays 16; raising it is already measured
  as a regression.
- Roughly 9.5% of train rows fall in the unknown slot under `prior_days` — the
  first day has no history at all. This is expected, not a bug.
