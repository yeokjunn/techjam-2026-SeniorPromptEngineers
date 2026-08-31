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

model = FMRanker(
    context.field_dimension + feature_dimension(spec),
    embedding_dim=int(parameters["k"]),
    learning_rate=float(parameters["learning_rate"]),
    l2=float(parameters.get("l2", 1e-6)),
    seed=int(parameters["seed"]),
)
```

Every hyperparameter comes from `parameters` — never hard-code one. `l2` is read with `.get`
because this family's grid does not name that axis, so no `l2` key is emitted for it and a
subscript would be a KeyError; 1e-6 is `FMRanker`'s own default.

Row width goes from 5 to `5 + g`. Train rows are scored from strictly earlier days; valid and
test rows are scored from all of train (see the time-respecting rule below), so the split label
changes the statistics, not just the length check.

Seven groups are available, each occupying 9 slots (8 quantile buckets computed on
train values only, plus a reserved slot for "no history / unknown"):

| group | value |
|---|---|
| `user_rate` | smoothed long_view rate of the user |
| `user_author` | smoothed rate of (user, author) — the DIN affinity signal |
| `user_tab` | smoothed rate of (user, tab) |
| `recency` | days since the user's last long_view, capped at 14 |
| `video_age` | days between the row's date and the video's upload date |
| `tab_cross` | smoothed rate of the (tab, duration-bucket) cell |
| `video_rate` | smoothed long_view rate of the video — the one video-side rate |

`video_rate` is the only video-side rate, and the strongest single signal measured here: the
official five fields carry a video's *identity* but no measure of how that video actually
performs, and ordering by this rate alone scores primary 0.5807 against 0.4827 for a random
ordering. It is built from train rows only — the kit's ready-made
`video_features_statistic_pure.csv` is not usable, because its counting window spans the test
dates — so a video with no train history lands in the unknown slot rather than in a bucket
derived from its own future rows.

The new columns are ordinary FM fields: they interact with `user_id`, `video_id`,
`author_id`, `tab` and `dur_bucket` through the existing second-order term, and
`fm_core.py` needs no change: it sums the gather over whatever row width you pass (verified
field-count agnostic at 5 and 11 fields).

Smoothing is `(positives + m·prior) / (count + m)` with `prior` the global train
long_view rate and `m = spec["smoothing"]`. All tables are built from train rows
only, and `spec["scheme"]` controls how a train row sees them. **`"prior_days"` — counting only
strictly earlier days — is now the only value this family's grid names**, so read `scheme` from
`parameters` like any other knob and expect exactly that one value; proposing another is
rejected by the sanitiser. The leave-one-out contrast (full tables minus the row's own
contribution) still exists in `src/models/features.py` for direct callers, but it was retired
from the search space rather than left as a dead axis to burn iterations on: CatBoost
(Prokhorenkova et al., NeurIPS 2018, https://arxiv.org/abs/1706.09516) measures exactly these
target-statistic schemes on click prediction and reports a relative logloss penalty of **+2.7 %
for leave-one-out against +1.5 % for a holdout scheme** (and +13 % for the greedy in-fold one).
`prior_days` is this repo's time-respecting holdout, so it is the better estimator on measured
grounds, not just on the leakage argument below.

Keep `k == 16`: this family's grid names no other value, by design. The *field set* is what
is under test, and a second free axis would make it unattributable — and this is already the
family closest to the experiment timeout. The kit's flat k-sweep (k = 8/16/32 gives
0.5895/0.5902/0.5887, `kuairand-starter-kit/README.en.md:133-139`) was measured under **pointwise
logloss**; the ranking-loss families reopen the axis on that basis, this one does not.

## Safe initial search space

- Either trusted same-user sampler; `build_features` is mandatory
- FM embedding dimension fixed at 16 — this family's grid names no other value (see above)
- Learning rate: 0.0003, 0.0005, or 0.001
- Batch size: 2048 or 4096; one or two negatives per positive
- Smoothing `m`: 5.0, 20.0, or 100.0
- Scheme: `prior_days` — the only value in this family's grid (see above)
- Any subset of the seven groups via `use_<group>`
- Epochs at most 20 — seven extra fields roughly double the gather/scatter cost
  (one FM epoch is about 12 s) against a 900 s experiment timeout

## Known failure modes

- User-side-only features are constant within a user. Because GAUC ranks within a
  user, they contribute exactly nothing through the first-order term and act only
  through crosses with video-side fields
  (`kuairand-starter-kit/README.en.md:141-148`). Enabling `user_rate` alone is
  expected to do very little. `video_rate` is the exception among the rate groups: it is keyed
  by the video, so it varies within a user and reorders that user's list on its own.
- `leave_one_out` let a train row see the same user's *later* days. The feature
  looked stronger during training and the gain did not transfer to validation,
  because every validation row is scored from strictly earlier data — and CatBoost
  measures it as the worse estimator on the same class of task (above). It is no
  longer in the grid; `prior_days` is the only proposable scheme.
- Low counts without smoothing memorise the label: a key seen once with a single
  positive has rate 1.0. That is what `m` is for; do not set it near zero.
- Unknown keys belong in the reserved slot, not the prior bucket. A key with no
  history and a key whose history happens to sit at the global rate are different
  statements, and collapsing them costs the model a distinction it can learn.
- More fields is not more capacity. `k` stays 16 here because the grid offers nothing else;
  capacity under a ranking loss is `bpr`/`group_softmax`'s axis to search, not this one's.
- Roughly 9.5% of train rows fall in the unknown slot under `prior_days` — the
  first day has no history at all. This is expected, not a bug.
