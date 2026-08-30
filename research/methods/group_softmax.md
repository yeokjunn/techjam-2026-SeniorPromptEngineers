# Same-user Group Softmax

## Primary sources

- Zhe Cao, Tao Qin, Tie-Yan Liu, Ming-Feng Tsai, Hang Li, "Learning to Rank:
  From Pairwise Approach to Listwise Approach," ICML 2007.
  https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/
- Sébastien Jean et al., "On Using Very Large Target Vocabulary for Neural
  Machine Translation," ACL 2015 (sampled normalization reference).
  https://aclanthology.org/P15-1001/

## Hypothesis

Ranking one positive against several negatives from the same user provides a
closer approximation to the evaluated within-user list than independent
pointwise examples or a single BPR pair.

## Objective

For one positive score and `K` same-user negative scores:

```text
logits = [positive, negative_1, ..., negative_K] / temperature
loss = -log_softmax(logits)[0]
```

The score gradient is `softmax(logits) - one_hot(positive)`, divided by the
temperature.

`src.models.sampling.sample_softmax_groups` is mandatory, and it returns **row indices that are
already grouped**:

```text
positives, negatives = sample_softmax_groups(train_users, train_y, rng, negatives_per_group)

positives   int64, shape (n_groups,)                       one positive row per group
negatives   int64, shape (n_groups, negatives_per_group)   ALREADY 2-D — do not reshape it
```

`negatives` is returned pre-shaped by `np.stack`, so calling `.reshape(-1, K)` on it is both
unnecessary and wrong: the flat size is `n_groups * negatives_per_group`, which only divides
cleanly by coincidence and raises `ValueError: cannot reshape array of size N` otherwise.
Row `k` of `negatives` holds the `K` negatives belonging to `positives[k]`, same user.

Gather features with `context.train_x[positives]` -> `(n_groups, n_fields)` and
`context.train_x[negatives]` -> `(n_groups, negatives_per_group, n_fields)`. Group `k`'s logits
are `positive_score[k]` concatenated with `negative_scores[k]`, giving `K + 1` entries.

Three properties to design around rather than assert against:

- Users without **both** a positive and a negative row are skipped entirely, so `positives`
  does not cover every user, nor every positive row. A test asserting full coverage will fail.
- When a user's negative pool is smaller than `negatives_per_group`, negatives are drawn with
  replacement, so a group can repeat an index; this is the duplicate-negative failure mode
  listed below, not a bug in the sampler.
- With no eligible user at all, `positives` is `(0,)` and `negatives` is
  `(0, negatives_per_group)`. Guard the division when averaging a loss over groups.

## Safe initial search space

- Same-user negatives only
- `K`: 4 or 8
- Temperature: 0.5, 1.0, or 2.0
- FM embedding dimension fixed at 16 for attribution
- Learning rate: 0.0003, 0.0005, or 0.001
- Batch size: 512, 1024, or 2048 groups

## Known failure modes

- Groups with duplicate negative rows reduce effective list size.
- Users without both labels must be skipped.
- Unstable exponentials require max-shifted softmax.
- Increasing `K` changes compute per step and must be reported.

