# Bayesian Personalized Ranking (BPR)

## Primary source

- Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, Lars Schmidt-Thieme,
  "BPR: Bayesian Personalized Ranking from Implicit Feedback," UAI 2009.
  https://arxiv.org/abs/1205.2618

## Hypothesis

The official FM uses pointwise binary cross-entropy even though evaluation ranks
items within a user. BPR instead optimizes the ordering of a positive interaction
above a negative interaction from the same user.

## Objective

For a user-specific positive item `i` and negative item `j`:

```text
d = score(user, i) - score(user, j)
loss = softplus(-d) = -log(sigmoid(d))
```

The derivative with respect to `d` is `sigmoid(d) - 1`.

`src.models.sampling.sample_bpr_pairs` is mandatory, and it returns **row indices**, not rows:

```text
positives, negatives = sample_bpr_pairs(train_users, train_y, rng, negatives_per_positive)

positives   int64, shape (n_pairs,)   index into context.train_x / train_y / train_users
negatives   int64, shape (n_pairs,)   parallel to positives: same user, label 0
n_pairs   = (eligible positives) * negatives_per_positive
```

The two arrays are parallel and one-dimensional; pair `k` is `(positives[k], negatives[k])`.
Gather features with `context.train_x[positives]` and `context.train_x[negatives]`, each
`(n_pairs, n_fields)`.

Three properties to design around rather than assert against:

- Users without **both** a positive and a negative row are skipped entirely, so `positives`
  does not cover every user, nor every positive row. A test asserting full coverage will fail.
- When a user's negative pool is smaller than `negatives_per_positive`, negatives are drawn
  with replacement, so repeated indices are expected and are not a bug.
- With no eligible user at all, both arrays are empty with shape `(0,)`. Guard the division
  when averaging a loss over pairs.

## Safe initial search space

- Same-user negative sampling only
- One or two negatives per positive
- FM embedding dimension fixed at 16 for attribution
- Learning rate: 0.0003, 0.0005, or 0.001
- Batch size: 2048 or 4096 pairs
- Resample negatives every epoch

## Known failure modes

- Sampling negatives from a different user misaligns the task.
- Users with only positives or only negatives cannot form pairs and must be skipped.
- First-order user bias and global bias cancel in score differences; this is expected.
- Very frequent items/users can dominate without balanced sampling.

