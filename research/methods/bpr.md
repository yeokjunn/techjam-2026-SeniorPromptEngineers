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

