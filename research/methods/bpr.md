# Bayesian Personalized Ranking (BPR)

## Primary source

- Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, Lars Schmidt-Thieme,
  "BPR: Bayesian Personalized Ranking from Implicit Feedback," UAI 2009.
  https://arxiv.org/abs/1205.2618
- Christopher J. C. Burges, "From RankNet to LambdaRank to LambdaMART: An Overview,"
  Microsoft Research Technical Report MSR-TR-2010-82, 2010 — the source of the
  `pair_weighting="delta_ndcg"` option below (LambdaRank weights a pairwise gradient by the
  metric change the pair's swap would cause).
  https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/

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

### `pair_weighting` — LambdaRank ΔnDCG weights

`parameters["pair_weighting"]` is `"none"` or `"delta_ndcg"`, and it is a weight on the loss, not
a change to the sampler or to the model. `"none"` is the plain BPR gradient above and is the
default. `"delta_ndcg"` multiplies pair `k`'s loss — and therefore its gradient, since the weight
is a constant with respect to the parameters — by `w_k = |ΔnDCG@5|`, the change in the user's
nDCG@5 that swapping the pair's two rows would cause in the user's **current** score order:

```text
d_k    = score(positives[k]) - score(negatives[k])
grad_k = w_k * (sigmoid(d_k) - 1)     # w_k == 1.0 reproduces pair_weighting="none" exactly
```

It is a few lines of numpy, computed per epoch from the scores you already have:

- Score every train row with the current model, then rank rows **within each user** by descending
  score (`np.argsort` over the user's slice, or one global lexsort on `(user, -score)`).
- Give rank `r` (1-based) the nDCG discount `D(r) = 1 / log2(1 + r)` for `r <= 5` and `D(r) = 0`
  beyond it — the metric is nDCG@**5**, so a swap that never touches the top 5 is worth nothing.
- Labels are binary here, so the gain term `|2^y_pos − 2^y_neg|` is always 1 and drops out. That
  leaves `w_k = |D(r_pos) − D(r_neg)| / IDCG_u`, with `IDCG_u = Σ_{r=1..min(5, n_pos_u)} D(r)` the
  ideal DCG@5 of that user's list, where `n_pos_u` is how many train rows of user `u` carry
  `y == 1` (so the sum runs over the top 5 of a perfect ranking, or over all of them when the user
  has fewer than 5 positives). Both terms are per-user, so build them once per epoch and
  gather by pair.
- Normalise the weights to mean 1 **over each minibatch, at the moment it is applied** — not once
  per epoch over the whole train set. Otherwise the effective
  step size shrinks by whatever the mean weight happens to be and the `learning_rate` grid stops
  meaning what it means for `"none"`.
- Guard the degenerate cases: a user whose positive and negative both sit outside the top 5 gets
  `w_k = 0` and contributes nothing, and an all-zero batch of weights must not become a division
  by zero in the normalisation.

Recomputing the order every epoch (rather than every batch) is the intended cost point — one extra
forward pass over train per epoch, no change to `sample_bpr_pairs`, no change to `FMRanker`.

The honest risk, worth stating because it decides whether the axis is worth a second iteration:
train users carry about 43 impressions while evaluation users carry about 5, so the ΔnDCG this
loss optimises is computed over a longer list than the one that is finally scored. The mechanism
is not excluded by the pointwise-vs-pairwise evidence — that gain came from gradient vanishing
under sparse positives, which is not this dataset's regime at 33.7 % positive — but the expected
gain is genuinely unknown, in the range 0 to +0.002. Run `"none"` and `"delta_ndcg"` at otherwise
identical parameters if you want the axis attributable.

## Safe initial search space

- Same-user negative sampling only
- One or two negatives per positive
- FM embedding dimension `k`: 8, 16, 32, or 64
- `l2`: 0.0, 1e-6, 1e-4, 1e-3, or 1e-2 (decoupled decay, so 1e-6 is the "off" end)
- Learning rate: 0.0003, 0.0005, 0.001, 0.002, or 0.005
- Batch size: 2048 or 4096 pairs
- Pair weighting: `none` (the default, plain BPR) or `delta_ndcg` (LambdaRank, see above)
- Resample negatives every epoch

`k`, `l2` and `learning_rate` are searchable here, not fixed: read each from `parameters`, and
treat this family's grid in `src/agent/families.py` as the authority on the permitted values. The
kit's flat k-sweep (k = 8/16/32 -> 0.5895/0.5902/0.5887,
`kuairand-starter-kit/README.en.md:133-139`) was measured under **pointwise logloss**, so it says
where a pointwise model saturates, not where a pairwise one does — and the loss is exactly what
this family changes. Move `learning_rate` with `k`: a wider embedding trained at a rate tuned for
k = 16 looks like a capacity dead end when it is an optimisation one.

## Known failure modes

- Sampling negatives from a different user misaligns the task.
- Users with only positives or only negatives cannot form pairs and must be skipped.
- First-order user bias and global bias cancel in score differences; this is expected.
- Very frequent items/users can dominate without balanced sampling.

