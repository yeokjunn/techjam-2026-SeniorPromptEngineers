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

