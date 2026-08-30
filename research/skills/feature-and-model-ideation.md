# Role Skill: Feature and Model Ideation

Use a broad method vocabulary when the experiment history has exhausted simple parameter changes.
Choose methods by a concrete ranking hypothesis, available train-only signals, runtime cost, and
compatibility with the registered candidate contract. Do not propose complexity merely for novelty.

## Feature and representation directions

- Nearest neighbours: user-user, item-item, or interaction-neighbour retrieval using cosine,
  Jaccard, co-occurrence, or learned embeddings. Convert neighbours into train-only aggregate
  scores, similarities, counts, or candidate-specific cross features. Fit the index on training
  interactions only and exclude the target row from train-row features.
- Clustering: k-means, mini-batch k-means, Gaussian mixtures, hierarchical or density clustering,
  spectral clustering, co-clustering, and graph/community methods such as Louvain or Leiden. Treat
  cluster IDs as representations and test cluster-by-item, cluster-by-author, or cluster-by-context
  interactions; a user-only cluster bias cannot change within-user ordering.
- Graph representations: bipartite user-video graphs, user-author/video-tag relations,
  co-watch/co-engagement graphs, connected components, centrality, personalized PageRank,
  random-walk embeddings, matrix factorization, and community assignments. Construct every edge
  from training events only; validation and test nodes may receive inductive/unknown features but
  must never contribute labelled edges.
- Statistical transforms: frequency/count encoding, smoothed target rates, recency/frequency,
  exposure and popularity trends, duration buckets, interaction crosses, quantiles, ratios, and
  train-fitted calibration. For each aggregate, specify its key, time window, smoothing, fallback,
  and leave-one-out or prior-time rule.
- Representation learning: PCA/SVD/NMF, autoencoders, contrastive embeddings, metric learning,
  node embeddings, and sequence encoders. Fit using training rows only and preserve row alignment.

## Model directions

- Classical supervised models: logistic/linear models, factorization machines, naive Bayes,
  k-nearest-neighbour classifiers, SVMs, decision trees, random forests, ExtraTrees, gradient
  boosting, XGBoost/LightGBM/CatBoost-style rankers, and calibrated or stacked ensembles.
- Ranking objectives: pointwise classification/regression, pairwise BPR or hinge losses,
  group-softmax/listwise losses, LambdaRank-style weighting, score blending, and multi-task losses.
  Sampling and grouping must remain within user.
- Neural recommenders: MLP/DeepFM, DCN, xDeepFM, AutoInt, NCF/two-tower retrieval, DIN/DIEN,
  GRU/LSTM/TCN sequence models, Transformers/SASRec/BERT4Rec-style encoders, mixture-of-experts,
  multi-task towers, and graph neural networks such as LightGCN, GraphSAGE, GAT, or heterogeneous
  message passing. Prefer small embeddings and shallow networks until the autonomous loop and
  baseline are reliable.
- Unsupervised/semi-supervised probes: clustering-derived features, graph propagation,
  self-supervised sequence masking, contrastive user/item views, pseudo-labels, and distillation.
  Pseudo-labels may come only from train-selected models and may not incorporate hidden-test or
  future information.

## Proposal discipline

Turn one idea into one falsifiable probe. State the signal, transformation/model, expected within-
user ranking effect, leakage boundary, parent configuration, resource estimate, and one primary
contrast. Prefer a cheap representation or ablation before a full architecture. Report cold-start
coverage, unknown rates, cluster/graph construction scope, and seed variance where relevant.

Use three feasibility levels:

1. **Executable now:** expressible by a registered family, approved parameter grid, trusted helper,
   and allowlisted imports. Propose it normally.
2. **Trusted extension required:** scientifically useful but missing a safe feature builder,
   sampler, model API, dependency, or parameter. Name the smallest required extension and do not
   pretend the current Builder can implement it.
3. **Out of scope:** violates leakage/hidden-test rules, budgets, evaluator conventions, immutable
   files, or within-user sampling. Do not propose it.

Generated `candidate.py` may use only its declared runtime contract. Knowledge of scikit-learn,
PyTorch, graph libraries, ANN indexes, or boosting packages does not authorize importing them.
External libraries may be used only after they are installed and explicitly exposed through a
trusted implementation or registered family. Never read raw logs from generated code to bypass a
missing helper.

Avoid known dead ends: adding all static fields, increasing FM embedding dimension alone, user-only
terms that are constant within a user's candidate set, indiscriminate six-group feature bundles,
or large neural architectures without a targeted hypothesis. Official validation GAUC, nDCG@5,
and their primary average remain the decision criteria.
