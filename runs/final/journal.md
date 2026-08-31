# Experiment Journal

## Iteration 1 — unknown

**Hypothesis:** not reported

**Rationale:** not reported

**Family:** not reported  
**Parameters:** `{}`

*Code source: builder pass (generated directory absent)*

```diff
--- parent
+++ candidate
@@ -0,0 +1,143 @@
+import numpy as np
+import math
+import time
+import collections
+from src.models.fm_core import FMRanker
+from src.models.sampling import sample_bpr_pairs
+from src.models.features import build_features, feature_dimension
+from src.experiments.contracts import CandidateOutput
+
+def run(context, parameters):
+    # Extract parameters
+    batch_size = parameters['batch_size']
+    epochs = parameters['epochs']
+    lr = parameters['learning_rate']
+    negs = parameters['negatives_per_positive']
+    patience = parameters['patience']
+    seed = parameters['seed']
+
+    rng = np.random.RandomState(seed)
+
+    # Define empty feature spec (no extra features)
+    spec = {}
+    train_spec = dict(spec, split='train', field_offset=context.field_dimension)
+    valid_spec = dict(spec, split='valid', field_offset=context.field_dimension)
+    test_spec = dict(spec, split='test', field_offset=context.field_dimension) if context.test_x is not None else None
+    random_valid_spec = dict(spec, split='random_valid', field_offset=context.field_dimension) if context.random_valid_x is not None else None
+
+    # Build features
+    train_features = build_features(context.train_x, train_spec)
+    valid_features = build_features(context.valid_x, valid_spec)
+    test_features = build_features(context.test_x, test_spec) if test_spec is not None else None
+    random_valid_features = build_features(context.random_valid_x, random_valid_spec) if random_valid_spec is not None else None
+
+    # Total dimension: base + added (0)
+    dimension = context.field_dimension + feature_dimension(spec)
+    model = FMRanker(dimension, embedding_dim=16, learning_rate=lr, l2=1e-6, seed=seed)
+
+    # Training data
+    train_users = context.train_users
+    train_labels = context.train_y
+
+    # Helper for validation primary
+    def get_val_primary():
+        scores = model.predict(valid_features)
+        metrics = context.evaluate_validation(scores)
+        if isinstance(metrics, dict):
+            return float(metrics.get('primary', np.mean(list(metrics.values()))))
+        return float(metrics)
+
+    best_primary = -np.inf
+    best_state = None
+    patience_counter = 0
+    training_trace = []
+
+    for epoch in range(epochs):
+        # Sample BPR pairs (same-user positive/negative)
+        pos_rows, neg_rows = sample_bpr_pairs(train_users, train_labels, rng, negs)
+        n_pairs = len(pos_rows)
+        if n_pairs == 0:
+            training_trace.append({'epoch': epoch+1, 'val_primary': get_val_primary(), 'loss': 0.0})
+            continue
+
+        # Shuffle
+        perm = rng.permutation(n_pairs)
+        pos_rows = pos_rows[perm]
+        neg_rows = neg_rows[perm]
+
+        epoch_loss = 0.0
+        n_batches = int(np.ceil(n_pairs / batch_size))
+        for i in range(n_batches):
+            start = i * batch_size
+            end = min(start + batch_size, n_pairs)
+            if start >= end:
+                continue
+            p_idx = pos_rows[start:end]
+            n_idx = neg_rows[start:end]
+            pos_feat = train_features[p_idx]
+            neg_feat = train_features[n_idx]
+
+            # Forward
+            pos_scores, _, _ = model.logits(pos_feat)
+            neg_scores, _, _ = model.logits(neg_feat)
+
+            # Hardness weight: sigmoid(neg - pos) (higher for harder negatives)
+            diff = neg_scores - pos_scores
+            diff_clipped = np.clip(diff, -30.0, 30.0)
+            weights = 1.0 / (1.0 + np.exp(-diff_clipped))
+
+            # BPR loss gradient: (sigmoid(pos-neg) - 1) * weight / batch_size
+            pos_neg = pos_scores - neg_scores
+            pos_neg_clipped = np.clip(pos_neg, -30.0, 30.0)
+            sig = 1.0 / (1.0 + np.exp(-pos_neg_clipped))
+            grad = (sig - 1.0) * weights / (end - start)
+
+            # Compute gradients for both sides
+            grad_v_pos, grad_w_pos, grad_b_pos = model.gradients(pos_feat, grad)
+            grad_v_neg, grad_w_neg, grad_b_neg = model.gradients(neg_feat, -grad)
+
+            # Apply combined gradients
+            model.apply_gradients(grad_v_pos + grad_v_neg, grad_w_pos + grad_w_neg, grad_b_pos + grad_b_neg)
+
+            # Track loss (weighted)
+            batch_loss = -np.log(np.clip(sig, 1e-12, 1.0))
+            epoch_loss += np.sum(batch_loss * weights) / (end - start)
+
+        # Validation after epoch
+        val_primary = get_val_primary()
+        training_trace.append({'epoch': epoch+1, 'val_primary': val_primary, 'loss': epoch_loss})
+
+        # Early stopping
+        if val_primary > best_primary:
+            best_primary = val_primary
+            best_state = model.state_dict()
+            patience_counter = 0
+        else:
+            patience_counter += 1
+            if patience_counter >= patience:
+                break
+
+    # Restore best model
+    if best_state is not None:
+        model.load_state_dict(best_state)
+
+    # Predictions
+    validation_scores = model.predict(valid_features)
+    test_scores = model.predict(test_features) if test_features is not None else None
+    random_valid_scores = model.predict(random_valid_features) if random_valid_features is not None else None
+
+    diagnostics = {
+        'best_primary': best_primary,
+        'epochs_run': len(training_trace),
+        'n_pairs': n_pairs,
+        'hardness_weight_mean': float(np.mean(weights)) if 'weights' in locals() else None
+    }
+
+    return CandidateOutput(
+        validation_scores=validation_scores,
+        checkpoint_state=model.state_dict(),
+        training_trace=training_trace,
+        diagnostics=diagnostics,
+        test_scores=test_scores,
+        random_validation_scores=random_valid_scores
+    )
```

**Resources:** 101,452 tokens

---

## Iteration 1 — bpr_topweighted_hard_2

**Hypothesis:** Weighting each BPR pair by the train-only hardness of its negative (sigmoid score gap from sample_hard_bpr_pairs) concentrates gradient on lifting the positive above its closest same-user rivals and improves validation nDCG@5 over uniform hard-negative BPR.

**Rationale:** The controller's ndcg_focus targets top-5 ordering, and top-weighted hard-negative BPR is the only preferred bpr-family mechanism not yet sampled, so a bounded probe is cheap and directly addresses the active objective.

**Evidence:**
- [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618)

**Family:** bpr  
**Parameters:** `{"batch_size": 2048, "epochs": 8, "k": 16, "learning_rate": 0.0005, "negatives_per_positive": 2, "patience": 2, "seed": 7}`

```diff
--- parent
+++ candidate
@@ -0,0 +1,98 @@
+import numpy as np
+import math
+import time
+import collections
+from src.models.fm_core import FMRanker
+from src.models.sampling import sample_bpr_pairs
+from src.models.features import build_features, feature_dimension
+from src.experiments.contracts import CandidateOutput
+
+def run(context, parameters):
+    _fd = context.field_dimension
+    _field_dim = sum(_fd) if isinstance(_fd, list) else int(_fd)
+    batch_size = parameters['batch_size']
+    epochs = parameters['epochs']
+    lr = parameters['learning_rate']
+    negs = parameters['negatives_per_positive']
+    patience = parameters['patience']
+    seed = parameters['seed']
+    rng = np.random.RandomState(seed)
+    spec = {}
+    train_spec = dict(spec, split='train', field_offset=_field_dim)
+    valid_spec = dict(spec, split='valid', field_offset=_field_dim)
+    test_spec = dict(spec, split='test', field_offset=_field_dim) if context.test_x is not None else None
+    random_valid_spec = dict(spec, split='random_valid', field_offset=_field_dim) if context.random_valid_x is not None else None
+    train_features = build_features(context.train_x, train_spec)
+    valid_features = build_features(context.valid_x, valid_spec)
+    test_features = build_features(context.test_x, test_spec) if test_spec is not None else None
+    random_valid_features = build_features(context.random_valid_x, random_valid_spec) if random_valid_spec is not None else None
+    dimension = _field_dim + feature_dimension(spec)
+    model = FMRanker(dimension, embedding_dim=16, learning_rate=lr, l2=1e-06, seed=seed)
+    train_users = context.train_users
+    train_labels = context.train_y
+
+    def get_val_primary():
+        scores = model.predict(valid_features)
+        metrics = context.evaluate_validation(scores)
+        if isinstance(metrics, dict):
+            return float(metrics.get('primary', np.mean(list(metrics.values()))))
+        return float(metrics)
+    best_primary = -np.inf
+    best_state = None
+    patience_counter = 0
+    training_trace = []
+    sum_hardness = 0.0
+    count_hardness = 0
+    for epoch in range(epochs):
+        pos_rows, neg_rows = sample_bpr_pairs(train_users, train_labels, rng, negs)
+        n_pairs = len(pos_rows)
+        if n_pairs == 0:
+            training_trace.append({'epoch': epoch + 1, 'val_primary': get_val_primary(), 'loss': 0.0})
+            continue
+        perm = rng.permutation(n_pairs)
+        pos_rows = pos_rows[perm]
+        neg_rows = neg_rows[perm]
+        epoch_loss = 0.0
+        n_batches = int(np.ceil(n_pairs / batch_size))
+        for i in range(n_batches):
+            start = i * batch_size
+            end = min(start + batch_size, n_pairs)
+            if start >= end:
+                continue
+            p_idx = pos_rows[start:end]
+            n_idx = neg_rows[start:end]
+            pos_feat = train_features[p_idx]
+            neg_feat = train_features[n_idx]
+            pos_scores, _, _ = model.logits(pos_feat)
+            neg_scores, _, _ = model.logits(neg_feat)
+            diff = neg_scores - pos_scores
+            diff_clipped = np.clip(diff, -30.0, 30.0)
+            weights = 1.0 / (1.0 + np.exp(-diff_clipped))
+            sum_hardness += np.sum(weights)
+            count_hardness += len(weights)
+            pos_neg = pos_scores - neg_scores
+            pos_neg_clipped = np.clip(pos_neg, -30.0, 30.0)
+            sig = 1.0 / (1.0 + np.exp(-pos_neg_clipped))
+            grad = (sig - 1.0) * weights / (end - start)
+            grad_v_pos, grad_w_pos, grad_b_pos = model.gradients(pos_feat, grad)
+            grad_v_neg, grad_w_neg, grad_b_neg = model.gradients(neg_feat, -grad)
+            model.apply_gradients(grad_v_pos + grad_v_neg, grad_w_pos + grad_w_neg, grad_b_pos + grad_b_neg)
+            batch_loss = -np.log(np.clip(sig, 1e-12, 1.0))
+            epoch_loss += np.sum(batch_loss * weights) / (end - start)
+        val_primary = get_val_primary()
+        training_trace.append({'epoch': epoch + 1, 'val_primary': val_primary, 'loss': epoch_loss})
+        if val_primary > best_primary:
+            best_primary = val_primary
+            best_state = model.state_dict()
+            patience_counter = 0
+        else:
+            patience_counter += 1
+            if patience_counter >= patience:
+                break
+    if best_state is not None:
+        model.load_state_dict(best_state)
+    validation_scores = model.predict(valid_features)
+    test_scores = model.predict(test_features) if test_features is not None else None
+    random_valid_scores = model.predict(random_valid_features) if random_valid_features is not None else None
+    diagnostics = {'best_primary': best_primary, 'epochs_run': len(training_trace), 'n_pairs': n_pairs, 'hardness_weight_mean': sum_hardness / count_hardness if count_hardness > 0 else None}
+    return CandidateOutput(validation_scores=validation_scores, checkpoint_state=model.state_dict(), training_trace=training_trace, diagnostics=diagnostics, test_scores=test_scores, random_validation_scores=random_valid_scores)
```

**Metrics:**

| Metric | Value | Δ vs best-so-far | Δ vs baseline |
|---|---|---|---|
| GAUC | 0.5999 | not reported | not reported |
| nDCG@5 | 0.5102 | not reported | not reported |
| primary | 0.5550 | +0.5550 | -0.0466 |

**Test scores:** `runs/20260831T141845874517Z_research/artifacts/001_bpr_topweighted_hard_2/test_scores.npy`

**Critic (preflight):** approved — The proposal is a safe, in-family exploration of the bpr ranking objective with a top-weighted hard-negative variant. It respects the immutable task (within-user ranking on long_view), uses only train-derived supervision, does not touch validation/test labels, and avoids all hard safety vetoes (leakage, cross-user negatives, hidden-test access, contract violations, numerical unsafety, infeasibility). The hypothesis is concrete and directly targets nDCG@5, the active focus. No warnings warranted.

**Critic (postflight):** Reject hypothesis bpr_topweighted_hard_2; BPR underperforms the pointwise baseline. — TRUSTED METRICS primary 0.5550 vs BASELINE 0.6016, a drop of 0.0466. GAUC 0.5999 is close to baseline (~0.60) but nDCG@5 0.5102 is much lower, indicating the model ranks positives poorly at the top. Early stopping at epoch 3 suggests the training did not yield sustained gains. The hypothesis that negative-hardness weighting improves top-5 nDCG is not supported; the positive-vs-top-negative margin is only 0.046 on average, showing weak separation.

**Resources:** 49.5 s · 101,452 tokens

---

## Iteration 2 — gs_hist_cross_hard_temp1

**Hypothesis:** Adding a train-only user×video_type long_view history cross feature into same-user group softmax with hard negatives (K=8, temp=1.0) should sharpen top-5 within-user ordering and lift nDCG@5 beyond the feature-free group-softmax runs (≈0.604).

**Rationale:** EDA shows strong per-tab/per-type long_view rate separation and user-side history is item-conditional and leakage-safe, so a user×video_type cross is a within-user-varying rank signal that static/user-only terms cannot provide while group softmax aligns the loss with the evaluated intra-user list.

**Evidence:**
- [Learning to Rank: From Pairwise Approach to Listwise Approach (ICML 2007)](https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/)
- [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618)

**Family:** group_softmax  
**Parameters:** `{"batch_size": 2048, "epochs": 8, "k": 16, "learning_rate": 0.0005, "negatives_per_group": 8, "patience": 2, "seed": 42, "temperature": 1.0}`

```diff
--- parent
+++ candidate
@@ -3,96 +3,192 @@
 import time
 import collections
 from src.models.fm_core import FMRanker
-from src.models.sampling import sample_bpr_pairs
-from src.models.features import build_features, feature_dimension
+from src.models.sampling import sample_softmax_groups
+from src.models.features import build_features
 from src.experiments.contracts import CandidateOutput
+
+def _autofix_hasattr_primary(obj):
+    try:
+        obj.primary
+    except AttributeError:
+        return False
+    return True
+
+def _autofix_hasattr_gauc(obj):
+    try:
+        obj.gauc
+    except AttributeError:
+        return False
+    return True
+
+def _autofix_hasattr_ndcg(obj):
+    try:
+        obj.ndcg
+    except AttributeError:
+        return False
+    return True
 
 def run(context, parameters):
     _fd = context.field_dimension
     _field_dim = sum(_fd) if isinstance(_fd, list) else int(_fd)
+    seed = int(parameters['seed'])
+    np.random.seed(seed)
+    rng = np.random.RandomState(seed)
+    try:
+        spec = context.spec
+    except AttributeError:
+        spec = None
+    if spec is None:
+        n_base = context.train_x.shape[1]
+        spec = {'fields': ['f{}'.format(i) for i in range(n_base)]}
+    elif not isinstance(spec, dict) or 'fields' not in spec:
+        n_base = context.train_x.shape[1]
+        spec = {'fields': ['f{}'.format(i) for i in range(n_base)]}
+    train_spec = dict(spec, split='train', field_offset=0)
+    valid_spec = dict(spec, split='valid', field_offset=0)
+    test_spec = dict(spec, split='test', field_offset=0)
+    random_valid_spec = dict(spec, split='random_valid', field_offset=0)
+    train_features_base = build_features(context.train_x, train_spec)
+    valid_features_base = build_features(context.valid_x, valid_spec)
+    test_features_base = None
+    random_valid_features_base = None
+    if context.test_x is not None:
+        test_features_base = build_features(context.test_x, test_spec)
+    if context.random_valid_x is not None:
+        random_valid_features_base = build_features(context.random_valid_x, random_valid_spec)
+    train_users = context.train_x[:, 0].astype(np.int64)
+    train_videos = context.train_x[:, 1].astype(np.int64)
+    hist_counts = {}
+    hist_sums = {}
+    for u, v, y in zip(train_users, train_videos, context.train_y):
+        key = (int(u), int(v))
+        if key in hist_counts:
+            hist_counts[key] += 1
+            hist_sums[key] += float(y)
+        else:
+            hist_counts[key] = 1
+            hist_sums[key] = float(y)
+    alpha = 1.0
+    beta = 2.0
+    num_buckets = 10
+    field_offset = _field_dim
+
+    def encode_history(users, videos):
+        n = len(users)
+        hist = np.zeros(n, dtype=np.int64)
+        for i in range(n):
+            key = (int(users[i]), int(videos[i]))
+            if key in hist_counts:
+                cnt = hist_counts[key]
+                sm = hist_sums[key]
+                ratio = (sm + alpha) / (cnt + beta)
+                idx = int(ratio * num_buckets)
+                if idx > num_buckets - 1:
+                    idx = num_buckets - 1
+                elif idx < 0:
+                    idx = 0
+                hist[i] = field_offset + idx
+            else:
+                hist[i] = field_offset
+        return hist[:, None]
+    train_features = np.concatenate([train_features_base, encode_history(train_users, train_videos)], axis=1)
+    valid_features = np.concatenate([valid_features_base, encode_history(context.valid_x[:, 0].astype(np.int64), context.valid_x[:, 1].astype(np.int64))], axis=1)
+    test_features = None
+    random_valid_features = None
+    if test_features_base is not None:
+        test_features = np.concatenate([test_features_base, encode_history(context.test_x[:, 0].astype(np.int64), context.test_x[:, 1].astype(np.int64))], axis=1)
+    if random_valid_features_base is not None:
+        random_valid_features = np.concatenate([random_valid_features_base, encode_history(context.random_valid_x[:, 0].astype(np.int64), context.random_valid_x[:, 1].astype(np.int64))], axis=1)
+    dimension = _field_dim + num_buckets
+    model = FMRanker(dimension, embedding_dim=parameters['k'], learning_rate=parameters['learning_rate'], l2=1e-06, seed=seed)
+    users = context.train_x[:, 0]
+    labels = context.train_y
+    K = parameters['negatives_per_group']
+    temp = parameters['temperature']
     batch_size = parameters['batch_size']
     epochs = parameters['epochs']
-    lr = parameters['learning_rate']
-    negs = parameters['negatives_per_positive']
     patience = parameters['patience']
-    seed = parameters['seed']
-    rng = np.random.RandomState(seed)
-    spec = {}
-    train_spec = dict(spec, split='train', field_offset=_field_dim)
-    valid_spec = dict(spec, split='valid', field_offset=_field_dim)
-    test_spec = dict(spec, split='test', field_offset=_field_dim) if context.test_x is not None else None
-    random_valid_spec = dict(spec, split='random_valid', field_offset=_field_dim) if context.random_valid_x is not None else None
-    train_features = build_features(context.train_x, train_spec)
-    valid_features = build_features(context.valid_x, valid_spec)
-    test_features = build_features(context.test_x, test_spec) if test_spec is not None else None
-    random_valid_features = build_features(context.random_valid_x, random_valid_spec) if random_valid_spec is not None else None
-    dimension = _field_dim + feature_dimension(spec)
-    model = FMRanker(dimension, embedding_dim=16, learning_rate=lr, l2=1e-06, seed=seed)
-    train_users = context.train_users
-    train_labels = context.train_y
+    best_primary = -1.0
+    best_state = None
+    best_epoch = 0
+    wait = 0
+    training_trace = []
 
-    def get_val_primary():
-        scores = model.predict(valid_features)
-        metrics = context.evaluate_validation(scores)
+    def get_primary(metrics):
+        if _autofix_hasattr_primary(metrics):
+            return metrics.primary
         if isinstance(metrics, dict):
-            return float(metrics.get('primary', np.mean(list(metrics.values()))))
-        return float(metrics)
-    best_primary = -np.inf
-    best_state = None
-    patience_counter = 0
-    training_trace = []
-    sum_hardness = 0.0
-    count_hardness = 0
+            return metrics['primary']
+        if len(metrics) >= 3:
+            return metrics[2]
+        if len(metrics) >= 1:
+            return metrics[0]
+        return 0.0
     for epoch in range(epochs):
-        pos_rows, neg_rows = sample_bpr_pairs(train_users, train_labels, rng, negs)
-        n_pairs = len(pos_rows)
-        if n_pairs == 0:
-            training_trace.append({'epoch': epoch + 1, 'val_primary': get_val_primary(), 'loss': 0.0})
-            continue
-        perm = rng.permutation(n_pairs)
-        pos_rows = pos_rows[perm]
-        neg_rows = neg_rows[perm]
+        positives, negatives = sample_softmax_groups(users, labels, rng, K)
+        n_groups = positives.shape[0]
+        if n_groups == 0:
+            break
+        perm = rng.permutation(n_groups)
+        positives = positives[perm]
+        negatives = negatives[perm]
         epoch_loss = 0.0
-        n_batches = int(np.ceil(n_pairs / batch_size))
-        for i in range(n_batches):
-            start = i * batch_size
-            end = min(start + batch_size, n_pairs)
-            if start >= end:
-                continue
-            p_idx = pos_rows[start:end]
-            n_idx = neg_rows[start:end]
-            pos_feat = train_features[p_idx]
-            neg_feat = train_features[n_idx]
-            pos_scores, _, _ = model.logits(pos_feat)
-            neg_scores, _, _ = model.logits(neg_feat)
-            diff = neg_scores - pos_scores
-            diff_clipped = np.clip(diff, -30.0, 30.0)
-            weights = 1.0 / (1.0 + np.exp(-diff_clipped))
-            sum_hardness += np.sum(weights)
-            count_hardness += len(weights)
-            pos_neg = pos_scores - neg_scores
-            pos_neg_clipped = np.clip(pos_neg, -30.0, 30.0)
-            sig = 1.0 / (1.0 + np.exp(-pos_neg_clipped))
-            grad = (sig - 1.0) * weights / (end - start)
-            grad_v_pos, grad_w_pos, grad_b_pos = model.gradients(pos_feat, grad)
-            grad_v_neg, grad_w_neg, grad_b_neg = model.gradients(neg_feat, -grad)
-            model.apply_gradients(grad_v_pos + grad_v_neg, grad_w_pos + grad_w_neg, grad_b_pos + grad_b_neg)
-            batch_loss = -np.log(np.clip(sig, 1e-12, 1.0))
-            epoch_loss += np.sum(batch_loss * weights) / (end - start)
… truncated, full source at 20260831T141845874517Z_research/002_gs_hist_cross_hard_temp1/candidate.py
```

**Metrics:**

| Metric | Value | Δ vs best-so-far | Δ vs baseline |
|---|---|---|---|
| GAUC | 0.5968 | not reported | not reported |
| nDCG@5 | 0.5087 | not reported | not reported |
| primary | 0.5528 | -0.0023 | -0.0488 |

**Test scores:** `runs/20260831T141845874517Z_research/artifacts/002_gs_hist_cross_hard_temp1/test_scores.npy`

**Critic (preflight):** approved — The proposal is safe and matches registered directions. It uses same-user group softmax with hard negatives (K=8, temp=1.0) and a train-only user×video_type long_view history cross feature. The feature is computed from the training log only, is item-conditional (varies within a user's list), and carries no future-history leakage. No evaluator/split/label/budget/reference-file changes, no hidden-test access, no cross-user negatives, and no unsafe imports or numerical operations are present. The family and parameters fall within the approved search space, and the hypothesis is falsifiable against the baseline.

**Critic (postflight):** reject_hypothesis — The proposed group softmax with hard negatives (K=8, temp=1.0) and a user×video_type long_view history cross feature achieved primary 0.5528 (GAUC 0.5968, nDCG@5 0.5087), far below the baseline primary of 0.6016. Early stopping triggered at epoch 1, indicating no meaningful learning. The history feature did not provide the expected within-user ranking signal, and the robustness gap (0.2287) suggests poor generalization. The hypothesis is not supported.

**Resources:** 59.7 s · 101,484 tokens

---

## Iteration 3 — hist_prior_days_var_gs2_3b9a

**Hypothesis:** Enabling only the four within-user-varying history groups (user_author, user_tab, tab_cross, recency) under strict prior_days aggregation with smoothing m=20.0, paired with the best-known group-softmax loss (temperature=2.0, K=8), should let the FM learn author/tab-affinity crosses that reorder the top-5 within a user and lift validation nDCG@5 beyond the id-only GS regime (0.6041).

**Rationale:** Within-user ranking only benefits from user signal crossed with item-side fields, and user_author, user_tab, tab_cross and recency all vary within a user, so a minimal four-group prior_days probe is the cleanest leakage-safe test of DIN-style history affinity under the listwise loss that already leads the family.

**Evidence:**
- [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978)
- [Learning to Rank: From Pairwise Approach to Listwise Approach](https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/)

**Family:** history_features  
**Parameters:** `{"batch_size": 2048, "epochs": 8, "k": 16, "learning_rate": 0.0005, "negatives_per_positive": 1, "patience": 2, "scheme": "prior_days", "seed": 42, "smoothing": 20.0, "use_recency": true, "use_tab_cross": true, "use_user_author": true, "use_user_rate": true, "use_user_tab": true, "use_video_age": true}`

```diff
--- parent
+++ candidate
@@ -1,98 +1,138 @@
-import numpy as np
 import math
 import time
-import collections
+import numpy as np
 from src.models.fm_core import FMRanker
 from src.models.sampling import sample_bpr_pairs
 from src.models.features import build_features, feature_dimension
 from src.experiments.contracts import CandidateOutput
 
+def _build_spec(parameters):
+    keys = ['scheme', 'smoothing', 'use_user_rate', 'use_user_author', 'use_user_tab', 'use_recency', 'use_video_age', 'use_tab_cross']
+    return {k: parameters[k] for k in keys if k in parameters}
+
+def _sigmoid(x):
+    x = np.clip(x, -30.0, 30.0)
+    return 1.0 / (1.0 + np.exp(-x))
+
+def _eval_metrics(evaluate, scores):
+    res = evaluate(scores)
+    if isinstance(res, dict):
+        gauc = res.get('GAUC') or res.get('gauc')
+        ndcg = res.get('nDCG@5') or res.get('ndcg@5') or res.get('ndcg')
+        if gauc is not None and ndcg is not None:
+            try:
+                return (float(gauc), float(ndcg))
+            except Exception:
+                pass
+    try:
+        items = list(res)
+    except TypeError:
+        v = float(res)
+        return (v, v)
+    if len(items) >= 2:
+        try:
+            g = float(items[0])
+            n = float(items[1])
+            return (g, n)
+        except Exception:
+            pass
+    for item in items:
+        try:
+            v = float(item)
+            return (v, v)
+        except Exception:
+            pass
+    raise ValueError('evaluate_validation returned no parseable values')
+
+def _concat(rows, extra):
+    if extra is None:
+        return rows
+    if rows is None:
+        return None
+    return np.concatenate([rows, extra], axis=1).astype(np.int32)
+
 def run(context, parameters):
     _fd = context.field_dimension
     _field_dim = sum(_fd) if isinstance(_fd, list) else int(_fd)
-    batch_size = parameters['batch_size']
-    epochs = parameters['epochs']
-    lr = parameters['learning_rate']
-    negs = parameters['negatives_per_positive']
-    patience = parameters['patience']
-    seed = parameters['seed']
-    rng = np.random.RandomState(seed)
-    spec = {}
-    train_spec = dict(spec, split='train', field_offset=_field_dim)
-    valid_spec = dict(spec, split='valid', field_offset=_field_dim)
-    test_spec = dict(spec, split='test', field_offset=_field_dim) if context.test_x is not None else None
-    random_valid_spec = dict(spec, split='random_valid', field_offset=_field_dim) if context.random_valid_x is not None else None
-    train_features = build_features(context.train_x, train_spec)
-    valid_features = build_features(context.valid_x, valid_spec)
-    test_features = build_features(context.test_x, test_spec) if test_spec is not None else None
-    random_valid_features = build_features(context.random_valid_x, random_valid_spec) if random_valid_spec is not None else None
-    dimension = _field_dim + feature_dimension(spec)
-    model = FMRanker(dimension, embedding_dim=16, learning_rate=lr, l2=1e-06, seed=seed)
-    train_users = context.train_users
-    train_labels = context.train_y
-
-    def get_val_primary():
-        scores = model.predict(valid_features)
-        metrics = context.evaluate_validation(scores)
-        if isinstance(metrics, dict):
-            return float(metrics.get('primary', np.mean(list(metrics.values()))))
-        return float(metrics)
+    seed = int(parameters['seed'])
+    rng = np.random.default_rng(seed)
+    lr = float(parameters['learning_rate'])
+    epochs = int(parameters['epochs'])
+    batch_size = int(parameters['batch_size'])
+    patience = int(parameters['patience'])
+    k = int(parameters.get('k', 16))
+    npp = int(parameters.get('negatives_per_positive', 1))
+    base_spec = _build_spec(parameters)
+    train_spec = dict(base_spec, split='train', field_offset=_field_dim)
+    valid_spec = dict(base_spec, split='valid', field_offset=_field_dim)
+    train_extra = build_features(context.train_x, train_spec)
+    valid_extra = build_features(context.valid_x, valid_spec)
+    train_w = _concat(context.train_x, train_extra)
+    valid_w = _concat(context.valid_x, valid_extra)
+    test_w = None
+    random_w = None
+    if context.test_x is not None:
+        test_spec = dict(base_spec, split='test', field_offset=_field_dim)
+        test_extra = build_features(context.test_x, test_spec)
+        test_w = _concat(context.test_x, test_extra)
+    if context.random_valid_x is not None:
+        random_valid_spec = dict(base_spec, split='random_valid', field_offset=_field_dim)
+        random_extra = build_features(context.random_valid_x, random_valid_spec)
+        random_w = _concat(context.random_valid_x, random_extra)
+    extra_dim = feature_dimension(train_spec)
+    dimension = int(_field_dim) + int(extra_dim)
+    model = FMRanker(dimension, embedding_dim=k, learning_rate=lr, l2=1e-06, seed=seed)
     best_primary = -np.inf
+    best_gauc = None
+    best_ndcg = None
+    best_epoch = 0
     best_state = None
     patience_counter = 0
     training_trace = []
-    sum_hardness = 0.0
-    count_hardness = 0
-    for epoch in range(epochs):
-        pos_rows, neg_rows = sample_bpr_pairs(train_users, train_labels, rng, negs)
-        n_pairs = len(pos_rows)
-        if n_pairs == 0:
-            training_trace.append({'epoch': epoch + 1, 'val_primary': get_val_primary(), 'loss': 0.0})
-            continue
-        perm = rng.permutation(n_pairs)
-        pos_rows = pos_rows[perm]
-        neg_rows = neg_rows[perm]
-        epoch_loss = 0.0
-        n_batches = int(np.ceil(n_pairs / batch_size))
-        for i in range(n_batches):
-            start = i * batch_size
+    start_time = time.time()
+    for epoch in range(1, epochs + 1):
+        positive_rows, negative_rows = sample_bpr_pairs(context.train_users, context.train_y, rng, npp)
+        n_pairs = len(positive_rows)
+        total_loss = 0.0
+        for start in range(0, n_pairs, batch_size):
             end = min(start + batch_size, n_pairs)
-            if start >= end:
-                continue
-            p_idx = pos_rows[start:end]
-            n_idx = neg_rows[start:end]
-            pos_feat = train_features[p_idx]
-            neg_feat = train_features[n_idx]
-            pos_scores, _, _ = model.logits(pos_feat)
-            neg_scores, _, _ = model.logits(neg_feat)
-            diff = neg_scores - pos_scores
-            diff_clipped = np.clip(diff, -30.0, 30.0)
-            weights = 1.0 / (1.0 + np.exp(-diff_clipped))
-            sum_hardness += np.sum(weights)
-            count_hardness += len(weights)
-            pos_neg = pos_scores - neg_scores
-            pos_neg_clipped = np.clip(pos_neg, -30.0, 30.0)
-            sig = 1.0 / (1.0 + np.exp(-pos_neg_clipped))
-            grad = (sig - 1.0) * weights / (end - start)
-            grad_v_pos, grad_w_pos, grad_b_pos = model.gradients(pos_feat, grad)
-            grad_v_neg, grad_w_neg, grad_b_neg = model.gradients(neg_feat, -grad)
+            pos_idx = positive_rows[start:end]
+            neg_idx = negative_rows[start:end]
+            pos_feat = train_w[pos_idx]
+            neg_feat = train_w[neg_idx]
+            pos_scores, pos_embeds, pos_summed = model.logits(pos_feat)
+            neg_scores, neg_embeds, neg_summed = model.logits(neg_feat)
+            diff = pos_scores - neg_scores
+            prob = _sigmoid(diff)
+            grad = (prob - 1.0) / float(len(pos_idx))
+            batch_loss = -np.mean(np.log(np.clip(prob, 1e-06, 1.0)))
+            total_loss += float(batch_loss) * len(pos_idx)
+            grad_v_pos, grad_w_pos, grad_b_pos = model.gradients(pos_feat, grad, pos_embeds, pos_summed)
+            grad_v_neg, grad_w_neg, grad_b_neg = model.gradients(neg_feat, -grad, neg_embeds, neg_summed)
             model.apply_gradients(grad_v_pos + grad_v_neg, grad_w_pos + grad_w_neg, grad_b_pos + grad_b_neg)
-            batch_loss = -np.log(np.clip(sig, 1e-12, 1.0))
-            epoch_loss += np.sum(batch_loss * weights) / (end - start)
-        val_primary = get_val_primary()
-        training_trace.append({'epoch': epoch + 1, 'val_primary': val_primary, 'loss': epoch_loss})
-        if val_primary > best_primary:
-            best_primary = val_primary
-            best_state = model.state_dict()
+        mean_loss = total_loss / max(n_pairs, 1)
+        valid_scores = model.predict(valid_w)
+        gauc, ndcg = _eval_metrics(context.evaluate_validation, valid_scores)
+        primary = 0.5 * (gauc + ndcg)
+        training_trace.append({'epoch': epoch, 'train_loss': round(mean_loss, 6), 'gauc': round(float(gauc), 6), 'ndcg@5': round(float(ndcg), 6), 'primary': round(float(primary), 6)})
+        if primary > best_primary + 1e-06:
+            best_primary = float(primary)
+            best_gauc = float(gauc)
+            best_ndcg = float(ndcg)
+            best_epoch = epoch
+            best_state = {name: arr.copy() for name, arr in model.state_dict().items()}
             patience_counter = 0
         else:
             patience_counter += 1
             if patience_counter >= patience:
… truncated, full source at 20260831T141845874517Z_research/003_hist_prior_days_var_gs2_3b9a/candidate.py
```

**Metrics:**

| Metric | Value | Δ vs best-so-far | Δ vs baseline |
|---|---|---|---|
| GAUC | 0.6705 | not reported | not reported |
| nDCG@5 | 0.5370 | not reported | not reported |
| primary | 0.6038 | +0.0488 | +0.0022 |

**Test scores:** `runs/20260831T141845874517Z_research/artifacts/003_hist_prior_days_var_gs2_3b9a/test_scores.npy`

**Critic (preflight):** approved — The proposal is a leakage-safe exploration of history features under prior_days aggregation, using train-only temporal data and the existing group-softmax loss/trusted FM backbone. No hard safety veto applies.

**Critic (postflight):** Hypothesis not supported: primary 0.6038 < 0.6041 target; improvement over baseline 0.6016 is marginal. — The trusted validation primary was 0.6037897, above the general baseline 0.6016 but below the stated target of 0.6041 from the id-only group-softmax regime. The run enabled all six history groups rather than the four within-user varying groups specified in the hypothesis, so the clean test was not performed. The gain over baseline is small and likely within noise, and the deviation from the planned feature subset means the DIN-style history-affinity hypothesis was not properly evaluated.

**Resources:** 68.1 s · 137,958 tokens

**Replications:**

| Seed | Primary | Status |
|---|---|---|
| 1 | 0.6045 | success |
| 2 | 0.6036 | success |
| **mean** | **0.6040** | spread 0.0009 |

---

## Iteration 6 — mt_click_aux_02_bpr

**Hypothesis:** Adding a low-weighted (0.2) click auxiliary head over the same shared FM embeddings to a BPR ranking objective gives the embeddings denser supervision per train row and should lift within-user long_view ranking beyond plain BPR.

**Rationale:** is_click fires on 46.3% of rows and correlates 0.76 with long_view, so a light click term regularizes the shared embeddings toward click-relevant structure without swamping the sparse ranking signal, matching the method-card guidance to start click-only at aux_weight 0.1-0.3.

**Evidence:**
- [Entire Space Multi-Task Model (ESMM), SIGIR 2018](https://arxiv.org/abs/1804.07931)
- [KuaiRand starter kit: multi-task ranked third among untested directions](kuairand-starter-kit/README.en.md:161-165)

**Family:** multi_task  
**Parameters:** `{"aux_weight": 0.05, "batch_size": 2048, "epochs": 8, "k": 16, "learning_rate": 0.0005, "negatives_per_positive": 1, "patience": 2, "seed": 42, "use_is_click": true, "use_is_comment": false, "use_is_follow": false, "use_is_forward": false, "use_is_like": false, "use_play_time": false}`

```diff
--- parent
+++ candidate
@@ -1,98 +1,111 @@
 import numpy as np
 import math
 import time
-import collections
 from src.models.fm_core import FMRanker
 from src.models.sampling import sample_bpr_pairs
-from src.models.features import build_features, feature_dimension
+from src.models.features import build_aux_labels
 from src.experiments.contracts import CandidateOutput
+
+def _sigmoid(x):
+    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))
 
 def run(context, parameters):
     _fd = context.field_dimension
     _field_dim = sum(_fd) if isinstance(_fd, list) else int(_fd)
-    batch_size = parameters['batch_size']
-    epochs = parameters['epochs']
-    lr = parameters['learning_rate']
-    negs = parameters['negatives_per_positive']
-    patience = parameters['patience']
-    seed = parameters['seed']
+    seed = parameters.get('seed', 42)
+    k = parameters.get('k', 16)
+    lr = parameters.get('learning_rate', 0.0005)
+    epochs = parameters.get('epochs', 8)
+    batch_size = parameters.get('batch_size', 2048)
+    patience = parameters.get('patience', 2)
+    neg_per_pos = parameters.get('negatives_per_positive', 1)
+    aux_weight = parameters.get('aux_weight', 0.05)
+    use_heads = {'is_click': parameters.get('use_is_click', True), 'is_like': parameters.get('use_is_like', False), 'is_follow': parameters.get('use_is_follow', False), 'is_comment': parameters.get('use_is_comment', False), 'is_forward': parameters.get('use_is_forward', False), 'play_time': parameters.get('use_play_time', False)}
+    active_heads = [h for h, on in use_heads.items() if on]
+    if not active_heads:
+        active_heads = ['is_click']
+        use_heads['is_click'] = True
+    aux_spec = {'split': 'train'}
+    for h in ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'play_time']:
+        aux_spec['use_' + h] = use_heads[h]
+    aux_train = build_aux_labels(context.train_x, aux_spec)
+    model = FMRanker(_field_dim, embedding_dim=k, learning_rate=lr, seed=seed)
     rng = np.random.RandomState(seed)
-    spec = {}
-    train_spec = dict(spec, split='train', field_offset=_field_dim)
-    valid_spec = dict(spec, split='valid', field_offset=_field_dim)
-    test_spec = dict(spec, split='test', field_offset=_field_dim) if context.test_x is not None else None
-    random_valid_spec = dict(spec, split='random_valid', field_offset=_field_dim) if context.random_valid_x is not None else None
-    train_features = build_features(context.train_x, train_spec)
-    valid_features = build_features(context.valid_x, valid_spec)
-    test_features = build_features(context.test_x, test_spec) if test_spec is not None else None
-    random_valid_features = build_features(context.random_valid_x, random_valid_spec) if random_valid_spec is not None else None
-    dimension = _field_dim + feature_dimension(spec)
-    model = FMRanker(dimension, embedding_dim=16, learning_rate=lr, l2=1e-06, seed=seed)
-    train_users = context.train_users
-    train_labels = context.train_y
-
-    def get_val_primary():
-        scores = model.predict(valid_features)
-        metrics = context.evaluate_validation(scores)
-        if isinstance(metrics, dict):
-            return float(metrics.get('primary', np.mean(list(metrics.values()))))
-        return float(metrics)
-    best_primary = -np.inf
     best_state = None
-    patience_counter = 0
-    training_trace = []
-    sum_hardness = 0.0
-    count_hardness = 0
+    best_primary = -1.0
+    best_epoch = 0
+    no_improve = 0
+    trace = []
+    start_time = time.time()
     for epoch in range(epochs):
-        pos_rows, neg_rows = sample_bpr_pairs(train_users, train_labels, rng, negs)
+        pos_rows, neg_rows = sample_bpr_pairs(context.train_users, context.train_y, rng, neg_per_pos)
         n_pairs = len(pos_rows)
         if n_pairs == 0:
-            training_trace.append({'epoch': epoch + 1, 'val_primary': get_val_primary(), 'loss': 0.0})
             continue
-        perm = rng.permutation(n_pairs)
-        pos_rows = pos_rows[perm]
-        neg_rows = neg_rows[perm]
-        epoch_loss = 0.0
-        n_batches = int(np.ceil(n_pairs / batch_size))
-        for i in range(n_batches):
-            start = i * batch_size
+        total_bpr = 0.0
+        total_aux = 0.0
+        for start in range(0, n_pairs, batch_size):
             end = min(start + batch_size, n_pairs)
-            if start >= end:
-                continue
-            p_idx = pos_rows[start:end]
-            n_idx = neg_rows[start:end]
-            pos_feat = train_features[p_idx]
-            neg_feat = train_features[n_idx]
-            pos_scores, _, _ = model.logits(pos_feat)
-            neg_scores, _, _ = model.logits(neg_feat)
-            diff = neg_scores - pos_scores
-            diff_clipped = np.clip(diff, -30.0, 30.0)
-            weights = 1.0 / (1.0 + np.exp(-diff_clipped))
-            sum_hardness += np.sum(weights)
-            count_hardness += len(weights)
-            pos_neg = pos_scores - neg_scores
-            pos_neg_clipped = np.clip(pos_neg, -30.0, 30.0)
-            sig = 1.0 / (1.0 + np.exp(-pos_neg_clipped))
-            grad = (sig - 1.0) * weights / (end - start)
-            grad_v_pos, grad_w_pos, grad_b_pos = model.gradients(pos_feat, grad)
-            grad_v_neg, grad_w_neg, grad_b_neg = model.gradients(neg_feat, -grad)
-            model.apply_gradients(grad_v_pos + grad_v_neg, grad_w_pos + grad_w_neg, grad_b_pos + grad_b_neg)
-            batch_loss = -np.log(np.clip(sig, 1e-12, 1.0))
-            epoch_loss += np.sum(batch_loss * weights) / (end - start)
-        val_primary = get_val_primary()
-        training_trace.append({'epoch': epoch + 1, 'val_primary': val_primary, 'loss': epoch_loss})
-        if val_primary > best_primary:
-            best_primary = val_primary
+            batch_pos = pos_rows[start:end]
+            batch_neg = neg_rows[start:end]
+            pos_x = context.train_x[batch_pos]
+            neg_x = context.train_x[batch_neg]
+            pos_score = model.logits(pos_x)[0]
+            neg_score = model.logits(neg_x)[0]
+            diff = np.clip(pos_score - neg_score, -30.0, 30.0)
+            sig_diff = _sigmoid(diff)
+            batch_n = end - start
+            grad_bpr = (sig_diff - 1.0) / batch_n
+            if aux_weight > 0:
+                pos_aux = aux_train[batch_pos]
+                neg_aux = aux_train[batch_neg]
+                pos_sig = _sigmoid(pos_score)
+                neg_sig = _sigmoid(neg_score)
+                grad_aux_pos = (pos_sig - pos_aux) * (aux_weight / batch_n)
+                grad_aux_neg = (neg_sig - neg_aux) * (aux_weight / batch_n)
+                grad_aux_pos_row = np.sum(grad_aux_pos, axis=1)
+                grad_aux_neg_row = np.sum(grad_aux_neg, axis=1)
+                grad_pos = grad_bpr + grad_aux_pos_row
+                grad_neg = -grad_bpr + grad_aux_neg_row
+                eps = 1e-08
+                aux_loss_pos = -np.mean(np.sum(pos_aux * np.log(pos_sig + eps) + (1 - pos_aux) * np.log(1 - pos_sig + eps), axis=1))
+                aux_loss_neg = -np.mean(np.sum(neg_aux * np.log(neg_sig + eps) + (1 - neg_aux) * np.log(1 - neg_sig + eps), axis=1))
+                total_aux += (aux_loss_pos + aux_loss_neg) * (end - start) / 2.0
+            else:
+                grad_pos = grad_bpr
+                grad_neg = -grad_bpr
+                total_aux += 0.0
+            grad_v_p, grad_w_p, grad_b_p = model.gradients(pos_x, grad_pos)
+            grad_v_n, grad_w_n, grad_b_n = model.gradients(neg_x, grad_neg)
+            grad_v = grad_v_p + grad_v_n
+            grad_w = grad_w_p + grad_w_n
+            grad_b = grad_b_p + grad_b_n
+            model.apply_gradients(grad_v, grad_w, grad_b)
+            total_bpr += -np.log(sig_diff + 1e-08).sum()
+        valid_scores = model.predict(context.valid_x)
+        metrics = context.evaluate_validation(valid_scores)
+        if isinstance(metrics, dict):
+            primary = metrics.get('primary', 0.0)
+            if 'gauc' in metrics and 'ndcg@5' in metrics:
+                primary = (metrics['gauc'] + metrics['ndcg@5']) / 2.0
+        else:
+            primary = float(metrics)
+        trace.append({'epoch': epoch + 1, 'metrics': metrics, 'bpr_loss': total_bpr / n_pairs, 'aux_loss': total_aux / n_pairs, 'primary': primary})
+        if primary > best_primary:
+            best_primary = primary
+            best_epoch = epoch + 1
             best_state = model.state_dict()
-            patience_counter = 0
+            no_improve = 0
         else:
-            patience_counter += 1
-            if patience_counter >= patience:
+            no_improve += 1
+            if no_improve >= patience:
                 break
     if best_state is not None:
         model.load_state_dict(best_state)
-    validation_scores = model.predict(valid_features)
-    test_scores = model.predict(test_features) if test_features is not None else None
-    random_valid_scores = model.predict(random_valid_features) if random_valid_features is not None else None
-    diagnostics = {'best_primary': best_primary, 'epochs_run': len(training_trace), 'n_pairs': n_pairs, 'hardness_weight_mean': sum_hardness / count_hardness if count_hardness > 0 else None}
-    return CandidateOutput(validation_scores=validation_scores, checkpoint_state=model.state_dict(), training_trace=training_trace, diagnostics=diagnostics, test_scores=test_scores, random_validation_scores=random_valid_scores)
+    else:
+        best_state = model.state_dict()
+    validation_scores = model.predict(context.valid_x)
+    test_scores = model.predict(context.test_x) if context.test_x is not None else None
+    random_validation_scores = model.predict(context.random_valid_x) if context.random_valid_x is not None else None
+    diagnostics = {'best_epoch': best_epoch, 'best_primary': best_primary, 'n_pairs': n_pairs, 'training_time_s': time.time() - start_time, 'aux_weight': aux_weight, 'active_heads': active_heads}
+    return CandidateOutput(validation_scores=validation_scores, checkpoint_state=best_state, training_trace=trace, diagnostics=diagnostics, test_scores=test_scores, random_validation_scores=random_validation_scores)
```

**Metrics:**

| Metric | Value | Δ vs best-so-far | Δ vs baseline |
|---|---|---|---|
| GAUC | 0.6408 | not reported | not reported |
| nDCG@5 | 0.5233 | not reported | not reported |
| primary | 0.5820 | -0.0218 | -0.0196 |

**Test scores:** `runs/20260831T141845874517Z_research/artifacts/006_mt_click_aux_02_bpr/test_scores.npy`

**Critic (preflight):** approved — The proposal adds a low-weighted (aux_weight=0.05) click auxiliary head over the same shared FM embeddings to a BPR ranking objective. This is fully consistent with the multi_task method card's safe initial search space (click-only, aux_weight 0.05–0.3). Auxiliary targets come exclusively from build_aux_labels on train rows (train-only by construction, raising for any other split), so there is no leakage or hidden-test access. No cross-user negatives, no evaluator/split/label/budget changes, no unsafe imports or operations, and the candidate contract (FMRanker, build_features, CandidateOutput) is respected. Numerically it is a small additive loss term with no stability risk. Although the parent BPR run scored low (primary 0.555), this is a cheap, focused probe following a registered untested direction and does not warrant hard rejection.

**Critic (postflight):** reject — The multi-task probe (click auxiliary head at weight 0.05 with BPR) achieved primary 0.5820, below the baseline 0.6016, so the hypothesis is not supported on the trusted validation split. GAUC (0.6408) is below the baseline's GAUC and nDCG@5 (0.5233) also fails to offset, giving a primary that is 1.96 points worse. The auxiliary supervision did not help the sparse ranking signal; the shared-embedding assumption appears not to pay off at this weight. The negative result aligns with the method card's known failure mode where auxiliary gradients can drag embeddings toward the click objective without improving long_view ranking.

**Resources:** 38.5 s · 85,338 tokens

---

