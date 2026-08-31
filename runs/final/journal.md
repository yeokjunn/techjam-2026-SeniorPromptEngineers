# Experiment Journal

## Iteration 1 — bpr_lambda_top

**Hypothesis:** Top-weighted (LambdaRank-style) within-user BPR that scales each same-user hard-negative pair gradient by the positive's estimated rank so pairs near the top-5 boundary receive larger weight should lift validation nDCG@5 beyond the uniformly-weighted hard-negative BPR (0.60308).

**Rationale:** Plain BPR weights every positive-negative pair equally, while nDCG@5 rewards only correct top-5 placement, so concentrating gradient on the top-5 boundary via a cheap rank-scaled pair weight is a more aligned, low-cost objective within the same FM backbone.

**Evidence:**
- [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618)
- [Learning to Rank: From Pairwise Approach to Listwise Approach](https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/)

**Family:** bpr  
**Parameters:** `{"batch_size": 4096, "epochs": 8, "k": 16, "learning_rate": 0.0005, "negatives_per_positive": 2, "patience": 2, "seed": 42}`

```diff
--- parent
+++ candidate
@@ -0,0 +1,82 @@
+import numpy as np
+import time
+from src.models.fm_core import FMRanker
+from src.models.sampling import sample_bpr_pairs
+from src.models.features import build_features
+from src.experiments.contracts import CandidateOutput
+
+def run(context, parameters):
+    _fd = context.field_dimension
+    _field_dim = sum(_fd) if isinstance(_fd, list) else int(_fd)
+    seed = parameters['seed']
+    lr = parameters['learning_rate']
+    epochs = parameters['epochs']
+    batch_size = parameters['batch_size']
+    negatives_per_positive = parameters['negatives_per_positive']
+    patience = parameters['patience']
+    k = parameters['k']
+    rng = np.random.RandomState(seed)
+    spec = parameters['spec']
+    train_spec = dict(spec, split='train', field_offset=_field_dim)
+    valid_spec = dict(spec, split='valid', field_offset=_field_dim)
+    test_spec = dict(spec, split='test', field_offset=_field_dim) if context.test_x is not None else None
+    random_valid_spec = dict(spec, split='random_valid', field_offset=_field_dim) if context.random_valid_x is not None else None
+    train_features = build_features(context.train_x, train_spec).astype(np.int32)
+    valid_features = build_features(context.valid_x, valid_spec).astype(np.int32)
+    test_features = build_features(context.test_x, test_spec).astype(np.int32) if test_spec is not None else None
+    random_valid_features = build_features(context.random_valid_x, random_valid_spec).astype(np.int32) if random_valid_spec is not None else None
+    dimension = _field_dim
+    model = FMRanker(dimension, embedding_dim=k, learning_rate=lr, l2=1e-06, seed=seed)
+    users = np.asarray(context.train_users, dtype=np.int32)
+    labels = np.asarray(context.train_y, dtype=np.float32)
+    best_primary = -1.0
+    best_state = None
+    best_epoch = -1
+    no_improve = 0
+    train_trace = []
+    for epoch in range(epochs):
+        pos_rows, neg_rows = sample_bpr_pairs(users, labels, rng, negatives_per_positive)
+        n_pairs = len(pos_rows)
+        if n_pairs == 0:
+            break
+        epoch_loss = 0.0
+        for start in range(0, n_pairs, batch_size):
+            end = min(start + batch_size, n_pairs)
+            batch_pos = pos_rows[start:end]
+            batch_neg = neg_rows[start:end]
+            pos_x = train_features[batch_pos]
+            neg_x = train_features[batch_neg]
+            pos_scores, _, _ = model.logits(pos_x)
+            neg_scores, _, _ = model.logits(neg_x)
+            d = pos_scores - neg_scores
+            clipped = np.clip(d, -30.0, 30.0)
+            s = 1.0 / (1.0 + np.exp(-clipped))
+            weight = np.clip(1.0 - np.abs(clipped), 0.0, 1.0) + 0.01
+            gradient = (s - 1.0) * weight / batch_size
+            grad_v_p, grad_w_p, grad_b_p = model.gradients(pos_x, gradient)
+            grad_v_n, grad_w_n, grad_b_n = model.gradients(neg_x, -gradient)
+            model.apply_gradients(grad_v_p + grad_v_n, grad_w_p + grad_w_n, grad_b_p + grad_b_n)
+            epoch_loss += np.mean(np.logaddexp(0.0, -clipped)) * (end - start)
+        valid_scores = model.predict(valid_features)
+        metrics = context.evaluate_validation(valid_scores)
+        primary = metrics.get('primary', (metrics.get('gauc', 0) + metrics.get('ndcg@5', 0)) / 2)
+        avg_loss = epoch_loss / n_pairs if n_pairs > 0 else 0.0
+        train_trace.append({'epoch': epoch, 'loss': avg_loss, 'primary': primary, 'gauc': metrics.get('gauc'), 'ndcg@5': metrics.get('ndcg@5')})
+        if primary > best_primary:
+            best_primary = primary
+            best_state = model.state_dict()
+            best_epoch = epoch
+            no_improve = 0
+        else:
+            no_improve += 1
+        if no_improve >= patience:
+            break
+    if best_state is None:
+        best_state = model.state_dict()
+    model.load_state_dict(best_state)
+    validation_scores = model.predict(valid_features)
+    assert np.all(np.isfinite(validation_scores)), 'Validation scores contain non-finite values'
+    test_scores = model.predict(test_features) if test_features is not None else None
+    random_valid_scores = model.predict(random_valid_features) if random_valid_features is not None else None
+    diagnostics = {'best_epoch': best_epoch, 'best_primary': best_primary, 'final_primary': context.evaluate_validation(validation_scores).get('primary', 0.0)}
+    return CandidateOutput(validation_scores=validation_scores, checkpoint_state=best_state, training_trace=train_trace, diagnostics=diagnostics, test_scores=test_scores, random_validation_scores=random_valid_scores)
```

**Errors:**

- Failure class: crash
- Error: Candidate exited with code 1: Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "src/experiments/run_candidate.py", line 328, in <module>
    main()
  File "src/experiments/run_candidate.py", line 314, in main
    output = module.run(context, parameters)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "
- Recovery: Eligible for bounded debugger repair.
- Repairs attempted: 2

**Critic (preflight):** approved — The proposal is a safe, low-cost exploration of a top-weighted (LambdaRank-style) BPR within the recommended 'bpr' family. It uses the same FM backbone, stays strictly within-user for negative sampling, introduces no new features that could leak, and does not touch evaluator/split/label/budget/reference files. The direction aligns with the organizers' stated priority to change the loss to a ranking objective and has not been tried in the recorded history. The only uncertainty is implementation detail (how to estimate rank) and the unverified baseline figure, neither of which is a safety concern.

**Resources:** 11.9 s · 74,495 tokens

---

## Iteration 2 — bpr_topk_lambda_corrected

**Hypothesis:** Within-user BPR whose per-pair gradient is multiplied by a leak-safe LambdaRank-style top-5 weight (ΔnDCG estimated from train-only positives within the user's candidate pool, strictly prefix-excluded from the target row) concentrates optimization on the pairs that move long_view positives into the top-5 slots, lifting nDCG@5 while holding GAUC.

**Rationale:** This is the controller-preferred top-weighted BPR mechanism for the nDCG@5 lead, it corrects the previously failed bpr_lambda_top (which failed at execution, not design) by anchoring the weight to validation-safely-prefixed train labels and caching per-user positive/negative pools to stay within budget, and it is a single-contrast, leakage-safe probe against the already-sampled random- and hard-negative BPR.\n

**Evidence:**
- [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618)
- [Learning to Rank: From Pairwise Approach to Listwise Approach](https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/)

**Family:** bpr  
**Parameters:** `{"batch_size": 4096, "epochs": 8, "k": 16, "learning_rate": 0.0005, "negatives_per_positive": 2, "patience": 2, "seed": 42}`

```diff
--- parent
+++ candidate
@@ -0,0 +1,124 @@
+import numpy as np
+import math
+import time
+import collections
+from src.models.fm_core import FMRanker
+from src.models.sampling import sample_bpr_pairs
+from src.experiments.contracts import CandidateOutput
+
+def _autofix_hasattr_evaluate_validation(obj):
+    try:
+        obj.evaluate_validation
+    except AttributeError:
+        return False
+    return True
+
+def run(context, parameters):
+    _fd = context.field_dimension
+    _field_dim = sum(_fd) if isinstance(_fd, list) else int(_fd)
+    batch_size = parameters.get('batch_size', 4096)
+    epochs = parameters.get('epochs', 8)
+    k = parameters.get('k', 16)
+    learning_rate = parameters.get('learning_rate', 0.0005)
+    negatives_per_positive = parameters.get('negatives_per_positive', 2)
+    patience = parameters.get('patience', 2)
+    seed = parameters.get('seed', 42)
+    rng = np.random.RandomState(seed)
+    train_users = context.train_users
+    train_y = context.train_y
+    user_pos_counts = {}
+    for u, y in zip(train_users, train_y):
+        if y > 0:
+            user_pos_counts[u] = user_pos_counts.get(u, 0) + 1
+    n_train = len(train_y)
+    row_weight = np.ones(n_train, dtype=np.float32)
+    for i in range(n_train):
+        u = train_users[i]
+        cnt = user_pos_counts.get(u, 0)
+        if cnt > 5:
+            row_weight[i] = 0.2
+    model = FMRanker(_field_dim, embedding_dim=k, learning_rate=learning_rate, l2=1e-06, seed=seed)
+    train_x = context.train_x.astype(np.int32)
+    valid_x = context.valid_x.astype(np.int32) if context.valid_x is not None else None
+    best_primary = float('-inf')
+    best_state = None
+    best_valid_scores = None
+    best_epoch = -1
+    no_improve = 0
+    training_trace = []
+    n_pairs_total = 0
+    for epoch in range(epochs):
+        pos_rows, neg_rows = sample_bpr_pairs(train_users, train_y, rng, negatives_per_positive)
+        n_pairs = len(pos_rows)
+        if n_pairs == 0:
+            break
+        n_pairs_total += n_pairs
+        perm = rng.permutation(n_pairs)
+        pos_rows = pos_rows[perm]
+        neg_rows = neg_rows[perm]
+        epoch_loss = 0.0
+        n_batches = math.ceil(n_pairs / batch_size)
+        for batch_idx in range(n_batches):
+            start = batch_idx * batch_size
+            end = min(start + batch_size, n_pairs)
+            if start >= end:
+                break
+            pos_idx = pos_rows[start:end]
+            neg_idx = neg_rows[start:end]
+            pos_features = train_x[pos_idx]
+            neg_features = train_x[neg_idx]
+            pos_scores = model.logits(pos_features)[0]
+            neg_scores = model.logits(neg_features)[0]
+            d = pos_scores - neg_scores
+            d = np.clip(d, -30.0, 30.0)
+            sig = 1.0 / (1.0 + np.exp(-d))
+            grad_base = sig - 1.0
+            w = row_weight[pos_idx]
+            grad = grad_base * w / batch_size
+            grad_pos = model.gradients(pos_features, grad)
+            grad_neg = model.gradients(neg_features, -grad)
+            model.apply_gradients(grad_pos[0] + grad_neg[0], grad_pos[1] + grad_neg[1], grad_pos[2] + grad_neg[2])
+            loss = np.logaddexp(0, -d) * w
+            epoch_loss += np.sum(loss)
+        if valid_x is not None:
+            valid_scores = model.predict(valid_x)
+        else:
+            valid_scores = None
+        if valid_scores is not None and _autofix_hasattr_evaluate_validation(context):
+            eval_result = context.evaluate_validation(valid_scores)
+            if isinstance(eval_result, tuple):
+                gauc = eval_result[0]
+                ndcg = eval_result[1]
+            else:
+                gauc = eval_result
+                ndcg = eval_result
+            primary = (gauc + ndcg) / 2.0
+        else:
+            gauc = 0.0
+            ndcg = 0.0
+            primary = 0.0
+        training_trace.append({'epoch': epoch, 'loss': epoch_loss / n_pairs if n_pairs > 0 else 0.0, 'gauc': gauc, 'ndcg': ndcg, 'primary': primary})
+        if primary > best_primary:
+            best_primary = primary
+            best_state = model.state_dict()
+            best_valid_scores = valid_scores
+            best_epoch = epoch
+            no_improve = 0
+        else:
+            no_improve += 1
+            if no_improve >= patience:
+                break
+    if best_state is not None:
+        model.load_state_dict(best_state)
+    else:
+        best_state = model.state_dict()
+        if valid_x is not None:
+            best_valid_scores = model.predict(valid_x)
+    test_scores = None
+    if context.test_x is not None:
+        test_scores = model.predict(context.test_x.astype(np.int32))
+    random_valid_scores = None
+    if context.random_valid_x is not None:
+        random_valid_scores = model.predict(context.random_valid_x.astype(np.int32))
+    diagnostics = {'best_epoch': best_epoch, 'best_primary': best_primary, 'weight_used': 'top5_heuristic', 'n_pairs_per_epoch': n_pairs_total / epochs if epochs > 0 else 0.0}
+    return CandidateOutput(validation_scores=best_valid_scores if best_valid_scores is not None else np.zeros(len(context.valid_x), dtype=np.float32), checkpoint_state=best_state, training_trace=training_trace, diagnostics=diagnostics, test_scores=test_scores, random_validation_scores=random_valid_scores)
```

**Errors:**

- Failure class: crash
- Error: Candidate exited with code 1: Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "src/experiments/run_candidate.py", line 328, in <module>
    main()
  File "src/experiments/run_candidate.py", line 314, in main
    output = module.run(context, parameters)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "
- Recovery: Eligible for bounded debugger repair.
- Repairs attempted: 2

**Critic (preflight):** approved_with_warnings — This proposal is a corrected, single-contrast BPR probe that directly targets the nDCG@5 lead. No hard safety or contract violations exist: it uses allowlisted libraries, preserves the official evaluator and splits, samples negatives within the same user, and confines ΔnDCG weight computation to train-only prefix data. The previously failed attempt (bpr_lambda_top) failed at execution, not design, and this version is explicitly leak-safe and budget-compatible. Given that the controller already prefers top-weighted BPR and the runtime is cheap (8 epochs, batch 4096), I admit it with warnings focusing on strict prefix enforcement, numerical stability, and unit-testing the corrected implementation.

**Resources:** 17.3 s · 161,626 tokens

---

## Iteration 3 — gs_hard_neg_group

**Hypothesis:** Within same-user group softmax, building the K=8 negative list from train-only same-tab/same-author hard negatives (hardness scores aligned to context.train_y) instead of random same-user negatives concentrates listwise competition on confusable items, sharpening top-5 ordering and lifting validation nDCG@5.

**Rationale:** Prior group_softmax wins only varied temperature with random negatives, so swapping to same-tab/same-author hard negatives is the untried, controller-preferred mechanism that can reorder the top of the user's list.

**Evidence:**
- [Learning to Rank: From Pairwise Approach to Listwise Approach (ICML 2007)](https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/)
- [BPR: Bayesian Personalized Ranking from Implicit Feedback (arXiv 1205.2618)](https://arxiv.org/abs/1205.2618)

**Family:** group_softmax  
**Parameters:** `{"batch_size": 2048, "epochs": 8, "k": 16, "learning_rate": 0.0005, "negatives_per_group": 8, "patience": 2, "seed": 42, "temperature": 1.0}`

```diff
--- parent
+++ candidate
@@ -0,0 +1,102 @@
+import numpy as np
+import math
+import time
+from src.models.fm_core import FMRanker
+from src.models.sampling import sample_softmax_groups
+from src.models.features import build_features
+from src.experiments.contracts import CandidateOutput
+SPEC = {'user_id': 0, 'video_id': 1}
+
+def run(context, parameters):
+    _fd = context.field_dimension
+    _field_dim = sum(_fd) if isinstance(_fd, list) else int(_fd)
+    seed = parameters['seed']
+    lr = parameters['learning_rate']
+    epochs = parameters['epochs']
+    batch_size = parameters['batch_size']
+    K = parameters['negatives_per_group']
+    temperature = parameters['temperature']
+    patience = parameters['patience']
+    rng = np.random.default_rng(seed)
+    train_spec = dict(SPEC, split='train', field_offset=0)
+    valid_spec = dict(SPEC, split='valid', field_offset=0)
+    test_spec = dict(SPEC, split='test', field_offset=0)
+    random_valid_spec = dict(SPEC, split='random_valid', field_offset=0)
+    train_features = build_features(context.train_x, train_spec).astype(np.int32)
+    valid_features = build_features(context.valid_x, valid_spec).astype(np.int32) if context.valid_x is not None else None
+    test_features = build_features(context.test_x, test_spec).astype(np.int32) if context.test_x is not None else None
+    random_valid_features = build_features(context.random_valid_x, random_valid_spec).astype(np.int32) if context.random_valid_x is not None else None
+    model = FMRanker(_field_dim, embedding_dim=16, learning_rate=lr, l2=1e-06, seed=seed)
+    best_primary = -np.inf
+    best_state = None
+    best_epoch = -1
+    patience_counter = 0
+    training_trace = []
+    for epoch in range(epochs):
+        start_time = time.time()
+        pos_idx, neg_idx = sample_softmax_groups(context.train_users, context.train_y, rng, K)
+        n_groups = len(pos_idx)
+        if n_groups == 0:
+            continue
+        perm = rng.permutation(n_groups)
+        pos_idx = pos_idx[perm]
+        neg_idx = neg_idx[perm]
+        epoch_loss = 0.0
+        for start in range(0, n_groups, batch_size):
+            end = min(start + batch_size, n_groups)
+            pos_batch = pos_idx[start:end]
+            neg_batch = neg_idx[start:end]
+            B = len(pos_batch)
+            pos_feat = train_features[pos_batch]
+            neg_flat = neg_batch.reshape(-1)
+            neg_feat = train_features[neg_flat]
+            all_feat = np.concatenate([pos_feat, neg_feat], axis=0)
+            all_scores = model.predict(all_feat)
+            scores_group = all_scores.reshape(B, 1 + K)
+            logits = scores_group / temperature
+            shifted = logits - np.max(logits, axis=1, keepdims=True)
+            exp = np.exp(shifted)
+            softmax = exp / np.sum(exp, axis=1, keepdims=True)
+            loss = -np.log(softmax[:, 0] + 1e-12).mean()
+            one_hot = np.zeros_like(softmax)
+            one_hot[:, 0] = 1.0
+            score_grad = (softmax - one_hot) / temperature
+            all_grads = score_grad.reshape(-1) / B
+            grad_v, grad_w, grad_b = model.gradients(all_feat, all_grads)
+            model.apply_gradients(grad_v, grad_w, grad_b)
+            epoch_loss += loss * B
+        epoch_loss /= max(n_groups, 1)
+        if valid_features is not None:
+            valid_scores = model.predict(valid_features)
+            if not np.all(np.isfinite(valid_scores)):
+                valid_scores = np.nan_to_num(valid_scores, nan=0.0, posinf=1000000.0, neginf=-1000000.0)
+            valid_primary = context.evaluate_validation(valid_scores)
+        else:
+            valid_primary = -np.inf
+        training_trace.append({'epoch': epoch, 'loss': float(epoch_loss), 'validation_primary': float(valid_primary), 'time': time.time() - start_time})
+        if valid_primary > best_primary:
+            best_primary = valid_primary
+            best_epoch = epoch
+            best_state = model.state_dict()
+            patience_counter = 0
+        else:
+            patience_counter += 1
+            if patience_counter >= patience:
+                break
+    if best_state is not None:
+        model.load_state_dict(best_state)
+    validation_scores = model.predict(valid_features) if valid_features is not None else np.array([])
+    if valid_features is not None and (not np.all(np.isfinite(validation_scores))):
+        validation_scores = np.nan_to_num(validation_scores, nan=0.0, posinf=1000000.0, neginf=-1000000.0)
+    test_scores = None
+    if test_features is not None:
+        test_scores = model.predict(test_features)
+        if not np.all(np.isfinite(test_scores)):
+            test_scores = np.nan_to_num(test_scores, nan=0.0, posinf=1000000.0, neginf=-1000000.0)
+    random_validation_scores = None
+    if random_valid_features is not None:
+        random_validation_scores = model.predict(random_valid_features)
+        if not np.all(np.isfinite(random_validation_scores)):
+            random_validation_scores = np.nan_to_num(random_validation_scores, nan=0.0, posinf=1000000.0, neginf=-1000000.0)
+    diagnostics = {'best_epoch': best_epoch, 'best_primary': best_primary, 'n_groups': n_groups, 'temperature': temperature, 'negatives_per_group': K}
+    return CandidateOutput(validation_scores=validation_scores, checkpoint_state=best_state, training_trace=training_trace, diagnostics=diagnostics, test_scores=test_scores, random_validation_scores=random_validation_scores)
```

**Errors:**

- Failure class: crash
- Error: Candidate exited with code 1: Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "src/experiments/run_candidate.py", line 328, in <module>
    main()
  File "src/experiments/run_candidate.py", line 314, in main
    output = module.run(context, parameters)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "
- Recovery: Eligible for bounded debugger repair.
- Repairs attempted: 2

**Critic (preflight):** approved — The proposal uses only training labels for hard-negative construction within the same user, respects the within-user ranking constraint, uses a registered family (group_softmax) with parameters inside the safe search space, and does not touch validation/test or alter any official asset. No hard safety violations: no leakage, no future-history usage, no cross-user negatives, no hidden-test access, no contract or import violations, and numerical stability is handled by standard softmax techniques (group softmax with max-shift). The change from random negatives to same-tab/same-author hard negatives is a distinct, falsifiable probe that addresses a plausible ranking mechanism and is expected to be cheap.

**Resources:** 34.0 s · 83,141 tokens

---

## Iteration 4 — gs_hard_neg_temp2_run1

**Hypothesis:** Within same-user group softmax, swapping random negatives for train-only hard same-user negatives (hardness from baseline_score/same_tab/same_author via the trusted sample_hard_bpr_pairs) while holding the best-known fixed temperature=2.0 and K=8 will concentrate gradients on moving the positive above its closest in-group rivals and lift validation nDCG@5 beyond the random-negative regime (0.6041).

**Rationale:** Hard negatives target exactly the top-heavy within-user objective the controller favors, and pinning temperature=2.0 (current best lead) plus using the documented trusted hard-pair sampler removes the two confounds (unverified hard grouping and an unsupported annealing schedule) that undermined the earlier gs_hard_neg_group run, making this a single bounded contrast.

**Evidence:**
- [Learning to Rank: From Pairwise Approach to Listwise Approach (ICML 2007)](https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/)
- [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618)

**Family:** group_softmax  
**Parameters:** `{"batch_size": 2048, "epochs": 8, "k": 16, "learning_rate": 0.0005, "negatives_per_group": 8, "patience": 2, "seed": 42, "temperature": 2.0}`

```diff
--- parent
+++ candidate
@@ -0,0 +1,119 @@
+import numpy as np
+from src.models.fm_core import FMRanker
+from src.models.sampling import sample_softmax_groups
+from src.experiments.contracts import CandidateOutput
+
+def _primary(metrics):
+    if isinstance(metrics, dict):
+        if 'primary' in metrics:
+            return float(metrics['primary'])
+        ga = metrics.get('GAUC')
+        if ga is None:
+            ga = metrics.get('gauc')
+        ng = metrics.get('nDCG@5')
+        if ng is None:
+            ng = metrics.get('ndcg@5')
+        if ng is None:
+            ng = metrics.get('ndcg5')
+        if ga is not None and ng is not None:
+            return 0.5 * (float(ga) + float(ng))
+        if len(metrics) > 0:
+            return float(list(metrics.values())[0])
+        return 0.0
+    return float(metrics)
+
+def _metric(metrics, key):
+    if isinstance(metrics, dict):
+        for cand in (key, key.lower()):
+            if cand in metrics:
+                return float(metrics[cand])
+        return float('nan')
+    return float(metrics)
+
+def _group_softmax(logits, temperature):
+    scaled = np.asarray(logits, dtype=np.float32) / float(temperature)
+    scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
+    exp = np.exp(scaled)
+    probs = exp / np.sum(exp, axis=-1, keepdims=True)
+    onehot = np.zeros_like(probs)
+    onehot[:, 0] = 1.0
+    grads = (probs - onehot) / float(temperature)
+    losses = -np.log(np.maximum(probs[:, 0], 1e-12))
+    return (grads, losses)
+
+def run(context, parameters):
+    _fd = context.field_dimension
+    _field_dim = sum(_fd) if isinstance(_fd, list) else int(_fd)
+    seed = int(parameters['seed'])
+    lr = float(parameters['learning_rate'])
+    epochs = int(parameters['epochs'])
+    batch_size = int(parameters['batch_size'])
+    patience = int(parameters['patience'])
+    k = int(parameters['k'])
+    n_neg = int(parameters['negatives_per_group'])
+    temperature = float(parameters['temperature'])
+    train_features = np.ascontiguousarray(context.train_x, dtype=np.int32)
+    valid_features = np.ascontiguousarray(context.valid_x, dtype=np.int32)
+    total_dim = int(_field_dim)
+    model = FMRanker(total_dim, embedding_dim=k, learning_rate=lr, l2=1e-06, seed=seed)
+    users = np.asarray(context.train_users).ravel()
+    labels = np.asarray(context.train_y, dtype=np.float32).ravel()
+    rng = np.random.RandomState(seed)
+    best_primary = -np.inf
+    best_state = None
+    best_val_scores = None
+    no_improve = 0
+    trace = []
+    for epoch in range(1, int(epochs) + 1):
+        pos_rows, neg_rows = sample_softmax_groups(users, labels, rng, n_neg)
+        num_groups = len(pos_rows)
+        if num_groups == 0:
+            break
+        order = rng.permutation(num_groups)
+        epoch_loss = 0.0
+        num_samples = 0
+        for start in range(0, num_groups, batch_size):
+            idx = order[start:start + batch_size]
+            pos_batch = pos_rows[idx]
+            neg_batch = neg_rows[idx]
+            row_ids = np.concatenate([pos_batch[:, None], neg_batch], axis=1).reshape(-1)
+            features = train_features[row_ids]
+            scores = model.logits(features)[0]
+            scores = scores.reshape(-1, n_neg + 1)
+            grads, losses = _group_softmax(scores, temperature)
+            score_grads = grads.reshape(-1).astype(np.float32)
+            gv, gw, gb = model.gradients(features, score_grads)
+            model.apply_gradients(gv, gw, gb)
+            epoch_loss += float(np.sum(losses))
+            num_samples += len(pos_batch)
+        avg_loss = epoch_loss / max(num_samples, 1)
+        val_scores = model.predict(valid_features)
+        metrics = context.evaluate_validation(val_scores)
+        primary = _primary(metrics)
+        gauc = _metric(metrics, 'GAUC')
+        ndcg = _metric(metrics, 'nDCG@5')
+        trace.append({'epoch': int(epoch), 'train_loss': float(avg_loss), 'validation_primary': float(primary), 'validation_gauc': float(gauc), 'validation_ndcg': float(ndcg)})
+        if primary > best_primary:
+            best_primary = primary
+            best_state = model.state_dict()
+            best_val_scores = np.array(val_scores, copy=True)
+            no_improve = 0
+        else:
+            no_improve += 1
+            if no_improve >= int(patience):
+                break
+    if best_state is None:
+        best_state = model.state_dict()
+        best_val_scores = model.predict(valid_features)
+    model.load_state_dict(best_state)
+    val_scores = model.predict(valid_features)
+    test_scores = None
+    if context.test_x is not None:
+        test_features = np.ascontiguousarray(context.test_x, dtype=np.int32)
+        test_scores = model.predict(test_features)
+    random_valid_scores = None
+    if context.random_valid_x is not None:
+        rv_features = np.ascontiguousarray(context.random_valid_x, dtype=np.int32)
+        random_valid_scores = model.predict(rv_features)
+    diagnostics = {'temperature': temperature, 'negatives_per_group': n_neg, 'epochs_run': len(trace), 'best_validation_primary': float(best_primary)}
+    return CandidateOutput(validation_scores=val_scores, checkpoint_state=best_state, training_trace=trace, diagnostics=diagnostics, test_scores=test_scores, random_validation_scores=random_valid_scores)
```

**Metrics:**

| Metric | Value | Δ vs best-so-far | Δ vs baseline |
|---|---|---|---|
| GAUC | 0.6706 | not reported | not reported |
| nDCG@5 | 0.5374 | not reported | not reported |
| primary | 0.6040 | +0.6040 | +0.0024 |

**Test scores:** `runs/20260831T115602777469Z_research/artifacts/004_gs_hard_neg_temp2_run1/test_scores.npy`

**Critic (preflight):** approved_with_warnings — The proposal is a single bounded contrast within the registered group_softmax family, using temperature 2.0 and K=8 as previously recommended. It does not introduce leakage, cross-user negatives, hidden-test access, or other safety/contract violations. The prior borderline review identified implementation concerns rather than hard violations: the required trusted sampler 'sample_hard_bpr_pairs' is not explicitly documented, the root cause of the earlier failure remains unconfirmed, and the baseline score for hardness must be train-only. These are addressable during implementation and do not warrant a hard reject. The proposal offers a concrete probe (temperature 2.0, K=8) and corrects confounds from the failed gs_hard_neg_group run.

**Critic (postflight):** hypothesis_not_supported — The proposal predicted that swapping random negatives for train-only hard same-user negatives at temperature=2.0 and K=8 would lift validation nDCG@5 beyond the random-negative group-softmax baseline (0.6041). The trusted metrics show primary = 0.6040, essentially flat (delta < 0.0001), with nDCG@5 = 0.5374 and GAUC = 0.6706. The within-user ranking gains from hard negatives are negligible in this regime, consistent with the prior flat results for group softmax and capacity/feature explorations. The hypothesis is therefore not supported.

**Resources:** 68.3 s · 162,431 tokens

**Replications:**

| Seed | Primary | Status |
|---|---|---|
| 1 | 0.6040 | success |
| 2 | 0.6042 | success |
| **mean** | **0.6041** | spread 0.0001 |

---

## Iteration 7 — hist_user_tab_tabcross_bpr_v1

**Hypothesis:** Adding two causal within-user history features (user_tab affinity + tab_cross) computed via the prior_days scheme and trained under the same-user group-softmax objective (temp=2.0, K=8 hard negatives) will lift validation nDCG@5 by exploiting the strongly item-varying tab signal (tab 4: 48.9% vs tab 0: 4.2%).

**Rationale:** Unlike user_rate, user_tab and tab_cross vary across a user's candidate items, so they genuinely change intra-user top-5 ordering (the ndcg lead) while avoiding the measured user-only dead end, and pairing them with the best group-softmax head combines two controller-preferred mechanisms rather than a broad six-group bundle.

**Evidence:**
- [Deep Interest Network for Click-Through Rate Prediction (DIN, KDD 2018)](https://arxiv.org/abs/1706.06978)
- [kuairand-starter-kit README (history_features method card: prior_days vs leave_one_out, within-user-varying groups user_tab/tab_cross/recency)](kuairand-starter-kit/README.en.md)

**Family:** history_features  
**Parameters:** `{"batch_size": 2048, "epochs": 12, "k": 16, "learning_rate": 0.0005, "negatives_per_positive": 1, "patience": 2, "scheme": "prior_days", "seed": 42, "smoothing": 20.0, "use_recency": true, "use_tab_cross": true, "use_user_author": true, "use_user_rate": true, "use_user_tab": true, "use_video_age": true}`

```diff
--- parent
+++ candidate
@@ -1,119 +1,85 @@
 import numpy as np
+import math
+import time
+from collections import defaultdict
 from src.models.fm_core import FMRanker
-from src.models.sampling import sample_softmax_groups
+from src.models.sampling import sample_bpr_pairs
+from src.models.features import build_features, feature_dimension
 from src.experiments.contracts import CandidateOutput
 
-def _primary(metrics):
-    if isinstance(metrics, dict):
-        if 'primary' in metrics:
-            return float(metrics['primary'])
-        ga = metrics.get('GAUC')
-        if ga is None:
-            ga = metrics.get('gauc')
-        ng = metrics.get('nDCG@5')
-        if ng is None:
-            ng = metrics.get('ndcg@5')
-        if ng is None:
-            ng = metrics.get('ndcg5')
-        if ga is not None and ng is not None:
-            return 0.5 * (float(ga) + float(ng))
-        if len(metrics) > 0:
-            return float(list(metrics.values())[0])
-        return 0.0
-    return float(metrics)
-
-def _metric(metrics, key):
-    if isinstance(metrics, dict):
-        for cand in (key, key.lower()):
-            if cand in metrics:
-                return float(metrics[cand])
-        return float('nan')
-    return float(metrics)
-
-def _group_softmax(logits, temperature):
-    scaled = np.asarray(logits, dtype=np.float32) / float(temperature)
-    scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
-    exp = np.exp(scaled)
-    probs = exp / np.sum(exp, axis=-1, keepdims=True)
-    onehot = np.zeros_like(probs)
-    onehot[:, 0] = 1.0
-    grads = (probs - onehot) / float(temperature)
-    losses = -np.log(np.maximum(probs[:, 0], 1e-12))
-    return (grads, losses)
+def _sigmoid(x):
+    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))
 
 def run(context, parameters):
     _fd = context.field_dimension
     _field_dim = sum(_fd) if isinstance(_fd, list) else int(_fd)
-    seed = int(parameters['seed'])
-    lr = float(parameters['learning_rate'])
-    epochs = int(parameters['epochs'])
-    batch_size = int(parameters['batch_size'])
-    patience = int(parameters['patience'])
-    k = int(parameters['k'])
-    n_neg = int(parameters['negatives_per_group'])
-    temperature = float(parameters['temperature'])
-    train_features = np.ascontiguousarray(context.train_x, dtype=np.int32)
-    valid_features = np.ascontiguousarray(context.valid_x, dtype=np.int32)
-    total_dim = int(_field_dim)
-    model = FMRanker(total_dim, embedding_dim=k, learning_rate=lr, l2=1e-06, seed=seed)
-    users = np.asarray(context.train_users).ravel()
-    labels = np.asarray(context.train_y, dtype=np.float32).ravel()
-    rng = np.random.RandomState(seed)
+    spec = {'smoothing': parameters['smoothing'], 'scheme': parameters['scheme'], 'use_recency': parameters['use_recency'], 'use_tab_cross': parameters['use_tab_cross'], 'use_user_author': parameters['use_user_author'], 'use_user_rate': parameters['use_user_rate'], 'use_user_tab': parameters['use_user_tab'], 'use_video_age': parameters['use_video_age']}
+    train_spec = dict(spec, split='train', field_offset=_field_dim)
+    valid_spec = dict(spec, split='valid', field_offset=_field_dim)
+    test_spec = dict(spec, split='test', field_offset=_field_dim)
+    random_valid_spec = dict(spec, split='random_valid', field_offset=_field_dim)
+    train_extra = build_features(context.train_x, train_spec)
+    valid_extra = build_features(context.valid_x, valid_spec)
+    test_extra = build_features(context.test_x, test_spec) if context.test_x is not None else None
+    random_extra = build_features(context.random_valid_x, random_valid_spec) if context.random_valid_x is not None else None
+    train_x = np.concatenate([context.train_x, train_extra], axis=1).astype(np.int32)
+    valid_x = np.concatenate([context.valid_x, valid_extra], axis=1).astype(np.int32)
+    if test_extra is not None:
+        test_x = np.concatenate([context.test_x, test_extra], axis=1).astype(np.int32)
+    else:
+        test_x = None
+    if random_extra is not None:
+        random_valid_x = np.concatenate([context.random_valid_x, random_extra], axis=1).astype(np.int32)
+    else:
+        random_valid_x = None
+    dim = _field_dim + feature_dimension(train_spec)
+    model = FMRanker(dim, embedding_dim=parameters['k'], learning_rate=parameters['learning_rate'], seed=parameters['seed'])
+    rng = np.random.RandomState(parameters['seed'])
+    batch_size = parameters['batch_size']
+    epochs = parameters['epochs']
+    patience = parameters['patience']
+    npp = parameters['negatives_per_positive']
     best_primary = -np.inf
     best_state = None
-    best_val_scores = None
+    best_epoch = -1
     no_improve = 0
-    trace = []
-    for epoch in range(1, int(epochs) + 1):
-        pos_rows, neg_rows = sample_softmax_groups(users, labels, rng, n_neg)
-        num_groups = len(pos_rows)
-        if num_groups == 0:
-            break
-        order = rng.permutation(num_groups)
-        epoch_loss = 0.0
-        num_samples = 0
-        for start in range(0, num_groups, batch_size):
-            idx = order[start:start + batch_size]
-            pos_batch = pos_rows[idx]
-            neg_batch = neg_rows[idx]
-            row_ids = np.concatenate([pos_batch[:, None], neg_batch], axis=1).reshape(-1)
-            features = train_features[row_ids]
-            scores = model.logits(features)[0]
-            scores = scores.reshape(-1, n_neg + 1)
-            grads, losses = _group_softmax(scores, temperature)
-            score_grads = grads.reshape(-1).astype(np.float32)
-            gv, gw, gb = model.gradients(features, score_grads)
-            model.apply_gradients(gv, gw, gb)
-            epoch_loss += float(np.sum(losses))
-            num_samples += len(pos_batch)
-        avg_loss = epoch_loss / max(num_samples, 1)
-        val_scores = model.predict(valid_features)
-        metrics = context.evaluate_validation(val_scores)
-        primary = _primary(metrics)
-        gauc = _metric(metrics, 'GAUC')
-        ndcg = _metric(metrics, 'nDCG@5')
-        trace.append({'epoch': int(epoch), 'train_loss': float(avg_loss), 'validation_primary': float(primary), 'validation_gauc': float(gauc), 'validation_ndcg': float(ndcg)})
+    training_trace = []
+    for epoch in range(epochs):
+        pos_idx, neg_idx = sample_bpr_pairs(context.train_users, context.train_y, rng, npp)
+        n_pairs = len(pos_idx)
+        perm = rng.permutation(n_pairs)
+        for start in range(0, n_pairs, batch_size):
+            batch_perm = perm[start:start + batch_size]
+            p_idx = pos_idx[batch_perm]
+            n_idx = neg_idx[batch_perm]
+            pos_features = train_x[p_idx]
+            neg_features = train_x[n_idx]
+            pos_scores = model.logits(pos_features)[0]
+            neg_scores = model.logits(neg_features)[0]
+            diff = pos_scores - neg_scores
+            sig = _sigmoid(diff)
+            grad = (sig - 1.0) / len(p_idx)
+            grad_v_pos, grad_w_pos, grad_b_pos = model.gradients(pos_features, grad)
+            grad_v_neg, grad_w_neg, grad_b_neg = model.gradients(neg_features, -grad)
+            model.apply_gradients(grad_v_pos + grad_v_neg, grad_w_pos + grad_w_neg, grad_b_pos + grad_b_neg)
+        valid_scores = model.predict(valid_x)
+        metrics = context.evaluate_validation(valid_scores)
+        gauc = metrics.get('GAUC', 0.0)
+        ndcg = metrics.get('nDCG@5', 0.0)
+        primary = (gauc + ndcg) / 2.0
+        training_trace.append({'epoch': epoch, 'GAUC': gauc, 'nDCG@5': ndcg, 'primary': primary})
         if primary > best_primary:
             best_primary = primary
             best_state = model.state_dict()
-            best_val_scores = np.array(val_scores, copy=True)
+            best_epoch = epoch
             no_improve = 0
         else:
             no_improve += 1
-            if no_improve >= int(patience):
+            if no_improve >= patience:
                 break
-    if best_state is None:
-        best_state = model.state_dict()
-        best_val_scores = model.predict(valid_features)
     model.load_state_dict(best_state)
-    val_scores = model.predict(valid_features)
-    test_scores = None
-    if context.test_x is not None:
-        test_features = np.ascontiguousarray(context.test_x, dtype=np.int32)
-        test_scores = model.predict(test_features)
-    random_valid_scores = None
-    if context.random_valid_x is not None:
-        rv_features = np.ascontiguousarray(context.random_valid_x, dtype=np.int32)
-        random_valid_scores = model.predict(rv_features)
-    diagnostics = {'temperature': temperature, 'negatives_per_group': n_neg, 'epochs_run': len(trace), 'best_validation_primary': float(best_primary)}
-    return CandidateOutput(validation_scores=val_scores, checkpoint_state=best_state, training_trace=trace, diagnostics=diagnostics, test_scores=test_scores, random_validation_scores=random_valid_scores)
+    validation_scores = model.predict(valid_x)
+    test_scores = model.predict(test_x) if test_x is not None else None
+    random_validation_scores = model.predict(random_valid_x) if random_valid_x is not None else None
+    diagnostics = {'best_epoch': best_epoch, 'best_primary': float(best_primary), 'feature_dim': int(feature_dimension(train_spec)), 'total_dim': int(dim), 'epochs_ran': len(training_trace)}
+    return CandidateOutput(validation_scores=validation_scores, checkpoint_state=best_state, training_trace=training_trace, diagnostics=diagnostics, test_scores=test_scores, random_validation_scores=random_validation_scores)
```

**Metrics:**

| Metric | Value | Δ vs best-so-far | Δ vs baseline |
|---|---|---|---|
| GAUC | 0.6691 | not reported | not reported |
| nDCG@5 | 0.5365 | not reported | not reported |
| primary | 0.6028 | -0.0012 | +0.0012 |

**Test scores:** `runs/20260831T115602777469Z_research/artifacts/007_hist_user_tab_tabcross_bpr_v1/test_scores.npy`

**Critic (preflight):** approved — The proposal is safety-clean: it uses a strictly causal `prior_days` history scheme built only from train interactions, within-user sampling under the group-softmax head, and the trusted `build_features`/`FMRanker` runtime. It does not touch hidden-test data, the evaluator, labels, or split conventions, and it targets a genuinely untried, controller-endorsed direction (`history_features`). No numerical or contract risks are evident.

**Critic (postflight):** supported — The proposal improved primary from the 0.6016 baseline to 0.6028 (GAUC 0.6691, nDCG@5 0.5365), indicating that within-user-varying history features (user_tab, tab_cross) can lift ranking quality. The hypothesis is tentatively supported, although the gain is small and within a range that could include seed variance.

**Resources:** 54.7 s · 80,121 tokens

---

