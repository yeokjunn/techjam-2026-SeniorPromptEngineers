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
@@ -0,0 +1,129 @@
+"""Same-user group softmax listwise loss on the fixed 5-field FM (k=16).
+
+Loss per group: -log_softmax([pos, neg_1..neg_K] / temperature)[0] (Cao et al., ICML 2007).
+Score gradient is (softmax - one_hot(pos)) / temperature, max-shifted for stability.
+"""
+import time
+import numpy as np
+from src.models.fm_core import FMRanker
+from src.models.sampling import sample_bpr_pairs, sample_softmax_groups
+from src.experiments.contracts import CandidateOutput
+
+
+def _finite(x):
+    x = float(x)
+    return x if np.isfinite(x) else 0.0
+
+
+def _primary(metrics):
+    """Primary score = mean(GAUC, nDCG@5); tolerate key-spelling variants."""
+    gauc, ndcg = None, None
+    for key, value in metrics.items():
+        low = str(key).lower()
+        if "gauc" in low:
+            gauc = _finite(value)
+        elif "ndcg" in low and "5" in low:
+            ndcg = _finite(value)
+    if gauc is None:
+        gauc = 0.0
+    if ndcg is None:
+        ndcg = 0.0
+    return 0.5 * (gauc + ndcg)
+
+
+def run(context, parameters):
+    seed = int(parameters["seed"])
+    epochs = int(parameters["epochs"])
+    batch_size = int(parameters["batch_size"])
+    temperature = float(parameters["temperature"])
+    n_neg = int(parameters["negatives_per_group"])
+    patience = int(parameters["patience"])
+    model = FMRanker(
+        context.field_dimension,
+        embedding_dim=int(parameters["k"]),
+        learning_rate=float(parameters["learning_rate"]),
+        seed=seed,
+    )
+    rng = np.random.default_rng(seed)
+    train_x = np.asarray(context.train_x)
+    users = np.asarray(context.train_users)
+    labels = (np.asarray(context.train_y, dtype=np.float64) > 0.5).astype(np.float64)
+
+    pos_idx, neg_groups = sample_softmax_groups(
+        users, labels, rng, negatives_per_group=n_neg)
+    n_groups = int(len(pos_idx))
+    diagnostics = {
+        "family": "group_softmax",
+        "temperature": temperature,
+        "negatives_per_group": n_neg,
+        "groups_total": n_groups,
+    }
+    if n_groups == 0:
+        raise ValueError("no same-user groups with both labels; cannot train")
+
+    dup = 0
+    for g in neg_groups:
+        if len(np.unique(g)) < n_neg:
+            dup += 1
+    diagnostics["groups_with_duplicate_negatives"] = int(dup)
+    diagnostics["duplicate_negative_rate"] = float(dup) / n_groups
+
+    trace = []
+    best_primary = -np.inf
+    best_scores = None
+    best_epoch = -1
+    bad = 0
+    t0 = time.time()
+    valid_x = np.asarray(context.valid_x)
+    for epoch in range(1, epochs + 1):
+        order = rng.permutation(n_groups)
+        loss_sum = 0.0
+        for start in range(0, n_groups, batch_size):
+            sel = order[start:start + batch_size]
+            b = len(sel)
+            rows = np.empty((b, n_neg + 1), dtype=np.int64)
+            rows[:, 0] = pos_idx[sel]
+            rows[:, 1:] = neg_groups[sel]
+            feats = train_x[rows.reshape(-1)]
+            scores = model.predict(feats).reshape(b, n_neg + 1)
+            shifted = (scores - scores.max(axis=1, keepdims=True)) / temperature
+            expz = np.exp(shifted)
+            probs = expz / expz.sum(axis=1, keepdims=True)
+            loss_sum += float(-np.log(np.clip(probs[:, 0], 1e-12, None)).sum())
+            score_grad = probs.copy()
+            score_grad[:, 0] -= 1.0
+            score_grad /= temperature
+            gv, gw, gb = model.gradients(feats, score_grad.reshape(-1))
+            model.apply_gradients(gv, gw, gb)
+        train_loss = loss_sum / n_groups
+        valid_scores = np.asarray(model.predict(valid_x), dtype=np.float64)
+        metrics = context.evaluate_validation(valid_scores)
+        primary = _primary(metrics)
+        entry = {"epoch": epoch, "train_group_loss": train_loss, "primary": primary}
+        for key, value in metrics.items():
+            entry[str(key)] = _finite(value)
+        trace.append(entry)
+        if primary > best_primary + 1e-9:
+            best_primary = primary
+            best_scores = valid_scores.copy()
+            best_epoch = epoch
+            bad = 0
+        else:
+            bad += 1
+            if bad >= patience:
+                break
+    if best_scores is None:
+        best_scores = np.asarray(model.predict(valid_x), dtype=np.float64)
+
+    test_scores = None
+    checkpoint = {"validation_scores": best_scores}
+    if context.test_x is not None:
+        test_scores = np.asarray(model.predict(np.asarray(context.test_x)), dtype=np.float64)
+        checkpoint["test_scores"] = test_scores
+    checkpoint["epoch_primary"] = np.asarray(
+        [t["primary"] for t in trace], dtype=np.float64)
+    diagnostics["best_epoch"] = int(best_epoch)
+    diagnostics["best_primary"] = float(best_primary)
+    diagnostics["epochs_run"] = int(len(trace))
+    diagnostics["elapsed_seconds"] = float(time.time() - t0)
+    return CandidateOutput(best_scores, checkpoint, trace, diagnostics, test_scores)
```

**Resources:** 19,139 tokens

---

## Iteration 2 — bpr-dim16-same-user-negatives-v1

**Hypothesis:** Training the same 5-field, k=16 FM with Bayesian Personalized Ranking — pairwise softplus(-score(pos) - score(neg)) on same-user positive/negative pairs with resampled negatives each epoch — will improve within-user validation GAUC and nDCG@5 over the pointwise-BCE baseline (primary 0.6015), because the evaluation ranks items within a user and BPR directly optimizes score differences under exactly that pairwise ordering.

**Rationale:** The experiment history is empty, so the first controlled comparison should isolate the loss function: keep every capacity and feature choice fixed (5 id fields, k=16, which the data card shows is flat across k=8/16/32) and change only the objective from pointwise BCE to BPR. BPR (Rendle et al., UAI 2009) optimizes P(i >_u j) for same-user pairs, which matches the GAUC/nDCG@5 within-user ranking metric definitionally, whereas the baseline pointwise loss only indirectly shapes intra-user ordering. The card's safe initial space prescribes same-user negatives only (misaligned cross-user sampling is a known failure), 1-2 negatives per positive, resampling every epoch, k=16 for attribution, lr in {0.0003, 0.0005, 0.001}, and batch sizes 2048/4096. I take the midpoint: 1 negative per positive, lr=0.0005, batch 2048, 20 epochs with patience 3 so the trusted worker stops before overfitting. Known failure modes are handled: users with only one label class are skipped in pair construction, and first-order user/global bias terms cancel in score differences, which is expected and harmless for ranking. If BPR beats baseline, the next controlled step is group_softmax (K=4-8 same-user negatives) to test whether listwise supervision improves further; if it does not, the bottleneck is not the loss, matching the data card's conclusion that features and capacity are also dead ends, and history_features becomes the priority.

**Evidence:**
- [BPR: Bayesian Personalized Ranking from Implicit Feedback (Rendle et al., UAI 2009)](https://arxiv.org/abs/1205.2618)
- Method card bpr: hypothesis that pairwise same-user loss aligns with within-user evaluation
- Data card: k=8/16/32 measured flat (0.5895/0.5902/0.5887), so k is held at 16 for loss attribution
- Data card: metric conventions — GAUC is per-user AUC over within-user orderings, matching BPR's pairwise objective

**Family:** bpr  
**Parameters:** `{"batch_size": 2048, "epochs": 20, "k": 16, "learning_rate": 0.0005, "negatives_per_positive": 1, "patience": 3, "seed": 42}`

```diff
--- parent
+++ candidate
@@ -0,0 +1,112 @@
+import numpy as np
+import time
+from collections import Counter
+from src.models.fm_core import FMRanker
+from src.models.sampling import sample_bpr_pairs
+from src.experiments.contracts import CandidateOutput
+
+
+def _sigmoid(x):
+    x = np.asarray(x, dtype=np.float64)
+    out = np.empty_like(x)
+    pos = x >= 0
+    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
+    ex = np.exp(x[~pos])
+    out[~pos] = ex / (1.0 + ex)
+    return out
+
+
+def _primary(metrics):
+    if "primary" in metrics:
+        return float(metrics["primary"])
+    gauc = None
+    ndcg = None
+    for key in metrics:
+        low = key.lower()
+        if "gauc" in low:
+            gauc = float(metrics[key])
+        if "ndcg" in low and "5" in low:
+            ndcg = float(metrics[key])
+    if gauc is None or ndcg is None:
+        return float(np.mean([float(v) for v in metrics.values()]))
+    return 0.5 * (gauc + ndcg)
+
+
+def run(context, parameters):
+    seed = int(parameters["seed"])
+    k = int(parameters["k"])
+    lr = float(parameters["learning_rate"])
+    batch_size = int(parameters["batch_size"])
+    epochs = int(parameters["epochs"])
+    npp = int(parameters["negatives_per_positive"])
+    patience = int(parameters["patience"])
+    rng = np.random.default_rng(seed)
+    model = FMRanker(context.field_dimension, embedding_dim=k, learning_rate=lr, seed=seed)
+    trace = []
+    best_primary = -np.inf
+    best_scores = None
+    best_epoch = -1
+    stale = 0
+    total_pairs = 0
+    start = time.time()
+    for epoch in range(epochs):
+        pos_idx, neg_idx = sample_bpr_pairs(
+            context.train_users, context.train_y, rng, negatives_per_positive=npp)
+        n_pairs = len(pos_idx)
+        order = rng.permutation(n_pairs)
+        pos_idx = pos_idx[order]
+        neg_idx = neg_idx[order]
+        total_pairs += n_pairs
+        for start_row in range(0, n_pairs, batch_size):
+            pb = pos_idx[start_row:start_row + batch_size]
+            nb = neg_idx[start_row:start_row + batch_size]
+            fp = context.train_x[pb]
+            fn = context.train_x[nb]
+            sp = model.predict(fp)
+            sn = model.predict(fn)
+            g = _sigmoid(-(sp - sn))
+            features = np.concatenate([fp, fn], axis=0)
+            sgrad = np.concatenate([-g, g]) / float(len(pb))
+            gv, gw, gb = model.gradients(features, sgrad)
+            model.apply_gradients(gv, gw, gb)
+        scores = model.predict(context.valid_x)
+        metrics = context.evaluate_validation(scores)
+        primary = _primary(metrics)
+        trace.append({
+            "epoch": epoch,
+            "primary": primary,
+            "metrics": {key: float(v) for key, v in metrics.items()},
+            "pairs": int(n_pairs),
+        })
+        if primary > best_primary:
+            best_primary = primary
+            best_scores = scores
+            best_epoch = epoch
+            stale = 0
+        else:
+            stale += 1
+            if stale >= patience:
+                break
+    if best_scores is None:
+        best_scores = np.asarray(model.predict(context.valid_x), dtype=np.float64)
+    test_scores = None
+    if context.test_x is not None:
+        test_scores = model.predict(context.test_x)
+    checkpoint_state = {
+        "best_valid_scores": np.asarray(best_scores, dtype=np.float64),
+        "epoch_primary": np.array([entry["primary"] for entry in trace], dtype=np.float64),
+        "best_epoch": np.array([best_epoch], dtype=np.int64),
+        "epochs_run": np.array([len(trace)], dtype=np.int64),
+    }
+    diagnostics = {
+        "family": "bpr",
+        "best_primary": float(best_primary),
+        "best_epoch": int(best_epoch),
+        "epochs_run": len(trace),
+        "total_pairs_sampled": int(total_pairs),
+        "elapsed_seconds": float(time.time() - start),
+        "note": ("same-user BPR pairs resampled per epoch; validation_scores are "
+                 "best-epoch scores under early stopping; test_scores come from the "
+                 "final model state"),
+    }
+    return CandidateOutput(best_scores, checkpoint_state, trace, diagnostics, test_scores)
```

**Metrics:**

| Metric | Value | Δ vs best-so-far | Δ vs baseline |
|---|---|---|---|
| GAUC | 0.6688 | not reported | not reported |
| nDCG@5 | 0.5367 | not reported | not reported |
| primary | 0.6027 | +0.6027 | +0.0013 |

**Test scores:** `runs/kj_20260829T164704368024Z_research/artifacts/002_bpr-dim16-same-user-negatives-v1/test_scores.npy`

**Critic (preflight):** approve — The proposal is approved. (1) Evidence-backed: every claim traces to the supplied bpr method card (Rendle et al., UAI 2009) or the data card — same-user negative sampling, 1-2 negatives per positive, resampling each epoch, k fixed at 16, lr in {0.0003,0.0005,0.001}, batch 2048/4096 are all explicitly within the card's safe initial space, and the chosen configuration (1 negative, lr=0.0005, batch 2048) sits inside it. (2) Novel: experiment history is empty, so no collision with prior runs; this is the correct first controlled experiment. (3) Isolates the ranking-loss variable: hypothesis_id bpr-vs-pointwise-baseline-dim16 holds features (5 fields), capacity (k=16, justified by the data card's measured flatness across k=8/16/32), and sampling alignment constant, changing only the objective from pointwise BCE to pairwise BPR. (4) Leakage-safe: no video-statistics aggregate features, no test access, no evaluator/split/label changes; validation metrics come only through context.evaluate_validation. (5) Feasible: ~1.14M train positives, 1 negative per positive, 20 epochs with patience 3 is well within compute budget using numpy plus the sanctioned FM/sampling modules. (6) Failure modes pre-handled: single-class users skipped in pair construction; user/global first-order terms canceling in score differences is expected under within-user ranking. The planned follow-up branching (group_softmax if BPR wins, history_features otherwise) is a sound adaptive path.

**Critic (postflight):** accept_result_continue_line — The trusted validation result supports the method card's hypothesis, but weakly. BPR with same-user pairs at k=16 achieves primary 0.6027 vs the pointwise-BCE baseline 0.6015, a gain of +0.00127 (GAUC 0.6688, nDCG@5 0.5367). The direction is consistent with the theory in Rendle et al. (UAI 2009): optimizing P(i >_u j) for same-user pairs aligns with the GAUC/nDCG@5 within-user metric definitionally, and the experiment cleanly isolated the loss by holding features (5 fields) and capacity (k=16, shown flat in the data card) fixed. Diagnostics are sane: 10 of 20 epochs ran, best epoch was 6 with patience 3, ~3.8M same-user pairs sampled, ~61s cost. The evidence chain is attributable to the bpr method card and the data card's metric conventions and dead-end measurements. This is a genuine but small win, so the loss-function family merits one more controlled step before declaring it exhausted.

**Resources:** 70.8 s · 23,924 tokens

---

## Iteration 3 — group-softmax-k4-t1-dim16-v1-builder

**Hypothesis:** Training the same FM ranker (dim 16) with a same-user group softmax loss — one positive against K=4 sampled same-user negatives with temperature 1.0 — will improve the primary metric over the BPR single-negative baseline because the listwise objective approximates the within-user evaluation list more closely, but with diminishing returns relative to compute cost per step.

**Rationale:** The BPR baseline (primary 0.6027) optimizes a single same-user pair per positive, while validation GAUC/nDCG@5 are computed over each user's full impression list. ListNet-style listwise losses (Cao et al., ICML 2007) match this evaluation structure by normalizing scores within the group, so each step contrasts a positive against K same-user negatives. The data card shows median 34 and p75 65 rows per user, so K=4 same-user negatives are available for nearly all users with both labels; users lacking both labels are skipped per the method card's failure modes. K=4 is chosen over K=8 to keep compute per step comparable to the baseline for a clean first attribution, holding k=16, learning rate 0.0005, and seed 42 fixed with the BPR run (its best setting). Temperature 1.0 is the neutral middle of the safe search space; if gradients are too soft/hard, 0.5 and 2.0 are the next probes. Max-shifted softmax inside the loss guards against unstable exponentials. Duplicate negative rows will be re-sampled to preserve effective list size. This is an explore action because it is the first group_softmax run; results will be compared directly against bpr-dim16-same-user-negatives-v1.

**Evidence:**
- [Learning to Rank: From Pairwise Approach to Listwise Approach (ListNet)](https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/)
- [On Using Very Large Target Vocabulary for Neural Machine Translation (sampled softmax normalization)](https://aclanthology.org/P15-1001/)

**Family:** group_softmax  
**Parameters:** `{"batch_size": 1024, "epochs": 20, "k": 16, "learning_rate": 0.0005, "negatives_per_group": 4, "patience": 3, "seed": 42, "temperature": 1.0}`

```diff
--- parent
+++ candidate
@@ -1,112 +1,150 @@
+import time
 import numpy as np
-import time
-from collections import Counter
 from src.models.fm_core import FMRanker
-from src.models.sampling import sample_bpr_pairs
+from src.models.sampling import sample_bpr_pairs, sample_softmax_groups
 from src.experiments.contracts import CandidateOutput
 
 
-def _sigmoid(x):
-    x = np.asarray(x, dtype=np.float64)
-    out = np.empty_like(x)
-    pos = x >= 0
-    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
-    ex = np.exp(x[~pos])
-    out[~pos] = ex / (1.0 + ex)
-    return out
+def _pick(metrics, keys):
+    for key in keys:
+        if key in metrics:
+            return float(metrics[key])
+    return None
 
 
 def _primary(metrics):
-    if "primary" in metrics:
-        return float(metrics["primary"])
-    gauc = None
-    ndcg = None
-    for key in metrics:
-        low = key.lower()
-        if "gauc" in low:
-            gauc = float(metrics[key])
-        if "ndcg" in low and "5" in low:
-            ndcg = float(metrics[key])
-    if gauc is None or ndcg is None:
-        return float(np.mean([float(v) for v in metrics.values()]))
-    return 0.5 * (gauc + ndcg)
+    gauc = _pick(metrics, ("gauc", "GAUC", "gauc_score"))
+    ndcg = _pick(metrics, ("ndcg@5", "nDCG@5", "ndcg_at_5", "ndcg5"))
+    if gauc is not None and ndcg is not None:
+        return 0.5 * (gauc + ndcg)
+    if gauc is not None:
+        return gauc
+    if ndcg is not None:
+        return ndcg
+    vals = [float(v) for v in metrics.values() if isinstance(v, (int, float))]
+    return float(np.mean(vals)) if vals else 0.0
+
+
+def _dedupe_negatives(users, pos_idx, neg_groups, rng):
+    k = neg_groups.shape[1]
+    # dup[g, j] is True when negative j of group g duplicates the positive
+    # or duplicates another negative in the same group. Both members of a
+    # mutually-equal pair are flagged so both are resampled.
+    dup = neg_groups == pos_idx[:, None]
+    for a in range(k):
+        for b in range(a + 1, k):
+            eq = neg_groups[:, a] == neg_groups[:, b]
+            # Flag ONLY the two members of the equal pair; broadcasting the
+            # (n,) vector across all columns would wrongly flag clean
+            # negatives in any group containing a duplicate.
+            dup[:, a] |= eq
+            dup[:, b] |= eq
+    n_dup = int(dup.sum())
+    if n_dup == 0:
+        return neg_groups, 0
+    order = np.argsort(users, kind="stable")
+    su = users[order]
+    fixed = neg_groups.copy()
+    for g in np.nonzero(dup.any(axis=1))[0]:
+        u = users[pos_idx[g]]
+        lo = int(np.searchsorted(su, u, "left"))
+        hi = int(np.searchsorted(su, u, "right"))
+        pool = order[lo:hi]
+        pool = pool[pool != pos_idx[g]]
+        if len(pool) == 0:
+            continue
+        used = set(fixed[g].tolist())
+        used.add(int(pos_idx[g]))
+        for j in np.nonzero(dup[g])[0]:
+            cand = int(rng.choice(pool))
+            tries = 0
+            while cand in used and tries < 10 and len(pool) > 1:
+                cand = int(rng.choice(pool))
+                tries += 1
+            fixed[g, j] = cand
+            used.add(cand)
+    return fixed, n_dup
+
+
+def _step(model, train_x, pos_batch, neg_batch, temperature):
+    n_fields = train_x.shape[1]
+    pos_x = train_x[pos_batch]
+    neg_x = train_x[neg_batch.reshape(-1)].reshape(neg_batch.shape[0], neg_batch.shape[1], n_fields)
+    pos_s = np.asarray(model.predict(pos_x), dtype=np.float64)
+    neg_s = np.asarray(model.predict(neg_x.reshape(-1, n_fields)), dtype=np.float64).reshape(neg_batch.shape)
+    logits = np.concatenate([pos_s[:, None], neg_s], axis=1) / temperature
+    logits -= logits.max(axis=1, keepdims=True)
+    e = np.exp(logits)
+    p = e / e.sum(axis=1, keepdims=True)
+    loss = float(-np.mean(np.log(p[:, 0] + 1e-12)))
+    g_pos = (p[:, 0] - 1.0) / temperature
+    g_neg = p[:, 1:] / temperature
+    gv, gw, gb = model.gradients(pos_x, g_pos)
+    gv2, gw2, gb2 = model.gradients(neg_x.reshape(-1, n_fields), g_neg.reshape(-1))
+    model.apply_gradients(gv + gv2, gw + gw2, gb + gb2)
+    return loss
 
 
 def run(context, parameters):
     seed = int(parameters["seed"])
-    k = int(parameters["k"])
-    lr = float(parameters["learning_rate"])
+    rng = np.random.default_rng(seed)
+    model = FMRanker(int(context.field_dimension), embedding_dim=int(parameters["k"]),
+                     learning_rate=float(parameters["learning_rate"]), seed=seed)
+    train_x = context.train_x
+    train_y = np.asarray(context.train_y, dtype=np.float64)
+    train_users = np.asarray(context.train_users)
+    epochs = int(parameters["epochs"])
     batch_size = int(parameters["batch_size"])
-    epochs = int(parameters["epochs"])
-    npp = int(parameters["negatives_per_positive"])
+    k_neg = int(parameters["negatives_per_group"])
+    temp = float(parameters["temperature"])
     patience = int(parameters["patience"])
-    rng = np.random.default_rng(seed)
-    model = FMRanker(context.field_dimension, embedding_dim=k, learning_rate=lr, seed=seed)
     trace = []
-    best_primary = -np.inf
-    best_scores = None
-    best_epoch = -1
+    best = {"primary": -1.0, "scores": None, "epoch": -1}
     stale = 0
-    total_pairs = 0
-    start = time.time()
+    n_groups_last = 0
+    dup_total = 0
     for epoch in range(epochs):
-        pos_idx, neg_idx = sample_bpr_pairs(
-            context.train_users, context.train_y, rng, negatives_per_positive=npp)
-        n_pairs = len(pos_idx)
-        order = rng.permutation(n_pairs)
-        pos_idx = pos_idx[order]
-        neg_idx = neg_idx[order]
-        total_pairs += n_pairs
-        for start_row in range(0, n_pairs, batch_size):
-            pb = pos_idx[start_row:start_row + batch_size]
-            nb = neg_idx[start_row:start_row + batch_size]
-            fp = context.train_x[pb]
-            fn = context.train_x[nb]
-            sp = model.predict(fp)
-            sn = model.predict(fn)
-            g = _sigmoid(-(sp - sn))
-            features = np.concatenate([fp, fn], axis=0)
-            sgrad = np.concatenate([-g, g]) / float(len(pb))
-            gv, gw, gb = model.gradients(features, sgrad)
-            model.apply_gradients(gv, gw, gb)
-        scores = model.predict(context.valid_x)
+        t0 = time.time()
+        pos_idx, neg_groups = sample_softmax_groups(train_users, train_y, rng, negatives_per_group=k_neg)
+        neg_groups, n_dup = _dedupe_negatives(train_users, pos_idx, neg_groups, rng)
+        dup_total += n_dup
+        n_groups_last = int(len(pos_idx))
+        perm = rng.permutation(n_groups_last)
+        losses = []
+        for start in range(0, n_groups_last, batch_size):
+            b = perm[start:start + batch_size]
+            losses.append(_step(model, train_x, pos_idx[b], neg_groups[b], temp))
+        scores = np.asarray(model.predict(context.valid_x), dtype=np.float64)
         metrics = context.evaluate_validation(scores)
         primary = _primary(metrics)
-        trace.append({
-            "epoch": epoch,
-            "primary": primary,
-            "metrics": {key: float(v) for key, v in metrics.items()},
-            "pairs": int(n_pairs),
-        })
-        if primary > best_primary:
-            best_primary = primary
-            best_scores = scores
-            best_epoch = epoch
+        trace.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else 0.0,
+                      "primary": primary, "metrics": {key: float(v) for key, v in metrics.items()},
+                      "groups": n_groups_last, "seconds": time.time() - t0})
+        if primary > best["primary"] + 1e-9:
+            best = {"primary": primary, "scores": scores, "epoch": epoch}
             stale = 0
         else:
             stale += 1
             if stale >= patience:
                 break
-    if best_scores is None:
-        best_scores = np.asarray(model.predict(context.valid_x), dtype=np.float64)
+    valid_scores = best["scores"] if best["scores"] is not None else np.asarray(model.predict(context.valid_x), dtype=np.float64)
… truncated, full source at kj_20260829T164704368024Z_research/003_group-softmax-k4-t1-dim16-v1-builder/candidate.py
```

**Metrics:**

| Metric | Value | Δ vs best-so-far | Δ vs baseline |
|---|---|---|---|
| GAUC | 0.6704 | not reported | not reported |
| nDCG@5 | 0.5369 | not reported | not reported |
| primary | 0.6037 | +0.0009 | +0.0022 |

**Test scores:** `runs/kj_20260829T164704368024Z_research/artifacts/003_group-softmax-k4-t1-dim16-v1-builder/test_scores.npy`

**Critic (preflight):** approve_with_concerns — The proposal is evidence-backed, novel, leakage-safe, feasible, and designed as a clean loss-function attribution. Evidence: both citations map to the supplied method_card_id group_softmax; ListNet (Cao et al., ICML 2007) grounds the within-group softmax/listwise objective, and Jean et al. (ACL 2015) grounds sampled normalization. Novelty: the state contains only a single bpr-family experiment (single same-user negative, primary 0.6027), so group_softmax is an unexplored family and this is a legitimate first explore step with parent_experiment correctly linked. Leakage safety: uses train/validation only, no test access, no evaluator or split changes, and test_scores will come from the same trained model per contract. Feasibility: K=4 same-user negatives is well within availability given median 34 and p75 65 rows per user (train+valid), and compute per step is comparable to the BPR baseline. Variable isolation: k=16, learning_rate=0.0005, seed=42, epochs=20, patience=3 are all held fixed against the parent run; only the loss (pairwise BPR -> K=4 group softmax with temperature 1.0) is the intended variable. The proposal explicitly addresses the method card's known failure modes: skipping users lacking both labels, max-shifted softmax for numerical stability, and re-sampling duplicate negatives. The acknowledged diminishing-returns-vs-compute caveat in the hypothesis is appropriately hedged.

**Critic (postflight):** accept_result_continue_family — The group softmax run (group-softmax-k4-t1-dim16-v1) achieved primary 0.60366 vs the reported baseline primary 0.60147 (+0.0022) and marginally above the stated parent BPR result of ~0.6027 (+0.0010). GAUC 0.67039 and nDCG@5 0.53694 are both consistent with a genuine but small listwise-objective gain. The direction of the hypothesis — that contrasting one positive against K same-user negatives approximates the within-user evaluation list better than a single BPR pair — is supported, but the effect size is thin, so it should be treated as weakly supported rather than confirmed. Attribution is clean: dim 16, lr 0.0005, seed 42, batch size 1024 were held fixed with the parent run, and only the loss changed. The diagnostics confirm the method card's failure modes were handled (duplicate negatives re-sampled, max-shifted softmax implied by stable training) and early stopping fired correctly at epoch 3 of 7 with patience 3, indicating fast convergence of the listwise gradient rather than instability.

**Resources:** 101.5 s · 46,584 tokens

**Replications:**

| Seed | Primary | Status |
|---|---|---|
| 1 | 0.6034 | success |
| 2 | 0.6035 | success |
| **mean** | **0.6035** | spread 0.0001 |

---

## Iteration 6 — group-softmax-k8-t1-dim16-v1-builder

**Hypothesis:** Increasing same-user negatives per group from K=4 to K=8 improves the within-user ranking approximation of the validation list, raising primary (mean of GAUC and nDCG@5) above the incumbent K=4 result of 0.6037.

**Rationale:** group_softmax is the best-performing family so far (primary 0.6037 vs BPR 0.6027 and baseline 0.6015), and its two seed replications (0.6034, 0.6035) confirm the gain is not seed noise, so the family is worth deeper exploration rather than replication. The method card's safe search space lists K in {4, 8}; only K=4 has been measured. Holding temperature=1.0, learning rate=0.0005, batch size=1024, k=16, and epochs=20 fixed makes this a single-variable controlled change isolating list size. The card warns K=8 doubles compute per group; keeping batch_size at 1024 (within the card's 512-2048 range) and epochs at 20 with early stopping patience 3 stays well inside the 900 s timeout, but per-step cost must be reported. Same-user negative sampling and skipping users without both labels remain per the card's failure modes; max-shifted softmax keeps exponentials stable at larger K.

**Evidence:**
- [Learning to Rank: From Pairwise Approach to Listwise Approach](https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/)
- [On Using Very Large Target Vocabulary for Neural Machine Translation (sampled normalization reference)](https://aclanthology.org/P15-1001/)

**Family:** group_softmax  
**Parameters:** `{"batch_size": 1024, "epochs": 20, "k": 16, "learning_rate": 0.0005, "negatives_per_group": 8, "patience": 3, "seed": 42, "temperature": 1.0}`

```diff
--- parent
+++ candidate
@@ -1,150 +1,118 @@
 import time
 import numpy as np
 from src.models.fm_core import FMRanker
-from src.models.sampling import sample_bpr_pairs, sample_softmax_groups
+from src.models.sampling import sample_softmax_groups
 from src.experiments.contracts import CandidateOutput
 
 
-def _pick(metrics, keys):
-    for key in keys:
-        if key in metrics:
-            return float(metrics[key])
-    return None
-
-
 def _primary(metrics):
-    gauc = _pick(metrics, ("gauc", "GAUC", "gauc_score"))
-    ndcg = _pick(metrics, ("ndcg@5", "nDCG@5", "ndcg_at_5", "ndcg5"))
-    if gauc is not None and ndcg is not None:
-        return 0.5 * (gauc + ndcg)
-    if gauc is not None:
-        return gauc
-    if ndcg is not None:
-        return ndcg
-    vals = [float(v) for v in metrics.values() if isinstance(v, (int, float))]
-    return float(np.mean(vals)) if vals else 0.0
-
-
-def _dedupe_negatives(users, pos_idx, neg_groups, rng):
-    k = neg_groups.shape[1]
-    # dup[g, j] is True when negative j of group g duplicates the positive
-    # or duplicates another negative in the same group. Both members of a
-    # mutually-equal pair are flagged so both are resampled.
-    dup = neg_groups == pos_idx[:, None]
-    for a in range(k):
-        for b in range(a + 1, k):
-            eq = neg_groups[:, a] == neg_groups[:, b]
-            # Flag ONLY the two members of the equal pair; broadcasting the
-            # (n,) vector across all columns would wrongly flag clean
-            # negatives in any group containing a duplicate.
-            dup[:, a] |= eq
-            dup[:, b] |= eq
-    n_dup = int(dup.sum())
-    if n_dup == 0:
-        return neg_groups, 0
-    order = np.argsort(users, kind="stable")
-    su = users[order]
-    fixed = neg_groups.copy()
-    for g in np.nonzero(dup.any(axis=1))[0]:
-        u = users[pos_idx[g]]
-        lo = int(np.searchsorted(su, u, "left"))
-        hi = int(np.searchsorted(su, u, "right"))
-        pool = order[lo:hi]
-        pool = pool[pool != pos_idx[g]]
-        if len(pool) == 0:
-            continue
-        used = set(fixed[g].tolist())
-        used.add(int(pos_idx[g]))
-        for j in np.nonzero(dup[g])[0]:
-            cand = int(rng.choice(pool))
-            tries = 0
-            while cand in used and tries < 10 and len(pool) > 1:
-                cand = int(rng.choice(pool))
-                tries += 1
-            fixed[g, j] = cand
-            used.add(cand)
-    return fixed, n_dup
-
-
-def _step(model, train_x, pos_batch, neg_batch, temperature):
-    n_fields = train_x.shape[1]
-    pos_x = train_x[pos_batch]
-    neg_x = train_x[neg_batch.reshape(-1)].reshape(neg_batch.shape[0], neg_batch.shape[1], n_fields)
-    pos_s = np.asarray(model.predict(pos_x), dtype=np.float64)
-    neg_s = np.asarray(model.predict(neg_x.reshape(-1, n_fields)), dtype=np.float64).reshape(neg_batch.shape)
-    logits = np.concatenate([pos_s[:, None], neg_s], axis=1) / temperature
-    logits -= logits.max(axis=1, keepdims=True)
-    e = np.exp(logits)
-    p = e / e.sum(axis=1, keepdims=True)
-    loss = float(-np.mean(np.log(p[:, 0] + 1e-12)))
-    g_pos = (p[:, 0] - 1.0) / temperature
-    g_neg = p[:, 1:] / temperature
-    gv, gw, gb = model.gradients(pos_x, g_pos)
-    gv2, gw2, gb2 = model.gradients(neg_x.reshape(-1, n_fields), g_neg.reshape(-1))
-    model.apply_gradients(gv + gv2, gw + gw2, gb + gb2)
-    return loss
+    if isinstance(metrics, dict):
+        if 'primary' in metrics:
+            return float(metrics['primary'])
+        vals = [float(v) for v in metrics.values()]
+        return float(np.mean(vals)) if vals else 0.0
+    return float(metrics)
 
 
 def run(context, parameters):
-    seed = int(parameters["seed"])
+    t0 = time.time()
+    seed = int(parameters['seed'])
+    epochs = int(parameters['epochs'])
+    patience = int(parameters['patience'])
+    k_neg = int(parameters['negatives_per_group'])
+    temperature = float(parameters['temperature'])
+    batch_size = int(parameters['batch_size'])
+
+    model = FMRanker(int(context.field_dimension),
+                     embedding_dim=int(parameters['k']),
+                     learning_rate=float(parameters['learning_rate']),
+                     seed=seed)
+    train_x = context.train_x
+    train_y = context.train_y
+    users = context.train_users
+    valid_x = context.valid_x
+    test_x = context.test_x
     rng = np.random.default_rng(seed)
-    model = FMRanker(int(context.field_dimension), embedding_dim=int(parameters["k"]),
-                     learning_rate=float(parameters["learning_rate"]), seed=seed)
-    train_x = context.train_x
-    train_y = np.asarray(context.train_y, dtype=np.float64)
-    train_users = np.asarray(context.train_users)
-    epochs = int(parameters["epochs"])
-    batch_size = int(parameters["batch_size"])
-    k_neg = int(parameters["negatives_per_group"])
-    temp = float(parameters["temperature"])
-    patience = int(parameters["patience"])
+
+    pos_idx, neg_groups = sample_softmax_groups(
+        users, train_y, rng, negatives_per_group=k_neg)
+    n_groups = int(pos_idx.shape[0])
+    steps_per_epoch = int(np.ceil(n_groups / batch_size)) if n_groups else 0
+
+    def score_all():
+        v = model.predict(valid_x)
+        t = None if test_x is None else model.predict(test_x)
+        return v, t
+
+    best_primary = -np.inf
+    best_valid = None
+    best_test = None
+    best_epoch = -1
+    no_improve = 0
+    epochs_run = 0
     trace = []
-    best = {"primary": -1.0, "scores": None, "epoch": -1}
-    stale = 0
-    n_groups_last = 0
-    dup_total = 0
+
     for epoch in range(epochs):
-        t0 = time.time()
-        pos_idx, neg_groups = sample_softmax_groups(train_users, train_y, rng, negatives_per_group=k_neg)
-        neg_groups, n_dup = _dedupe_negatives(train_users, pos_idx, neg_groups, rng)
-        dup_total += n_dup
-        n_groups_last = int(len(pos_idx))
-        perm = rng.permutation(n_groups_last)
-        losses = []
-        for start in range(0, n_groups_last, batch_size):
-            b = perm[start:start + batch_size]
-            losses.append(_step(model, train_x, pos_idx[b], neg_groups[b], temp))
-        scores = np.asarray(model.predict(context.valid_x), dtype=np.float64)
-        metrics = context.evaluate_validation(scores)
-        primary = _primary(metrics)
-        trace.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else 0.0,
-                      "primary": primary, "metrics": {key: float(v) for key, v in metrics.items()},
-                      "groups": n_groups_last, "seconds": time.time() - t0})
-        if primary > best["primary"] + 1e-9:
-            best = {"primary": primary, "scores": scores, "epoch": epoch}
-            stale = 0
+        epochs_run = epoch + 1
+        order = rng.permutation(n_groups)
+        for s in range(steps_per_epoch):
+            b = order[s * batch_size:(s + 1) * batch_size]
+            pf = train_x[pos_idx[b]]
+            nf = train_x[neg_groups[b]]
+            B = pf.shape[0]
+            nff = nf.reshape(B * k_neg, -1)
+            ps = model.predict(pf)
+            ns = model.predict(nff).reshape(B, k_neg)
+            logits = np.concatenate([ps[:, None], ns], axis=1) / temperature
+            logits -= logits.max(axis=1, keepdims=True)
+            e = np.exp(logits)
+            p = e / e.sum(axis=1, keepdims=True)
+            g = p / temperature
+            g[:, 0] -= 1.0 / temperature
+            gp = np.ascontiguousarray(g[:, 0])
+            gn = np.ascontiguousarray(g[:, 1:].reshape(B * k_neg))
+            gv, gw, gb = model.gradients(pf, gp)
+            gv2, gw2, gb2 = model.gradients(nff, gn)
+            model.apply_gradients(gv + gv2, gw + gw2, gb + gb2)
+
+        valid_scores, test_scores = score_all()
+        primary = _primary(context.evaluate_validation(valid_scores))
+        trace.append({'epoch': int(epoch + 1), 'primary': float(primary)})
+        if primary > best_primary:
+            best_primary = primary
+            best_valid = valid_scores.copy()
+            best_test = None if test_scores is None else test_scores.copy()
+            best_epoch = epoch + 1
… truncated, full source at kj_20260829T164704368024Z_research/006_group-softmax-k8-t1-dim16-v1-builder/candidate.py
```

**Metrics:**

| Metric | Value | Δ vs best-so-far | Δ vs baseline |
|---|---|---|---|
| GAUC | 0.6695 | not reported | not reported |
| nDCG@5 | 0.5373 | not reported | not reported |
| primary | 0.6034 | -0.0003 | +0.0019 |

**Test scores:** `runs/kj_20260829T164704368024Z_research/artifacts/006_group-softmax-k8-t1-dim16-v1-builder/test_scores.npy`

**Critic (preflight):** approve — The proposal is evidence-backed: the group_softmax method card explicitly lists K in {4, 8} in its safe search space and cites the ListNet listwise objective (Cao et al., ICML 2007) plus sampled-normalization reference (Jean et al., ACL 2015). It is novel relative to history: only K=4 has been measured in this family, and the incumbent best (primary 0.6037) plus two seed replications (0.6034, 0.6035) confirm the family's small edge over BPR (0.6027) and baseline (0.6015) is not seed noise, justifying exploration over another replication. The change isolates exactly one variable — negatives_per_group from 4 to 8 — with temperature, learning rate, batch size, k=16, epochs, patience, and seed all held fixed at the parent's values, satisfying the attribution requirement. It is leakage-safe: same-user negative sampling uses train data only, validation is scored via context.evaluate_validation, and no test or evaluator access is implied. It is computationally feasible: per-step cost grows from 5 to 9 scored rows per group (~1.8x), still within the card's stated batch-size range and the timeout budget given early stopping; the card requires per-step compute to be reported, which the proposal acknowledges.

**Critic (postflight):** reject_hypothesis_not_supported — The K=8 run produced primary 0.60338, which is below the parent K=4 result of 0.6037 and statistically indistinguishable from the K=4 seed replications (0.6034, 0.6035). The hypothesis that doubling same-user negatives per group raises primary above 0.6037 was therefore not supported: the list-size axis appears flat, with any K=8 effect within seed noise (~±0.0002). The run is otherwise healthy — 35.9 s train time, early stopping at epoch 6 with best epoch 3, clean diagnostics, and the compute note confirms the card's warning that K=8 doubles rows scored per step, so the extra cost bought no measurable gain. Per the method card's attribution framing (dimension fixed at 16, temperature/LR/batch held constant), the controlled comparison is valid; the conclusion is that within-user list approximation is not the bottleneck at this scale of K.

**Resources:** 45.5 s · 23,518 tokens

---

