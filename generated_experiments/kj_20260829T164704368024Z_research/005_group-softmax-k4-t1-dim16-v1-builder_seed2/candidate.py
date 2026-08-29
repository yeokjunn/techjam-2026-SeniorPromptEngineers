import time
import numpy as np
from src.models.fm_core import FMRanker
from src.models.sampling import sample_bpr_pairs, sample_softmax_groups
from src.experiments.contracts import CandidateOutput


def _pick(metrics, keys):
    for key in keys:
        if key in metrics:
            return float(metrics[key])
    return None


def _primary(metrics):
    gauc = _pick(metrics, ("gauc", "GAUC", "gauc_score"))
    ndcg = _pick(metrics, ("ndcg@5", "nDCG@5", "ndcg_at_5", "ndcg5"))
    if gauc is not None and ndcg is not None:
        return 0.5 * (gauc + ndcg)
    if gauc is not None:
        return gauc
    if ndcg is not None:
        return ndcg
    vals = [float(v) for v in metrics.values() if isinstance(v, (int, float))]
    return float(np.mean(vals)) if vals else 0.0


def _dedupe_negatives(users, pos_idx, neg_groups, rng):
    k = neg_groups.shape[1]
    # dup[g, j] is True when negative j of group g duplicates the positive
    # or duplicates another negative in the same group. Both members of a
    # mutually-equal pair are flagged so both are resampled.
    dup = neg_groups == pos_idx[:, None]
    for a in range(k):
        for b in range(a + 1, k):
            eq = neg_groups[:, a] == neg_groups[:, b]
            # Flag ONLY the two members of the equal pair; broadcasting the
            # (n,) vector across all columns would wrongly flag clean
            # negatives in any group containing a duplicate.
            dup[:, a] |= eq
            dup[:, b] |= eq
    n_dup = int(dup.sum())
    if n_dup == 0:
        return neg_groups, 0
    order = np.argsort(users, kind="stable")
    su = users[order]
    fixed = neg_groups.copy()
    for g in np.nonzero(dup.any(axis=1))[0]:
        u = users[pos_idx[g]]
        lo = int(np.searchsorted(su, u, "left"))
        hi = int(np.searchsorted(su, u, "right"))
        pool = order[lo:hi]
        pool = pool[pool != pos_idx[g]]
        if len(pool) == 0:
            continue
        used = set(fixed[g].tolist())
        used.add(int(pos_idx[g]))
        for j in np.nonzero(dup[g])[0]:
            cand = int(rng.choice(pool))
            tries = 0
            while cand in used and tries < 10 and len(pool) > 1:
                cand = int(rng.choice(pool))
                tries += 1
            fixed[g, j] = cand
            used.add(cand)
    return fixed, n_dup


def _step(model, train_x, pos_batch, neg_batch, temperature):
    n_fields = train_x.shape[1]
    pos_x = train_x[pos_batch]
    neg_x = train_x[neg_batch.reshape(-1)].reshape(neg_batch.shape[0], neg_batch.shape[1], n_fields)
    pos_s = np.asarray(model.predict(pos_x), dtype=np.float64)
    neg_s = np.asarray(model.predict(neg_x.reshape(-1, n_fields)), dtype=np.float64).reshape(neg_batch.shape)
    logits = np.concatenate([pos_s[:, None], neg_s], axis=1) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    p = e / e.sum(axis=1, keepdims=True)
    loss = float(-np.mean(np.log(p[:, 0] + 1e-12)))
    g_pos = (p[:, 0] - 1.0) / temperature
    g_neg = p[:, 1:] / temperature
    gv, gw, gb = model.gradients(pos_x, g_pos)
    gv2, gw2, gb2 = model.gradients(neg_x.reshape(-1, n_fields), g_neg.reshape(-1))
    model.apply_gradients(gv + gv2, gw + gw2, gb + gb2)
    return loss


def run(context, parameters):
    seed = int(parameters["seed"])
    rng = np.random.default_rng(seed)
    model = FMRanker(int(context.field_dimension), embedding_dim=int(parameters["k"]),
                     learning_rate=float(parameters["learning_rate"]), seed=seed)
    train_x = context.train_x
    train_y = np.asarray(context.train_y, dtype=np.float64)
    train_users = np.asarray(context.train_users)
    epochs = int(parameters["epochs"])
    batch_size = int(parameters["batch_size"])
    k_neg = int(parameters["negatives_per_group"])
    temp = float(parameters["temperature"])
    patience = int(parameters["patience"])
    trace = []
    best = {"primary": -1.0, "scores": None, "epoch": -1}
    stale = 0
    n_groups_last = 0
    dup_total = 0
    for epoch in range(epochs):
        t0 = time.time()
        pos_idx, neg_groups = sample_softmax_groups(train_users, train_y, rng, negatives_per_group=k_neg)
        neg_groups, n_dup = _dedupe_negatives(train_users, pos_idx, neg_groups, rng)
        dup_total += n_dup
        n_groups_last = int(len(pos_idx))
        perm = rng.permutation(n_groups_last)
        losses = []
        for start in range(0, n_groups_last, batch_size):
            b = perm[start:start + batch_size]
            losses.append(_step(model, train_x, pos_idx[b], neg_groups[b], temp))
        scores = np.asarray(model.predict(context.valid_x), dtype=np.float64)
        metrics = context.evaluate_validation(scores)
        primary = _primary(metrics)
        trace.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else 0.0,
                      "primary": primary, "metrics": {key: float(v) for key, v in metrics.items()},
                      "groups": n_groups_last, "seconds": time.time() - t0})
        if primary > best["primary"] + 1e-9:
            best = {"primary": primary, "scores": scores, "epoch": epoch}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    valid_scores = best["scores"] if best["scores"] is not None else np.asarray(model.predict(context.valid_x), dtype=np.float64)
    test_scores = None
    if context.test_x is not None:
        test_scores = np.asarray(model.predict(context.test_x), dtype=np.float64)
    checkpoint_state = {
        "validation_scores": np.asarray(valid_scores, dtype=np.float64),
        "test_scores": np.asarray(test_scores, dtype=np.float64) if test_scores is not None else np.zeros(0),
        "best_epoch": np.array(best["epoch"]),
        "best_primary": np.array(best["primary"]),
        "seed": np.array(seed),
    }
    diagnostics = {
        "family": "group_softmax", "parameters": dict(parameters),
        "negatives_per_group": k_neg, "temperature": temp, "batch_size": batch_size,
        "learning_rate": float(parameters["learning_rate"]), "embedding_dim": int(parameters["k"]),
        "groups_per_epoch": n_groups_last, "duplicate_negatives_resampled": dup_total,
        "epochs_run": len(trace), "best_epoch": best["epoch"],
        "best_primary": best["primary"], "early_stopped": len(trace) < epochs,
        "valid_rows": int(np.asarray(valid_scores).shape[0]),
    }
    return CandidateOutput(valid_scores, checkpoint_state, trace, diagnostics, test_scores)