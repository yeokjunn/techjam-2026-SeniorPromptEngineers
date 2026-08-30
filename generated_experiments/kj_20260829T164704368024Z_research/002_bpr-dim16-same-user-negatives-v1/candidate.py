import numpy as np
import time
from collections import Counter
from src.models.fm_core import FMRanker
from src.models.sampling import sample_bpr_pairs
from src.experiments.contracts import CandidateOutput


def _sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _primary(metrics):
    if "primary" in metrics:
        return float(metrics["primary"])
    gauc = None
    ndcg = None
    for key in metrics:
        low = key.lower()
        if "gauc" in low:
            gauc = float(metrics[key])
        if "ndcg" in low and "5" in low:
            ndcg = float(metrics[key])
    if gauc is None or ndcg is None:
        return float(np.mean([float(v) for v in metrics.values()]))
    return 0.5 * (gauc + ndcg)


def run(context, parameters):
    seed = int(parameters["seed"])
    k = int(parameters["k"])
    lr = float(parameters["learning_rate"])
    batch_size = int(parameters["batch_size"])
    epochs = int(parameters["epochs"])
    npp = int(parameters["negatives_per_positive"])
    patience = int(parameters["patience"])
    rng = np.random.default_rng(seed)
    model = FMRanker(context.field_dimension, embedding_dim=k, learning_rate=lr, seed=seed)
    trace = []
    best_primary = -np.inf
    best_scores = None
    best_epoch = -1
    stale = 0
    total_pairs = 0
    start = time.time()
    for epoch in range(epochs):
        pos_idx, neg_idx = sample_bpr_pairs(
            context.train_users, context.train_y, rng, negatives_per_positive=npp)
        n_pairs = len(pos_idx)
        order = rng.permutation(n_pairs)
        pos_idx = pos_idx[order]
        neg_idx = neg_idx[order]
        total_pairs += n_pairs
        for start_row in range(0, n_pairs, batch_size):
            pb = pos_idx[start_row:start_row + batch_size]
            nb = neg_idx[start_row:start_row + batch_size]
            fp = context.train_x[pb]
            fn = context.train_x[nb]
            sp = model.predict(fp)
            sn = model.predict(fn)
            g = _sigmoid(-(sp - sn))
            features = np.concatenate([fp, fn], axis=0)
            sgrad = np.concatenate([-g, g]) / float(len(pb))
            gv, gw, gb = model.gradients(features, sgrad)
            model.apply_gradients(gv, gw, gb)
        scores = model.predict(context.valid_x)
        metrics = context.evaluate_validation(scores)
        primary = _primary(metrics)
        trace.append({
            "epoch": epoch,
            "primary": primary,
            "metrics": {key: float(v) for key, v in metrics.items()},
            "pairs": int(n_pairs),
        })
        if primary > best_primary:
            best_primary = primary
            best_scores = scores
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_scores is None:
        best_scores = np.asarray(model.predict(context.valid_x), dtype=np.float64)
    test_scores = None
    if context.test_x is not None:
        test_scores = model.predict(context.test_x)
    checkpoint_state = {
        "best_valid_scores": np.asarray(best_scores, dtype=np.float64),
        "epoch_primary": np.array([entry["primary"] for entry in trace], dtype=np.float64),
        "best_epoch": np.array([best_epoch], dtype=np.int64),
        "epochs_run": np.array([len(trace)], dtype=np.int64),
    }
    diagnostics = {
        "family": "bpr",
        "best_primary": float(best_primary),
        "best_epoch": int(best_epoch),
        "epochs_run": len(trace),
        "total_pairs_sampled": int(total_pairs),
        "elapsed_seconds": float(time.time() - start),
        "note": ("same-user BPR pairs resampled per epoch; validation_scores are "
                 "best-epoch scores under early stopping; test_scores come from the "
                 "final model state"),
    }
    return CandidateOutput(best_scores, checkpoint_state, trace, diagnostics, test_scores)
