import time
import numpy as np
from src.models.fm_core import FMRanker
from src.models.sampling import sample_softmax_groups
from src.experiments.contracts import CandidateOutput


def _primary(metrics):
    if isinstance(metrics, dict):
        if 'primary' in metrics:
            return float(metrics['primary'])
        vals = [float(v) for v in metrics.values()]
        return float(np.mean(vals)) if vals else 0.0
    return float(metrics)


def run(context, parameters):
    t0 = time.time()
    seed = int(parameters['seed'])
    epochs = int(parameters['epochs'])
    patience = int(parameters['patience'])
    k_neg = int(parameters['negatives_per_group'])
    temperature = float(parameters['temperature'])
    batch_size = int(parameters['batch_size'])

    model = FMRanker(int(context.field_dimension),
                     embedding_dim=int(parameters['k']),
                     learning_rate=float(parameters['learning_rate']),
                     seed=seed)
    train_x = context.train_x
    train_y = context.train_y
    users = context.train_users
    valid_x = context.valid_x
    test_x = context.test_x
    rng = np.random.default_rng(seed)

    pos_idx, neg_groups = sample_softmax_groups(
        users, train_y, rng, negatives_per_group=k_neg)
    n_groups = int(pos_idx.shape[0])
    steps_per_epoch = int(np.ceil(n_groups / batch_size)) if n_groups else 0

    def score_all():
        v = model.predict(valid_x)
        t = None if test_x is None else model.predict(test_x)
        return v, t

    best_primary = -np.inf
    best_valid = None
    best_test = None
    best_epoch = -1
    no_improve = 0
    epochs_run = 0
    trace = []

    for epoch in range(epochs):
        epochs_run = epoch + 1
        order = rng.permutation(n_groups)
        for s in range(steps_per_epoch):
            b = order[s * batch_size:(s + 1) * batch_size]
            pf = train_x[pos_idx[b]]
            nf = train_x[neg_groups[b]]
            B = pf.shape[0]
            nff = nf.reshape(B * k_neg, -1)
            ps = model.predict(pf)
            ns = model.predict(nff).reshape(B, k_neg)
            logits = np.concatenate([ps[:, None], ns], axis=1) / temperature
            logits -= logits.max(axis=1, keepdims=True)
            e = np.exp(logits)
            p = e / e.sum(axis=1, keepdims=True)
            g = p / temperature
            g[:, 0] -= 1.0 / temperature
            gp = np.ascontiguousarray(g[:, 0])
            gn = np.ascontiguousarray(g[:, 1:].reshape(B * k_neg))
            gv, gw, gb = model.gradients(pf, gp)
            gv2, gw2, gb2 = model.gradients(nff, gn)
            model.apply_gradients(gv + gv2, gw + gw2, gb + gb2)

        valid_scores, test_scores = score_all()
        primary = _primary(context.evaluate_validation(valid_scores))
        trace.append({'epoch': int(epoch + 1), 'primary': float(primary)})
        if primary > best_primary:
            best_primary = primary
            best_valid = valid_scores.copy()
            best_test = None if test_scores is None else test_scores.copy()
            best_epoch = epoch + 1
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_valid is None:
        best_valid, best_test = score_all()
        best_epoch = epochs_run
        best_primary = _primary(context.evaluate_validation(best_valid))

    diagnostics = {
        'negatives_per_group': int(k_neg),
        'groups_per_epoch': n_groups,
        'steps_per_epoch': steps_per_epoch,
        'epochs_run': int(epochs_run),
        'best_epoch': int(best_epoch),
        'best_primary': float(best_primary),
        'temperature': float(temperature),
        'batch_size': int(batch_size),
        'embedding_dim': int(parameters['k']),
        'compute_note': 'K=8 doubles rows scored per step vs K=4 at fixed batch_size',
        'train_seconds': float(time.time() - t0),
    }
    checkpoint = {
        'best_valid_scores': np.asarray(best_valid, dtype=np.float64),
        'best_primary': np.array(float(best_primary)),
        'best_epoch': np.array(int(best_epoch)),
    }
    if best_test is not None:
        checkpoint['best_test_scores'] = np.asarray(best_test, dtype=np.float64)
    return CandidateOutput(np.asarray(best_valid, dtype=np.float64),
                           checkpoint, trace, diagnostics, best_test)
