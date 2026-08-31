import numpy as np
from src.models.fm_core import FMRanker
from src.models.sampling import sample_softmax_groups
from src.experiments.contracts import CandidateOutput

def _primary(metrics):
    if isinstance(metrics, dict):
        if 'primary' in metrics:
            return float(metrics['primary'])
        ga = metrics.get('GAUC')
        if ga is None:
            ga = metrics.get('gauc')
        ng = metrics.get('nDCG@5')
        if ng is None:
            ng = metrics.get('ndcg@5')
        if ng is None:
            ng = metrics.get('ndcg5')
        if ga is not None and ng is not None:
            return 0.5 * (float(ga) + float(ng))
        if len(metrics) > 0:
            return float(list(metrics.values())[0])
        return 0.0
    return float(metrics)

def _metric(metrics, key):
    if isinstance(metrics, dict):
        for cand in (key, key.lower()):
            if cand in metrics:
                return float(metrics[cand])
        return float('nan')
    return float(metrics)

def _group_softmax(logits, temperature):
    scaled = np.asarray(logits, dtype=np.float32) / float(temperature)
    scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp = np.exp(scaled)
    probs = exp / np.sum(exp, axis=-1, keepdims=True)
    onehot = np.zeros_like(probs)
    onehot[:, 0] = 1.0
    grads = (probs - onehot) / float(temperature)
    losses = -np.log(np.maximum(probs[:, 0], 1e-12))
    return (grads, losses)

def run(context, parameters):
    _fd = context.field_dimension
    _field_dim = sum(_fd) if isinstance(_fd, list) else int(_fd)
    seed = int(parameters['seed'])
    lr = float(parameters['learning_rate'])
    epochs = int(parameters['epochs'])
    batch_size = int(parameters['batch_size'])
    patience = int(parameters['patience'])
    k = int(parameters['k'])
    n_neg = int(parameters['negatives_per_group'])
    temperature = float(parameters['temperature'])
    train_features = np.ascontiguousarray(context.train_x, dtype=np.int32)
    valid_features = np.ascontiguousarray(context.valid_x, dtype=np.int32)
    total_dim = int(_field_dim)
    model = FMRanker(total_dim, embedding_dim=k, learning_rate=lr, l2=1e-06, seed=seed)
    users = np.asarray(context.train_users).ravel()
    labels = np.asarray(context.train_y, dtype=np.float32).ravel()
    rng = np.random.RandomState(seed)
    best_primary = -np.inf
    best_state = None
    best_val_scores = None
    no_improve = 0
    trace = []
    for epoch in range(1, int(epochs) + 1):
        pos_rows, neg_rows = sample_softmax_groups(users, labels, rng, n_neg)
        num_groups = len(pos_rows)
        if num_groups == 0:
            break
        order = rng.permutation(num_groups)
        epoch_loss = 0.0
        num_samples = 0
        for start in range(0, num_groups, batch_size):
            idx = order[start:start + batch_size]
            pos_batch = pos_rows[idx]
            neg_batch = neg_rows[idx]
            row_ids = np.concatenate([pos_batch[:, None], neg_batch], axis=1).reshape(-1)
            features = train_features[row_ids]
            scores = model.logits(features)[0]
            scores = scores.reshape(-1, n_neg + 1)
            grads, losses = _group_softmax(scores, temperature)
            score_grads = grads.reshape(-1).astype(np.float32)
            gv, gw, gb = model.gradients(features, score_grads)
            model.apply_gradients(gv, gw, gb)
            epoch_loss += float(np.sum(losses))
            num_samples += len(pos_batch)
        avg_loss = epoch_loss / max(num_samples, 1)
        val_scores = model.predict(valid_features)
        metrics = context.evaluate_validation(val_scores)
        primary = _primary(metrics)
        gauc = _metric(metrics, 'GAUC')
        ndcg = _metric(metrics, 'nDCG@5')
        trace.append({'epoch': int(epoch), 'train_loss': float(avg_loss), 'validation_primary': float(primary), 'validation_gauc': float(gauc), 'validation_ndcg': float(ndcg)})
        if primary > best_primary:
            best_primary = primary
            best_state = model.state_dict()
            best_val_scores = np.array(val_scores, copy=True)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(patience):
                break
    if best_state is None:
        best_state = model.state_dict()
        best_val_scores = model.predict(valid_features)
    model.load_state_dict(best_state)
    val_scores = model.predict(valid_features)
    test_scores = None
    if context.test_x is not None:
        test_features = np.ascontiguousarray(context.test_x, dtype=np.int32)
        test_scores = model.predict(test_features)
    random_valid_scores = None
    if context.random_valid_x is not None:
        rv_features = np.ascontiguousarray(context.random_valid_x, dtype=np.int32)
        random_valid_scores = model.predict(rv_features)
    diagnostics = {'temperature': temperature, 'negatives_per_group': n_neg, 'epochs_run': len(trace), 'best_validation_primary': float(best_primary)}
    return CandidateOutput(validation_scores=val_scores, checkpoint_state=best_state, training_trace=trace, diagnostics=diagnostics, test_scores=test_scores, random_validation_scores=random_valid_scores)
