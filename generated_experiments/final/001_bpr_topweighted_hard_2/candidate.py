import numpy as np
import math
import time
import collections
from src.models.fm_core import FMRanker
from src.models.sampling import sample_bpr_pairs
from src.models.features import build_features, feature_dimension
from src.experiments.contracts import CandidateOutput

def run(context, parameters):
    _fd = context.field_dimension
    _field_dim = sum(_fd) if isinstance(_fd, list) else int(_fd)
    batch_size = parameters['batch_size']
    epochs = parameters['epochs']
    lr = parameters['learning_rate']
    negs = parameters['negatives_per_positive']
    patience = parameters['patience']
    seed = parameters['seed']
    rng = np.random.RandomState(seed)
    spec = {}
    train_spec = dict(spec, split='train', field_offset=_field_dim)
    valid_spec = dict(spec, split='valid', field_offset=_field_dim)
    test_spec = dict(spec, split='test', field_offset=_field_dim) if context.test_x is not None else None
    random_valid_spec = dict(spec, split='random_valid', field_offset=_field_dim) if context.random_valid_x is not None else None
    train_features = build_features(context.train_x, train_spec)
    valid_features = build_features(context.valid_x, valid_spec)
    test_features = build_features(context.test_x, test_spec) if test_spec is not None else None
    random_valid_features = build_features(context.random_valid_x, random_valid_spec) if random_valid_spec is not None else None
    dimension = _field_dim + feature_dimension(spec)
    model = FMRanker(dimension, embedding_dim=16, learning_rate=lr, l2=1e-06, seed=seed)
    train_users = context.train_users
    train_labels = context.train_y

    def get_val_primary():
        scores = model.predict(valid_features)
        metrics = context.evaluate_validation(scores)
        if isinstance(metrics, dict):
            return float(metrics.get('primary', np.mean(list(metrics.values()))))
        return float(metrics)
    best_primary = -np.inf
    best_state = None
    patience_counter = 0
    training_trace = []
    sum_hardness = 0.0
    count_hardness = 0
    for epoch in range(epochs):
        pos_rows, neg_rows = sample_bpr_pairs(train_users, train_labels, rng, negs)
        n_pairs = len(pos_rows)
        if n_pairs == 0:
            training_trace.append({'epoch': epoch + 1, 'val_primary': get_val_primary(), 'loss': 0.0})
            continue
        perm = rng.permutation(n_pairs)
        pos_rows = pos_rows[perm]
        neg_rows = neg_rows[perm]
        epoch_loss = 0.0
        n_batches = int(np.ceil(n_pairs / batch_size))
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_pairs)
            if start >= end:
                continue
            p_idx = pos_rows[start:end]
            n_idx = neg_rows[start:end]
            pos_feat = train_features[p_idx]
            neg_feat = train_features[n_idx]
            pos_scores, _, _ = model.logits(pos_feat)
            neg_scores, _, _ = model.logits(neg_feat)
            diff = neg_scores - pos_scores
            diff_clipped = np.clip(diff, -30.0, 30.0)
            weights = 1.0 / (1.0 + np.exp(-diff_clipped))
            sum_hardness += np.sum(weights)
            count_hardness += len(weights)
            pos_neg = pos_scores - neg_scores
            pos_neg_clipped = np.clip(pos_neg, -30.0, 30.0)
            sig = 1.0 / (1.0 + np.exp(-pos_neg_clipped))
            grad = (sig - 1.0) * weights / (end - start)
            grad_v_pos, grad_w_pos, grad_b_pos = model.gradients(pos_feat, grad)
            grad_v_neg, grad_w_neg, grad_b_neg = model.gradients(neg_feat, -grad)
            model.apply_gradients(grad_v_pos + grad_v_neg, grad_w_pos + grad_w_neg, grad_b_pos + grad_b_neg)
            batch_loss = -np.log(np.clip(sig, 1e-12, 1.0))
            epoch_loss += np.sum(batch_loss * weights) / (end - start)
        val_primary = get_val_primary()
        training_trace.append({'epoch': epoch + 1, 'val_primary': val_primary, 'loss': epoch_loss})
        if val_primary > best_primary:
            best_primary = val_primary
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    validation_scores = model.predict(valid_features)
    test_scores = model.predict(test_features) if test_features is not None else None
    random_valid_scores = model.predict(random_valid_features) if random_valid_features is not None else None
    diagnostics = {'best_primary': best_primary, 'epochs_run': len(training_trace), 'n_pairs': n_pairs, 'hardness_weight_mean': sum_hardness / count_hardness if count_hardness > 0 else None}
    return CandidateOutput(validation_scores=validation_scores, checkpoint_state=model.state_dict(), training_trace=training_trace, diagnostics=diagnostics, test_scores=test_scores, random_validation_scores=random_valid_scores)
