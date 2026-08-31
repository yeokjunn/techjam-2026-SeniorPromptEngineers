import numpy as np
import time
from src.models.fm_core import FMRanker
from src.models.sampling import sample_bpr_pairs
from src.models.features import build_features
from src.experiments.contracts import CandidateOutput

def run(context, parameters):
    _fd = context.field_dimension
    _field_dim = sum(_fd) if isinstance(_fd, list) else int(_fd)
    seed = parameters['seed']
    lr = parameters['learning_rate']
    epochs = parameters['epochs']
    batch_size = parameters['batch_size']
    negatives_per_positive = parameters['negatives_per_positive']
    patience = parameters['patience']
    k = parameters['k']
    rng = np.random.RandomState(seed)
    spec = parameters['spec']
    train_spec = dict(spec, split='train', field_offset=_field_dim)
    valid_spec = dict(spec, split='valid', field_offset=_field_dim)
    test_spec = dict(spec, split='test', field_offset=_field_dim) if context.test_x is not None else None
    random_valid_spec = dict(spec, split='random_valid', field_offset=_field_dim) if context.random_valid_x is not None else None
    train_features = build_features(context.train_x, train_spec).astype(np.int32)
    valid_features = build_features(context.valid_x, valid_spec).astype(np.int32)
    test_features = build_features(context.test_x, test_spec).astype(np.int32) if test_spec is not None else None
    random_valid_features = build_features(context.random_valid_x, random_valid_spec).astype(np.int32) if random_valid_spec is not None else None
    dimension = _field_dim
    model = FMRanker(dimension, embedding_dim=k, learning_rate=lr, l2=1e-06, seed=seed)
    users = np.asarray(context.train_users, dtype=np.int32)
    labels = np.asarray(context.train_y, dtype=np.float32)
    best_primary = -1.0
    best_state = None
    best_epoch = -1
    no_improve = 0
    train_trace = []
    for epoch in range(epochs):
        pos_rows, neg_rows = sample_bpr_pairs(users, labels, rng, negatives_per_positive)
        n_pairs = len(pos_rows)
        if n_pairs == 0:
            break
        epoch_loss = 0.0
        for start in range(0, n_pairs, batch_size):
            end = min(start + batch_size, n_pairs)
            batch_pos = pos_rows[start:end]
            batch_neg = neg_rows[start:end]
            pos_x = train_features[batch_pos]
            neg_x = train_features[batch_neg]
            pos_scores, _, _ = model.logits(pos_x)
            neg_scores, _, _ = model.logits(neg_x)
            d = pos_scores - neg_scores
            clipped = np.clip(d, -30.0, 30.0)
            s = 1.0 / (1.0 + np.exp(-clipped))
            weight = np.clip(1.0 - np.abs(clipped), 0.0, 1.0) + 0.01
            gradient = (s - 1.0) * weight / batch_size
            grad_v_p, grad_w_p, grad_b_p = model.gradients(pos_x, gradient)
            grad_v_n, grad_w_n, grad_b_n = model.gradients(neg_x, -gradient)
            model.apply_gradients(grad_v_p + grad_v_n, grad_w_p + grad_w_n, grad_b_p + grad_b_n)
            epoch_loss += np.mean(np.logaddexp(0.0, -clipped)) * (end - start)
        valid_scores = model.predict(valid_features)
        metrics = context.evaluate_validation(valid_scores)
        primary = metrics.get('primary', (metrics.get('gauc', 0) + metrics.get('ndcg@5', 0)) / 2)
        avg_loss = epoch_loss / n_pairs if n_pairs > 0 else 0.0
        train_trace.append({'epoch': epoch, 'loss': avg_loss, 'primary': primary, 'gauc': metrics.get('gauc'), 'ndcg@5': metrics.get('ndcg@5')})
        if primary > best_primary:
            best_primary = primary
            best_state = model.state_dict()
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)
    validation_scores = model.predict(valid_features)
    assert np.all(np.isfinite(validation_scores)), 'Validation scores contain non-finite values'
    test_scores = model.predict(test_features) if test_features is not None else None
    random_valid_scores = model.predict(random_valid_features) if random_valid_features is not None else None
    diagnostics = {'best_epoch': best_epoch, 'best_primary': best_primary, 'final_primary': context.evaluate_validation(validation_scores).get('primary', 0.0)}
    return CandidateOutput(validation_scores=validation_scores, checkpoint_state=best_state, training_trace=train_trace, diagnostics=diagnostics, test_scores=test_scores, random_validation_scores=random_valid_scores)
