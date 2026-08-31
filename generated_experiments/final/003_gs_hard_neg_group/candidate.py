import numpy as np
import math
import time
from src.models.fm_core import FMRanker
from src.models.sampling import sample_softmax_groups
from src.models.features import build_features
from src.experiments.contracts import CandidateOutput
SPEC = {'user_id': 0, 'video_id': 1}

def run(context, parameters):
    _fd = context.field_dimension
    _field_dim = sum(_fd) if isinstance(_fd, list) else int(_fd)
    seed = parameters['seed']
    lr = parameters['learning_rate']
    epochs = parameters['epochs']
    batch_size = parameters['batch_size']
    K = parameters['negatives_per_group']
    temperature = parameters['temperature']
    patience = parameters['patience']
    rng = np.random.default_rng(seed)
    train_spec = dict(SPEC, split='train', field_offset=0)
    valid_spec = dict(SPEC, split='valid', field_offset=0)
    test_spec = dict(SPEC, split='test', field_offset=0)
    random_valid_spec = dict(SPEC, split='random_valid', field_offset=0)
    train_features = build_features(context.train_x, train_spec).astype(np.int32)
    valid_features = build_features(context.valid_x, valid_spec).astype(np.int32) if context.valid_x is not None else None
    test_features = build_features(context.test_x, test_spec).astype(np.int32) if context.test_x is not None else None
    random_valid_features = build_features(context.random_valid_x, random_valid_spec).astype(np.int32) if context.random_valid_x is not None else None
    model = FMRanker(_field_dim, embedding_dim=16, learning_rate=lr, l2=1e-06, seed=seed)
    best_primary = -np.inf
    best_state = None
    best_epoch = -1
    patience_counter = 0
    training_trace = []
    for epoch in range(epochs):
        start_time = time.time()
        pos_idx, neg_idx = sample_softmax_groups(context.train_users, context.train_y, rng, K)
        n_groups = len(pos_idx)
        if n_groups == 0:
            continue
        perm = rng.permutation(n_groups)
        pos_idx = pos_idx[perm]
        neg_idx = neg_idx[perm]
        epoch_loss = 0.0
        for start in range(0, n_groups, batch_size):
            end = min(start + batch_size, n_groups)
            pos_batch = pos_idx[start:end]
            neg_batch = neg_idx[start:end]
            B = len(pos_batch)
            pos_feat = train_features[pos_batch]
            neg_flat = neg_batch.reshape(-1)
            neg_feat = train_features[neg_flat]
            all_feat = np.concatenate([pos_feat, neg_feat], axis=0)
            all_scores = model.predict(all_feat)
            scores_group = all_scores.reshape(B, 1 + K)
            logits = scores_group / temperature
            shifted = logits - np.max(logits, axis=1, keepdims=True)
            exp = np.exp(shifted)
            softmax = exp / np.sum(exp, axis=1, keepdims=True)
            loss = -np.log(softmax[:, 0] + 1e-12).mean()
            one_hot = np.zeros_like(softmax)
            one_hot[:, 0] = 1.0
            score_grad = (softmax - one_hot) / temperature
            all_grads = score_grad.reshape(-1) / B
            grad_v, grad_w, grad_b = model.gradients(all_feat, all_grads)
            model.apply_gradients(grad_v, grad_w, grad_b)
            epoch_loss += loss * B
        epoch_loss /= max(n_groups, 1)
        if valid_features is not None:
            valid_scores = model.predict(valid_features)
            if not np.all(np.isfinite(valid_scores)):
                valid_scores = np.nan_to_num(valid_scores, nan=0.0, posinf=1000000.0, neginf=-1000000.0)
            valid_primary = context.evaluate_validation(valid_scores)
        else:
            valid_primary = -np.inf
        training_trace.append({'epoch': epoch, 'loss': float(epoch_loss), 'validation_primary': float(valid_primary), 'time': time.time() - start_time})
        if valid_primary > best_primary:
            best_primary = valid_primary
            best_epoch = epoch
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    validation_scores = model.predict(valid_features) if valid_features is not None else np.array([])
    if valid_features is not None and (not np.all(np.isfinite(validation_scores))):
        validation_scores = np.nan_to_num(validation_scores, nan=0.0, posinf=1000000.0, neginf=-1000000.0)
    test_scores = None
    if test_features is not None:
        test_scores = model.predict(test_features)
        if not np.all(np.isfinite(test_scores)):
            test_scores = np.nan_to_num(test_scores, nan=0.0, posinf=1000000.0, neginf=-1000000.0)
    random_validation_scores = None
    if random_valid_features is not None:
        random_validation_scores = model.predict(random_valid_features)
        if not np.all(np.isfinite(random_validation_scores)):
            random_validation_scores = np.nan_to_num(random_validation_scores, nan=0.0, posinf=1000000.0, neginf=-1000000.0)
    diagnostics = {'best_epoch': best_epoch, 'best_primary': best_primary, 'n_groups': n_groups, 'temperature': temperature, 'negatives_per_group': K}
    return CandidateOutput(validation_scores=validation_scores, checkpoint_state=best_state, training_trace=training_trace, diagnostics=diagnostics, test_scores=test_scores, random_validation_scores=random_validation_scores)
