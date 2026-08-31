import numpy as np
import math
import time
from collections import defaultdict
from src.models.fm_core import FMRanker
from src.models.sampling import sample_bpr_pairs
from src.models.features import build_features, feature_dimension
from src.experiments.contracts import CandidateOutput

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

def run(context, parameters):
    _fd = context.field_dimension
    _field_dim = sum(_fd) if isinstance(_fd, list) else int(_fd)
    spec = {'smoothing': parameters['smoothing'], 'scheme': parameters['scheme'], 'use_recency': parameters['use_recency'], 'use_tab_cross': parameters['use_tab_cross'], 'use_user_author': parameters['use_user_author'], 'use_user_rate': parameters['use_user_rate'], 'use_user_tab': parameters['use_user_tab'], 'use_video_age': parameters['use_video_age']}
    train_spec = dict(spec, split='train', field_offset=_field_dim)
    valid_spec = dict(spec, split='valid', field_offset=_field_dim)
    test_spec = dict(spec, split='test', field_offset=_field_dim)
    random_valid_spec = dict(spec, split='random_valid', field_offset=_field_dim)
    train_extra = build_features(context.train_x, train_spec)
    valid_extra = build_features(context.valid_x, valid_spec)
    test_extra = build_features(context.test_x, test_spec) if context.test_x is not None else None
    random_extra = build_features(context.random_valid_x, random_valid_spec) if context.random_valid_x is not None else None
    train_x = np.concatenate([context.train_x, train_extra], axis=1).astype(np.int32)
    valid_x = np.concatenate([context.valid_x, valid_extra], axis=1).astype(np.int32)
    if test_extra is not None:
        test_x = np.concatenate([context.test_x, test_extra], axis=1).astype(np.int32)
    else:
        test_x = None
    if random_extra is not None:
        random_valid_x = np.concatenate([context.random_valid_x, random_extra], axis=1).astype(np.int32)
    else:
        random_valid_x = None
    dim = _field_dim + feature_dimension(train_spec)
    model = FMRanker(dim, embedding_dim=parameters['k'], learning_rate=parameters['learning_rate'], seed=parameters['seed'])
    rng = np.random.RandomState(parameters['seed'])
    batch_size = parameters['batch_size']
    epochs = parameters['epochs']
    patience = parameters['patience']
    npp = parameters['negatives_per_positive']
    best_primary = -np.inf
    best_state = None
    best_epoch = -1
    no_improve = 0
    training_trace = []
    for epoch in range(epochs):
        pos_idx, neg_idx = sample_bpr_pairs(context.train_users, context.train_y, rng, npp)
        n_pairs = len(pos_idx)
        perm = rng.permutation(n_pairs)
        for start in range(0, n_pairs, batch_size):
            batch_perm = perm[start:start + batch_size]
            p_idx = pos_idx[batch_perm]
            n_idx = neg_idx[batch_perm]
            pos_features = train_x[p_idx]
            neg_features = train_x[n_idx]
            pos_scores = model.logits(pos_features)[0]
            neg_scores = model.logits(neg_features)[0]
            diff = pos_scores - neg_scores
            sig = _sigmoid(diff)
            grad = (sig - 1.0) / len(p_idx)
            grad_v_pos, grad_w_pos, grad_b_pos = model.gradients(pos_features, grad)
            grad_v_neg, grad_w_neg, grad_b_neg = model.gradients(neg_features, -grad)
            model.apply_gradients(grad_v_pos + grad_v_neg, grad_w_pos + grad_w_neg, grad_b_pos + grad_b_neg)
        valid_scores = model.predict(valid_x)
        metrics = context.evaluate_validation(valid_scores)
        gauc = metrics.get('GAUC', 0.0)
        ndcg = metrics.get('nDCG@5', 0.0)
        primary = (gauc + ndcg) / 2.0
        training_trace.append({'epoch': epoch, 'GAUC': gauc, 'nDCG@5': ndcg, 'primary': primary})
        if primary > best_primary:
            best_primary = primary
            best_state = model.state_dict()
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    model.load_state_dict(best_state)
    validation_scores = model.predict(valid_x)
    test_scores = model.predict(test_x) if test_x is not None else None
    random_validation_scores = model.predict(random_valid_x) if random_valid_x is not None else None
    diagnostics = {'best_epoch': best_epoch, 'best_primary': float(best_primary), 'feature_dim': int(feature_dimension(train_spec)), 'total_dim': int(dim), 'epochs_ran': len(training_trace)}
    return CandidateOutput(validation_scores=validation_scores, checkpoint_state=best_state, training_trace=training_trace, diagnostics=diagnostics, test_scores=test_scores, random_validation_scores=random_validation_scores)
