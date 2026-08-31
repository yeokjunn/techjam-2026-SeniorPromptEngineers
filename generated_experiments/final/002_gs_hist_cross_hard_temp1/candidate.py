import numpy as np
import math
import time
import collections
from src.models.fm_core import FMRanker
from src.models.sampling import sample_softmax_groups
from src.models.features import build_features
from src.experiments.contracts import CandidateOutput

def _autofix_hasattr_primary(obj):
    try:
        obj.primary
    except AttributeError:
        return False
    return True

def _autofix_hasattr_gauc(obj):
    try:
        obj.gauc
    except AttributeError:
        return False
    return True

def _autofix_hasattr_ndcg(obj):
    try:
        obj.ndcg
    except AttributeError:
        return False
    return True

def run(context, parameters):
    _fd = context.field_dimension
    _field_dim = sum(_fd) if isinstance(_fd, list) else int(_fd)
    seed = int(parameters['seed'])
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    try:
        spec = context.spec
    except AttributeError:
        spec = None
    if spec is None:
        n_base = context.train_x.shape[1]
        spec = {'fields': ['f{}'.format(i) for i in range(n_base)]}
    elif not isinstance(spec, dict) or 'fields' not in spec:
        n_base = context.train_x.shape[1]
        spec = {'fields': ['f{}'.format(i) for i in range(n_base)]}
    train_spec = dict(spec, split='train', field_offset=0)
    valid_spec = dict(spec, split='valid', field_offset=0)
    test_spec = dict(spec, split='test', field_offset=0)
    random_valid_spec = dict(spec, split='random_valid', field_offset=0)
    train_features_base = build_features(context.train_x, train_spec)
    valid_features_base = build_features(context.valid_x, valid_spec)
    test_features_base = None
    random_valid_features_base = None
    if context.test_x is not None:
        test_features_base = build_features(context.test_x, test_spec)
    if context.random_valid_x is not None:
        random_valid_features_base = build_features(context.random_valid_x, random_valid_spec)
    train_users = context.train_x[:, 0].astype(np.int64)
    train_videos = context.train_x[:, 1].astype(np.int64)
    hist_counts = {}
    hist_sums = {}
    for u, v, y in zip(train_users, train_videos, context.train_y):
        key = (int(u), int(v))
        if key in hist_counts:
            hist_counts[key] += 1
            hist_sums[key] += float(y)
        else:
            hist_counts[key] = 1
            hist_sums[key] = float(y)
    alpha = 1.0
    beta = 2.0
    num_buckets = 10
    field_offset = _field_dim

    def encode_history(users, videos):
        n = len(users)
        hist = np.zeros(n, dtype=np.int64)
        for i in range(n):
            key = (int(users[i]), int(videos[i]))
            if key in hist_counts:
                cnt = hist_counts[key]
                sm = hist_sums[key]
                ratio = (sm + alpha) / (cnt + beta)
                idx = int(ratio * num_buckets)
                if idx > num_buckets - 1:
                    idx = num_buckets - 1
                elif idx < 0:
                    idx = 0
                hist[i] = field_offset + idx
            else:
                hist[i] = field_offset
        return hist[:, None]
    train_features = np.concatenate([train_features_base, encode_history(train_users, train_videos)], axis=1)
    valid_features = np.concatenate([valid_features_base, encode_history(context.valid_x[:, 0].astype(np.int64), context.valid_x[:, 1].astype(np.int64))], axis=1)
    test_features = None
    random_valid_features = None
    if test_features_base is not None:
        test_features = np.concatenate([test_features_base, encode_history(context.test_x[:, 0].astype(np.int64), context.test_x[:, 1].astype(np.int64))], axis=1)
    if random_valid_features_base is not None:
        random_valid_features = np.concatenate([random_valid_features_base, encode_history(context.random_valid_x[:, 0].astype(np.int64), context.random_valid_x[:, 1].astype(np.int64))], axis=1)
    dimension = _field_dim + num_buckets
    model = FMRanker(dimension, embedding_dim=parameters['k'], learning_rate=parameters['learning_rate'], l2=1e-06, seed=seed)
    users = context.train_x[:, 0]
    labels = context.train_y
    K = parameters['negatives_per_group']
    temp = parameters['temperature']
    batch_size = parameters['batch_size']
    epochs = parameters['epochs']
    patience = parameters['patience']
    best_primary = -1.0
    best_state = None
    best_epoch = 0
    wait = 0
    training_trace = []

    def get_primary(metrics):
        if _autofix_hasattr_primary(metrics):
            return metrics.primary
        if isinstance(metrics, dict):
            return metrics['primary']
        if len(metrics) >= 3:
            return metrics[2]
        if len(metrics) >= 1:
            return metrics[0]
        return 0.0
    for epoch in range(epochs):
        positives, negatives = sample_softmax_groups(users, labels, rng, K)
        n_groups = positives.shape[0]
        if n_groups == 0:
            break
        perm = rng.permutation(n_groups)
        positives = positives[perm]
        negatives = negatives[perm]
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_groups, batch_size):
            end = min(start + batch_size, n_groups)
            batch_pos = positives[start:end]
            batch_neg = negatives[start:end]
            pos_feat = train_features[batch_pos]
            neg_flat = batch_neg.reshape(-1)
            neg_feat = train_features[neg_flat]
            all_rows = np.concatenate([pos_feat, neg_feat], axis=0)
            scores_all = model.logits(all_rows)[0]
            pos_scores = scores_all[:batch_pos.shape[0]]
            neg_scores = scores_all[batch_pos.shape[0]:].reshape(batch_pos.shape[0], K)
            logits = np.concatenate([pos_scores[:, None], neg_scores], axis=1) / temp
            logits_max = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            p = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            loss = -np.log(p[:, 0] + 1e-12)
            mean_loss = np.mean(loss)
            epoch_loss += mean_loss
            n_batches += 1
            scale = 1.0 / (batch_pos.shape[0] * temp)
            grad_pos = (p[:, 0] - 1.0) * scale
            grad_neg = p[:, 1:] * scale
            grad_neg_flat = grad_neg.reshape(-1)
            score_grads_all = np.concatenate([grad_pos, grad_neg_flat])
            grad_v, grad_w, grad_b = model.gradients(all_rows, score_grads_all)
            model.apply_gradients(grad_v, grad_w, grad_b)
        if n_batches == 0:
            break
        avg_loss = epoch_loss / n_batches
        valid_scores = model.predict(valid_features)
        metrics = context.evaluate_validation(valid_scores)
        primary = get_primary(metrics)
        trace = {'epoch': epoch + 1, 'loss': avg_loss, 'primary': primary}
        if _autofix_hasattr_gauc(metrics):
            trace['gauc'] = metrics.gauc
        if _autofix_hasattr_ndcg(metrics):
            trace['ndcg'] = metrics.ndcg
        training_trace.append(trace)
        if primary > best_primary:
            best_primary = primary
            best_state = model.state_dict()
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    valid_scores_best = model.predict(valid_features)
    test_scores = None
    if test_features is not None:
        test_scores = model.predict(test_features)
    random_valid_scores = None
    if random_valid_features is not None:
        random_valid_scores = model.predict(random_valid_features)
    diagnostics = {'best_epoch': best_epoch, 'best_primary': best_primary, 'history_buckets': num_buckets, 'history_type': 'user x video long_view rate (train only)', 'negatives_per_group': K, 'temperature': temp}
    return CandidateOutput(validation_scores=valid_scores_best, checkpoint_state=best_state if best_state is not None else model.state_dict(), training_trace=training_trace, diagnostics=diagnostics, test_scores=test_scores, random_validation_scores=random_valid_scores)
