import math
import time
import numpy as np
from src.models.fm_core import FMRanker
from src.models.sampling import sample_bpr_pairs
from src.models.features import build_features, feature_dimension
from src.experiments.contracts import CandidateOutput

def _build_spec(parameters):
    keys = ['scheme', 'smoothing', 'use_user_rate', 'use_user_author', 'use_user_tab', 'use_recency', 'use_video_age', 'use_tab_cross']
    return {k: parameters[k] for k in keys if k in parameters}

def _sigmoid(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))

def _eval_metrics(evaluate, scores):
    res = evaluate(scores)
    if isinstance(res, dict):
        gauc = res.get('GAUC') or res.get('gauc')
        ndcg = res.get('nDCG@5') or res.get('ndcg@5') or res.get('ndcg')
        if gauc is not None and ndcg is not None:
            try:
                return (float(gauc), float(ndcg))
            except Exception:
                pass
    try:
        items = list(res)
    except TypeError:
        v = float(res)
        return (v, v)
    if len(items) >= 2:
        try:
            g = float(items[0])
            n = float(items[1])
            return (g, n)
        except Exception:
            pass
    for item in items:
        try:
            v = float(item)
            return (v, v)
        except Exception:
            pass
    raise ValueError('evaluate_validation returned no parseable values')

def _concat(rows, extra):
    if extra is None:
        return rows
    if rows is None:
        return None
    return np.concatenate([rows, extra], axis=1).astype(np.int32)

def run(context, parameters):
    _fd = context.field_dimension
    _field_dim = sum(_fd) if isinstance(_fd, list) else int(_fd)
    seed = int(parameters['seed'])
    rng = np.random.default_rng(seed)
    lr = float(parameters['learning_rate'])
    epochs = int(parameters['epochs'])
    batch_size = int(parameters['batch_size'])
    patience = int(parameters['patience'])
    k = int(parameters.get('k', 16))
    npp = int(parameters.get('negatives_per_positive', 1))
    base_spec = _build_spec(parameters)
    train_spec = dict(base_spec, split='train', field_offset=_field_dim)
    valid_spec = dict(base_spec, split='valid', field_offset=_field_dim)
    train_extra = build_features(context.train_x, train_spec)
    valid_extra = build_features(context.valid_x, valid_spec)
    train_w = _concat(context.train_x, train_extra)
    valid_w = _concat(context.valid_x, valid_extra)
    test_w = None
    random_w = None
    if context.test_x is not None:
        test_spec = dict(base_spec, split='test', field_offset=_field_dim)
        test_extra = build_features(context.test_x, test_spec)
        test_w = _concat(context.test_x, test_extra)
    if context.random_valid_x is not None:
        random_valid_spec = dict(base_spec, split='random_valid', field_offset=_field_dim)
        random_extra = build_features(context.random_valid_x, random_valid_spec)
        random_w = _concat(context.random_valid_x, random_extra)
    extra_dim = feature_dimension(train_spec)
    dimension = int(_field_dim) + int(extra_dim)
    model = FMRanker(dimension, embedding_dim=k, learning_rate=lr, l2=1e-06, seed=seed)
    best_primary = -np.inf
    best_gauc = None
    best_ndcg = None
    best_epoch = 0
    best_state = None
    patience_counter = 0
    training_trace = []
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        positive_rows, negative_rows = sample_bpr_pairs(context.train_users, context.train_y, rng, npp)
        n_pairs = len(positive_rows)
        total_loss = 0.0
        for start in range(0, n_pairs, batch_size):
            end = min(start + batch_size, n_pairs)
            pos_idx = positive_rows[start:end]
            neg_idx = negative_rows[start:end]
            pos_feat = train_w[pos_idx]
            neg_feat = train_w[neg_idx]
            pos_scores, pos_embeds, pos_summed = model.logits(pos_feat)
            neg_scores, neg_embeds, neg_summed = model.logits(neg_feat)
            diff = pos_scores - neg_scores
            prob = _sigmoid(diff)
            grad = (prob - 1.0) / float(len(pos_idx))
            batch_loss = -np.mean(np.log(np.clip(prob, 1e-06, 1.0)))
            total_loss += float(batch_loss) * len(pos_idx)
            grad_v_pos, grad_w_pos, grad_b_pos = model.gradients(pos_feat, grad, pos_embeds, pos_summed)
            grad_v_neg, grad_w_neg, grad_b_neg = model.gradients(neg_feat, -grad, neg_embeds, neg_summed)
            model.apply_gradients(grad_v_pos + grad_v_neg, grad_w_pos + grad_w_neg, grad_b_pos + grad_b_neg)
        mean_loss = total_loss / max(n_pairs, 1)
        valid_scores = model.predict(valid_w)
        gauc, ndcg = _eval_metrics(context.evaluate_validation, valid_scores)
        primary = 0.5 * (gauc + ndcg)
        training_trace.append({'epoch': epoch, 'train_loss': round(mean_loss, 6), 'gauc': round(float(gauc), 6), 'ndcg@5': round(float(ndcg), 6), 'primary': round(float(primary), 6)})
        if primary > best_primary + 1e-06:
            best_primary = float(primary)
            best_gauc = float(gauc)
            best_ndcg = float(ndcg)
            best_epoch = epoch
            best_state = {name: arr.copy() for name, arr in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    if best_state is None:
        best_state = {name: arr.copy() for name, arr in model.state_dict().items()}
    else:
        model.load_state_dict(best_state)
    valid_scores = model.predict(valid_w)
    test_scores = model.predict(test_w) if test_w is not None else None
    random_scores = model.predict(random_w) if random_w is not None else None
    enabled = sorted((g for g in ('user_rate', 'user_author', 'user_tab', 'recency', 'video_age', 'tab_cross') if parameters.get('use_' + g, False)))
    diagnostics = {'feature_dimension': int(extra_dim), 'enabled_groups': enabled, 'scheme': parameters.get('scheme'), 'smoothing': parameters.get('smoothing'), 'seed': seed, 'learning_rate': lr, 'batch_size': batch_size, 'epochs_used': training_trace[-1]['epoch'] if training_trace else 0, 'best_epoch': int(best_epoch), 'best_gauc': None if best_gauc is None else round(best_gauc, 6), 'best_ndcg@5': None if best_ndcg is None else round(best_ndcg, 6), 'best_primary': None if math.isinf(best_primary) else round(best_primary, 6), 'elapsed_seconds': round(time.time() - start_time, 3)}
    return CandidateOutput(validation_scores=valid_scores, checkpoint_state=best_state, training_trace=training_trace, diagnostics=diagnostics, test_scores=test_scores, random_validation_scores=random_scores)
