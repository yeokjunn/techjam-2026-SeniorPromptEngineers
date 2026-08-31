import numpy as np
import math
import time
from src.models.fm_core import FMRanker
from src.models.sampling import sample_bpr_pairs
from src.models.features import build_aux_labels
from src.experiments.contracts import CandidateOutput

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

def run(context, parameters):
    _fd = context.field_dimension
    _field_dim = sum(_fd) if isinstance(_fd, list) else int(_fd)
    seed = parameters.get('seed', 42)
    k = parameters.get('k', 16)
    lr = parameters.get('learning_rate', 0.0005)
    epochs = parameters.get('epochs', 8)
    batch_size = parameters.get('batch_size', 2048)
    patience = parameters.get('patience', 2)
    neg_per_pos = parameters.get('negatives_per_positive', 1)
    aux_weight = parameters.get('aux_weight', 0.05)
    use_heads = {'is_click': parameters.get('use_is_click', True), 'is_like': parameters.get('use_is_like', False), 'is_follow': parameters.get('use_is_follow', False), 'is_comment': parameters.get('use_is_comment', False), 'is_forward': parameters.get('use_is_forward', False), 'play_time': parameters.get('use_play_time', False)}
    active_heads = [h for h, on in use_heads.items() if on]
    if not active_heads:
        active_heads = ['is_click']
        use_heads['is_click'] = True
    aux_spec = {'split': 'train'}
    for h in ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'play_time']:
        aux_spec['use_' + h] = use_heads[h]
    aux_train = build_aux_labels(context.train_x, aux_spec)
    model = FMRanker(_field_dim, embedding_dim=k, learning_rate=lr, seed=seed)
    rng = np.random.RandomState(seed)
    best_state = None
    best_primary = -1.0
    best_epoch = 0
    no_improve = 0
    trace = []
    start_time = time.time()
    for epoch in range(epochs):
        pos_rows, neg_rows = sample_bpr_pairs(context.train_users, context.train_y, rng, neg_per_pos)
        n_pairs = len(pos_rows)
        if n_pairs == 0:
            continue
        total_bpr = 0.0
        total_aux = 0.0
        for start in range(0, n_pairs, batch_size):
            end = min(start + batch_size, n_pairs)
            batch_pos = pos_rows[start:end]
            batch_neg = neg_rows[start:end]
            pos_x = context.train_x[batch_pos]
            neg_x = context.train_x[batch_neg]
            pos_score = model.logits(pos_x)[0]
            neg_score = model.logits(neg_x)[0]
            diff = np.clip(pos_score - neg_score, -30.0, 30.0)
            sig_diff = _sigmoid(diff)
            batch_n = end - start
            grad_bpr = (sig_diff - 1.0) / batch_n
            if aux_weight > 0:
                pos_aux = aux_train[batch_pos]
                neg_aux = aux_train[batch_neg]
                pos_sig = _sigmoid(pos_score)
                neg_sig = _sigmoid(neg_score)
                grad_aux_pos = (pos_sig - pos_aux) * (aux_weight / batch_n)
                grad_aux_neg = (neg_sig - neg_aux) * (aux_weight / batch_n)
                grad_aux_pos_row = np.sum(grad_aux_pos, axis=1)
                grad_aux_neg_row = np.sum(grad_aux_neg, axis=1)
                grad_pos = grad_bpr + grad_aux_pos_row
                grad_neg = -grad_bpr + grad_aux_neg_row
                eps = 1e-08
                aux_loss_pos = -np.mean(np.sum(pos_aux * np.log(pos_sig + eps) + (1 - pos_aux) * np.log(1 - pos_sig + eps), axis=1))
                aux_loss_neg = -np.mean(np.sum(neg_aux * np.log(neg_sig + eps) + (1 - neg_aux) * np.log(1 - neg_sig + eps), axis=1))
                total_aux += (aux_loss_pos + aux_loss_neg) * (end - start) / 2.0
            else:
                grad_pos = grad_bpr
                grad_neg = -grad_bpr
                total_aux += 0.0
            grad_v_p, grad_w_p, grad_b_p = model.gradients(pos_x, grad_pos)
            grad_v_n, grad_w_n, grad_b_n = model.gradients(neg_x, grad_neg)
            grad_v = grad_v_p + grad_v_n
            grad_w = grad_w_p + grad_w_n
            grad_b = grad_b_p + grad_b_n
            model.apply_gradients(grad_v, grad_w, grad_b)
            total_bpr += -np.log(sig_diff + 1e-08).sum()
        valid_scores = model.predict(context.valid_x)
        metrics = context.evaluate_validation(valid_scores)
        if isinstance(metrics, dict):
            primary = metrics.get('primary', 0.0)
            if 'gauc' in metrics and 'ndcg@5' in metrics:
                primary = (metrics['gauc'] + metrics['ndcg@5']) / 2.0
        else:
            primary = float(metrics)
        trace.append({'epoch': epoch + 1, 'metrics': metrics, 'bpr_loss': total_bpr / n_pairs, 'aux_loss': total_aux / n_pairs, 'primary': primary})
        if primary > best_primary:
            best_primary = primary
            best_epoch = epoch + 1
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_state = model.state_dict()
    validation_scores = model.predict(context.valid_x)
    test_scores = model.predict(context.test_x) if context.test_x is not None else None
    random_validation_scores = model.predict(context.random_valid_x) if context.random_valid_x is not None else None
    diagnostics = {'best_epoch': best_epoch, 'best_primary': best_primary, 'n_pairs': n_pairs, 'training_time_s': time.time() - start_time, 'aux_weight': aux_weight, 'active_heads': active_heads}
    return CandidateOutput(validation_scores=validation_scores, checkpoint_state=best_state, training_trace=trace, diagnostics=diagnostics, test_scores=test_scores, random_validation_scores=random_validation_scores)
