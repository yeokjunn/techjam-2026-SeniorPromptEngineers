import numpy as np
import math
import time
import collections
from src.models.fm_core import FMRanker
from src.models.sampling import sample_bpr_pairs
from src.experiments.contracts import CandidateOutput

def _autofix_hasattr_evaluate_validation(obj):
    try:
        obj.evaluate_validation
    except AttributeError:
        return False
    return True

def run(context, parameters):
    _fd = context.field_dimension
    _field_dim = sum(_fd) if isinstance(_fd, list) else int(_fd)
    batch_size = parameters.get('batch_size', 4096)
    epochs = parameters.get('epochs', 8)
    k = parameters.get('k', 16)
    learning_rate = parameters.get('learning_rate', 0.0005)
    negatives_per_positive = parameters.get('negatives_per_positive', 2)
    patience = parameters.get('patience', 2)
    seed = parameters.get('seed', 42)
    rng = np.random.RandomState(seed)
    train_users = context.train_users
    train_y = context.train_y
    user_pos_counts = {}
    for u, y in zip(train_users, train_y):
        if y > 0:
            user_pos_counts[u] = user_pos_counts.get(u, 0) + 1
    n_train = len(train_y)
    row_weight = np.ones(n_train, dtype=np.float32)
    for i in range(n_train):
        u = train_users[i]
        cnt = user_pos_counts.get(u, 0)
        if cnt > 5:
            row_weight[i] = 0.2
    model = FMRanker(_field_dim, embedding_dim=k, learning_rate=learning_rate, l2=1e-06, seed=seed)
    train_x = context.train_x.astype(np.int32)
    valid_x = context.valid_x.astype(np.int32) if context.valid_x is not None else None
    best_primary = float('-inf')
    best_state = None
    best_valid_scores = None
    best_epoch = -1
    no_improve = 0
    training_trace = []
    n_pairs_total = 0
    for epoch in range(epochs):
        pos_rows, neg_rows = sample_bpr_pairs(train_users, train_y, rng, negatives_per_positive)
        n_pairs = len(pos_rows)
        if n_pairs == 0:
            break
        n_pairs_total += n_pairs
        perm = rng.permutation(n_pairs)
        pos_rows = pos_rows[perm]
        neg_rows = neg_rows[perm]
        epoch_loss = 0.0
        n_batches = math.ceil(n_pairs / batch_size)
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_pairs)
            if start >= end:
                break
            pos_idx = pos_rows[start:end]
            neg_idx = neg_rows[start:end]
            pos_features = train_x[pos_idx]
            neg_features = train_x[neg_idx]
            pos_scores = model.logits(pos_features)[0]
            neg_scores = model.logits(neg_features)[0]
            d = pos_scores - neg_scores
            d = np.clip(d, -30.0, 30.0)
            sig = 1.0 / (1.0 + np.exp(-d))
            grad_base = sig - 1.0
            w = row_weight[pos_idx]
            grad = grad_base * w / batch_size
            grad_pos = model.gradients(pos_features, grad)
            grad_neg = model.gradients(neg_features, -grad)
            model.apply_gradients(grad_pos[0] + grad_neg[0], grad_pos[1] + grad_neg[1], grad_pos[2] + grad_neg[2])
            loss = np.logaddexp(0, -d) * w
            epoch_loss += np.sum(loss)
        if valid_x is not None:
            valid_scores = model.predict(valid_x)
        else:
            valid_scores = None
        if valid_scores is not None and _autofix_hasattr_evaluate_validation(context):
            eval_result = context.evaluate_validation(valid_scores)
            if isinstance(eval_result, tuple):
                gauc = eval_result[0]
                ndcg = eval_result[1]
            else:
                gauc = eval_result
                ndcg = eval_result
            primary = (gauc + ndcg) / 2.0
        else:
            gauc = 0.0
            ndcg = 0.0
            primary = 0.0
        training_trace.append({'epoch': epoch, 'loss': epoch_loss / n_pairs if n_pairs > 0 else 0.0, 'gauc': gauc, 'ndcg': ndcg, 'primary': primary})
        if primary > best_primary:
            best_primary = primary
            best_state = model.state_dict()
            best_valid_scores = valid_scores
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_state = model.state_dict()
        if valid_x is not None:
            best_valid_scores = model.predict(valid_x)
    test_scores = None
    if context.test_x is not None:
        test_scores = model.predict(context.test_x.astype(np.int32))
    random_valid_scores = None
    if context.random_valid_x is not None:
        random_valid_scores = model.predict(context.random_valid_x.astype(np.int32))
    diagnostics = {'best_epoch': best_epoch, 'best_primary': best_primary, 'weight_used': 'top5_heuristic', 'n_pairs_per_epoch': n_pairs_total / epochs if epochs > 0 else 0.0}
    return CandidateOutput(validation_scores=best_valid_scores if best_valid_scores is not None else np.zeros(len(context.valid_x), dtype=np.float32), checkpoint_state=best_state, training_trace=training_trace, diagnostics=diagnostics, test_scores=test_scores, random_validation_scores=random_valid_scores)
