import numpy as np
import time
from src.models.fm_core import FMRanker
from src.models.sampling import sample_bpr_pairs
from src.experiments.contracts import CandidateOutput

def _primary_score(val):
    """Extract scalar primary score from evaluation result."""
    if isinstance(val, dict):
        if 'primary' in val:
            return val['primary']
        if 'gauc' in val and 'ndcg@5' in val:
            return (val['gauc'] + val['ndcg@5']) / 2.0
        # fallback: use first value (should not happen in practice)
        return float(next(iter(val.values())))
    return float(val)

def run(context, parameters):
    seed = parameters['seed']
    k = parameters['k']
    lr = parameters['learning_rate']
    epochs = parameters['epochs']
    batch_size = parameters['batch_size']
    patience = parameters['patience']
    npp = parameters['negatives_per_positive']

    rng = np.random.default_rng(seed)
    model = FMRanker(context.field_dimension, embedding_dim=k, learning_rate=lr, l2=1e-6, seed=seed)

    best_score = -np.inf
    best_state = None
    patience_counter = 0
    train_trace = []

    for epoch in range(epochs):
        epoch_start = time.time()
        pos_rows, neg_rows = sample_bpr_pairs(list(context.train_users), context.train_y, rng, npp)
        if len(pos_rows) == 0:
            continue

        # shuffle pairs
        n_pairs = len(pos_rows)
        indices = np.arange(n_pairs)
        rng.shuffle(indices)
        pos_rows = pos_rows[indices]
        neg_rows = neg_rows[indices]

        total_loss = 0.0
        n_batches = (n_pairs + batch_size - 1) // batch_size
        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, n_pairs)
            if start >= end:
                break
            batch_pos = pos_rows[start:end]
            batch_neg = neg_rows[start:end]

            pos_x = context.train_x[batch_pos]
            neg_x = context.train_x[batch_neg]

            pos_scores = model.logits(pos_x)[0]
            neg_scores = model.logits(neg_x)[0]

            diff = pos_scores - neg_scores
            grad = 1.0 / (1.0 + np.exp(-diff)) - 1.0
            grad = grad / len(batch_pos)

            grad_v_p, grad_w_p, grad_b_p = model.gradients(pos_x, grad)
            grad_v_n, grad_w_n, grad_b_n = model.gradients(neg_x, -grad)

            model.apply_gradients(
                grad_v_p + grad_v_n,
                grad_w_p + grad_w_n,
                grad_b_p + grad_b_n
            )

            loss = np.logaddexp(0, -diff).mean()
            total_loss += loss * len(batch_pos)

        valid_scores = model.predict(context.valid_x)
        val_result = context.evaluate_validation(valid_scores)
        val_score = _primary_score(val_result)

        if val_score > best_score:
            best_score = val_score
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        train_trace.append({
            'epoch': epoch,
            'train_loss': total_loss / n_pairs if n_pairs > 0 else None,
            'val_score': val_score,
            'time': time.time() - epoch_start
        })

    if best_state is not None:
        model.load_state_dict(best_state)

    validation_scores = model.predict(context.valid_x)
    test_scores = model.predict(context.test_x) if context.test_x is not None else None

    checkpoint_state = model.state_dict() if best_state is None else best_state

    diagnostics = {
        'epochs_used': len(train_trace),
        'early_stopped': patience_counter >= patience,
        'best_val_score': best_score
    }

    return CandidateOutput(
        validation_scores=validation_scores,
        checkpoint_state=checkpoint_state,
        training_trace=train_trace,
        diagnostics=diagnostics,
        test_scores=test_scores
    )
