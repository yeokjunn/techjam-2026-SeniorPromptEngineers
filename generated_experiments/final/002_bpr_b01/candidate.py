import numpy as np
import math
import time
from src.models.fm_core import FMRanker
from src.models.sampling import sample_bpr_pairs
from src.experiments.contracts import CandidateOutput

def run(context, parameters):
    seed = parameters['seed']
    rng = np.random.default_rng(seed)
    model = FMRanker(context.field_dimension, embedding_dim=parameters['k'],
                     learning_rate=parameters['learning_rate'], l2=1e-6, seed=seed)
    batch_size = parameters['batch_size']
    epochs = parameters['epochs']
    npp = parameters['negatives_per_positive']
    patience = parameters['patience']

    best_metric = -np.inf
    best_state = None
    patience_counter = 0
    training_trace = []

    for epoch in range(epochs):
        epoch_start = time.time()
        pos_idx, neg_idx = sample_bpr_pairs(context.train_users, context.train_y, rng, npp)
        n_pairs = len(pos_idx)
        if n_pairs == 0:
            training_trace.append({'epoch': epoch, 'train_loss': None, 'val_primary': None})
            break
        perm = rng.permutation(n_pairs)
        pos_idx = pos_idx[perm]
        neg_idx = neg_idx[perm]
        total_loss = 0.0
        n_batches = math.ceil(n_pairs / batch_size)
        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, n_pairs)
            if start >= end:
                continue
            batch_pos = pos_idx[start:end]
            batch_neg = neg_idx[start:end]
            pos_x = context.train_x[batch_pos]
            neg_x = context.train_x[batch_neg]
            pos_scores = model.logits(pos_x)[0]
            neg_scores = model.logits(neg_x)[0]
            diff = pos_scores - neg_scores
            loss = np.logaddexp(0, -diff).mean()
            total_loss += loss * (end - start)
            grad = (1.0 / (1.0 + np.exp(-diff)) - 1.0) / (end - start)
            grad_v_p, grad_w_p, grad_b_p = model.gradients(pos_x, grad)
            grad_v_n, grad_w_n, grad_b_n = model.gradients(neg_x, -grad)
            model.apply_gradients(grad_v_p + grad_v_n, grad_w_p + grad_w_n, grad_b_p + grad_b_n)
        train_loss = total_loss / n_pairs if n_pairs > 0 else None
        val_scores = model.predict(context.valid_x)
        val_metrics = context.evaluate_validation(val_scores)
        if isinstance(val_metrics, dict):
            val_primary = val_metrics.get('primary', val_metrics)
        else:
            val_primary = val_metrics
        training_trace.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_primary': float(val_primary),
            'elapsed': time.time() - epoch_start
        })
        if val_primary > best_metric:
            best_metric = val_primary
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_scores = model.predict(context.valid_x)
    test_scores = None
    if context.test_x is not None:
        test_scores = model.predict(context.test_x)

    final_metric = context.evaluate_validation(final_val_scores)
    if isinstance(final_metric, dict):
        final_primary = final_metric.get('primary', final_metric)
    else:
        final_primary = final_metric

    diagnostics = {
        'epochs_run': len(training_trace),
        'final_val_primary': float(final_primary),
        'best_metric': float(best_metric),
        'model_v_shape': model.state_dict()['V'].shape,
        'model_w_shape': model.state_dict()['W'].shape
    }

    return CandidateOutput(
        validation_scores=final_val_scores,
        checkpoint_state=model.state_dict(),
        training_trace=training_trace,
        diagnostics=diagnostics,
        test_scores=test_scores
    )