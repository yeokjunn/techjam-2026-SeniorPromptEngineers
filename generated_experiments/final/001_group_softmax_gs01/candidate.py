import numpy as np
import time
import math
from collections import defaultdict
from src.models.fm_core import FMRanker
from src.models.sampling import sample_softmax_groups
from src.experiments.contracts import CandidateOutput

def run(context, parameters):
    seed = parameters.get('seed', 42)
    k = parameters['k']
    lr = parameters['learning_rate']
    epochs = parameters['epochs']
    batch_size = parameters['batch_size']
    patience = parameters['patience']
    negatives_per_group = parameters['negatives_per_group']
    temperature = parameters['temperature']

    rng = np.random.default_rng(seed)
    model = FMRanker(context.field_dimension, embedding_dim=k, learning_rate=lr, seed=seed)

    best_primary = -1.0
    best_state = None
    patience_counter = 0
    training_trace = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0

        positives, negatives = sample_softmax_groups(
            context.train_users, context.train_y, rng, negatives_per_group
        )
        n_groups = len(positives)
        if n_groups == 0:
            continue

        perm = rng.permutation(n_groups)
        positives = positives[perm]
        negatives = negatives[perm]

        for start in range(0, n_groups, batch_size):
            end = min(start + batch_size, n_groups)
            pos_idx = positives[start:end]
            neg_idx = negatives[start:end]

            pos_x = context.train_x[pos_idx]
            neg_x = context.train_x[neg_idx]

            pos_scores = model.logits(pos_x)[0]
            neg_scores = model.logits(neg_x.reshape(-1, neg_x.shape[2]))[0]
            neg_scores = neg_scores.reshape(-1, negatives_per_group)

            logits = np.concatenate([pos_scores[:, None], neg_scores], axis=1) / temperature
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            loss = -np.log(probs[:, 0] + 1e-12).mean()
            epoch_loss += loss
            n_batches += 1

            grad_pos = (probs[:, 0] - 1.0) / temperature
            grad_neg = probs[:, 1:] / temperature

            grad_v_pos, grad_w_pos, grad_b_pos = model.gradients(pos_x, grad_pos)
            neg_x_flat = neg_x.reshape(-1, neg_x.shape[2])
            grad_neg_flat = grad_neg.reshape(-1)
            grad_v_neg, grad_w_neg, grad_b_neg = model.gradients(neg_x_flat, grad_neg_flat)

            model.apply_gradients(
                grad_v_pos + grad_v_neg,
                grad_w_pos + grad_w_neg,
                grad_b_pos + grad_b_neg
            )

        if n_batches == 0:
            continue

        avg_loss = epoch_loss / n_batches
        valid_scores = model.predict(context.valid_x)
        primary = context.evaluate_validation(valid_scores)

        training_trace.append({
            'epoch': epoch,
            'train_loss': avg_loss,
            'primary': primary,
            'n_groups': n_groups
        })

        if primary > best_primary:
            best_primary = primary
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    valid_scores = model.predict(context.valid_x)
    test_scores = None
    if context.test_x is not None:
        test_scores = model.predict(context.test_x)

    diagnostics = {
        'best_primary': best_primary,
        'epochs_run': len(training_trace),
        'final_train_loss': training_trace[-1]['train_loss'] if training_trace else None,
        'total_time_sec': time.time() - start_time
    }

    return CandidateOutput(
        validation_scores=valid_scores,
        checkpoint_state=best_state if best_state is not None else model.state_dict(),
        training_trace=training_trace,
        diagnostics=diagnostics,
        test_scores=test_scores
    )
