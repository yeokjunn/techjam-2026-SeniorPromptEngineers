import numpy as np
from src.models.fm_core import FMRanker
from src.models.sampling import sample_softmax_groups
from src.experiments.contracts import CandidateOutput


def run(context, parameters):
    # Extract parameters
    seed = parameters['seed']
    k = parameters['k']
    lr = parameters['learning_rate']
    epochs = parameters['epochs']
    batch_size = parameters['batch_size']
    neg_per_group = parameters['negatives_per_group']
    temperature = parameters['temperature']
    patience = parameters['patience']

    rng = np.random.default_rng(seed)
    model = FMRanker(context.field_dimension, embedding_dim=k, learning_rate=lr, l2=1e-6, seed=seed)

    # Sample same-user groups
    positives, negatives = sample_softmax_groups(context.train_users, context.train_y, rng, neg_per_group)
    n_groups = len(positives)

    def extract_primary(metric_result):
        if isinstance(metric_result, dict):
            return float(metric_result['primary'])
        elif isinstance(metric_result, (tuple, list)):
            return float(metric_result[0])
        else:
            return float(metric_result)

    best_metric = -np.inf
    best_state = None
    patience_counter = 0
    training_trace = []

    if n_groups > 0:
        for epoch in range(epochs):
            perm = rng.permutation(n_groups)
            total_loss = 0.0
            total_samples = 0

            for start in range(0, n_groups, batch_size):
                idx = perm[start:start + batch_size]
                if len(idx) == 0:
                    continue
                bsz = len(idx)

                pos_idx = positives[idx]
                neg_idx = negatives[idx]  # (bsz, K)

                pos_x = context.train_x[pos_idx]  # (bsz, fields)
                neg_x = context.train_x[neg_idx.reshape(-1)]  # (bsz*K, fields)

                # Forward pass
                pos_scores = model.logits(pos_x)[0]  # (bsz,)
                neg_scores_flat = model.logits(neg_x)[0]  # (bsz*K,)
                neg_scores = neg_scores_flat.reshape(bsz, neg_per_group)  # (bsz, K)

                logits = np.concatenate([pos_scores[:, None], neg_scores], axis=1) / temperature
                # Stable softmax (max-shifted)
                max_logits = np.max(logits, axis=1, keepdims=True)
                exp_logits = np.exp(logits - max_logits)
                probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # (bsz, K+1)

                # Group softmax loss (mean over batch)
                loss = -np.log(probs[:, 0] + 1e-12).mean()
                total_loss += loss * bsz
                total_samples += bsz

                # Gradients for positive and negatives, scaled by batch size
                grad_pos = (probs[:, 0] - 1.0) / temperature / bsz
                grad_neg = probs[:, 1:] / temperature / bsz  # (bsz, K)

                # Backward
                grad_v_p, grad_w_p, grad_b_p = model.gradients(pos_x, grad_pos)
                grad_v_n, grad_w_n, grad_b_n = model.gradients(neg_x, grad_neg.reshape(-1))

                # Accumulate gradients (sum positive and negative contributions)
                grad_v = grad_v_p + grad_v_n
                grad_w = grad_w_p + grad_w_n
                grad_b = grad_b_p + grad_b_n
                model.apply_gradients(grad_v, grad_w, grad_b)

            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

            # Validation
            valid_scores = model.predict(context.valid_x)
            metric_result = context.evaluate_validation(valid_scores)
            primary = extract_primary(metric_result)

            training_trace.append({'epoch': epoch + 1, 'loss': avg_loss, 'primary': primary})

            # Early stopping
            if primary > best_metric:
                best_metric = primary
                best_state = model.state_dict()  # returns copies
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    else:
        # No eligible groups; use untrained model
        valid_scores = model.predict(context.valid_x)
        metric_result = context.evaluate_validation(valid_scores)
        primary = extract_primary(metric_result)
        training_trace.append({'epoch': 0, 'loss': 0.0, 'primary': primary})
        best_state = model.state_dict()

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_state = model.state_dict()

    # Final validation scores
    valid_scores = model.predict(context.valid_x)

    # Test scores
    if context.test_x is not None:
        test_scores = model.predict(context.test_x)
    else:
        test_scores = None

    diagnostics = {
        'n_groups': n_groups,
        'total_train_rows': len(context.train_x),
        'best_primary': best_metric,
        'temperature': temperature,
        'negatives_per_group': neg_per_group,
        'patience': patience,
    }

    return CandidateOutput(
        validation_scores=valid_scores,
        checkpoint_state=best_state,
        training_trace=training_trace,
        diagnostics=diagnostics,
        test_scores=test_scores,
    )
