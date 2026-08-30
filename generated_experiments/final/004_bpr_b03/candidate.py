import numpy as np
import time
from src.models.fm_core import FMRanker
from src.models.sampling import sample_bpr_pairs
from src.experiments.contracts import CandidateOutput


def _extract_primary(metrics):
    if isinstance(metrics, dict):
        if 'primary' in metrics:
            return float(metrics['primary'])
        for value in metrics.values():
            if isinstance(value, (int, float, np.floating)):
                return float(value)
        return float('nan')
    return float(metrics)


def run(context, parameters):
    seed = int(parameters.get('seed', 42))
    k = int(parameters.get('k', 16))
    lr = float(parameters.get('learning_rate', 0.001))
    epochs = int(parameters.get('epochs', 20))
    batch_size = int(parameters.get('batch_size', 2048))
    patience = int(parameters.get('patience', 5))
    npp = int(parameters.get('negatives_per_positive', 1))
    train_labels = np.asarray(context.train_y)
    model = FMRanker(context.field_dimension, embedding_dim=k, learning_rate=lr, l2=1e-6, seed=seed)
    rng = np.random.default_rng(seed)
    best_primary = -np.inf
    best_epoch = -1
    best_state = None
    patience_count = 0
    training_trace = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        positives, negatives = sample_bpr_pairs(context.train_users, train_labels, rng, npp)
        epoch_loss_sum = 0.0
        epoch_count = 0
        n_pairs = len(positives)
        if n_pairs == 0:
            valid_scores = model.predict(context.valid_x)
            metrics = context.evaluate_validation(valid_scores)
            primary = _extract_primary(metrics)
            best_primary = primary
            best_epoch = epoch
            best_state = {name: np.array(value, copy=True) for name, value in model.state_dict().items()}
            training_trace.append({'epoch': epoch, 'train_loss': None, 'primary': primary})
            break

        for start in range(0, n_pairs, batch_size):
            end = min(start + batch_size, n_pairs)
            pos_idx = positives[start:end]
            neg_idx = negatives[start:end]
            pos_x = context.train_x[pos_idx]
            neg_x = context.train_x[neg_idx]
            pos_scores = model.logits(pos_x)[0]
            neg_scores = model.logits(neg_x)[0]
            delta = pos_scores - neg_scores
            loss = np.logaddexp(0.0, -delta)
            epoch_loss_sum += float(np.sum(loss))
            epoch_count += len(delta)
            grad = -1.0 / (1.0 + np.exp(np.clip(delta, -50.0, 50.0)))
            grad = grad / float(len(delta))
            grad_v_pos, grad_w_pos, _ = model.gradients(pos_x, grad)
            grad_v_neg, grad_w_neg, _ = model.gradients(neg_x, -grad)
            model.apply_gradients(grad_v_pos + grad_v_neg, grad_w_pos + grad_w_neg)
        train_loss = epoch_loss_sum / max(1, epoch_count)

        valid_scores = model.predict(context.valid_x)
        metrics = context.evaluate_validation(valid_scores)
        primary = _extract_primary(metrics)
        trace_entry = {'epoch': epoch, 'train_loss': train_loss, 'primary': primary}
        if isinstance(metrics, dict):
            for name, value in metrics.items():
                if isinstance(value, (int, float, np.floating)):
                    trace_entry[name] = float(value)
        training_trace.append(trace_entry)

        if primary > best_primary + 1e-12:
            best_primary = primary
            best_epoch = epoch
            best_state = {name: np.array(value, copy=True) for name, value in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    elapsed = time.time() - start_time
    diagnostics = {'best_epoch': best_epoch, 'best_primary': float(best_primary), 'elapsed_seconds': float(elapsed)}
    if best_state is None:
        best_state = {name: np.array(value, copy=True) for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    validation_scores = np.asarray(model.predict(context.valid_x), dtype=np.float64)
    test_scores = None
    if context.test_x is not None:
        test_scores = np.asarray(model.predict(context.test_x), dtype=np.float64)
    checkpoint_state = {name: np.array(value, copy=True) for name, value in best_state.items()}
    return CandidateOutput(
        validation_scores=validation_scores,
        checkpoint_state=checkpoint_state,
        training_trace=training_trace,
        diagnostics=diagnostics,
        test_scores=test_scores,
    )