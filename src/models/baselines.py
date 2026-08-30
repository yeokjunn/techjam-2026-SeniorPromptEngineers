from __future__ import annotations

import collections
import time
from pathlib import Path

import numpy as np

from src.evaluation.holdout import split_users
from src.evaluation.official import REPO_ROOT, official_evaluate, starter_modules


def run_random(splits: dict, seed: int) -> dict:
    rows = splits["valid"]
    scores = np.random.default_rng(seed).random(len(rows))
    return {
        "metrics": official_evaluate(
            [row[1] for row in rows], [row[6] for row in rows], scores
        ),
        "epoch_trace": [],
        "artifact_path": None,
    }


def run_popularity(splits: dict, prior: float) -> dict:
    positives: collections.Counter = collections.Counter()
    impressions: collections.Counter = collections.Counter()
    for row in splits["train"]:
        impressions[row[2]] += 1
        positives[row[2]] += row[6]
    global_mean = sum(positives.values()) / sum(impressions.values())

    def score(video_id: str) -> float:
        if not impressions[video_id]:
            return global_mean
        return (positives[video_id] + prior * global_mean) / (
            impressions[video_id] + prior
        )

    rows = splits["valid"]
    scores = [score(row[2]) for row in rows]
    return {
        "metrics": official_evaluate(
            [row[1] for row in rows], [row[6] for row in rows], scores
        ),
        "epoch_trace": [],
        "artifact_path": None,
    }


def run_fm(splits: dict, parameters: dict, artifact_dir: Path) -> dict:
    data_module, _, baseline_module = starter_modules()
    encoded, dimension = data_module.encode(splits)
    train_x, train_y, _ = encoded["train"]
    valid_x, valid_y, valid_users = encoded["valid"]

    seed = int(parameters.get("seed", 0))
    model = baseline_module.FM(
        dimension,
        k=int(parameters.get("k", 16)),
        lr=float(parameters.get("learning_rate", 0.001)),
        seed=seed,
    )
    epochs = int(parameters.get("epochs", 40))
    batch_size = int(parameters.get("batch_size", 8192))
    patience = int(parameters.get("patience", 4))
    rng = np.random.default_rng(seed)
    best_score = float("-inf")
    best_state = None
    bad_epochs = 0
    trace: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        epoch_started = time.monotonic()
        indices = rng.permutation(len(train_y))
        losses = []
        for offset in range(0, len(indices), batch_size):
            batch = indices[offset : offset + batch_size]
            losses.append(model.step(train_x[batch], train_y[batch]))
        metrics = official_evaluate(valid_users, valid_y, model.predict(valid_x))
        trace.append(
            {
                "epoch": float(epoch),
                "loss": float(np.mean(losses)),
                "GAUC": metrics["GAUC"],
                "nDCG@5": metrics["nDCG@5"],
                "primary": metrics["primary"],
                "duration_seconds": time.monotonic() - epoch_started,
            }
        )
        print(
            f"epoch={epoch} loss={np.mean(losses):.4f} "
            f"GAUC={metrics['GAUC']:.4f} nDCG@5={metrics['nDCG@5']:.4f} "
            f"primary={metrics['primary']:.4f}",
            flush=True,
        )
        if metrics["primary"] > best_score + 1e-5:
            best_score = metrics["primary"]
            bad_epochs = 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("FM training completed without a valid checkpoint.")
    model.V, model.W, model.b = best_state
    final_scores = model.predict(valid_x)
    final_metrics = official_evaluate(valid_users, valid_y, final_scores)
    # Score the same checkpoint on the selection/reporting halves so a candidate's
    # `report_primary` has something to be compared against. Without this the two sides are
    # measured on different populations and the delta is meaningless -- the reporting half is
    # simply the easier of the two (candidates score ~+0.003 higher on it), so comparing a
    # candidate's report_primary to the baseline's full-validation primary overstates the gain.
    #
    # Additive only: `metrics["primary"]` and the training loop above are untouched, so the
    # baseline stays bit-identical to the committed reference. Note the asymmetry that remains
    # -- the baseline early-stops on full validation (changing that would alter its primary),
    # while candidates now early-stop on the selection half. That biases the baseline's
    # reporting-half number slightly upward, which understates any candidate's delta rather
    # than flattering it.
    for name, mask in zip(("select", "report"), split_users(valid_users)):
        subset_users = [user for user, keep in zip(valid_users, mask) if keep]
        if subset_users:
            final_metrics[f"{name}_primary"] = official_evaluate(
                subset_users, valid_y[mask], final_scores[mask]
            )["primary"]
    checkpoint = artifact_dir / "model.npz"
    np.savez_compressed(checkpoint, V=model.V, W=model.W, b=model.b)
    return {
        "metrics": final_metrics,
        "epoch_trace": trace,
        "artifact_path": checkpoint.relative_to(REPO_ROOT).as_posix(),
    }
