from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.official import (
    TEST_ROWS,
    classify_primary,
    load_random_validation,
    load_test_meta,
    load_train_valid,
    official_evaluate,
    starter_modules,
)
from src.experiments.contracts import CandidateContext, CandidateOutput
from src.agent.safety import validate_source


RESERVED_DIAGNOSTIC_KEYS = {"metrics", "gauc", "ndcg", "ndcg@5", "primary"}

REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_relative(path: Path) -> str:
    """Repo-relative POSIX when under the repo root, else absolute POSIX."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size > 1000:
            raise ValueError("Diagnostics may not contain large arrays.")
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"Value is not JSON serializable: {type(value).__name__}")


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_candidate(path: Path):
    source = path.read_text(encoding="utf-8")
    validate_source(source)
    spec = importlib.util.spec_from_file_location("generated_candidate", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not import candidate: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "run", None)):
        raise ValueError("Candidate must define callable run(context, parameters).")
    return module


def _dcg(labels: np.ndarray) -> float:
    gains = (np.power(2.0, labels.astype(np.float64)) - 1.0)
    discounts = 1.0 / np.log2(np.arange(2, len(labels) + 2, dtype=np.float64))
    return float(np.sum(gains * discounts))


def topk_diagnostics(
    users: tuple[str, ...] | list[str],
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    k: int = 5,
    high_gauc_threshold: float = 0.9,
    low_ndcg_threshold: float = 0.5,
) -> dict[str, Any]:
    """Validation-only top-k diagnostics computed inside the trusted runner."""
    by_user: dict[str, list[int]] = {}
    for index, user in enumerate(users):
        by_user.setdefault(str(user), []).append(index)

    ndcgs: list[float] = []
    positive_hits = 0
    possible_hits = 0
    high_gauc_low_ndcg = 0
    margins: list[float] = []
    by_size: dict[str, list[float]] = {}
    by_positive_count: dict[str, list[float]] = {}

    for indices in by_user.values():
        idx = np.asarray(indices, dtype=np.int64)
        user_labels = labels[idx].astype(np.float64)
        user_scores = scores[idx].astype(np.float64)
        order = np.lexsort((idx, -user_scores))
        top = order[:k]
        ideal = np.argsort(-user_labels, kind="stable")[:k]
        ideal_dcg = _dcg(user_labels[ideal])
        ndcg = 0.0 if ideal_dcg <= 0 else _dcg(user_labels[top]) / ideal_dcg
        ndcgs.append(float(ndcg))

        positives = int(np.sum(user_labels > 0.5))
        possible_hits += min(k, positives)
        positive_hits += int(np.sum(user_labels[top] > 0.5))
        size_bucket = "1-5" if len(idx) <= 5 else "6-20" if len(idx) <= 20 else "21+"
        pos_bucket = "0" if positives == 0 else "1" if positives == 1 else "2-4" if positives <= 4 else "5+"
        by_size.setdefault(size_bucket, []).append(float(ndcg))
        by_positive_count.setdefault(pos_bucket, []).append(float(ndcg))

        positive_scores = user_scores[user_labels > 0.5]
        negative_scores = user_scores[user_labels <= 0.5]
        if len(positive_scores) and len(negative_scores):
            margins.append(float(np.max(positive_scores) - np.max(negative_scores)))
            comparisons = (positive_scores[:, None] > negative_scores[None, :]).mean()
            ties = (positive_scores[:, None] == negative_scores[None, :]).mean()
            user_gauc = float(comparisons + 0.5 * ties)
            if user_gauc >= high_gauc_threshold and ndcg < low_ndcg_threshold:
                high_gauc_low_ndcg += 1

    def summary(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "p10": float(np.quantile(arr, 0.10)),
            "p50": float(np.quantile(arr, 0.50)),
            "p90": float(np.quantile(arr, 0.90)),
        }

    return {
        "topk": int(k),
        "per_user_ndcg": summary(ndcgs),
        "top5_positive_hits": int(positive_hits),
        "top5_possible_positive_hits": int(possible_hits),
        "top5_hit_rate": float(positive_hits / possible_hits) if possible_hits else 0.0,
        "high_gauc_low_ndcg_users": int(high_gauc_low_ndcg),
        "positive_vs_top_negative_margin": summary(margins),
        "ndcg_by_impression_count": {key: summary(value) for key, value in sorted(by_size.items())},
        "ndcg_by_positive_count": {key: summary(value) for key, value in sorted(by_positive_count.items())},
    }


def validate_and_persist_output(
    output: CandidateOutput,
    valid_users: tuple[str, ...] | list[str],
    valid_y: np.ndarray,
    artifact_dir: Path,
    *,
    expected_test_rows: int | None = None,
    random_valid_users: tuple[str, ...] | list[str] | None = None,
    random_valid_y: np.ndarray | None = None,
) -> dict[str, Any]:
    if not isinstance(output, CandidateOutput):
        raise TypeError("Candidate run() must return CandidateOutput.")
    scores = np.asarray(output.validation_scores)
    if scores.ndim != 1 or len(scores) != len(valid_y):
        raise ValueError("Validation scores have the wrong shape.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Validation scores contain NaN or Inf.")
    metrics = official_evaluate(valid_users, valid_y, scores)
    topk_report = topk_diagnostics(valid_users, valid_y, scores, k=5)

    checkpoint: dict[str, np.ndarray] = {}
    total_elements = 0
    for key, value in output.checkpoint_state.items():
        if not key.replace("_", "").isalnum():
            raise ValueError(f"Unsafe checkpoint key: {key!r}")
        array = np.asarray(value)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Checkpoint array {key!r} contains NaN or Inf.")
        total_elements += int(array.size)
        checkpoint[key] = array
    if total_elements > 50_000_000:
        raise ValueError("Checkpoint exceeds the 50M-element safety limit.")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "model.npz"
    np.savez_compressed(checkpoint_path, **checkpoint)
    valid_scores_path = artifact_dir / "validation_scores.npy"
    np.save(valid_scores_path, scores)

    # Test scores are validated, never trusted from candidate diagnostics, and
    # never raise: a bad or absent array is reported so the runner can classify
    # the outcome without losing the metrics above.
    test_scores_path: Path | None = None
    if expected_test_rows is None:
        test_scores_status = "not_required"
    elif output.test_scores is None:
        test_scores_status = "missing"
    else:
        try:
            test_scores = np.asarray(output.test_scores, dtype=np.float64)
        except (TypeError, ValueError):
            test_scores = None
        if (
            test_scores is None
            or test_scores.ndim != 1
            or len(test_scores) != expected_test_rows
            or not np.all(np.isfinite(test_scores))
        ):
            test_scores_status = "invalid"
        else:
            test_scores_status = "ok"
            test_scores_path = artifact_dir / "test_scores.npy"
            np.save(test_scores_path, test_scores)

    diagnostics = {
        key: _json_safe(value)
        for key, value in output.diagnostics.items()
        if str(key).lower() not in RESERVED_DIAGNOSTIC_KEYS
    }
    diagnostic_metrics: dict[str, dict[str, float]] = {}
    if random_valid_y is not None and random_valid_users is not None:
        random_scores = output.random_validation_scores
        if random_scores is None:
            diagnostics["random_exposure_status"] = "not_scored"
        else:
            random_scores = np.asarray(random_scores, dtype=np.float64)
            if (
                random_scores.ndim != 1
                or len(random_scores) != len(random_valid_y)
                or not np.all(np.isfinite(random_scores))
            ):
                raise ValueError(
                    "Random-exposure validation scores have the wrong shape or contain NaN/Inf."
                )
            random_metrics = official_evaluate(
                random_valid_users, random_valid_y, random_scores
            )
            random_metrics.update(
                {
                    "users": float(len(set(random_valid_users))),
                    "rows": float(len(random_valid_y)),
                    "robustness_gap": float(metrics["primary"])
                    - float(random_metrics["primary"]),
                }
            )
            diagnostic_metrics["random_exposure"] = random_metrics
            diagnostics["random_exposure_status"] = "scored"
    metrics.update({"users": float(len(set(valid_users))), "rows": float(len(valid_y))})
    return {
        "metrics": metrics,
        "training_trace": _json_safe(output.training_trace),
        "diagnostics": {**diagnostics, "topk_diagnostics": _json_safe(topk_report)},
        "topk_diagnostics": _json_safe(topk_report),
        "diagnostic_metrics": diagnostic_metrics,
        "artifact_path": checkpoint_path.as_posix(),
        "validation_scores_path": _repo_relative(valid_scores_path),
        "test_scores_status": test_scores_status,
        "test_scores_path": (
            _repo_relative(test_scores_path) if test_scores_path is not None else None
        ),
        # Classified, never raised: result.json still gets written so the
        # ledger keeps the number; the runner decides what it means.
        "sanity_class": classify_primary(float(metrics["primary"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute one generated candidate safely.")
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    args = parser.parse_args()

    parameters = json.loads(args.spec.read_text(encoding="utf-8"))["parameters"]
    splits = load_train_valid(args.data_dir)
    splits["random_valid"] = load_random_validation(args.data_dir)
    # Test features only: the kit derives bucket edges and every vocab from
    # splits['train'] alone, so this third key changes nothing about train/valid
    # and reproduces the kit's own test encoding. The placeholder label column
    # is dropped below and never handed to the candidate.
    splits["test"] = list(load_test_meta(args.data_dir, expected_rows=TEST_ROWS).rows)
    data_module, _, _ = starter_modules()
    encoded, dimension = data_module.encode(splits)
    train_x, train_y, train_users = encoded["train"]
    valid_x, valid_y, valid_users = encoded["valid"]
    random_valid_x, random_valid_y, random_valid_users = encoded["random_valid"]
    test_x = encoded["test"][0]

    def evaluate_validation(scores: np.ndarray) -> dict[str, float]:
        scores = np.asarray(scores)
        if scores.ndim != 1 or len(scores) != len(valid_y):
            raise ValueError("Validation scores have the wrong shape.")
        if not np.all(np.isfinite(scores)):
            raise ValueError("Validation scores contain NaN or Inf.")
        return official_evaluate(valid_users, valid_y, scores)

    context = CandidateContext(
        train_x=train_x,
        train_y=train_y,
        train_users=tuple(train_users),
        valid_x=valid_x,
        valid_users=tuple(valid_users),
        field_dimension=dimension,
        evaluate_validation=evaluate_validation,
        test_x=test_x,
        random_valid_x=random_valid_x,
    )
    module = _load_candidate(args.candidate)
    output = module.run(context, parameters)
    payload = validate_and_persist_output(
        output,
        tuple(valid_users),
        valid_y,
        args.artifact_dir,
        expected_test_rows=int(test_x.shape[0]),
        random_valid_users=tuple(random_valid_users),
        random_valid_y=random_valid_y,
    )
    _write_json_atomic(args.result, payload)


if __name__ == "__main__":
    main()
