"""Automated model ensembling and rank/score blending (MLE-STAR style).

Maintains a pool of diverse candidate predictions across families (BPR, Group Softmax,
History Sequences, Multi-Task), learns optimal blending weights strictly on the validation set,
and writes out blended test predictions when the ensemble strictly outperforms every single model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation.official import official_evaluate, load_train_valid

REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


@dataclass(frozen=True)
class CandidatePrediction:
    experiment_id: str
    family: str
    candidate_dir: str
    primary_metric: float
    validation_scores: np.ndarray
    test_scores: np.ndarray


@dataclass
class EnsembleResult:
    status: str  # "ok", "skipped", "error"
    reason: str | None = None
    ensemble_node_dir: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    candidates_used: list[dict[str, Any]] = field(default_factory=list)
    single_best_primary: float | None = None
    ensemble_primary: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Standardize scores to zero mean and unit variance for scale compatibility."""
    arr = np.asarray(scores, dtype=np.float64)
    std = float(np.std(arr))
    if std < 1e-8:
        return np.zeros_like(arr)
    return (arr - float(np.mean(arr))) / std


def rank_transform(scores: np.ndarray) -> np.ndarray:
    """Convert scores to fractional ranks [0, 1]. Invariance to monotonic logit scale."""
    arr = np.asarray(scores, dtype=np.float64)
    n = len(arr)
    if n <= 1:
        return np.zeros_like(arr)
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, num=n, endpoint=True)
    return ranks


def blend_predictions(
    predictions: list[np.ndarray], weights: list[float], method: str = "rank"
) -> np.ndarray:
    """Blend an array of predictions using normalized or ranked weights."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / np.sum(w)
    blended = np.zeros_like(predictions[0], dtype=np.float64)
    for p, weight in zip(predictions, w):
        transformed = rank_transform(p) if method == "rank" else normalize_scores(p)
        blended += weight * transformed
    return blended


def find_optimal_weights(
    valid_users: tuple[str, ...] | list[str],
    valid_y: np.ndarray,
    candidate_valid_scores: list[np.ndarray],
    method: str = "rank",
    max_steps: int = 20,
) -> tuple[list[float], dict[str, float]]:
    """Simplex coordinate search finding non-negative weights maximizing validation primary score."""
    k = len(candidate_valid_scores)
    if k == 1:
        metrics = official_evaluate(valid_users, valid_y, candidate_valid_scores[0])
        return [1.0], metrics

    weights = np.ones(k, dtype=np.float64) / k
    best_weights = weights.copy()
    current_blend = blend_predictions(candidate_valid_scores, best_weights, method=method)
    best_metrics = official_evaluate(valid_users, valid_y, current_blend)
    best_score = float(best_metrics["primary"])

    deltas = [0.2, 0.1, 0.05, 0.02]
    for delta in deltas:
        for _ in range(max_steps):
            improved = False
            for i in range(k):
                for step in (+delta, -delta):
                    candidate_w = best_weights.copy()
                    candidate_w[i] = max(0.0, candidate_w[i] + step)
                    total = np.sum(candidate_w)
                    if total <= 1e-9:
                        continue
                    candidate_w = candidate_w / total
                    blend = blend_predictions(candidate_valid_scores, candidate_w, method=method)
                    metrics = official_evaluate(valid_users, valid_y, blend)
                    score = float(metrics["primary"])
                    if score > best_score + 1e-6:
                        best_score = score
                        best_weights = candidate_w
                        best_metrics = metrics
                        improved = True
            if not improved:
                break

    return [float(w) for w in best_weights], best_metrics


def _resolve_repo_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _score_paths_for_node(node: Any) -> tuple[Path | None, Path | None]:
    """Resolve score files from persisted run artifacts, with legacy fallbacks."""
    val_path = _resolve_repo_path(getattr(node, "validation_scores_path", None))
    test_path = _resolve_repo_path(getattr(node, "test_scores_path", None))
    if val_path is not None and test_path is not None:
        return val_path, test_path

    artifact_path = _resolve_repo_path(getattr(node, "artifact_path", None))
    if artifact_path is not None:
        artifact_dir = artifact_path.parent
        artifact_val = artifact_dir / "validation_scores.npy"
        artifact_test = artifact_dir / "test_scores.npy"
        if val_path is None:
            val_path = artifact_val
        if test_path is None:
            test_path = artifact_test
        if val_path.is_file() and test_path.is_file():
            return val_path, test_path

    candidate_dir = _resolve_repo_path(getattr(node, "candidate_dir", None))
    if candidate_dir is not None:
        if val_path is None:
            val_path = candidate_dir / "validation_scores.npy"
        if test_path is None:
            test_path = candidate_dir / "test_scores.npy"
    return val_path, test_path


def select_candidate_pool(
    nodes: list[Any],
    generated_root: Path,
    max_candidates: int = 4,
) -> list[CandidatePrediction]:
    """Select the top candidates across diverse model families."""
    candidates_by_family: dict[str, list[tuple[float, Any]]] = {}
    for node in nodes:
        if getattr(node, "status", None) != "success":
            continue
        metrics = getattr(node, "metrics", None) or {}
        primary = float(metrics.get("primary", 0.0))
        if primary <= 0.0:
            continue
        family = getattr(node, "family", "unknown")
        candidates_by_family.setdefault(family, []).append((primary, node))

    chosen_nodes: list[Any] = []
    # Pick the best candidate from each family first (diversity priority)
    for family in sorted(candidates_by_family.keys()):
        candidates_by_family[family].sort(key=lambda x: x[0], reverse=True)
        chosen_nodes.append(candidates_by_family[family][0][1])

    # If we have room and fewer than max_candidates, fill with next best overall
    if len(chosen_nodes) < max_candidates:
        remaining = []
        for family, node_list in candidates_by_family.items():
            remaining.extend(node_list[1:])
        remaining.sort(key=lambda x: x[0], reverse=True)
        for _, node in remaining:
            if len(chosen_nodes) >= max_candidates:
                break
            chosen_nodes.append(node)

    results: list[CandidatePrediction] = []
    for node in chosen_nodes:
        cand_dir = getattr(node, "candidate_dir", None)
        val_path, test_path = _score_paths_for_node(node)
        if not (
            val_path is not None
            and test_path is not None
            and val_path.is_file()
            and test_path.is_file()
        ):
            continue

        try:
            val_scores = np.load(val_path)
            test_scores = np.load(test_path)
        except Exception:
            continue

        if (
            val_scores.ndim != 1
            or test_scores.ndim != 1
            or not np.all(np.isfinite(val_scores))
            or not np.all(np.isfinite(test_scores))
        ):
            continue

        metrics = getattr(node, "metrics", {}) or {}
        results.append(
            CandidatePrediction(
                experiment_id=str(getattr(node, "experiment_id", "")),
                family=str(getattr(node, "family", "")),
                candidate_dir=str(cand_dir or val_path.parent),
                primary_metric=float(metrics.get("primary", 0.0)),
                validation_scores=val_scores,
                test_scores=test_scores,
            )
        )

    # Sort descending by single model score
    results.sort(key=lambda x: x.primary_metric, reverse=True)
    return results[:max_candidates]


def try_blend_candidates(
    run_dir: Path,
    state: Any,
    data_dir: Path,
    generated_root: Path,
    min_candidates: int = 2,
    max_candidates: int = 4,
) -> EnsembleResult:
    """Evaluate ensemble blending of candidate models. Accepts if strictly beats single best."""
    try:
        pool = select_candidate_pool(
            getattr(state, "nodes", []),
            generated_root=generated_root,
            max_candidates=max_candidates,
        )
        if len(pool) < min_candidates:
            return EnsembleResult(
                status="skipped",
                reason=f"Insufficient candidates with valid scores (found {len(pool)}, need {min_candidates}).",
            )

        splits = load_train_valid(data_dir)
        valid_rows = splits["valid"]
        valid_users = [row[1] for row in valid_rows]
        valid_y = np.asarray([int(row[6]) for row in valid_rows], dtype=np.int32)

        candidate_val_scores = [c.validation_scores for c in pool]
        candidate_test_scores = [c.test_scores for c in pool]

        # Test both rank blending and z-score blending, pick whichever performs best on validation
        rank_weights, rank_metrics = find_optimal_weights(
            valid_users, valid_y, candidate_val_scores, method="rank"
        )
        norm_weights, norm_metrics = find_optimal_weights(
            valid_users, valid_y, candidate_val_scores, method="normalize"
        )

        if float(rank_metrics["primary"]) >= float(norm_metrics["primary"]):
            best_weights = rank_weights
            best_metrics = rank_metrics
            best_method = "rank"
        else:
            best_weights = norm_weights
            best_metrics = norm_metrics
            best_method = "normalize"

        single_best_primary = float(getattr(state, "best_metrics", {}).get("primary", 0.0))
        ensemble_primary = float(best_metrics["primary"])

        # Strictly require improvement on validation over the single best checkpoint
        if ensemble_primary <= single_best_primary + 1e-6:
            return EnsembleResult(
                status="skipped",
                reason="Ensemble validation score does not beat single best candidate.",
                single_best_primary=single_best_primary,
                ensemble_primary=ensemble_primary,
            )

        # Blend test scores using the validation-optimal weights
        blended_test = blend_predictions(candidate_test_scores, best_weights, method=best_method)
        blended_val = blend_predictions(candidate_val_scores, best_weights, method=best_method)

        ensemble_node_dir = generated_root / state.run_id / "ensemble"
        ensemble_node_dir.mkdir(parents=True, exist_ok=True)

        np.save(ensemble_node_dir / "test_scores.npy", blended_test)
        np.save(ensemble_node_dir / "validation_scores.npy", blended_val)

        weights_dict = {
            f"{c.experiment_id} ({c.family})": float(w)
            for c, w in zip(pool, best_weights)
        }
        candidates_used = [
            {
                "experiment_id": c.experiment_id,
                "family": c.family,
                "primary": c.primary_metric,
                "weight": float(w),
            }
            for c, w in zip(pool, best_weights)
        ]

        manifest = {
            "status": "ok",
            "blending_method": best_method,
            "metrics": best_metrics,
            "single_best_primary": single_best_primary,
            "ensemble_primary": ensemble_primary,
            "delta_primary": ensemble_primary - single_best_primary,
            "weights": weights_dict,
            "candidates": candidates_used,
        }
        (ensemble_node_dir / "ensemble_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        return EnsembleResult(
            status="ok",
            ensemble_node_dir=_repo_relative(ensemble_node_dir),
            metrics=best_metrics,
            weights=weights_dict,
            candidates_used=candidates_used,
            single_best_primary=single_best_primary,
            ensemble_primary=ensemble_primary,
        )
    except Exception as exc:
        return EnsembleResult(status="error", reason=f"{type(exc).__name__}: {exc}")
