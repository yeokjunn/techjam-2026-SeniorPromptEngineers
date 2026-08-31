from __future__ import annotations

import collections

import numpy as np


def eligible_user_indices(
    users: tuple[str, ...] | list[str], labels: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    positives: dict[str, list[int]] = collections.defaultdict(list)
    negatives: dict[str, list[int]] = collections.defaultdict(list)
    for index, (user, label) in enumerate(zip(users, labels)):
        (positives if float(label) > 0.5 else negatives)[user].append(index)
    return {
        user: (
            np.asarray(positives[user], dtype=np.int64),
            np.asarray(negatives[user], dtype=np.int64),
        )
        for user in positives.keys() & negatives.keys()
    }


def sample_bpr_pairs(
    users: tuple[str, ...] | list[str],
    labels: np.ndarray,
    rng: np.random.Generator,
    negatives_per_positive: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be positive.")
    positive_indices: list[int] = []
    negative_indices: list[int] = []
    for positive_pool, negative_pool in eligible_user_indices(users, labels).values():
        for positive in positive_pool:
            sampled = rng.choice(
                negative_pool, size=negatives_per_positive, replace=len(negative_pool) < negatives_per_positive
            )
            positive_indices.extend([int(positive)] * negatives_per_positive)
            negative_indices.extend(int(index) for index in sampled)
    return (
        np.asarray(positive_indices, dtype=np.int64),
        np.asarray(negative_indices, dtype=np.int64),
    )


def sample_hard_bpr_pairs(
    users: tuple[str, ...] | list[str],
    labels: np.ndarray,
    rng: np.random.Generator,
    hardness_scores: np.ndarray,
    negatives_per_positive: int = 1,
    *,
    top_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample same-user negatives from the highest-scoring negative pool.

    ``hardness_scores`` must be train-only model or prior scores in the same row
    order as ``labels``. The sampler never crosses users and falls back to the
    user's full negative pool when the hard slice would be empty.
    """
    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be positive.")
    hardness = np.asarray(hardness_scores, dtype=np.float64)
    if hardness.shape[0] != len(labels):
        raise ValueError("hardness_scores must align with labels.")
    if not np.all(np.isfinite(hardness)):
        raise ValueError("hardness_scores must be finite.")
    if not (0.0 < top_fraction <= 1.0):
        raise ValueError("top_fraction must be in (0, 1].")

    positive_indices: list[int] = []
    negative_indices: list[int] = []
    for positive_pool, negative_pool in eligible_user_indices(users, labels).values():
        keep = max(1, int(np.ceil(len(negative_pool) * top_fraction)))
        ranked = negative_pool[np.argsort(-hardness[negative_pool], kind="stable")[:keep]]
        pool = ranked if len(ranked) else negative_pool
        for positive in positive_pool:
            sampled = rng.choice(
                pool,
                size=negatives_per_positive,
                replace=len(pool) < negatives_per_positive,
            )
            positive_indices.extend([int(positive)] * negatives_per_positive)
            negative_indices.extend(int(index) for index in sampled)
    return (
        np.asarray(positive_indices, dtype=np.int64),
        np.asarray(negative_indices, dtype=np.int64),
    )


def sample_constrained_hard_bpr_pairs(
    users: tuple[str, ...] | list[str],
    labels: np.ndarray,
    rng: np.random.Generator,
    hardness_scores: np.ndarray,
    constraint_values: np.ndarray,
    negatives_per_positive: int = 1,
    *,
    top_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Same-user hard negatives, preferring negatives with the same constraint key."""
    constraints = np.asarray(constraint_values)
    if constraints.shape[0] != len(labels):
        raise ValueError("constraint_values must align with labels.")
    base_positive, base_negative = sample_hard_bpr_pairs(
        users,
        labels,
        rng,
        hardness_scores,
        negatives_per_positive,
        top_fraction=top_fraction,
    )
    if len(base_positive) == 0:
        return base_positive, base_negative

    by_user_constraint: dict[tuple[str, object], list[int]] = collections.defaultdict(list)
    for index, (user, label) in enumerate(zip(users, labels)):
        if float(label) <= 0.5:
            by_user_constraint[(str(user), constraints[index].item() if hasattr(constraints[index], "item") else constraints[index])].append(index)

    positive_indices: list[int] = []
    negative_indices: list[int] = []
    hardness = np.asarray(hardness_scores, dtype=np.float64)
    for positive in base_positive[::negatives_per_positive]:
        key = constraints[positive].item() if hasattr(constraints[positive], "item") else constraints[positive]
        pool = by_user_constraint.get((str(users[int(positive)]), key), [])
        if not pool:
            continue
        pool_arr = np.asarray(pool, dtype=np.int64)
        keep = max(1, int(np.ceil(len(pool_arr) * top_fraction)))
        hard_pool = pool_arr[np.argsort(-hardness[pool_arr], kind="stable")[:keep]]
        sampled = rng.choice(
            hard_pool,
            size=negatives_per_positive,
            replace=len(hard_pool) < negatives_per_positive,
        )
        positive_indices.extend([int(positive)] * negatives_per_positive)
        negative_indices.extend(int(index) for index in sampled)
    if not positive_indices:
        return base_positive, base_negative
    return (
        np.asarray(positive_indices, dtype=np.int64),
        np.asarray(negative_indices, dtype=np.int64),
    )


def sample_softmax_groups(
    users: tuple[str, ...] | list[str],
    labels: np.ndarray,
    rng: np.random.Generator,
    negatives_per_group: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    if negatives_per_group < 1:
        raise ValueError("negatives_per_group must be positive.")
    positive_indices: list[int] = []
    negative_groups: list[np.ndarray] = []
    for positive_pool, negative_pool in eligible_user_indices(users, labels).values():
        for positive in positive_pool:
            sampled = rng.choice(
                negative_pool,
                size=negatives_per_group,
                replace=len(negative_pool) < negatives_per_group,
            )
            positive_indices.append(int(positive))
            negative_groups.append(np.asarray(sampled, dtype=np.int64))
    negatives = (
        np.stack(negative_groups)
        if negative_groups
        else np.empty((0, negatives_per_group), dtype=np.int64)
    )
    return np.asarray(positive_indices, dtype=np.int64), negatives

