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
    # sorted(): set iteration order is permuted by PYTHONHASHSEED, and this
    # dict's order is what the samplers below feed to rng.choice — unsorted,
    # byte-identical candidates score differently run to run.
    return {
        user: (
            np.asarray(positives[user], dtype=np.int64),
            np.asarray(negatives[user], dtype=np.int64),
        )
        for user in sorted(positives.keys() & negatives.keys())
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

