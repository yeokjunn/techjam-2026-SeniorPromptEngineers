from __future__ import annotations

from typing import Any

from .types import ExperimentNode, RunState


FAMILIES = {"bpr", "group_softmax"}


def successful_families(state: RunState) -> set[str]:
    return {node.family for node in state.nodes if node.status == "success"}


def required_family(state: RunState) -> str | None:
    completed = successful_families(state)
    if not completed:
        return None
    missing = FAMILIES - completed
    return next(iter(missing)) if len(missing) == 1 else None


def coverage_complete(state: RunState) -> bool:
    return FAMILIES.issubset(successful_families(state))


def sanitize_parameters(family: str, raw: dict[str, Any]) -> dict[str, Any]:
    parameters = {
        "seed": int(raw.get("seed", 0)),
        "k": int(raw.get("k", 16)),
        "learning_rate": float(raw.get("learning_rate", 0.001)),
        "epochs": int(raw.get("epochs", 40)),
        "batch_size": int(raw.get("batch_size", 2048)),
        "patience": int(raw.get("patience", 4)),
    }
    if parameters["k"] != 16:
        raise ValueError("Ranking-loss attribution requires k=16 in the first research run.")
    if parameters["learning_rate"] not in {0.0003, 0.0005, 0.001}:
        raise ValueError("learning_rate is outside the approved method-card search space.")
    if not 1 <= parameters["epochs"] <= 40:
        raise ValueError("epochs must be between 1 and 40.")
    if not 1 <= parameters["patience"] <= 6:
        raise ValueError("patience must be between 1 and 6.")
    if parameters["seed"] < 0:
        raise ValueError("seed must be non-negative.")

    if family == "bpr":
        parameters["negatives_per_positive"] = int(raw.get("negatives_per_positive") or 1)
        if parameters["negatives_per_positive"] not in {1, 2}:
            raise ValueError("BPR negatives_per_positive must be 1 or 2.")
        if parameters["batch_size"] not in {2048, 4096}:
            raise ValueError("BPR batch_size must be 2048 or 4096.")
    elif family == "group_softmax":
        parameters["negatives_per_group"] = int(raw.get("negatives_per_group") or 4)
        parameters["temperature"] = float(raw.get("temperature") or 1.0)
        if parameters["negatives_per_group"] not in {4, 8}:
            raise ValueError("Group-softmax negatives_per_group must be 4 or 8.")
        if parameters["temperature"] not in {0.5, 1.0, 2.0}:
            raise ValueError("Group-softmax temperature must be 0.5, 1.0, or 2.0.")
        if parameters["batch_size"] not in {512, 1024, 2048}:
            raise ValueError("Group-softmax batch_size must be 512, 1024, or 2048.")
    else:
        raise ValueError(f"Unsupported family: {family}")
    return parameters


class SearchPolicy:
    def __init__(self, epsilon: float, patience: int, replication_seeds: list[int]):
        self.epsilon = float(epsilon)
        self.patience = int(patience)
        self.replication_seeds = [int(seed) for seed in replication_seeds]

    def observe_success(self, state: RunState, node: ExperimentNode) -> None:
        assert node.metrics is not None
        score = float(node.metrics["primary"])
        previous_meaningful = state.meaningful_best
        if previous_meaningful is None or score > previous_meaningful + self.epsilon:
            state.meaningful_best = score
            state.stagnant_iterations = 0
        else:
            state.stagnant_iterations += 1

        if state.best_metrics is None or score > float(state.best_metrics["primary"]):
            state.best_metrics = dict(node.metrics)
            state.best_experiment_id = node.experiment_id
            state.best_artifact_path = node.artifact_path
            state.best_candidate_dir = node.candidate_dir

        improvement = score - state.baseline_primary
        if improvement > self.epsilon and node.action != "replicate":
            existing_sources = {item.get("source_experiment") for item in state.pending_replications}
            if node.experiment_id not in existing_sources:
                for seed in self.replication_seeds:
                    state.pending_replications.append(
                        {"source_experiment": node.experiment_id, "seed": seed}
                    )

    def should_stop(self, state: RunState) -> bool:
        return coverage_complete(state) and state.stagnant_iterations >= self.patience and not state.pending_replications
