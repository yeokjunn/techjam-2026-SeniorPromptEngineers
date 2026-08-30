from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.models.ensemble import (
    CandidatePrediction,
    blend_predictions,
    find_optimal_weights,
    normalize_scores,
    rank_transform,
    select_candidate_pool,
    try_blend_candidates,
)
from src.evaluation.gate import run_gate
from tests.test_gate import KIT_DIR, write_synthetic_data


@dataclass
class MockNode:
    experiment_id: str
    family: str
    action: str
    status: str
    candidate_dir: str
    metrics: dict[str, float]


@dataclass
class MockState:
    run_id: str
    nodes: list[MockNode]
    best_metrics: dict[str, float]
    best_candidate_dir: str


class EnsembleTests(unittest.TestCase):
    def test_normalize_scores(self):
        scores = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        normed = normalize_scores(scores)
        self.assertAlmostEqual(float(np.mean(normed)), 0.0, places=6)
        self.assertAlmostEqual(float(np.std(normed)), 1.0, places=6)

        constant = np.asarray([3.0, 3.0, 3.0])
        self.assertTrue(np.all(normalize_scores(constant) == 0.0))

    def test_rank_transform_is_monotonic_and_bounded(self):
        scores = np.asarray([10.0, -5.0, 2.5, 100.0])
        ranks = rank_transform(scores)
        self.assertEqual(len(ranks), 4)
        self.assertAlmostEqual(ranks[1], 0.0)  # -5.0 is minimum
        self.assertAlmostEqual(ranks[3], 1.0)  # 100.0 is maximum
        self.assertTrue(ranks[1] < ranks[2] < ranks[0] < ranks[3])

    def test_find_optimal_weights_improves_metrics(self):
        # Two users, each with two positive and two negative impressions
        users = ["u1", "u1", "u1", "u1", "u2", "u2", "u2", "u2"]
        labels = np.asarray([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int32)

        # Model A gets u1 right but confuses u2
        scores_a = np.asarray([3.0, 2.0, 1.0, 0.0, 0.0, 1.0, 2.0, 3.0])
        # Model B gets u2 right but confuses u1
        scores_b = np.asarray([0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0])

        weights, metrics = find_optimal_weights(
            users, labels, [scores_a, scores_b], method="rank"
        )
        self.assertEqual(len(weights), 2)
        self.assertAlmostEqual(sum(weights), 1.0, places=5)
        # The blend of both models should perform better than either model alone
        self.assertGreater(metrics["primary"], 0.5)

    def test_try_blend_candidates_skips_when_fewer_than_min_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state = MockState(run_id="run1", nodes=[], best_metrics={"primary": 0.6}, best_candidate_dir="")
            res = try_blend_candidates(root, state, root, root, min_candidates=2)
            self.assertEqual(res.status, "skipped")
            self.assertIn("Insufficient candidates", res.reason)

    def test_try_blend_candidates_accepts_when_improved_and_gates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_dir = write_synthetic_data(root)

            # Synthetic validation rows in write_synthetic_data:
            # 20220410 (train), 20220425 (valid): 1 valid row (u3,v3) + others
            from src.evaluation.official import load_train_valid, load_test_meta
            splits = load_train_valid(data_dir)
            n_valid = len(splits["valid"])
            test_split = load_test_meta(data_dir)
            n_test = len(test_split.rows)

            cand1_dir = root / "cand1"
            cand1_dir.mkdir()
            val1 = np.linspace(0.1, 0.8, n_valid)
            test1 = np.linspace(0.1, 0.8, n_test)
            np.save(cand1_dir / "validation_scores.npy", val1)
            np.save(cand1_dir / "test_scores.npy", test1)

            cand2_dir = root / "cand2"
            cand2_dir.mkdir()
            val2 = np.linspace(0.2, 0.9, n_valid)
            test2 = np.linspace(0.2, 0.9, n_test)
            np.save(cand2_dir / "validation_scores.npy", val2)
            np.save(cand2_dir / "test_scores.npy", test2)

            nodes = [
                MockNode("cand1", "bpr", "propose", "success", str(cand1_dir), {"primary": 0.50}),
                MockNode("cand2", "history_features", "propose", "success", str(cand2_dir), {"primary": 0.50}),
            ]
            state = MockState(
                run_id="test_run",
                nodes=nodes,
                best_metrics={"primary": 0.45},  # lower so ensemble improves
                best_candidate_dir=str(cand1_dir),
            )

            result = try_blend_candidates(
                run_dir=root,
                state=state,
                data_dir=data_dir,
                generated_root=root / "generated",
            )
            self.assertEqual(result.status, "ok")
            self.assertIsNotNone(result.ensemble_node_dir)
            ensemble_dir = root / "generated" / "test_run" / "ensemble"
            self.assertTrue((ensemble_dir / "test_scores.npy").is_file())
            self.assertTrue((ensemble_dir / "validation_scores.npy").is_file())
            self.assertTrue((ensemble_dir / "ensemble_manifest.json").is_file())

            # Verify that run_gate can successfully validate this ensemble node
            gate_res = run_gate(
                run_dir=root,
                node_dir=ensemble_dir,
                data_dir=data_dir,
                kit_dir=KIT_DIR,
            )
            self.assertEqual(gate_res.status, "ok")


if __name__ == "__main__":
    unittest.main()

