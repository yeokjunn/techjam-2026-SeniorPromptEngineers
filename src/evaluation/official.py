from __future__ import annotations

import csv
import importlib
import sys
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
STARTER_DIR = REPO_ROOT / "kuairand-starter-kit"
TRAIN_START = 20220408
TRAIN_END = 20220421
VALID_START = 20220422
VALID_END = 20220428


@lru_cache(maxsize=1)
def starter_modules():
    """Import the untouched starter-kit modules without copying their logic."""
    starter = str(STARTER_DIR)
    sys.path.insert(0, starter)
    try:
        data_module = importlib.import_module("data")
        evaluate_module = importlib.import_module("evaluate")
        baseline_module = importlib.import_module("baseline")
    finally:
        if sys.path and sys.path[0] == starter:
            sys.path.pop(0)
    return data_module, evaluate_module, baseline_module


def load_train_valid(data_dir: Path) -> dict[str, list[tuple]]:
    """Load only train/validation dates; rows after 2022-04-28 are never parsed."""
    video_to_author: dict[str, str] = {}
    with (data_dir / "video_features_basic_pure.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            video_to_author[row["video_id"]] = row["author_id"]

    splits: dict[str, list[tuple]] = {"train": [], "valid": []}
    sources = (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    )
    for filename in sources:
        with (data_dir / filename).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                date = int(row["date"])
                if TRAIN_START <= date <= TRAIN_END:
                    split = "train"
                elif VALID_START <= date <= VALID_END:
                    split = "valid"
                else:
                    # Crucially skip before reading the relevance label.
                    continue
                splits[split].append(
                    (
                        date,
                        row["user_id"],
                        row["video_id"],
                        video_to_author.get(row["video_id"], "UNK"),
                        row["tab"],
                        float(row["duration_ms"]),
                        1 if row["long_view"] != "0" else 0,
                    )
                )
    return splits


def official_evaluate(user_ids, labels, scores) -> dict[str, float]:
    _, evaluate_module, _ = starter_modules()
    result = evaluate_module.evaluate(user_ids, labels, scores)
    return {name: float(value) for name, value in result.items()}

