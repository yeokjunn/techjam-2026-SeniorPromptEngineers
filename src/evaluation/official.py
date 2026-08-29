from __future__ import annotations

import csv
import importlib
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
STARTER_DIR = REPO_ROOT / "kuairand-starter-kit"
TRAIN_START = 20220408
TRAIN_END = 20220421
VALID_START = 20220422
VALID_END = 20220428
TEST_START = 20220429
TEST_END = 20220508
TEST_ROWS = 170_588
LABEL_PLACEHOLDER = -1
SANITY_FLOOR = 0.47
SANITY_CEILING = 0.80
OFFICIAL_VALIDATION_BASELINE = 0.6016
BASELINE_TOLERANCE = 0.003


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


@dataclass(frozen=True)
class TestSplit:
    """Test-split rows with identifiers but never labels.

    ``meta`` carries ``(row_id, user_id, video_id)`` per row in the kit's own
    ``load()['test']`` order (both loaders read the same two files and filter by
    date, preserving file order, so the 0-based index within the split *is* the
    kit's row order). ``rows`` are kit-shaped 7-tuples whose label slot is
    ``LABEL_PLACEHOLDER`` so they can flow through the kit's ``encode`` for
    features only.
    """

    meta: tuple[tuple[int, str, str], ...]
    rows: tuple[tuple, ...]


def load_test_meta(data_dir: Path, *, expected_rows: int | None = None) -> TestSplit:
    """Load the test split's identifiers and features; the label is never read.

    Same structure as ``load_train_valid`` with the filter inverted: the date is
    checked before any other column is touched, and the label slot is filled
    with ``LABEL_PLACEHOLDER`` instead of a parsed value.
    """
    video_to_author: dict[str, str] = {}
    with (data_dir / "video_features_basic_pure.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            video_to_author[row["video_id"]] = row["author_id"]

    rows: list[tuple] = []
    sources = (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    )
    for filename in sources:
        with (data_dir / filename).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                date = int(row["date"])
                if not (TEST_START <= date <= TEST_END):
                    # Crucially skip before reading any other column.
                    continue
                rows.append(
                    (
                        date,
                        row["user_id"],
                        row["video_id"],
                        video_to_author.get(row["video_id"], "UNK"),
                        row["tab"],
                        float(row["duration_ms"]),
                        LABEL_PLACEHOLDER,
                    )
                )
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(
            f"Test split has {len(rows)} rows; expected {expected_rows}."
        )
    meta = tuple((index, row[1], row[2]) for index, row in enumerate(rows))
    return TestSplit(meta=meta, rows=tuple(rows))


def classify_primary(primary: float) -> str | None:
    """Sanity band for a trusted validation primary (review I11).

    Below the floor the run learned nothing; above the ceiling the number is
    more plausibly a leak than a result (the oracle primary is 0.8484 on
    validation, and 27.1% of test users are all-negative). ``None`` means the
    value is plausible. Purely a classifier — callers decide what to do.
    """
    if primary < SANITY_FLOOR:
        return "low_score"
    if primary > SANITY_CEILING:
        return "leak"
    return None


def within_baseline_tolerance(
    primary: float,
    official: float = OFFICIAL_VALIDATION_BASELINE,
    tolerance: float = BASELINE_TOLERANCE,
) -> bool:
    """Two-sided baseline predicate: a reproduction must match, not merely clear.

    The old one-sided gate accepted anything >= official - 0.002, so a leaked
    0.85 counted as a successful baseline reproduction. The cushion keeps the
    boundary inclusive under binary floats (|0.5986 - 0.6016| computes a hair
    above 0.003, not exactly it).
    """
    return abs(primary - official) <= tolerance + 1e-12


def official_evaluate(user_ids, labels, scores) -> dict[str, float]:
    _, evaluate_module, _ = starter_modules()
    result = evaluate_module.evaluate(user_ids, labels, scores)
    return {name: float(value) for name, value in result.items()}

