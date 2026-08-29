from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.official import load_train_valid
from src.ui.loaders import REPO_ROOT, load_dashboard_config


def _quantiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    points = np.quantile(np.asarray(values, dtype=np.float64), [0, 0.25, 0.5, 0.75, 0.95, 1])
    return {name: float(value) for name, value in zip(("min", "p25", "p50", "p75", "p95", "max"), points)}


def build_eda_profile(data_dir: Path) -> dict[str, Any]:
    splits = load_train_valid(data_dir)
    split_summary: dict[str, Any] = {}
    date_rows: Counter[tuple[str, int]] = Counter()
    date_positives: Counter[tuple[str, int]] = Counter()
    durations: list[float] = []
    for split, rows in splits.items():
        users: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
        positives = 0
        for date, user_id, _video_id, _author_id, _tab, duration_ms, label in rows:
            positives += int(label)
            users[user_id][0] += 1
            users[user_id][1] += int(label)
            date_rows[(split, int(date))] += 1
            date_positives[(split, int(date))] += int(label)
            durations.append(float(duration_ms) / 1000.0)
        split_summary[split] = {
            "rows": len(rows),
            "users": len(users),
            "positives": positives,
            "positive_rate": positives / len(rows) if rows else 0.0,
            "impressions_per_user": _quantiles([value[0] for value in users.values()]),
            "positives_per_user": _quantiles([value[1] for value in users.values()]),
        }
    activity = [
        {
            "split": split,
            "date": date,
            "rows": count,
            "positives": date_positives[(split, date)],
            "positive_rate": date_positives[(split, date)] / count if count else 0.0,
        }
        for (split, date), count in sorted(date_rows.items())
    ]
    edges = np.asarray([0, 5, 10, 20, 30, 60, 120, 300, 600, np.inf], dtype=np.float64)
    counts, _ = np.histogram(np.asarray(durations, dtype=np.float64), bins=edges)
    duration_histogram = [
        {
            "seconds": f"{int(left)}–{'∞' if np.isinf(right) else int(right)}",
            "rows": int(count),
        }
        for left, right, count in zip(edges[:-1], edges[1:], counts)
    ]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": (
            "Aggregates from the trusted train split (2022-04-08..21) and validation split "
            "(2022-04-22..28). Rows after 2022-04-28 are skipped before labels are read."
        ),
        "splits": split_summary,
        "activity_by_date": activity,
        "duration_histogram": duration_histogram,
    }


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build aggregate-only KuaiRand-Pure EDA data.")
    parser.add_argument("--config", type=Path, default=Path("configs/ui.json"))
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = load_dashboard_config(config_path)
    profile = build_eda_profile(config.data_dir)
    _write_json_atomic(config.eda_profile_path, profile)
    print(f"Wrote aggregate EDA profile to {config.eda_profile_path}")


if __name__ == "__main__":
    main()
