from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.evaluation.official import load_train_valid
from src.models.baselines import run_fm, run_popularity, run_random


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one validation-only baseline.")
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    splits = load_train_valid(args.data_dir)
    print(
        f"loaded train={len(splits['train'])} valid={len(splits['valid'])}; "
        "test rows were not loaded",
        flush=True,
    )
    kind = spec["kind"]
    parameters = spec.get("parameters", {})
    if kind == "random":
        payload = run_random(splits, seed=int(parameters.get("seed", 0)))
    elif kind == "popularity":
        payload = run_popularity(splits, prior=float(parameters.get("prior", 20.0)))
    elif kind == "fm":
        payload = run_fm(splits, parameters, args.artifact_dir)
    else:
        raise ValueError(f"Unsupported baseline kind: {kind}")
    _write_json_atomic(args.result, payload)


if __name__ == "__main__":
    main()

