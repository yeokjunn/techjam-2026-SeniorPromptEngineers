"""Scratchpad: gate input loading for the T3 sanity-gate experiment.

Loads the persisted candidate artifacts and hands them to the gate in a
single container. To be absorbed into gate.py once T3 starts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_gate_inputs(path: str, expected_rows: int) -> dict:
    """Load gate inputs from a persisted .npz bundle.

    Returns a dict with keys ``x`` (feature matrix), ``meta`` (row ids),
    and ``rows`` (row count).
    """
    with np.load(Path(path)) as data:
        x = np.asarray(data["x"], dtype=np.float32)
        meta = list(data["meta"])
        rows = int(x.shape[0])
    return {"x": x, "meta": meta, "rows": rows}
