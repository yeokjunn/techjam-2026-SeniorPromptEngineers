"""Within-user residual correlation of engagement signals with long_view.

This is the cheap, decisive diagnostic that decides whether the multi-task path
(direction B) can transfer at all, before spending any hours on it.

The reasoning (from the starter kit + AGENTS.md):
  The task is *within-user ranking*. A signal that varies only *between* users
  (e.g. is_click's marginal correlation ~0.76 with long_view) is dominated by
  base-rate differences between users — and within a user that variation is a
  constant, so it contributes exactly 0 to the ranking. The multi-task bet
  only works if each auxiliary signal has *within-user* (conditional) residual
  correlation with long_view: once you remove each user's mean from both the
  signal and long_view, is there still a relationship across the pooled
  residuals?

  We measure that Pearson r over the discriminative users (those with both a
  positive and a negative long_view, the population GAUC ranks), pooled across
  users after per-user mean-centring. A signal with |r| < 0.15 here is a
  noise-level multi-task probe and the family should be abandoned rather than
  swept.

Outputs both the *marginal* (uncentred) correlation (to show the 0.76-style
headline number) and the *within-user residual* correlation (the number that
matters), plus a per-signal verdict and an overall recommendation.

Pure numpy + stdlib. Reads only TRAIN dates (20220408-20220421) so no
validation/test label is ever touched.

Usage (from the repository root):

    python -m scripts.within_user_correlation
    python -m scripts.within_user_correlation --data_dir data/KuaiRand-Pure/data
    python -m scripts.within_user_correlation --min-user-rows 5 --transfer-floor 0.15
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.official import TRAIN_END, TRAIN_START  # noqa: E402

# The six engagement signals the multi-task direction would use as aux heads.
# long_view is the ranking target (not an aux head, but reported for reference).
SIGNALS = ("is_click", "is_like", "is_follow", "is_comment", "is_forward", "play_time_ms")
TARGET = "long_view"

DEFAULT_TRANSFER_FLOOR = 0.15  # |r| below this within-user is a noise-level probe.
DEFAULT_MIN_USER_ROWS = 2  # need >=2 rows to centre; GAUC needs discriminative users anyway.


def _resolve_data_dir(arg: str | None) -> Path:
    raw = arg or os.environ.get("KUAIRAND_DATA_DIR") or REPO_ROOT / "data" / "KuaiRand-Pure" / "data"
    return Path(raw)


def load_train_engagement(data_dir: Path) -> list[dict]:
    """Read the train-date rows with all engagement columns + long_view.

    The date is checked *before* any other column is read (the kit's discipline),
    so no valid/test signal is ever parsed. Returns one dict per train row.
    """
    rows: list[dict] = []
    sources = (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    )
    for filename in sources:
        path = data_dir / filename
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                date = int(row["date"])
                if not (TRAIN_START <= date <= TRAIN_END):
                    continue  # skip before reading any signal column
                rows.append(row)
    return rows


def _safe_float(value: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    return x if math.isfinite(x) else 0.0


def build_arrays(rows: list[dict]):
    """Return per-row arrays: user_id, long_view, and each signal (binary/float)."""
    n = len(rows)
    users = [row["user_id"] for row in rows]
    long_view = np.fromiter(
        (1.0 if row["long_view"] != "0" else 0.0 for row in rows), dtype=np.float64, count=n
    )
    signals: dict[str, np.ndarray] = {}
    for name in SIGNALS:
        if name == "play_time_ms":
            # play_time is heavy-tailed; correlate against a log1p transform so one
            # outlier row doesn't dominate the Pearson r.
            signals[name] = np.log1p(np.clip(
                np.fromiter((_safe_float(r[name]) for r in rows), dtype=np.float64, count=n),
                0.0, None,
            ))
        else:
            signals[name] = np.fromiter(
                (1.0 if row[name] != "0" else 0.0 for row in rows), dtype=np.float64, count=n
            )
    return users, long_view, signals


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    if denom == 0.0:
        return 0.0
    return float((x * y).sum() / denom)


def marginal_correlation(signal: np.ndarray, target: np.ndarray) -> float:
    """The headline number (uncentred across all rows) — the misleading one."""
    return _pearson(signal, target)


def within_user_residual_correlation(
    users: list[str],
    signal: np.ndarray,
    target: np.ndarray,
    min_user_rows: int = DEFAULT_MIN_USER_ROWS,
) -> tuple[float, int]:
    """Per-user mean-centred, pooled Pearson r — the number that matters for ranking.

    Only discriminative users (both a positive and a negative long_view) are
    pooled, matching the GAUC population; a non-discriminative user has zero
    target variance so centring them yields nothing.

    Returns (r, n_pooled_rows).
    """
    # Group row indices by user.
    by_user: dict[str, list[int]] = {}
    for i, u in enumerate(users):
        by_user.setdefault(u, []).append(i)

    sx = sy = sxx = syy = sxy = 0.0
    n_pooled = 0
    for idxs in by_user.values():
        if len(idxs) < min_user_rows:
            continue
        ix = np.fromiter(idxs, dtype=np.int64)
        y = target[ix]
        if y.min() == y.max():
            continue  # not discriminative; within-user target variance is zero
        x = signal[ix]
        x = x - x.mean()
        y = y - y.mean()
        sx += x.sum(); sy += y.sum()
        sxx += float((x * x).sum()); syy += float((y * y).sum())
        sxy += float((x * y).sum())
        n_pooled += len(idxs)

    # sx/sy are ~0 by construction (each block is centred) but keep the formula exact.
    cov = sxy - (sx * sy / n_pooled) if n_pooled else 0.0
    var_x = sxx - (sx * sx / n_pooled) if n_pooled else 0.0
    var_y = syy - (sy * sy / n_pooled) if n_pooled else 0.0
    denom = math.sqrt(var_x * var_y)
    if denom == 0.0:
        return 0.0, n_pooled
    return float(cov / denom), n_pooled


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default=None, help="KuaiRand-Pure data dir (default: data/KuaiRand-Pure/data)")
    ap.add_argument("--transfer-floor", type=float, default=DEFAULT_TRANSFER_FLOOR,
                    help=f"|within-user r| below this is a noise-level multi-task probe (default {DEFAULT_TRANSFER_FLOOR})")
    ap.add_argument("--min-user-rows", type=int, default=DEFAULT_MIN_USER_ROWS,
                    help="skip users with fewer rows than this (default %(default)s)")
    args = ap.parse_args()

    data_dir = _resolve_data_dir(args.data_dir)
    print(f"loading train engagement from {data_dir} ...")
    rows = load_train_engagement(data_dir)
    if not rows:
        print("ERROR: no train rows found. Check --data_dir and that the dataset is downloaded.")
        return 1
    print(f"  {len(rows):,} train rows (dates {TRAIN_START}-{TRAIN_END})")

    users, long_view, signals = build_arrays(rows)
    n_users = len(set(users))
    pos = float(long_view.mean())
    print(f"  {n_users:,} users | long_view positive rate {pos:.4f}")

    print(f"\n{'signal':14} {'marginal r':>11} {'within-user r':>13} {'pooled rows':>12}  verdict")
    print("-" * 70)
    transferable: list[str] = []
    for name in SIGNALS:
        sig = signals[name]
        r_marg = marginal_correlation(sig, long_view)
        r_wu, n_pooled = within_user_residual_correlation(
            users, sig, long_view, min_user_rows=args.min_user_rows
        )
        verdict = "TRANSFERABLE" if abs(r_wu) >= args.transfer_floor else "noise-level"
        if abs(r_wu) >= args.transfer_floor:
            transferable.append(name)
        print(f"{name:14} {r_marg:>11.4f} {r_wu:>13.4f} {n_pooled:>12,}  {verdict}")

    print("-" * 70)
    print(f"transfer floor |r| = {args.transfer_floor}  |  transferable signals: "
          f"{transferable or 'none'}")
    print()
    if transferable:
        print("VERDICT: multi-task has within-user signal -> worth building (direction B).")
        print("         strongest aux head(s) by |within-user r| should be added first.")
    else:
        print("VERDICT: NO within-user residual correlation -> multi-task will NOT transfer")
        print("         to within-user ranking. Abandon direction B; pivot hard to sequences (A/C).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
