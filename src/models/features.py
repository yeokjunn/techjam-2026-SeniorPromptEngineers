"""Trusted, train-only history features for the ``history_features`` family (review I8).

The kit freezes the model at five id fields; user-behaviour history is its own #2 untested
direction. This module builds that history in *trusted* code so a generated candidate never
touches a raw log, and so the leakage rules are enforced here rather than hoped for in a prompt.

**Leakage invariant.** The relevance label (element 6 of a kit row) is read only while iterating
the train split -- see ``_compute_train_state``, the one function that touches it. Valid and test
rows contribute *keys* (user, author, tab, duration, date) and never a label. ``load_train_valid``
already skips test dates before reading the label, and ``load_test_meta`` fills the label slot
with a placeholder, so no test label is reachable at all.

**Time-respecting rule.** For a train row on day *d*, ``scheme="prior_days"`` counts only that
key's rows on days strictly earlier than *d*; valid and test rows use all of train. Every
valid/test row is therefore scored from statistics built out of strictly earlier data, and
building train rows the same way keeps the train-time and inference-time feature distributions
aligned. ``scheme="leave_one_out"`` is offered as the contrast -- same tables, minus the row's
own contribution -- but it lets a train row see the same user's *later* days, so the model
over-trusts the feature and the gain does not transfer. It exists to be measured, not preferred.

**Layout.** Each enabled group contributes one int32 column of ``SLOTS_PER_GROUP`` slots: eight
value buckets from seven interior quantile edges computed on train values only (the kit's trick,
``kuairand-starter-kit/data.py:32-33``), plus a reserved slot for "no history / unknown". Unknown
is deliberately its own slot rather than the prior bucket, so the model can learn it separately.

**The ``video_rate`` group.** The per-video train-window ``long_view`` rate is the strongest
single signal measured on this dataset: ordering by it alone scores primary 0.5807 against
0.4827 for a random ordering, and blended with the FM it added +0.0021. The kit ships a
ready-made version of it in ``video_features_statistic_pure.csv``, which a leakage audit
rejected -- that file's counting window spans the test dates, so reading it is reading a label
from the future. This group is the train-only replacement, and it also *predicts test behaviour
better* than the leaky file (Spearman 0.869 vs 0.831). It is an ordinary ``RATE_GROUPS`` member
and so inherits the scheme, smoothing and unknown-slot discipline above with no new mechanism:
in particular a video seen only in valid/test rows has no train table entry and lands in
``UNKNOWN_SLOT``, never in a bucket derived from its own future rows.

``build_aux_labels`` serves the sibling ``multi_task`` family with auxiliary train targets
(``is_click``, ``is_like``, scaled ``play_time``). It is train-only by construction: a loss touches
train rows only, so no valid/test path exists and none may be added.
"""

from __future__ import annotations

import bisect
import csv
import datetime as dt
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.official import (
    REPO_ROOT,
    TRAIN_END,
    TRAIN_START,
    load_test_meta,
    load_train_valid,
)


#: Group order fixes column order, so ``video_rate`` is *appended*: inserting it would shift the
#: field layout of every proposal already expressed against the six original groups.
GROUPS = (
    "user_rate",
    "user_author",
    "user_tab",
    "recency",
    "video_age",
    "tab_cross",
    "video_rate",
)
#: Groups whose value is a smoothed long_view rate over a key; the rest are computed per row.
RATE_GROUPS = ("user_rate", "user_author", "user_tab", "tab_cross", "video_rate")

SLOTS_PER_GROUP = 9
VALUE_BUCKETS = SLOTS_PER_GROUP - 1  # -> VALUE_BUCKETS - 1 interior quantile edges
UNKNOWN_SLOT = SLOTS_PER_GROUP - 1

#: Auxiliary training targets for the ``multi_task`` family (review I8, spec appendix A idea 9).
#: is_click covers 44% of rows and is strongly linked to long_view.
AUX_HEADS = ("is_click", "is_like", "is_follow", "is_comment", "is_forward", "play_time")
AUX_SOURCES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)

RECENCY_CAP_DAYS = 14
DEFAULT_SMOOTHING = 20.0
DEFAULT_SCHEME = "prior_days"
SCHEMES = ("prior_days", "leave_one_out")
SPLITS = ("train", "valid", "test")

# Kit row layout: (date, user_id, video_id, author_id, tab, duration_ms, long_view)
_DATE, _USER, _VIDEO, _AUTHOR, _TAB, _DURATION, _LABEL = range(7)


class FeatureDataUnavailable(RuntimeError):
    """The trusted rows backing a split could not be loaded; the caller may degrade gracefully."""


# --------------------------------------------------------------------------------------------
# spec helpers (pure -- no data, so a candidate can size its model before loading anything)
# --------------------------------------------------------------------------------------------


def enabled_groups(spec: dict) -> tuple[str, ...]:
    return tuple(name for name in GROUPS if bool(spec.get(f"use_{name}", True)))


def feature_dimension(spec: dict) -> int:
    """Width of the index space ``build_features`` emits into: 9 slots per enabled group."""
    return SLOTS_PER_GROUP * len(enabled_groups(spec))


# --------------------------------------------------------------------------------------------
# trusted loaders
# --------------------------------------------------------------------------------------------


def _resolve_data_dir(spec: dict) -> Path:
    raw = spec.get("data_dir") or os.environ.get("KUAIRAND_DATA_DIR")
    return Path(raw) if raw else REPO_ROOT / "data" / "KuaiRand-Pure" / "data"


@lru_cache(maxsize=1)
def _cached_train_valid(data_dir: str) -> dict[str, list[tuple]]:
    """One ~10s parse serves every call in a worker process."""
    return load_train_valid(Path(data_dir))


@lru_cache(maxsize=1)
def _cached_test_rows(data_dir: str, expected_rows: int) -> tuple[tuple, ...]:
    return load_test_meta(Path(data_dir), expected_rows=expected_rows).rows


@lru_cache(maxsize=1)
def _cached_upload_ordinals(data_dir: str) -> dict[str, int]:
    """``video_id -> upload day ordinal``; ``load_train_valid`` keeps only the author column."""
    path = Path(data_dir) / "video_features_basic_pure.csv"
    ordinals: dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get("upload_dt") or "").strip()[:10]
            try:
                ordinals[row["video_id"]] = dt.date.fromisoformat(raw).toordinal()
            except ValueError:
                continue  # missing or malformed upload_dt -> the unknown slot
    return ordinals


def _date_ordinal(value: Any) -> int:
    packed = int(value)
    return dt.date(packed // 10000, (packed // 100) % 100, packed % 100).toordinal()


def _split_rows(spec: dict, split: str, expected: int) -> tuple[list, list, dict[str, int]]:
    """Return ``(train_rows, target_rows, upload_ordinals)`` from trusted sources only."""
    override = spec.get("history_rows")
    if override is not None:
        # Documented test override: synthetic rows, so unit tests need no dataset.
        for required in ("train", split):
            if required not in override:
                raise ValueError(f"spec['history_rows'] must provide a {required!r} entry.")
        # ...and an unguarded *write* channel into the train table if it is not bounded
        # here. Candidate code can reach labelled validation rows (this module re-exports
        # ``load_train_valid``), and both ``safety.validate_source`` and
        # ``validate_family_contract`` accept a spec that hands them over as "train"
        # history — which would fold valid labels into every rate group and so into the
        # promotion decision. Bounded with the same discipline the rest of the module
        # uses (``build_aux_labels`` at :492, ``official.py:60``): the date is checked
        # *before* any other column is read, so a non-train label is never touched.
        train_rows = [
            row for row in override["train"] if TRAIN_START <= int(row[_DATE]) <= TRAIN_END
        ]
        return (
            train_rows,
            # The train split scores against its own rows, so it gets the bounded list
            # too — otherwise the row-count check below would compare a clamped history
            # against an unclamped target.
            train_rows if split == "train" else list(override[split]),
            dict(spec.get("video_upload_dates") or {}),
        )

    data_dir = _resolve_data_dir(spec)
    try:
        splits = _cached_train_valid(str(data_dir))
        target = (
            _cached_test_rows(str(data_dir), expected) if split == "test" else splits[split]
        )
        uploads = _cached_upload_ordinals(str(data_dir))
    except (OSError, KeyError) as exc:
        raise FeatureDataUnavailable(
            f"Could not load trusted rows for split {split!r}: {exc}"
        ) from exc
    return list(splits["train"]), list(target), uploads


# --------------------------------------------------------------------------------------------
# feature tables
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _TrainState:
    """Everything derived from the train split: tables, bucket edges and the train slots."""

    tables: dict[str, dict[Any, list[float]]]
    positive_days: dict[str, list[int]]
    duration_edges: np.ndarray
    prior: float
    edges: dict[str, np.ndarray]
    slots: dict[str, np.ndarray]


def _duration_buckets(edges: np.ndarray, rows: list) -> np.ndarray:
    """Bucket every row's duration in one vectorised call.

    Done up front rather than per row: a scalar ``np.searchsorted`` per row dominated the
    build (38s -> 14s on the real train split for the same output).
    """
    if not rows:
        return np.empty(0, dtype=np.int32)
    durations = np.asarray([row[_DURATION] for row in rows], dtype=np.float64)
    return np.searchsorted(edges, durations).astype(np.int32)


def _rate_key(group: str, row: tuple, duration_bucket: int) -> Any:
    if group == "user_rate":
        return row[_USER]
    if group == "user_author":
        return (row[_USER], row[_AUTHOR])
    if group == "user_tab":
        return (row[_USER], row[_TAB])
    if group == "video_rate":
        # The one video-side rate: keyed by the video alone, so it is *not* constant within a
        # user and therefore contributes through GAUC's within-user ranking directly, unlike
        # the user-side groups which only act through crosses.
        return row[_VIDEO]
    if group == "tab_cross":
        return (row[_TAB], duration_bucket)
    # Explicit, so a group added to RATE_GROUPS without its own branch fails here
    # instead of silently sharing tab_cross's table.
    raise ValueError(f"No rate key defined for group {group!r}.")


def _smoothed(positives: float, count: float, prior: float, smoothing: float) -> float:
    return (positives + smoothing * prior) / (count + smoothing)


def _recency_value(positive_days: dict[str, list[int]], row: tuple, day: int) -> float:
    """Days since the user's last long_view strictly before this row's day, capped.

    Scheme-independent by construction: requiring a *strictly earlier* day already excludes the
    row itself, so ``leave_one_out`` and ``prior_days`` agree here.
    """
    days = positive_days.get(row[_USER])
    if not days:
        return np.nan
    position = bisect.bisect_left(days, day)
    if position == 0:
        return np.nan
    return float(min(day - days[position - 1], RECENCY_CAP_DAYS))


def _video_age_value(uploads: dict[str, int], row: tuple, day: int) -> float:
    uploaded = uploads.get(row[_VIDEO])
    if uploaded is None:
        return np.nan
    return float(day - uploaded)


def _bucket(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    slots = np.full(len(values), UNKNOWN_SLOT, dtype=np.int16)
    known = np.isfinite(values)
    if known.any():
        slots[known] = (
            np.searchsorted(edges, values[known]).astype(np.int16) if len(edges) else 0
        )
    return slots


def _quantile_edges(values: np.ndarray) -> np.ndarray:
    known = values[np.isfinite(values)]
    if not len(known):
        return np.empty(0)
    # The kit's trick (data.py:32-33): interior quantiles of the TRAIN values only.
    return np.quantile(known, np.linspace(0, 1, VALUE_BUCKETS + 1)[1:-1])


def _compute_train_state(
    train_rows: list, uploads: dict[str, int], scheme: str, smoothing: float
) -> _TrainState:
    """Build every table from train rows. This is the only place a label is read."""
    duration_edges = (
        np.quantile(
            np.asarray([row[_DURATION] for row in train_rows], dtype=np.float64),
            np.linspace(0, 1, 11)[1:-1],
        )
        if train_rows
        else np.empty(0)
    )
    buckets = _duration_buckets(duration_edges, train_rows)
    ordinals = [_date_ordinal(row[_DATE]) for row in train_rows]

    labels = np.asarray([float(row[_LABEL]) for row in train_rows], dtype=np.float64)
    prior = float(labels.mean()) if len(labels) else 0.0

    tables: dict[str, dict[Any, list[float]]] = {group: {} for group in RATE_GROUPS}
    positive_days: dict[str, list[int]] = {}
    values: dict[str, np.ndarray] = {group: np.full(len(train_rows), np.nan) for group in GROUPS}

    order = sorted(range(len(train_rows)), key=lambda index: train_rows[index][_DATE])

    def fold(index: int) -> None:
        row = train_rows[index]
        label = labels[index]
        bucket = buckets[index]
        for group in RATE_GROUPS:
            entry = tables[group].setdefault(_rate_key(group, row, bucket), [0.0, 0.0])
            entry[0] += label
            entry[1] += 1.0
        if label > 0.5:
            positive_days.setdefault(row[_USER], []).append(ordinals[index])

    def score(index: int) -> None:
        row = train_rows[index]
        bucket = buckets[index]
        for group in RATE_GROUPS:
            entry = tables[group].get(_rate_key(group, row, bucket))
            if entry is not None and entry[1] > 0:
                values[group][index] = _smoothed(entry[0], entry[1], prior, smoothing)

    if scheme == "prior_days":
        # Expanding window: score a whole day from the tables, *then* fold that day in.
        start = 0
        while start < len(order):
            day = train_rows[order[start]][_DATE]
            end = start
            while end < len(order) and train_rows[order[end]][_DATE] == day:
                end += 1
            for index in order[start:end]:
                score(index)
            for index in order[start:end]:
                fold(index)
            start = end
    else:  # leave_one_out: full tables minus the row's own contribution
        for index in order:
            fold(index)
        for index in range(len(train_rows)):
            row = train_rows[index]
            label = labels[index]
            bucket = buckets[index]
            for group in RATE_GROUPS:
                entry = tables[group][_rate_key(group, row, bucket)]
                count = entry[1] - 1.0
                if count > 0:
                    values[group][index] = _smoothed(entry[0] - label, count, prior, smoothing)

    for days in positive_days.values():
        days.sort()

    for index, row in enumerate(train_rows):
        day = ordinals[index]
        values["recency"][index] = _recency_value(positive_days, row, day)
        values["video_age"][index] = _video_age_value(uploads, row, day)

    edges = {group: _quantile_edges(values[group]) for group in GROUPS}
    slots = {group: _bucket(values[group], edges[group]) for group in GROUPS}
    return _TrainState(tables, positive_days, duration_edges, prior, edges, slots)


@lru_cache(maxsize=4)
def _cached_train_state(data_dir: str, scheme: str, smoothing: float) -> _TrainState:
    rows = _cached_train_valid(data_dir)["train"]
    return _compute_train_state(rows, _cached_upload_ordinals(data_dir), scheme, smoothing)


def _target_slots(
    state: _TrainState,
    target_rows: list,
    uploads: dict[str, int],
    group: str,
    smoothing: float,
) -> np.ndarray:
    """Score non-train rows against the *full* train tables. No label is read here."""
    values = np.full(len(target_rows), np.nan)
    if group in RATE_GROUPS:
        table = state.tables[group]
        buckets = _duration_buckets(state.duration_edges, target_rows)
        for index, row in enumerate(target_rows):
            entry = table.get(_rate_key(group, row, buckets[index]))
            if entry is not None and entry[1] > 0:
                values[index] = _smoothed(entry[0], entry[1], state.prior, smoothing)
    elif group == "recency":
        for index, row in enumerate(target_rows):
            values[index] = _recency_value(state.positive_days, row, _date_ordinal(row[_DATE]))
    else:
        for index, row in enumerate(target_rows):
            values[index] = _video_age_value(uploads, row, _date_ordinal(row[_DATE]))
    return _bucket(values, state.edges[group])


# --------------------------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------------------------


def build_features(rows, spec: dict) -> np.ndarray:
    """Build history feature columns for one split.

    ``rows`` is the caller's encoded id matrix for the split (``context.train_x`` /
    ``valid_x`` / ``test_x``); it is used for its length, which must match the trusted row
    count. Returns ``(len(rows), g)`` int32 indices already offset by ``spec["field_offset"]``,
    where ``g`` is the number of enabled groups, so the array concatenates straight onto the
    caller's matrix and every value is a valid FM field index.
    """
    split = str(spec.get("split", "train"))
    if split not in SPLITS:
        raise ValueError(f"spec['split'] must be one of {SPLITS}; got {split!r}")
    scheme = str(spec.get("scheme", DEFAULT_SCHEME))
    if scheme not in SCHEMES:
        raise ValueError(f"spec['scheme'] must be one of {SCHEMES}; got {scheme!r}")
    smoothing = float(spec.get("smoothing", DEFAULT_SMOOTHING))
    if smoothing <= 0:
        raise ValueError("spec['smoothing'] must be positive.")
    field_offset = int(spec.get("field_offset", 0))
    groups = enabled_groups(spec)

    expected = len(rows)
    train_rows, target_rows, uploads = _split_rows(spec, split, expected)
    if len(target_rows) != expected:
        raise ValueError(
            f"Split {split!r} has {len(target_rows)} trusted rows but {expected} were passed; "
            "row order and length must match the kit's."
        )

    if spec.get("history_rows") is not None:
        state = _compute_train_state(train_rows, uploads, scheme, smoothing)
    else:
        state = _cached_train_state(str(_resolve_data_dir(spec)), scheme, smoothing)

    features = np.empty((expected, len(groups)), dtype=np.int32)
    for column, group in enumerate(groups):
        slots = (
            state.slots[group]
            if split == "train"
            else _target_slots(state, target_rows, uploads, group, smoothing)
        )
        # Cast before the add: slots are int16, and a realistic field_offset (40260 on
        # KuaiRand-Pure) overflows int16 under NumPy 2's NEP 50 promotion rules.
        base = field_offset + column * SLOTS_PER_GROUP
        features[:, column] = slots.astype(np.int32) + base
    return features


# --------------------------------------------------------------------------------------------
# auxiliary training targets (the ``multi_task`` family)
# --------------------------------------------------------------------------------------------


def enabled_aux_heads(spec: dict) -> tuple[str, ...]:
    return tuple(name for name in AUX_HEADS if bool(spec.get(f"use_{name}", True)))


def aux_dimension(spec: dict) -> int:
    """Number of auxiliary target columns ``build_aux_labels`` will emit."""
    return len(enabled_aux_heads(spec))


@lru_cache(maxsize=1)
def _cached_aux_columns(data_dir: str) -> dict[str, np.ndarray]:
    """``is_click`` / ``is_like`` / ``is_follow`` / ``is_comment`` / ``is_forward`` / ``play_time_ms`` for TRAIN dates only.

    A separate reader because ``load_train_valid``'s 7-tuple drops these columns. Same
    discipline as ``official.py``: the date is checked *before* any other column is touched, so
    no auxiliary signal is ever read from a valid or test row. Rows come out in
    ``load_train_valid``'s train order -- the two loaders read the same files in the same order
    with the same filter -- and ``build_aux_labels`` asserts the lengths agree.
    """
    clicks: list[float] = []
    likes: list[float] = []
    follows: list[float] = []
    comments: list[float] = []
    forwards: list[float] = []
    plays: list[float] = []
    for filename in AUX_SOURCES:
        with (Path(data_dir) / filename).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if not (TRAIN_START <= int(row["date"]) <= TRAIN_END):
                    # Crucially skip before reading any auxiliary label.
                    continue
                clicks.append(1.0 if row["is_click"] != "0" else 0.0)
                likes.append(1.0 if row["is_like"] != "0" else 0.0)
                follows.append(1.0 if row["is_follow"] != "0" else 0.0)
                comments.append(1.0 if row["is_comment"] != "0" else 0.0)
                forwards.append(1.0 if row["is_forward"] != "0" else 0.0)
                plays.append(float(row["play_time_ms"]))
    return {
        "is_click": np.asarray(clicks, dtype=np.float64),
        "is_like": np.asarray(likes, dtype=np.float64),
        "is_follow": np.asarray(follows, dtype=np.float64),
        "is_comment": np.asarray(comments, dtype=np.float64),
        "is_forward": np.asarray(forwards, dtype=np.float64),
        "play_time": np.asarray(plays, dtype=np.float64),
    }


def _aux_columns(spec: dict) -> dict[str, np.ndarray]:
    override = spec.get("aux_rows")
    if override is not None:
        # Documented test override: (is_click, is_like, is_follow, is_comment, is_forward, play_time_ms) or 3-tuple fallback per train row.
        rows = [tuple(item) for item in override]
        if rows and len(rows[0]) >= 6:
            return {
                "is_click": np.asarray([r[0] for r in rows], dtype=np.float64),
                "is_like": np.asarray([r[1] for r in rows], dtype=np.float64),
                "is_follow": np.asarray([r[2] for r in rows], dtype=np.float64),
                "is_comment": np.asarray([r[3] for r in rows], dtype=np.float64),
                "is_forward": np.asarray([r[4] for r in rows], dtype=np.float64),
                "play_time": np.asarray([r[5] for r in rows], dtype=np.float64),
            }
        # Fallback for 3-tuple legacy test overrides
        return {
            "is_click": np.asarray([r[0] for r in rows], dtype=np.float64),
            "is_like": np.asarray([r[1] for r in rows], dtype=np.float64),
            "is_follow": np.zeros(len(rows), dtype=np.float64),
            "is_comment": np.zeros(len(rows), dtype=np.float64),
            "is_forward": np.zeros(len(rows), dtype=np.float64),
            "play_time": np.asarray([r[2] for r in rows], dtype=np.float64),
        }
    try:
        return _cached_aux_columns(str(_resolve_data_dir(spec)))
    except (OSError, KeyError) as exc:
        raise FeatureDataUnavailable(f"Could not load auxiliary train columns: {exc}") from exc


def _scaled_play_time(values: np.ndarray) -> np.ndarray:
    """``log1p`` then min-max on train. play_time_ms is heavy-tailed and censored at video length."""
    compressed = np.log1p(np.clip(values, 0.0, None))
    low = float(compressed.min()) if len(compressed) else 0.0
    high = float(compressed.max()) if len(compressed) else 0.0
    if high <= low:
        return np.zeros_like(compressed)
    return (compressed - low) / (high - low)


def build_aux_labels(rows, spec: dict) -> np.ndarray:
    """Auxiliary targets for a multi-task loss, read from train dates only.

    Returns ``(len(rows), t)`` float32 in ``[0, 1]``, one column per enabled head in ``AUX_HEADS``
    order. ``is_click`` and ``is_like`` are binary; ``play_time`` is ``log1p`` compressed and
    min-max scaled on train.

    **Train-only by construction.** A loss touches train rows only, so no valid/test path exists
    and none may be added -- this raises for any other split rather than silently returning
    something a scorer could misuse.
    """
    split = str(spec.get("split", "train"))
    if split != "train":
        raise ValueError(
            f"build_aux_labels is train-only; auxiliary targets have no {split!r} path."
        )
    heads = enabled_aux_heads(spec)
    if not heads:
        raise ValueError("At least one auxiliary head must be enabled.")

    expected = len(rows)
    columns = _aux_columns(spec)
    available = len(columns["is_click"])
    if available != expected:
        raise ValueError(
            f"Auxiliary train columns have {available} rows but {expected} were passed; "
            "row order and length must match the kit's."
        )

    targets = np.empty((expected, len(heads)), dtype=np.float32)
    for index, head in enumerate(heads):
        values = columns[head]
        targets[:, index] = (
            _scaled_play_time(values) if head == "play_time" else values
        ).astype(np.float32)
    return targets
