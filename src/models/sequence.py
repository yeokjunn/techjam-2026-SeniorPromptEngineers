"""Trusted, leakage-safe user-behaviour sequence builder (DIN/DIEN input).

The kit freezes the model at five id fields; user-behaviour *sequences* are the
committee's #2 unexplored direction ("each user has hundreds-to-thousands of
interactions in train; interest modelling of the DIN/SIM variety is a
completely blank direction"). ``features.py`` already builds leakage-safe
*aggregated* history (rate tables, recency); this module builds the *item
sequence* a target-attention model needs, in trusted code, so a generated
candidate never touches a raw log and the leakage rules are enforced here.

**Leakage invariant.** The relevance label (element 6 of a kit row) is never
read by this builder — sequences use only ``date`` (element 0), ``user_id``
(element 1) and ``video_id`` (element 2), from train rows. The candidate row's
own label is irrelevant here and is never dereferenced.

**Time-respecting rule (mirrors ``features.py`` prior_days).**
  - A *train* target row on day ``d`` sees only that user's interactions on days
    *strictly* earlier than ``d`` (same-day excluded: two rows on the same day
    cannot see each other, matching ``features.py``'s day-by-day fold).
  - A *valid*/*test* target row sees *all* of that user's train interactions
    (days ``TRAIN_START..TRAIN_END``). It never sees valid/test-period
    interactions or labels.
Because packed dates are ``YYYYMMDD`` integers, numeric comparison *is*
chronological, so no date-ordinal conversion is needed.

**Vocabulary.** ``video_id -> int`` is built from the *train* split only, with a
reserved UNK slot at the end (the kit's trick, ``data.py:44-50``). Items absent
from the train vocab (unseen in training) map to UNK, so a valid/test candidate
that never appeared in train still gets a defined embedding.

**Layout.** Per target row, the most recent ``seq_len`` history items are kept
(chronological order, oldest first within the unmasked prefix) and the rest is
zero-padded at the end; ``history_mask`` is ``1.0`` over real history and
``0.0`` over padding. Output item indices are raw ``[0, vocab_size)`` (the
sequence model owns its own embedding table, separate from the FM fields) so no
field-offset bookkeeping is needed.

This is the input producer only. The attention / pooling math lives in the
candidate model (PyTorch, outside the numpy sandbox) and in a pure-numpy
attention helper added later; neither belongs in trusted build code.
"""

from __future__ import annotations

import csv
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

# Kit row layout: (date, user_id, video_id, author_id, tab, duration_ms, long_view)
_DATE, _USER, _VIDEO = 0, 1, 2  # the only elements this builder reads

DEFAULT_SEQ_LEN = 50
SPLITS = ("train", "valid", "test")


class SequenceDataUnavailable(RuntimeError):
    """The trusted rows backing a split could not be loaded; caller may degrade."""


@dataclass(frozen=True)
class UserSequences:
    """The padded history + mask + candidate indices for one target split.

    - ``history_items``: ``(N, L)`` int32 video-id indices, UNK-padded at the end.
    - ``history_mask``: ``(N, L)`` float32, ``1.0`` over real history, ``0.0`` over padding.
    - ``candidate_items``: ``(N,)`` int32 — the target video's vocab index (the
      attention query). Never UNK-padded; an unseen candidate maps to the UNK slot.
    - ``field_dimension``: the item-vocabulary size (including the UNK slot) for
      the model's history/candidate embedding tables.
    """

    history_items: np.ndarray
    history_mask: np.ndarray
    candidate_items: np.ndarray
    field_dimension: int


def _resolve_data_dir(spec: dict[str, Any]) -> Path:
    raw = spec.get("data_dir") or os.environ.get("KUAIRAND_DATA_DIR")
    return Path(raw) if raw else REPO_ROOT / "data" / "KuaiRand-Pure" / "data"


@lru_cache(maxsize=1)
def _cached_train_valid(data_dir: str) -> dict[str, list[tuple]]:
    return load_train_valid(Path(data_dir))


@lru_cache(maxsize=1)
def _cached_test_rows(data_dir: str, expected_rows: int) -> tuple[tuple, ...]:
    return load_test_meta(Path(data_dir), expected_rows=expected_rows).rows


def _history_rows(spec: dict[str, Any]) -> list:
    """Return the train rows used as the *history source*.

    Honours the documented ``history_rows`` override (synthetic rows for tests,
    so unit tests need no dataset). Unlike ``features._split_rows`` the *target*
    rows are the caller's own kit rows (they carry the user/video/date needed to
    build a sequence), so only the train history source comes from here.
    """
    override = spec.get("history_rows")
    if override is not None:
        if "train" not in override:
            raise ValueError("spec['history_rows'] must provide a 'train' entry.")
        return list(override["train"])

    data_dir = _resolve_data_dir(spec)
    try:
        splits = _cached_train_valid(str(data_dir))
    except (OSError, KeyError) as exc:
        raise SequenceDataUnavailable(
            f"Could not load trusted train rows: {exc}"
        ) from exc
    return list(splits["train"])


def _assert_aligned(spec: dict[str, Any], split: str, expected: int) -> None:
    """Defensive: the caller's target rows must match the trusted split length.

    Skipped under the synthetic override (tests may build sequences for a
    subset of rows). On the real path this guards row-order/length drift.
    """
    if spec.get("history_rows") is not None:
        return
    data_dir = _resolve_data_dir(spec)
    try:
        if split == "test":
            trusted = _cached_test_rows(str(data_dir), expected)
            if len(trusted) != expected:
                raise ValueError(
                    f"Test split has {len(trusted)} trusted rows but {expected} were passed."
                )
        else:
            splits = _cached_train_valid(str(data_dir))
            if len(splits[split]) != expected:
                raise ValueError(
                    f"Split {split!r} has {len(splits[split])} trusted rows but "
                    f"{expected} were passed; row order and length must match the kit's."
                )
    except (OSError, KeyError) as exc:
        raise SequenceDataUnavailable(
            f"Could not load trusted rows for split {split!r}: {exc}"
        ) from exc


def _build_video_vocab(train_rows: list) -> tuple[dict[str, int], int]:
    """``video_id -> int`` from train only, with a trailing UNK slot."""
    vocab: dict[str, int] = {}
    for row in train_rows:
        vid = row[_VIDEO]
        if vid not in vocab:
            vocab[vid] = len(vocab)
    unk = len(vocab)
    return vocab, unk + 1  # field_dimension = unique train videos + UNK


def _build_user_history(train_rows: list) -> dict[str, list[tuple[int, str]]]:
    """Per user, the chronologically ordered list of ``(date, video_id)`` train rows.

    Packed dates are YYYYMMDD ints and compare chronologically, so a plain sort
    on the packed int gives chronological order. Sorted once here so each target
    row can slice its prefix cheaply.
    """
    by_user: dict[str, list[tuple[int, str]]] = {}
    for row in train_rows:
        by_user.setdefault(row[_USER], []).append((row[_DATE], row[_VIDEO]))
    for items in by_user.values():
        items.sort(key=lambda pair: pair[0])
    return by_user


def _history_for_row(
    user: str,
    date: int,
    split: str,
    user_history: dict[str, list[tuple[int, str]]],
    seq_len: int,
) -> tuple[list[str], int]:
    """Return ``(history_video_ids, n_real)`` — the most recent ``seq_len`` prior items.

    Train target rows use strictly-earlier-days history; valid/test use all train.
    The slice keeps the last ``seq_len`` items (most recent) when history is long.
    """
    items = user_history.get(user)
    if not items:
        return [], 0
    if split == "train":
        # Strictly earlier DAYS: same-day rows cannot see each other.
        lo, hi = 0, len(items)
        # binary search the first index with date >= `date`
        while lo < hi:
            mid = (lo + hi) // 2
            if items[mid][0] < date:
                lo = mid + 1
            else:
                hi = mid
        prior = items[:lo]
    else:
        prior = items  # all train (valid/test target -> train-only history)
    if len(prior) > seq_len:
        prior = prior[-seq_len:]
    return [vid for _, vid in prior], len(prior)


def build_user_sequences(target_rows, spec: dict[str, Any]) -> UserSequences:
    """Build item-id history sequences for one split.

    ``target_rows`` is the caller's per-split row list (only its length is
    trusted against the loader; row order matches the kit's). ``spec`` keys:
      - ``split``: ``"train"`` | ``"valid"`` | ``"test"`` (default ``"train"``)
      - ``seq_len``: max history length ``L`` (default ``50``)
      - ``history_rows``: optional synthetic override ``{"train":[...], split:[...]}``
      - ``video_vocab``: optional override ``video_id -> int`` (tests; otherwise
        built from the train entry)
      - ``data_dir``: optional data dir
    """
    split = str(spec.get("split", "train"))
    if split not in SPLITS:
        raise ValueError(f"spec['split'] must be one of {SPLITS}; got {split!r}")
    seq_len = int(spec.get("seq_len", DEFAULT_SEQ_LEN))
    if seq_len < 1:
        raise ValueError("spec['seq_len'] must be positive.")

    expected = len(target_rows)
    _assert_aligned(spec, split, expected)
    train_rows = _history_rows(spec)

    override_vocab = spec.get("video_vocab")
    if override_vocab is not None:
        vocab: dict[str, int] = dict(override_vocab)
        field_dimension = len(vocab) + 1  # +1 for UNK
    else:
        vocab, field_dimension = _build_video_vocab(train_rows)
    unk = field_dimension - 1

    user_history = _build_user_history(train_rows)

    history_items = np.zeros((expected, seq_len), dtype=np.int32)
    history_mask = np.zeros((expected, seq_len), dtype=np.float32)
    candidate_items = np.full(expected, unk, dtype=np.int32)

    for i, row in enumerate(target_rows):
        # Candidate query: the target video's vocab index (UNK if unseen in train).
        candidate_items[i] = vocab.get(row[_VIDEO], unk)
        hist, n_real = _history_for_row(row[_USER], row[_DATE], split, user_history, seq_len)
        if n_real:
            history_items[i, :n_real] = [vocab.get(vid, unk) for vid in hist]
            history_mask[i, :n_real] = 1.0

    return UserSequences(
        history_items=history_items,
        history_mask=history_mask,
        candidate_items=candidate_items,
        field_dimension=field_dimension,
    )
