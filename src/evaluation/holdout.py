"""Split validation into a selection half and a reporting half.

Why this exists. Every candidate early-stops on validation and then reports that same validation
number, and the run reports the best of ~14 candidates -- roughly 112 selections against the same
124,909 rows. Measured over 21 candidates, the reported score sits +0.0025 above the median of
the epoch curve it was drawn from, while the claimed improvement over the baseline is +0.0024.
The improvement and the selection artefact are the same size, so on the current numbers a real
gain cannot be distinguished from the top of the noise. It is also why a 0.6039 candidate
replicated at 0.6025 and 0.6024.

The split is by **user**, never by row: GAUC is computed per user and weighted by that user's
positive count, and nDCG@5 ranks within a user's impressions, so splitting rows would tear a
user's list in half and score fragments of it on both sides.

The split is deterministic given the user id -- a hash, not an RNG -- so it is stable across
processes, resumes and machines without threading a seed through the worker, and a given user
lands on the same side in every run.

Nothing here changes what `metrics["primary"]` means: the trusted worker still reports full
validation, which is what the official baseline is measured on and what the judged delta uses.
These halves are reported alongside it so the selection gap is visible rather than assumed.
"""

from __future__ import annotations

import hashlib

import numpy as np

#: Fraction of users held out for reporting. Half keeps both estimates equally noisy; a smaller
#: reporting half would make the honest number noisier than the biased one it is meant to check.
REPORT_SHARE = 0.5


def _bucket(user: str) -> float:
    """Stable [0, 1) position for a user, independent of process hash randomisation.

    ``hash()`` is salted per process in Python 3, so it cannot be used: the same user would
    land on different sides in the worker and in any later analysis.
    """
    digest = hashlib.blake2b(str(user).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def selection_mask(users) -> np.ndarray:
    """Boolean mask over rows: True for the selection half (early stopping, tuning).

    The reporting half is its complement, so callers never have to keep two masks in sync.
    """
    return np.fromiter(
        (_bucket(user) >= REPORT_SHARE for user in users), dtype=bool, count=len(users)
    )


def split_users(users) -> tuple[np.ndarray, np.ndarray]:
    """``(selection_mask, reporting_mask)`` over the rows of ``users``."""
    selection = selection_mask(users)
    return selection, ~selection
