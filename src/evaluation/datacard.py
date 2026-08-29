"""Data card renderer (review item I15)."""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

_TRAIN_START = 20220408
_TRAIN_END = 20220421
_VALID_START = 20220422
_VALID_END = 20220428

_LOG_NAMES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)
_FEAT_NAMES = (
    "video_features_basic_pure.csv",
    "user_features_pure.csv",
    "video_features_statistic_pure.csv",
)
_REQUIRED = _LOG_NAMES + _FEAT_NAMES

_LABELS = (
    "long_view", "is_click", "is_like", "is_follow",
    "is_comment", "is_forward", "is_hate", "is_profile_enter",
)


def _fmt(n: float) -> str:
    if abs(n) >= 1e9:
        return f"{n / 1e9:.2f} B"
    if abs(n) >= 1e6:
        return f"{n / 1e6:.2f} M"
    return f"{n:,.0f}"


def render_data_card(data_dir: Path) -> str:
    """Render a deterministic Markdown data card for the Researcher prompt."""
    data_dir = Path(data_dir)
    for name in _REQUIRED:
        if not (data_dir / name).is_file():
            return ""

    # ── Scan interaction logs ───────────────────────────────────
    sp_rows = {"train": 0, "valid": 0, "test": 0}
    sp_users: dict[str, set[str]] = {"train": set(), "valid": set()}
    sp_vids: dict[str, set[str]] = {"train": set(), "valid": set()}
    pos = {s: {lb: 0 for lb in _LABELS} for s in ("train", "valid")}
    tab_d: dict[str, list[int]] = {}
    u_counts: dict[str, int] = defaultdict(int)
    zdur_train = 0
    zdur_vids: set[str] = set()
    rand_vals: set[str] = set()
    min_date = 99999999

    dup_set: set[bytes] = set()
    dup_n = 0

    for li, ln in enumerate(_LOG_NAMES):
        with (data_dir / ln).open(encoding="utf-8", newline="") as fh:
            hl = next(fh)
            hdr = next(csv.reader([hl]))
            ci = {n: i for i, n in enumerate(hdr)}

            for raw in fh:
                if li == 0:
                    dup_n += 1
                    dup_set.add(
                        hashlib.blake2b(raw.encode(), digest_size=16).digest()
                    )

                r = raw.rstrip("\r\n").split(",")
                d = int(r[ci["date"]])
                if d < min_date:
                    min_date = d

                if _TRAIN_START <= d <= _TRAIN_END:
                    sp = "train"
                elif _VALID_START <= d <= _VALID_END:
                    sp = "valid"
                else:
                    sp_rows["test"] += 1
                    continue

                sp_rows[sp] += 1
                uid, vid = r[ci["user_id"]], r[ci["video_id"]]
                sp_users[sp].add(uid)
                sp_vids[sp].add(vid)
                u_counts[uid] += 1

                for lb in _LABELS:
                    if r[ci[lb]] != "0":
                        pos[sp][lb] += 1

                if sp == "train":
                    tb = r[ci["tab"]]
                    if tb not in tab_d:
                        tab_d[tb] = [0, 0, 0]
                    e = tab_d[tb]
                    e[0] += 1
                    if r[ci["is_click"]] != "0":
                        e[1] += 1
                    if r[ci["long_view"]] != "0":
                        e[2] += 1

                if float(r[ci["duration_ms"]]) == 0:
                    if sp == "train":
                        zdur_train += 1
                    zdur_vids.add(vid)

                rand_vals.add(r[ci["is_rand"]])

    dups = dup_n - len(dup_set)

    rand_path = data_dir / "log_random_4_22_to_5_08_pure.csv"
    rand_rows = 0
    if rand_path.is_file():
        with rand_path.open(encoding="utf-8", newline="") as fh:
            hl = next(fh)
            hdr = next(csv.reader([hl]))
            ci_r = {n: i for i, n in enumerate(hdr)}
            for raw in fh:
                rand_rows += 1
                r = raw.rstrip("\r\n").split(",")
                if float(r[ci_r["duration_ms"]]) == 0:
                    zdur_vids.add(r[ci_r["video_id"]])
    total_logged = sum(sp_rows.values()) + rand_rows

    # ── Scan feature tables ─────────────────────────────────────
    all_vids = sp_vids["train"] | sp_vids["valid"]
    all_uids = set(u_counts)

    vf_ids: set[str] = set()
    unk_vt = 0
    unk_ut = 0
    vis_vals: set[str] = set()
    with (data_dir / _FEAT_NAMES[0]).open(encoding="utf-8", newline="") as fh:
        rdr = csv.reader(fh)
        hdr = next(rdr)
        ci = {n: i for i, n in enumerate(hdr)}
        for r in rdr:
            vf_ids.add(r[ci["video_id"]])
            if r[ci["video_type"]] == "UNKNOWN":
                unk_vt += 1
            if r[ci["upload_type"]] == "UNKNOWN":
                unk_ut += 1
            vis_vals.add(r[ci["visible_status"]])

    uf_ids: set[str] = set()
    uf_n = 0
    ls_neg = 0
    unk_ad = 0
    la_vals: set[str] = set()
    with (data_dir / _FEAT_NAMES[1]).open(encoding="utf-8", newline="") as fh:
        rdr = csv.reader(fh)
        hdr = next(rdr)
        ci = {n: i for i, n in enumerate(hdr)}
        for r in rdr:
            uf_n += 1
            uf_ids.add(r[ci["user_id"]])
            if r[ci["is_live_streamer"]] == "-124":
                ls_neg += 1
            if r[ci["user_active_degree"]] == "UNKNOWN":
                unk_ad += 1
            la_vals.add(r[ci["is_lowactive_period"]])

    sf_ids: set[str] = set()
    implied_shows = 0.0
    cmin = float("inf")
    cmax = float("-inf")
    with (data_dir / _FEAT_NAMES[2]).open(encoding="utf-8", newline="") as fh:
        rdr = csv.reader(fh)
        hdr = next(rdr)
        ci = {n: i for i, n in enumerate(hdr)}
        for r in rdr:
            sf_ids.add(r[ci["video_id"]])
            ct = float(r[ci["counts"]])
            implied_shows += float(r[ci["show_cnt"]]) * ct
            if ct < cmin:
                cmin = ct
            if ct > cmax:
                cmax = ct

    v_cov = len(all_vids & vf_ids) / len(all_vids) * 100 if all_vids else 0
    u_cov = len(all_uids & uf_ids) / len(all_uids) * 100 if all_uids else 0

    # ── Derived stats ───────────────────────────────────────────
    ua = np.fromiter(u_counts.values(), dtype=np.int64)
    pcts = (25, 50, 75, 90, 95, 99)
    pv = np.percentile(ua, pcts) if len(ua) else np.zeros(len(pcts))

    cc = []
    if len(la_vals) == 1:
        cc.append("is_lowactive_period")
    if len(vis_vals) == 1:
        cc.append("visible_status")
    if len(rand_vals) == 1:
        cc.append("is_rand")

    # ── Render card ─────────────────────────────────────────────
    tr = sp_rows["train"]
    vl = sp_rows["valid"]
    L: list[str] = []
    a = L.append

    a("# Dataset Profile")
    a("")
    a("## Splits")
    a("")
    a("| Split | Rows | Users | Videos |")
    a("|---|---|---|---|")
    a(f"| train | {tr:,} | {len(sp_users['train']):,} | {len(sp_vids['train']):,} |")
    a(f"| valid | {vl:,} | {len(sp_users['valid']):,} | {len(sp_vids['valid']):,} |")
    a(f"| test | {sp_rows['test']:,} | — | — |")
    a("")
    a("## Label Rates")
    a("")
    a("| Label | Train Rate | Valid Rate |")
    a("|---|---|---|")
    for lb in _LABELS:
        t_r = pos["train"][lb] / tr * 100 if tr else 0
        v_r = pos["valid"][lb] / vl * 100 if vl else 0
        a(f"| {lb} | {t_r:.4f} % | {v_r:.4f} % |")
    a("")
    a("## Tab Breakdown (train)")
    a("")
    a("| Tab | Rows | Share | Click Rate | Long View Rate |")
    a("|---|---|---|---|---|")
    for tb in sorted(tab_d, key=lambda t: tab_d[t][0], reverse=True):
        e = tab_d[tb]
        sh = e[0] / tr * 100 if tr else 0
        cr = e[1] / e[0] * 100 if e[0] else 0
        lv = e[2] / e[0] * 100 if e[0] else 0
        a(f"| {tb} | {e[0]:,} | {sh:.2f} % | {cr:.2f} % | {lv:.2f} % |")
    a("")
    a("## Rows per User (train + valid)")
    a("")
    a("| Percentile | Count |")
    a("|---|---|")
    for p, v in zip(pcts, pv):
        a(f"| p{p} | {int(v)} |")
    a(f"| max | {int(ua.max()) if len(ua) else 0} |")
    a("")
    a("## Data Quality")
    a("")
    a(f"- {len(zdur_vids):,} videos with zero duration ({zdur_train:,} train rows affected)")
    a(f"- {dups:,} exact duplicate rows in the training-period log")
    a(f"- Sentinel: is_live_streamer = -124 on {ls_neg:,} of {uf_n:,} user rows")
    a(f"- UNKNOWN values: user_active_degree ({unk_ad}), video_type ({unk_vt}), upload_type ({unk_ut})")
    if cc:
        a(f"- Constant columns: {', '.join(cc)}")
    a("- hourmin encodes hour times 100 in UTC+8")
    if min_date > _TRAIN_START:
        a(f"- The training-period log begins on {min_date}, not {_TRAIN_START}")
    a("")
    a("## Feature Coverage")
    a("")
    a(f"- Video features: {v_cov:.3f} % of train+valid video IDs found in the feature table")
    a(f"- User features: {u_cov:.3f} % of train+valid user IDs found in the profile table")
    a("")
    a("## Leakage Flag")
    a("")
    ratio = implied_shows / total_logged if total_logged else 0
    a("The video statistics table is a period aggregate over full-platform traffic.")
    a(f"Implied total shows: {_fmt(implied_shows)} vs {_fmt(float(total_logged))} logged")
    a(f"exposures (ratio approximately {ratio:,.0f} to one). Observation periods per video")
    a(f"range {int(cmin)} to {int(cmax)}, exceeding the interaction log span.")
    a("Legal for use, but any feature derived from it must be caveated as leaky.")
    a("")
    a("## Metric Conventions")
    a("")
    a("- Task: within-user ranking (each user's impressions ranked against each other)")
    a("- nDCG@5: sorts by score with a stable sort (ties fall back to row order)")
    a("- AUC: averages ranks over ties (Mann-Whitney U)")
    a("- GAUC: per-user AUC, only users with 0 < positives < impressions,")
    a("  weighted by positive count")
    a("- Zero-positive users: nDCG recorded as 0.0 and included in the average")
    a("- Primary score = mean(GAUC, nDCG@5)")
    a("")
    a("## Measured Dead Ends")
    a("")
    a("- Adding all 13 static feature fields: primary 0.5940 vs 0.5950 for the default 5")
    a("- Embedding dimension k = 8 / 16 / 32: primary 0.5895 / 0.5902 / 0.5887 (flat)")
    a("- The bottleneck is not features or capacity")
    a("- First-order user-only terms contribute exactly 0 to within-user ranking")
    a("  (any term constant within a user does not change intra-group ordering)")

    return "\n".join(L) + "\n"


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.evaluation.datacard <data_dir>", file=sys.stderr)
        sys.exit(1)
    card = render_data_card(Path(sys.argv[1]))
    if card:
        print(card, end="")
    else:
        print("(required files missing)", file=sys.stderr)
