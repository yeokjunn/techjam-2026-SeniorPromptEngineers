"""The negative-result artifact: what this run measured *flat*, and against what.

A search that reports only its best number is unfalsifiable. This module renders
``<run_dir>/falsified.md`` from the run's own nodes: which families and which
parameter axes produced no separation, the noise the separation would have had to
clear (the shipped sigma and margin constants, not fresh guesses), and a static
table of published effect sizes so "flat" can be read against what a real gain in
this literature actually looks like.

Report-side only. Nothing here is read by a decision; ``research_controller``
calls it inside the same containment as ``render_reports``, so a fault costs this
file and nothing else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .policy import DEFAULT_PROMOTION_MARGIN, MEASURED_SEED_SIGMA
from .types import RunState


#: Where the fully cited version of the context table lives (checked in, static).
REFERENCE_EFFECT_SIZES_PATH = "research/reference_effect_sizes.md"

#: A compact restatement of that file, inlined so the artifact is readable on its
#: own. Every row is transcribed from the project's external-benchmark survey
#: (`.superpowers/...`, which is git-excluded); the citation a reader can actually
#: follow is the reference file above, whose Sources section carries the URLs.
PUBLISHED_EFFECT_SIZES: tuple[tuple[str, str, str], ...] = (
    ("Wide&Deep over deep BaseModel", "Alibaba production (DIN, KDD'18)", "+0.0007 GAUC"),
    ("DeepFM over deep BaseModel", "Alibaba production (DIN, KDD'18)", "+0.0023 GAUC"),
    ("DIN over BaseModel (history attention)", "Alibaba production (DIN, KDD'18)", "+0.0059 GAUC"),
    ("DIEN over DIN", "KuaiRand-1K (VQL, 2025)", "+0.0005 AUC"),
    ("BCE -> BCE + pairwise ranking loss", "industrial ads (Lin et al., KDD'24)", "+0.00077 AUC"),
    ("SMES-L (510M params) over Rankmixer", "KuaiRand-1K (SMES, 2026)", "+0.0029 GAUC"),
    ("AutoIFS over HiNet", "KuaiRand-Pure, long-view (2025)", "+0.0024 AUC"),
    ("The field's own significance bar", "Guo et al. IJCAI'17, quoted by AutoIFS and VQL", "+0.001"),
)

#: E[max] of N draws from the null (every candidate truly equal to the incumbent)
#: at the measured sigma -- the number that says how much "best of N" is worth
#: before any real effect exists. Expected value of the maximum of N standard
#: normals, times sigma. Source: investigation/2-noise-floor.md section 4.
_EXPECTED_MAX_UNDER_NULL: tuple[tuple[int, float], ...] = (
    (3, 0.00077),
    (5, 0.00106),
    (10, 0.00140),
    (20, 0.00170),
    (50, 0.00205),
)


def _scored(state: RunState) -> list[Any]:
    return [
        node
        for node in state.nodes
        if node.status == "success" and node.metrics and "primary" in node.metrics
    ]


def _primary(node: Any) -> float:
    return float(node.metrics["primary"])


def _family_bands(state: RunState) -> dict[str, dict[str, Any]]:
    """Per family: how many scored nodes, and the band of primaries they span."""
    bands: dict[str, dict[str, Any]] = {}
    for node in _scored(state):
        band = bands.setdefault(
            node.family, {"n": 0, "primaries": [], "experiment_ids": []}
        )
        band["n"] += 1
        band["primaries"].append(_primary(node))
        band["experiment_ids"].append(node.experiment_id)
    for band in bands.values():
        band["min"] = min(band["primaries"])
        band["max"] = max(band["primaries"])
    return bands


def _sorted_values(by_value: dict[Any, list[float]]) -> str:
    """The values an axis took, in their own order rather than in `repr` order.

    `_grid_property` guarantees every grid key is type-homogeneous, so the plain
    sort is the correct one in every shipping case; the `repr` fallback only
    catches a heterogeneous key no family registers today.
    """
    try:
        ordered = sorted(by_value)
    except TypeError:
        ordered = sorted(by_value, key=repr)
    return ", ".join(str(value) for value in ordered)


def _axis_rows(state: RunState, sigma: float) -> list[tuple[str, str, str, float, bool]]:
    """Per (family, parameter) actually varied: the values tried and the spread.

    An axis counts as measured *flat* when it took at least two distinct values
    and the primaries across them span less than one sigma -- i.e. moving the knob
    did not move the metric further than repeating the same configuration does.

    Only the flat verdict is reported, and deliberately: the grouping is one key
    at a time while every *other* knob varies freely inside the group, so a large
    spread cannot be attributed to this axis -- but a small one still bounds it,
    whatever else moved. `seed` is skipped outright; it is the noise axis section
    3 accounts for with sigma, and reporting it as a finding would contradict that
    section in the same document.
    """
    grouped: dict[tuple[str, str], dict[Any, list[float]]] = {}
    for node in _scored(state):
        for key, value in (node.parameters or {}).items():
            if key == "seed":  # the noise axis, accounted for by sigma in section 3
                continue
            try:
                hash(value)
            except TypeError:  # a list-valued knob: not an axis this can group on
                continue
            grouped.setdefault((node.family, key), {}).setdefault(value, []).append(
                _primary(node)
            )
    rows: list[tuple[str, str, str, float, bool]] = []
    for (family, key), by_value in sorted(grouped.items(), key=lambda item: item[0]):
        if len(by_value) < 2:
            continue
        primaries = [score for scores in by_value.values() for score in scores]
        spread = max(primaries) - min(primaries)
        rows.append((family, key, _sorted_values(by_value), spread, spread < sigma))
    return rows


def flat_families(
    state: RunState, *, promotion_margin: float = DEFAULT_PROMOTION_MARGIN
) -> list[str]:
    """Families every one of whose scored candidates stayed inside the margin.

    Reused for the campaign digest's ``falsified:`` line, so the two artifacts
    cannot disagree about what this run ruled out.
    """
    baseline = float(state.baseline_primary)
    return sorted(
        family
        for family, band in _family_bands(state).items()
        if band["max"] - baseline < promotion_margin
    )


def render_falsified(
    state: RunState,
    *,
    promotion_margin: float = DEFAULT_PROMOTION_MARGIN,
    epsilon: float | None = None,
    sigma: float = MEASURED_SEED_SIGMA,
) -> str:
    """The whole artifact as Markdown."""
    baseline = float(state.baseline_primary)
    bands = _family_bands(state)
    flat = set(flat_families(state, promotion_margin=promotion_margin))
    lines: list[str] = [
        "# What this run falsified",
        "",
        f"Run `{state.run_id}`. Baseline primary **{baseline:.6f}**.",
        "",
        "A result is only evidence if it could have come out the other way. This file is the",
        "other way: the directions this run measured and found *flat*, stated with the noise",
        "they would have had to clear and the effect sizes the published literature reports",
        "for changes of the same kind.",
        "",
        "## 1. Families measured flat",
        "",
    ]

    if not bands:
        lines += [
            "No candidate was scored in this run, so nothing was measured and nothing is",
            "falsified. An empty section here is not a null result.",
            "",
        ]
    else:
        lines += [
            "| Family | Scored | Score band (primary) | Best Δ vs baseline | Verdict |",
            "|---|---|---|---|---|",
        ]
        for family in sorted(bands):
            band = bands[family]
            best_delta = band["max"] - baseline
            verdict = (
                f"**flat** (< margin {promotion_margin:.4f})"
                if family in flat
                else f"cleared margin {promotion_margin:.4f}"
            )
            lines.append(
                f"| `{family}` | {band['n']} | {band['min']:.6f} – {band['max']:.6f} "
                f"| {best_delta:+.6f} | {verdict} |"
            )
        lines.append("")
        if flat:
            lines += [
                "Falsified this run: "
                + ", ".join(f"`{family}`" for family in sorted(flat))
                + ". Every scored candidate in each of these families stayed inside the",
                f"promotion margin ({promotion_margin:.4f}), so this run has no evidence that any",
                "of them beats the incumbent.",
                "",
            ]
        else:
            lines += [
                "No family was measured flat: at least one candidate in each cleared the",
                f"promotion margin ({promotion_margin:.4f}).",
                "",
            ]

    axis_rows = _axis_rows(state, sigma)
    lines += ["## 2. Parameter axes measured flat", ""]
    if not axis_rows:
        lines += [
            "No parameter took two distinct values across the scored candidates, so no axis",
            "was measured. Nothing is claimed about any knob this run did not vary.",
            "",
        ]
    else:
        lines += [
            "An axis is flat when the primaries across the values tried span less than one",
            f"sigma ({sigma:.5f}) — moving the knob moved the metric less than re-running the",
            "same configuration does.",
            "",
            "| Family | Axis | Values tried | Primary spread | Verdict |",
            "|---|---|---|---|---|",
        ]
        for family, key, values, spread, is_flat in axis_rows:
            # Never "separated": with every other knob free inside the group, a
            # large spread cannot be attributed to this axis. A small one can.
            verdict = "**flat** (< 1σ)" if is_flat else "not flat (≥ 1σ)"
            lines.append(
                f"| `{family}` | `{key}` | {values} | {spread:.6f} | {verdict} |"
            )
        lines.append("")

    lines += [
        "## 3. The noise this run had to clear",
        "",
        f"- **Measured seed sigma: {sigma:.5f}** (`policy.MEASURED_SEED_SIGMA`). Sample sigma",
        "  over three runs of byte-identical candidate code with identical parameters and the",
        "  same seed. The spread comes from per-process string-hash ordering inside",
        "  `sampling.eligible_user_indices`, not from the `seed` parameter, so it is not",
        "  removable by fixing a seed.",
        f"- **Promotion margin: {promotion_margin:.4f}** (`policy.DEFAULT_PROMOTION_MARGIN`,",
        f"  ≈{promotion_margin / sigma:.1f}σ). A candidate must beat the incumbent by this much",
        "  before it may displace it; a bare `>` promotes noise.",
    ]
    if epsilon is not None:
        lines.append(
            f"- **Convergence epsilon: {float(epsilon):.4f}** (≈{float(epsilon) / sigma:.1f}σ), the"
        )
        lines.append("  stagnation and official-convergence threshold.")
    lines += [
        "- **Best-of-N inflation.** Under the null — every candidate truly equal to the",
        "  incumbent — the expected maximum of N draws at this sigma is:",
        "",
        "| Scored candidates N | E[best Δ] under the null |",
        "|---|---|",
    ]
    for count, expected in _EXPECTED_MAX_UNDER_NULL:
        lines.append(f"| {count} | +{expected:.5f} |")
    scored_count = len(_scored(state))
    lines += [
        "",
        f"This run scored **{scored_count}** candidate(s). Any headline delta smaller than the",
        "row above it is indistinguishable from taking the maximum of that many coin flips.",
        "",
        "## 4. How big a real gain is, in this literature",
        "",
        "Static context, so a flat result can be read against something. Full citations with",
        f"URLs: [`{REFERENCE_EFFECT_SIZES_PATH}`](../../{REFERENCE_EFFECT_SIZES_PATH}).",
        "",
        "| Change | Setting | Reported effect |",
        "|---|---|---|",
    ]
    for change, setting, effect in PUBLISHED_EFFECT_SIZES:
        lines.append(f"| {change} | {setting} | {effect} |")
    lines += [
        "",
        "Read the last row first: the field's own significance bar is **+0.001 absolute**, and",
        f"this run's seed sigma is {sigma:.5f}. A gain that would be publishable here is roughly",
        "the size of this harness's own measurement noise — which is why a flat family is the",
        "expected outcome of most iterations, and why saying so is a result rather than an",
        "excuse.",
        "",
    ]
    return "\n".join(lines) + "\n"


def write_falsified(
    run_dir: Path,
    state: RunState,
    *,
    promotion_margin: float = DEFAULT_PROMOTION_MARGIN,
    epsilon: float | None = None,
    sigma: float = MEASURED_SEED_SIGMA,
) -> Path:
    """Render the artifact into ``<run_dir>/falsified.md`` and return its path."""
    path = Path(run_dir) / "falsified.md"
    path.write_text(
        render_falsified(
            state, promotion_margin=promotion_margin, epsilon=epsilon, sigma=sigma
        ),
        encoding="utf-8",
    )
    return path
