from __future__ import annotations

import html
import json
import math
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Streamlit adds the entry-point directory to sys.path. When this nested file is
# launched directly, make the repository package root explicit before importing
# project modules. This also makes absolute-path launches independent of cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import altair as alt
import pandas as pd
import streamlit as st

from src.agent.activity import STAGE_ORDER
from src.ui.loaders import (
    activity_age_seconds,
    discover_runs,
    load_call_prompt,
    load_candidate_files,
    load_dashboard_config,
    load_patch_text,
    load_role_passes,
    load_run_snapshot,
    validate_submission,
)
from src.ui.models import (
    DebuggerEvent,
    EDAArtifact,
    LLMCall,
    RolePass,
    RunSnapshot,
    StageTransition,
)


CONFIG_PATH = REPO_ROOT / "configs" / "ui.json"

STAGE_LABELS = {
    "initializing": "Initialize",
    "eda_researcher": "EDA plan",
    "eda_builder": "EDA report",
    "researcher": "Research",
    "critic_preflight": "Preflight",
    "builder": "Build",
    "safety_tests": "Safety + tests",
    "debugger": "Repair",
    "training_evaluation": "Train + validate",
    "critic_postflight": "Reflect",
    "persistence": "Persist",
    "completed": "Complete",
}

ROLE_LABELS = {
    "eda_researcher": "EDA researcher",
    "eda_builder": "EDA builder",
    "researcher": "Researcher",
    "researcher_web": "Researcher (web)",
    "critic_preflight": "Critic · preflight",
    "critic_postflight": "Critic · postflight",
    "builder": "Builder",
    "debugger": "Debugger",
}

# Role families and their fixed categorical slots (validated adjacency order in
# both light and dark mode with the dataviz palette gates). Color follows the
# entity: a family keeps its hue in every chart, chip, and legend.
FAMILY_ORDER = ("researcher", "builder", "eda", "critic", "debugger", "other")
FAMILY_LABELS = {
    "researcher": "Researcher",
    "builder": "Builder",
    "eda": "EDA",
    "critic": "Critic",
    "debugger": "Debugger",
    "other": "Other",
}
FAMILY_COLORS = {
    "researcher": "#2a78d6",
    "builder": "#eb6834",
    "eda": "#1baf7a",
    "critic": "#eda100",
    "debugger": "#e87ba4",
    "other": "#8a9995",
}

CHART_PRIMARY = "#2a78d6"
CHART_BASELINE = "#8a9995"

# Fixed-order series colors for built-in charts (the validated categorical
# slots). More series than slots -> let Streamlit assign rather than crash.
_SERIES_SLOTS = ("#2a78d6", "#1baf7a", "#e87ba4", "#eda100", "#eb6834")


def _series_colors(count: int) -> list[str] | None:
    if 0 < count <= len(_SERIES_SLOTS):
        return list(_SERIES_SLOTS[:count])
    return None


# ---------------------------------------------------------------------------
# Theme
#
# Architecture: `.streamlit/config.toml` is the ONLY palette authority.
# Streamlit paints every ground, ink, and piece of chrome from its
# [theme.light] / [theme.dark] sections and repaints them LIVE when the
# viewer flips the theme in the app menu — no reload, no rerun.
#
# The custom CSS below therefore never names a mode-specific color and the
# server never chooses colors: every custom tone is derived in CSS, at the
# element where it is used, from `currentColor` — the live theme ink — via
# color-mix(). The few fixed hues (accent + status + chart series) are
# mid-tones validated on both grounds, and the text-facing variants are
# mixed with currentColor so they deepen on porcelain and lighten on slate
# by themselves. Flipping the in-menu theme restyles everything instantly.
#
# Forbidden here forever: server-side theme readbacks (st.context), palette
# dicts keyed by mode, and OS-preference (prefers-color-scheme) color
# branching — the OS preference and Streamlit's own setting can disagree,
# and anything the server picks goes stale the moment the viewer toggles.
# ---------------------------------------------------------------------------


def _css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root{
  /* Fixed identity hues — mid-tones that read on both grounds. Everything
     mode-dependent below is mixed from currentColor (the live theme ink),
     so the in-menu theme toggle restyles the page with no reload. */
  --accent-h:#2E8C99; --ok-h:#1F9D62; --warn-h:#B8860B; --fail-h:#C75844;
  --accent:color-mix(in srgb, var(--accent-h) 72%, currentColor);
  --ok:color-mix(in srgb, var(--ok-h) 70%, currentColor);
  --warn:color-mix(in srgb, var(--warn-h) 68%, currentColor);
  --fail:color-mix(in srgb, var(--fail-h) 72%, currentColor);
  --live:var(--ok);
  --muted:color-mix(in srgb, currentColor 62%, transparent);
  --faint:color-mix(in srgb, currentColor 45%, transparent);
  --line:color-mix(in srgb, currentColor 16%, transparent);
  --line-strong:color-mix(in srgb, currentColor 32%, transparent);
  --tint:color-mix(in srgb, currentColor 3.5%, transparent);
  --accent-soft:color-mix(in srgb, var(--accent-h) 11%, transparent);
  --ok-soft:color-mix(in srgb, var(--ok-h) 11%, transparent);
  --warn-soft:color-mix(in srgb, var(--warn-h) 13%, transparent);
  --fail-soft:color-mix(in srgb, var(--fail-h) 11%, transparent);
  --mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
  --sans:'Archivo',-apple-system,'Segoe UI',sans-serif;
}

.block-container{max-width:1280px; padding-top:3.6rem; padding-bottom:4rem;}
h1,h2,h3,h4{letter-spacing:-0.015em;}
h3{font-size:1.22rem !important;}
[data-testid="stCaptionContainer"] p{color:var(--muted);}
code, pre{font-family:var(--mono) !important;}

[data-testid="stTabs"] [role="tablist"]{gap:1.45rem; border-bottom:1px solid var(--line);}
[data-testid="stTab"]{padding:.15rem 0 .4rem;}
[data-testid="stTab"] p{font-family:var(--mono); text-transform:uppercase;
  letter-spacing:.09em; font-size:.76rem !important; color:var(--muted);}
[data-testid="stTab"][aria-selected="true"] p{color:var(--accent); font-weight:600;}
[data-testid="stTab"] .react-aria-SelectionIndicator{background:var(--accent);}

[data-testid="stMetric"]{border:1px solid var(--line); border-radius:10px; padding:.75rem 1rem;}
[data-testid="stMetricLabel"] p{font-family:var(--mono); text-transform:uppercase;
  letter-spacing:.09em; font-size:.66rem !important; color:var(--muted);}
[data-testid="stMetricValue"]{font-family:var(--mono); font-size:1.35rem;}
[data-testid="stMetricDelta"]{font-family:var(--mono); font-size:.85rem;}

/* Section switcher inside a destination: quiet mono pills, no radio dots. */
[class*="st-key-section-"] [role="radiogroup"]{gap:.15rem;}
[class*="st-key-section-"] [data-testid="stRadioOption"]{border:1px solid transparent;
  border-radius:999px; padding:.18rem .75rem .22rem; margin-right:.1rem;}
[class*="st-key-section-"] [data-testid="stRadioOption"]
  div:has(> [data-testid="stMarkdownContainer"]) > div:first-child{display:none;}
[class*="st-key-section-"] [data-testid="stRadioOption"] p{font-family:var(--mono);
  font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; color:var(--muted);}
[class*="st-key-section-"] [data-testid="stRadioOption"][data-selected="true"]{
  border-color:var(--accent); background:var(--accent-soft);}
[class*="st-key-section-"] [data-testid="stRadioOption"][data-selected="true"] p{
  color:var(--accent); font-weight:600;}
/* The hidden dot carried the focus ring; give it back to the pill itself. */
[class*="st-key-section-"] [data-testid="stRadioOption"]:has(input:focus-visible){
  outline:2px solid var(--accent); outline-offset:2px;}

/* Graphviz labels follow the live ink; strokes are fixed both-ground tones. */
[data-testid="stGraphVizChart"] svg text{fill:color-mix(in srgb, currentColor 78%, transparent) !important;}

.eyebrow{font-family:var(--mono); color:var(--muted); font-size:.68rem;
  letter-spacing:.14em; text-transform:uppercase;}
.src{font-family:var(--mono); font-size:.63rem; letter-spacing:.1em; text-transform:uppercase;
  color:var(--faint); margin:.05rem 0 .45rem;}

.mast-title{font-size:2rem; font-weight:800; letter-spacing:-.03em; line-height:1.02;
  margin:.25rem 0 .3rem;}
.mast-sub{color:var(--muted); font-size:.88rem; max-width:44rem; line-height:1.4;}
.mast-rule{border-bottom:2px solid currentColor; margin:.55rem 0 1.05rem;}
.chip-row{display:flex; gap:.4rem; justify-content:flex-end; flex-wrap:wrap; margin-top:.45rem;}
.chip{display:inline-block; font-family:var(--mono); font-size:.67rem; letter-spacing:.06em;
  padding:.14rem .55rem; border-radius:999px; border:1px solid var(--line-strong);
  color:var(--muted); max-width:22rem; overflow:hidden; text-overflow:ellipsis;
  white-space:nowrap;}
.chip.live{color:var(--ok); border-color:var(--ok); font-weight:600;}
.chip.warn{color:var(--warn); border-color:var(--warn); font-weight:600;}
.chip.fail{color:var(--fail); border-color:var(--fail); font-weight:600;}

.ledger{border:1px solid var(--line-strong); border-radius:12px;
  overflow:hidden; margin:.35rem 0 1rem;}
.ledger-grid{display:grid; grid-template-columns:repeat(6,1fr);}
@media (max-width:1100px){.ledger-grid{grid-template-columns:repeat(3,1fr);}}
@media (max-width:640px){.ledger-grid{grid-template-columns:repeat(2,1fr);}}
.lcell{padding:.78rem .9rem .66rem; border-right:1px solid var(--line);
  border-bottom:1px solid var(--line); min-height:5.9rem;}
.ledger-grid .lcell:nth-child(6n){border-right:none;}
@media (max-width:1100px){
  .ledger-grid .lcell:nth-child(6n){border-right:1px solid var(--line);}
  .ledger-grid .lcell:nth-child(3n){border-right:none;}
}
@media (max-width:640px){
  .ledger-grid .lcell:nth-child(3n){border-right:1px solid var(--line);}
  .ledger-grid .lcell:nth-child(2n){border-right:none;}
}
.lcell .lk{font-family:var(--mono); font-size:.61rem; letter-spacing:.12em;
  text-transform:uppercase; color:var(--muted); margin-bottom:.4rem;}
.lcell .lv{font-family:var(--mono); font-size:1.04rem; font-weight:600;
  line-height:1.25; overflow-wrap:anywhere;}
.lcell .ln{font-size:.75rem; color:var(--muted); margin-top:.28rem; line-height:1.35;}
.lv.ok{color:var(--ok);} .lv.warn{color:var(--warn);} .lv.fail{color:var(--fail);} .lv.live{color:var(--ok);}

.ledger-rail{display:grid; grid-template-columns:repeat(3,1fr); gap:1.15rem;
  padding:.72rem .9rem .82rem; background:var(--tint); border-top:1px solid var(--line);}
@media (max-width:900px){.ledger-rail{grid-template-columns:1fr;}}
.rail-note{padding:.4rem .9rem .55rem; background:var(--tint); border-top:1px solid var(--line);
  font-family:var(--mono); font-size:.62rem; letter-spacing:.08em; text-transform:uppercase;
  color:var(--muted);}
.meter-head{display:flex; justify-content:space-between; gap:.6rem; font-family:var(--mono);
  font-size:.66rem; letter-spacing:.07em; text-transform:uppercase; color:var(--muted);
  margin-bottom:.32rem;}
.meter-head b{color:inherit; font-weight:600;}
.meter-track{height:5px; border-radius:99px; background:var(--line); overflow:hidden;}
.meter-fill{height:100%; border-radius:99px; background:var(--accent);}
.meter-fill.warn{background:var(--warn);} .meter-fill.fail{background:var(--fail);}

.tiles{display:grid; grid-template-columns:repeat(6,1fr); gap:.55rem; margin:.35rem 0 .9rem;}
@media (max-width:1100px){.tiles{grid-template-columns:repeat(3,1fr);}}
@media (max-width:640px){.tiles{grid-template-columns:repeat(2,1fr);}}
.tile{border:1px solid var(--line); border-radius:10px; padding:.62rem .8rem;}
.tile .tk{font-family:var(--mono); font-size:.6rem; letter-spacing:.11em; text-transform:uppercase;
  color:var(--muted); margin-bottom:.3rem;}
.tile .tv{font-family:var(--mono); font-size:1.12rem; font-weight:600;}
.tile .tn{font-size:.7rem; color:var(--muted); margin-top:.12rem;}

.chain{border:1px solid var(--line); border-left:3px solid var(--warn); border-radius:8px;
  padding:.55rem .8rem; margin:.35rem 0;}
.chain.error{border-left-color:var(--fail);}
.chain.repair{border-left-color:var(--accent);}
.chain .ck{font-family:var(--mono); font-size:.65rem; letter-spacing:.1em;
  text-transform:uppercase; color:var(--warn); font-weight:600;}
.chain.error .ck{color:var(--fail);}
.chain.repair .ck{color:var(--accent);}
.chain .cbody{font-size:.78rem; margin-top:.25rem; font-family:var(--mono);
  white-space:pre-wrap; word-break:break-word;}

.empty-panel{border:1px dashed var(--line-strong); border-radius:10px; padding:.95rem 1.1rem;
  color:var(--muted); font-size:.87rem;}
.note-panel{border:1px solid var(--line); border-left:3px solid var(--accent); border-radius:8px;
  padding:.62rem .9rem; font-size:.85rem; margin:.3rem 0;}
.note-panel.ok{border-left-color:var(--ok);}
.note-panel.warn{border-left-color:var(--warn);}
.note-panel.fail{border-left-color:var(--fail);}

.live-overlay{border:1px solid var(--line); border-left:3px solid var(--ok);
  border-radius:12px; background:var(--tint); padding:.95rem 1.15rem; margin:.2rem 0 1rem;}
.live-title{font-size:1.12rem; font-weight:650; margin:.18rem 0;}
.live-meta{color:var(--muted); font-size:.9rem;}
.pulse{display:inline-block; width:.6rem; height:.6rem; border-radius:50%; background:var(--ok);
  animation:pulse 1.8s infinite; margin-right:.45rem;}
@keyframes pulse{
  0%{box-shadow:0 0 0 0 color-mix(in srgb, var(--ok-h) 42%, transparent)}
  70%{box-shadow:0 0 0 9px transparent}
  100%{box-shadow:0 0 0 0 transparent}
}
@media (prefers-reduced-motion: reduce){.pulse{animation:none;}}

.stage-row{display:flex; flex-wrap:wrap; gap:.4rem; margin:.75rem 0 .2rem;}
.stage{padding:.32rem .62rem; border-radius:999px; border:1px solid var(--line);
  color:var(--muted); font-size:.72rem; font-family:var(--mono);}
.stage.done{color:var(--ok); background:var(--ok-soft); border-color:transparent;}
.stage.active{color:var(--accent); background:var(--accent-soft); border-color:var(--accent); font-weight:600;}
.stage.failed{color:var(--fail); background:var(--fail-soft); border-color:var(--fail);}

.story{border:1.5px solid currentColor; border-radius:14px;
  overflow:hidden; margin:.4rem 0 .55rem;}
.story-grid{display:grid; grid-template-columns:1.02fr 1.28fr 1fr;}
@media (max-width:980px){.story-grid{grid-template-columns:1fr;}}
.scell{padding:1.3rem 1.5rem 1.15rem; border-right:1px solid var(--line); min-height:9.3rem;}
.scell:last-child{border-right:none;}
@media (max-width:980px){
  .scell{border-right:none; border-bottom:1px solid var(--line); min-height:0;}
  .scell:last-child{border-bottom:none;}
}
.scell .sk{font-family:var(--mono); font-size:.63rem; letter-spacing:.14em;
  text-transform:uppercase; color:var(--muted);}
.scell .sv{font-family:var(--sans); font-size:1.64rem; font-weight:700; letter-spacing:-.02em;
  line-height:1.12; margin:.55rem 0 .5rem;}
.scell .sv.num{font-family:var(--mono); font-size:2.3rem; font-weight:600; letter-spacing:-.01em;}
.scell .sv.ok{color:var(--ok);} .scell .sv.warn{color:var(--warn);}
.scell .sv.fail{color:var(--fail);} .scell .sv.live{color:var(--ok);}
.scell .sl{font-size:.85rem; color:var(--muted); line-height:1.5;}
.scell .sl b{color:inherit; font-weight:600;}
.scell .sl b.ok{color:var(--ok);} .scell .sl b.warn{color:var(--warn);}
.story-meters{display:flex; flex-direction:column; gap:.72rem; margin-top:.9rem;}
.wayfinding{font-family:var(--mono); font-size:.68rem; letter-spacing:.05em; color:var(--muted);
  margin:.75rem 0 .35rem; line-height:2;}
.wayfinding b{color:inherit; font-weight:600;}

@media print{
  [data-testid="stHeader"],[data-testid="stToolbar"]{display:none !important;}
  .block-container{max-width:100%; padding-top:0;}
}
</style>
"""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _parse_time(value: str) -> datetime | None:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result.replace(tzinfo=result.tzinfo or timezone.utc)
    except (TypeError, ValueError):
        return None


def _fmt_metric(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


def _fmt_duration(seconds: Any) -> str:
    try:
        total = int(float(seconds))
    except (TypeError, ValueError):
        return "—"
    if total < 0:
        total = 0
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _fmt_clock(iso_value: str) -> str:
    parsed = _parse_time(iso_value)
    return parsed.strftime("%H:%M:%S") if parsed else "—"


def _empty(text: str) -> None:
    st.markdown(f'<div class="empty-panel">{html.escape(text)}</div>', unsafe_allow_html=True)


def _note(text: str, tone: str = "") -> None:
    tone_class = f" {tone}" if tone in {"ok", "warn", "fail"} else ""
    st.markdown(
        f'<div class="note-panel{tone_class}">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def _source_line(text: str) -> None:
    st.markdown(f'<div class="src">{html.escape(text)}</div>', unsafe_allow_html=True)


def _guard(render: Callable[[], None], panel: str) -> None:
    """A mid-write run directory must never crash a panel; it retries next poll."""
    try:
        render()
    except Exception as exc:  # noqa: BLE001 - deliberate containment boundary
        st.markdown(
            f'<div class="note-panel fail">The {html.escape(panel)} panel could not render '
            f"({html.escape(type(exc).__name__)}). A half-written artifact clears on the "
            "next refresh; a message that persists here is a real rendering bug.</div>",
            unsafe_allow_html=True,
        )


def _elapsed_label(activity: StageTransition) -> str:
    started = _parse_time(activity.started_at)
    if started is None:
        return "elapsed time unavailable"
    seconds = max(0, int((datetime.now(timezone.utc) - started).total_seconds()))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


# ---------------------------------------------------------------------------
# Cached loading (a 50-iteration run has hundreds of pass files; the cache is
# keyed by the newest artifact mtime so an idle 5 s refresh costs one stat pass)
# ---------------------------------------------------------------------------


_FINGERPRINT_FILES = (
    "state.json",
    "summary.json",
    "activity.json",
    "activity.jsonl",
    "iterations.jsonl",
    "research_memory.jsonl",
    "debugger_memory.jsonl",
    "resources.json",
    "interventions.jsonl",
    "baseline_selection.json",
    "journal.md",
    "results.md",
)


def _run_fingerprint(run_dir: Path) -> int:
    latest = 0
    for name in _FINGERPRINT_FILES:
        try:
            latest = max(latest, (run_dir / name).stat().st_mtime_ns)
        except OSError:
            continue
    for sub in ("passes", "eda", "changes"):
        try:
            latest = max(latest, (run_dir / sub).stat().st_mtime_ns)
        except OSError:
            continue
    return latest


@st.cache_data(show_spinner=False, max_entries=8)
def _snapshot_cached(run_dir: str, fingerprint: int, baseline: float) -> RunSnapshot:
    return load_run_snapshot(Path(run_dir), baseline)


def _snapshot(run_dir: Path, baseline: float) -> RunSnapshot:
    return _snapshot_cached(str(run_dir), _run_fingerprint(run_dir), baseline)


# ---------------------------------------------------------------------------
# Signature element: the verdict ledger + budget rail
# ---------------------------------------------------------------------------


def _ledger_cell(label: str, value: str, note: str, tone: str = "") -> str:
    tone_class = f" {tone}" if tone in {"ok", "warn", "fail", "live"} else ""
    return (
        f'<div class="lcell"><div class="lk">{html.escape(label)}</div>'
        f'<div class="lv{tone_class}">{html.escape(value)}</div>'
        f'<div class="ln">{html.escape(note)}</div></div>'
    )


def _meter(label: str, used_text: str, cap_text: str, fraction: float | None) -> str:
    if fraction is None:
        fill = ""
        head_value = f"{used_text} · {cap_text}"
    else:
        pct = max(0.0, min(1.0, fraction)) * 100.0
        if fraction >= 1.0:
            tone_class = " fail"
        elif fraction >= 0.85:
            tone_class = " warn"
        else:
            tone_class = ""
        fill = f'<div class="meter-fill{tone_class}" style="width:{pct:.1f}%"></div>'
        head_value = f"{used_text} / {cap_text}"
    return (
        f'<div class="meter"><div class="meter-head"><span>{html.escape(label)}</span>'
        f"<b>{html.escape(head_value)}</b></div>"
        f'<div class="meter-track">{fill}</div></div>'
    )


def _resource_metrics(snapshot: RunSnapshot) -> dict[str, Any]:
    resources = snapshot.resources or {}
    tokens = resources.get("token_usage") or {}
    return {
        "total_tokens": tokens.get("total_tokens", 0),
        "input_tokens": tokens.get("input_tokens", 0),
        "output_tokens": tokens.get("output_tokens", 0),
        "cached_tokens": tokens.get("cached_tokens", 0),
        "wall_clock_seconds": resources.get("wall_clock_seconds", 0.0),
        "training_attempts": resources.get("training_attempts", len(snapshot.iterations)),
        "iteration_count": resources.get("iteration_count", len(snapshot.iterations)),
        "gpu_hours": resources.get("gpu_hours", 0.0),
        "manual_interventions": resources.get("manual_interventions", 0),
    }


def _stale_age(snapshot: RunSnapshot, stale_after: int) -> float | None:
    """Seconds since the last activity write, when a "running" run has gone
    quiet past the threshold; ``None`` for fresh, finished, or unknowable."""
    if snapshot.status != "running":
        return None
    age = activity_age_seconds(snapshot.activity)
    if age is not None and age > stale_after:
        return age
    return None


def _wall_clock_seconds(snapshot: RunSnapshot, stale_after: int = 1800) -> float:
    recorded = _resource_metrics(snapshot)["wall_clock_seconds"]
    try:
        recorded = float(recorded)
    except (TypeError, ValueError):
        recorded = 0.0
    # Extend by live elapsed time only while the run is demonstrably alive;
    # a stale "running" state must not inflate the meter into a fake overrun.
    if (
        snapshot.status == "running"
        and snapshot.started_at
        and _stale_age(snapshot, stale_after) is None
        and activity_age_seconds(snapshot.activity) is not None
    ):
        started = _parse_time(snapshot.started_at)
        if started is not None:
            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            return max(recorded, elapsed)
    return recorded


def _verdict_ledger_html(snapshot: RunSnapshot, stale_after: int = 1200) -> str:
    baseline = snapshot.baseline_primary
    best_metrics = snapshot.best_metrics or {}
    best_primary = best_metrics.get("primary")
    usage = _resource_metrics(snapshot)
    last_iteration = max(
        (item.iteration for item in snapshot.iterations),
        default=usage["iteration_count"] or 0,
    )

    # 1 · Run status & stop reason
    if snapshot.status == "running":
        stale_age = _stale_age(snapshot, stale_after)
        if stale_age is not None:
            cell_status = _ledger_cell(
                "Run · stop reason",
                "RUNNING?",
                f"no activity for {_fmt_duration(stale_age)} — possibly interrupted",
                "warn",
            )
        else:
            cell_status = _ledger_cell(
                "Run · stop reason",
                "RUNNING",
                "no stop recorded yet — the loop is still deciding",
                "live",
            )
    else:
        stop = (snapshot.stop_reason or "not recorded").replace("_", " ")
        cell_status = _ledger_cell(
            "Run · stop reason", stop, f"status: {snapshot.status}",
        )

    # 2 · The organizers' convergence rule, reported beside the stop — never
    #     in place of it.
    convergence = snapshot.run_config.get("convergence") or {}
    rule_label = "Official rule"
    if convergence.get("epsilon") is not None and convergence.get("patience") is not None:
        rule_label = (
            f"Official rule · ε {convergence.get('epsilon')}"
            f" · patience {convergence.get('patience')}"
        )
    if snapshot.converged_official is True:
        fired_at = snapshot.converged_official_iteration
        value = f"fired @ iter {fired_at}" if fired_at is not None else "fired"
        if fired_at is not None and last_iteration and last_iteration > fired_at:
            note = f"harness continued to iter {last_iteration} for coverage"
        else:
            note = "the stop followed the rule"
        cell_rule = _ledger_cell(rule_label, value, note, "ok")
    elif snapshot.converged_official is False:
        cell_rule = _ledger_cell(
            rule_label, "never fired", "no ε-plateau within patience on record"
        )
    else:
        cell_rule = _ledger_cell(
            rule_label, "pending", "verdict lands in summary.json at completion"
        )

    # 3 · The margin-gated claim. When nothing has been scored in-run, the
    # "best" on file is the adopted baseline artifact — say so, never let an
    # inherited number read as this run's achievement.
    max_scored = snapshot.max_scored_primary
    if best_primary is not None and max_scored is None:
        cell_claim = _ledger_cell(
            "Claimed best · margin-gated",
            _fmt_metric(best_primary),
            f"{snapshot.best_experiment_id or 'candidate'} · adopted baseline artifact — "
            "no scored in-run candidate yet",
        )
    elif best_primary is not None:
        delta = float(best_primary) - baseline
        note = (
            f"{snapshot.best_experiment_id or 'candidate'} · Δ {delta:+.4f} "
            f"vs run baseline {baseline:.4f}"
        )
        cell_claim = _ledger_cell(
            "Claimed best · margin-gated",
            _fmt_metric(best_primary),
            note,
            "ok" if delta > 0 else "",
        )
    else:
        cell_claim = _ledger_cell(
            "Claimed best · margin-gated", "—", "no promoted candidate yet"
        )

    # 4 · The raw maximum, so the gap the margin creates is visible not silent
    if max_scored is not None:
        if best_primary is None:
            note = "raw maximum among scored candidates"
        elif max_scored - float(best_primary) > 1e-9:
            note = f"+{max_scored - float(best_primary):.4f} above the claim — the margin held the claim back"
        else:
            note = "equals the claim — the margin gave nothing away"
        cell_max = _ledger_cell("Max measured · raw", _fmt_metric(max_scored), note)
    else:
        cell_max = _ledger_cell("Max measured · raw", "—", "no scored candidate yet")

    # 5 · Replication of the best draw
    replication = snapshot.best_replicated or {}
    if replication.get("n"):
        note = (
            f"median {_fmt_metric(replication.get('median_primary'))} · "
            f"spread {_fmt_metric(replication.get('spread'))}"
        )
        cell_rep = _ledger_cell("Best replicated", f"n = {replication['n']}", note, "ok")
    else:
        cell_rep = _ledger_cell(
            "Best replicated",
            "not replicated",
            "repeatability unknown — no invented spread",
        )

    # 6 · The interventions ledger (autonomy evidence)
    if snapshot.interventions_recorded:
        count = len(snapshot.interventions)
        if count == 0:
            cell_iv = _ledger_cell(
                "Manual interventions",
                "0",
                "ledger present and empty — the loop ran untouched",
                "ok",
            )
        else:
            cell_iv = _ledger_cell(
                "Manual interventions",
                str(count),
                "manual entries on the record — see Results",
                "warn",
            )
    else:
        reported = usage["manual_interventions"]
        cell_iv = _ledger_cell(
            "Manual interventions",
            "—",
            f"no ledger file recorded; resources.json reports {reported}",
        )

    # Budget rail
    budgets = snapshot.run_config.get("budgets") or {}
    total_tokens = usage["total_tokens"]
    token_cap = _positive_cap(snapshot.token_cap)
    if token_cap:
        meter_tokens = _meter(
            "LLM tokens", _fmt_int(total_tokens), _fmt_int(token_cap),
            _safe_float(total_tokens) / token_cap,
        )
    else:
        meter_tokens = _meter(
            "LLM tokens", _fmt_int(total_tokens), "cap not recorded", None
        )
    max_iters = _positive_cap(budgets.get("max_iterations"))
    if max_iters:
        meter_iters = _meter(
            "Iterations", _fmt_int(usage["iteration_count"]), _fmt_int(max_iters),
            _safe_float(usage["iteration_count"]) / max_iters,
        )
    else:
        meter_iters = _meter(
            "Iterations", _fmt_int(usage["iteration_count"]), "cap not recorded", None
        )
    wall_used = _wall_clock_seconds(snapshot, max(stale_after, 1800))
    wall_cap = _positive_cap(budgets.get("max_wall_clock_seconds"))
    if wall_cap:
        meter_wall = _meter(
            "Wall clock", _fmt_duration(wall_used), _fmt_duration(wall_cap),
            _safe_float(wall_used) / wall_cap,
        )
    else:
        meter_wall = _meter("Wall clock", _fmt_duration(wall_used), "cap not recorded", None)

    rail_note = ""
    if not token_cap and not max_iters and not wall_cap:
        rail_note = (
            '<div class="rail-note">caps not recorded in run artifacts — '
            "run_config.json absent for this run</div>"
        )
    return (
        '<div class="ledger"><div class="ledger-grid">'
        + cell_status + cell_rule + cell_claim + cell_max + cell_rep + cell_iv
        + '</div><div class="ledger-rail">'
        + meter_tokens + meter_iters + meter_wall
        + "</div>" + rail_note + "</div>"
    )


def _verdict_ledger(snapshot: RunSnapshot, stale_after: int = 1200) -> None:
    _source_line(
        "verdict · summary.json + state.json + research_memory.jsonl + "
        "run_config.json + interventions.jsonl + resources.json"
    )
    st.markdown(_verdict_ledger_html(snapshot, stale_after), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Overview — the 60-second story
#
# One calm screen, three statements: is the run healthy, what score can a
# judge trust, and how much budget is left. Every deeper number lives one
# click away; nothing is deleted, only deferred.
# ---------------------------------------------------------------------------


def _story_cell(label: str, value: str, lines: list[str], tone: str = "", numeric: bool = False) -> str:
    classes = "sv"
    if numeric:
        classes += " num"
    if tone in {"ok", "warn", "fail", "live"}:
        classes += f" {tone}"
    body = "".join(f'<div class="sl">{line}</div>' for line in lines)
    return (
        f'<div class="scell"><div class="sk">{html.escape(label)}</div>'
        f'<div class="{classes}">{html.escape(value)}</div>{body}</div>'
    )


def _story_health_cell(snapshot: RunSnapshot, stale_after: int) -> str:
    usage = _resource_metrics(snapshot)
    last_iteration = max(
        (item.iteration for item in snapshot.iterations),
        default=usage["iteration_count"] or 0,
    )
    lines: list[str] = []
    if snapshot.status == "running":
        stale_age = _stale_age(snapshot, stale_after)
        activity = snapshot.activity
        stage = (
            STAGE_LABELS.get(activity.stage, activity.stage.replace("_", " ").title())
            if activity
            else None
        )
        if stale_age is not None:
            value, tone = f"Quiet {_fmt_duration(stale_age)}", "warn"
            lines.append("still marked running — possibly interrupted")
            if activity and stage:
                lines.append(
                    f"last seen: iteration {activity.iteration} · {html.escape(stage)}"
                )
        else:
            value, tone = "Live", "live"
            if activity and stage:
                lines.append(
                    f"iteration {activity.iteration} · <b>{html.escape(stage)}</b> in progress"
                )
            lines.append("no stop recorded yet — the loop is still deciding")
    else:
        value = snapshot.status.title() or "Unknown"
        tone = "fail" if snapshot.status == "failed" else ""
        stop = (snapshot.stop_reason or "not recorded").replace("_", " ")
        if snapshot.converged_official is True:
            fired = snapshot.converged_official_iteration
            rule = (
                f"official rule <b class=\"ok\">fired @ iteration {fired}</b>"
                if fired is not None
                else 'official rule <b class="ok">fired</b>'
            )
            if fired is not None and last_iteration and last_iteration > fired:
                rule += f" of {last_iteration}"
            if tone == "":
                tone = "ok"
        elif snapshot.converged_official is False:
            rule = "official rule never fired"
        else:
            rule = "official rule verdict not recorded"
        lines.append(f"stopped: <b>{html.escape(stop)}</b> · {rule}")
        trust_bits: list[str] = []
        gate = (snapshot.gate_info or {}).get("status")
        if gate:
            gate_str = html.escape(str(gate))
            trust_bits.append(
                f'gate <b class="ok">{gate_str}</b>'
                if str(gate) == "ok"
                else f'gate <b class="warn">{gate_str}</b>'
            )
        if snapshot.interventions_recorded:
            count = len(snapshot.interventions)
            trust_bits.append(
                '<b class="ok">loop ran untouched</b>'
                if count == 0
                else f'<b class="warn">{count} manual intervention(s)</b>'
            )
        else:
            trust_bits.append("no interventions ledger on file")
        lines.append(" · ".join(trust_bits))
    return _story_cell("Run health", value, lines, tone)


def _story_claim_cell(snapshot: RunSnapshot, official: float) -> str:
    best_metrics = snapshot.best_metrics or {}
    best_primary = best_metrics.get("primary")
    max_scored = snapshot.max_scored_primary
    label = "Claimed best · margin-gated"
    if best_primary is None:
        return _story_cell(label, "—", ["no promoted candidate yet"])
    if max_scored is None:
        return _story_cell(
            label,
            _fmt_metric(best_primary),
            [
                '<b class="warn">adopted baseline artifact</b> — not measured by this run',
                "no scored in-run candidate yet",
            ],
            "warn",
            numeric=True,
        )
    delta = float(best_primary) - official
    lines = [
        f"<b>Δ {delta:+.4f}</b> vs official baseline {official:.4f}"
    ]
    replication = snapshot.best_replicated or {}
    if replication.get("n"):
        lines.append(
            f'replicated <b class="ok">n = {html.escape(str(replication["n"]))}</b> · median '
            f"{_fmt_metric(replication.get('median_primary'))} · spread "
            f"{_fmt_metric(replication.get('spread'))}"
        )
    else:
        lines.append("not replicated — single measurement, spread unknown")
    gap = max_scored - float(best_primary)
    if gap > 1e-9:
        lines.append(f"raw max {_fmt_metric(max_scored)} — the margin held back {gap:+.4f}")
    return _story_cell(
        label,
        _fmt_metric(best_primary),
        lines,
        "ok" if delta > 0 else "",
        numeric=True,
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _positive_cap(value: Any) -> float | None:
    """A budget cap straight from run_config: usable only as a finite positive
    number — anything else reads as 'no cap on record', never a crash."""
    cap = _safe_float(value, 0.0)
    return cap if cap > 0 else None


def _budget_entries(
    snapshot: RunSnapshot, stale_after: int
) -> list[tuple[str, str, str, float]]:
    """(label, used, cap, used-fraction) for every cap actually on record."""
    usage = _resource_metrics(snapshot)
    budgets = snapshot.run_config.get("budgets") or {}
    wall_used = _wall_clock_seconds(snapshot, max(stale_after, 1800))
    entries: list[tuple[str, str, str, float]] = []
    token_cap = _positive_cap(snapshot.token_cap)
    if token_cap:
        entries.append(
            (
                "LLM tokens",
                _fmt_int(usage["total_tokens"]),
                _fmt_int(token_cap),
                _safe_float(usage["total_tokens"]) / token_cap,
            )
        )
    max_iters = _positive_cap(budgets.get("max_iterations"))
    if max_iters:
        entries.append(
            (
                "Iterations",
                _fmt_int(usage["iteration_count"]),
                _fmt_int(max_iters),
                _safe_float(usage["iteration_count"]) / max_iters,
            )
        )
    wall_cap = _positive_cap(budgets.get("max_wall_clock_seconds"))
    if wall_cap:
        entries.append(
            (
                "Wall clock",
                _fmt_duration(wall_used),
                _fmt_duration(wall_cap),
                _safe_float(wall_used) / wall_cap,
            )
        )
    return entries


def _story_budget_cell(snapshot: RunSnapshot, stale_after: int) -> str:
    usage = _resource_metrics(snapshot)
    wall_used = _wall_clock_seconds(snapshot, max(stale_after, 1800))
    entries = _budget_entries(snapshot, stale_after)
    if not entries:
        lines = [
            "no caps recorded — run_config.json absent for this run",
            f"spent so far: <b>{_fmt_int(usage['total_tokens'])}</b> tokens · "
            f"<b>{_fmt_int(usage['iteration_count'])}</b> iterations · "
            f"<b>{html.escape(_fmt_duration(wall_used))}</b>",
        ]
        return _story_cell("Budget remaining", "No caps on record", lines)
    tight_label, _, _, tight_frac = max(entries, key=lambda entry: entry[3])
    left = min(1.0, max(0.0, 1.0 - tight_frac))
    tone = "" if left > 0.15 else ("warn" if left > 0.0 else "fail")
    tone_class = f" {tone}" if tone else ""
    meters = "".join(
        _meter(entry_label, used, cap, frac) for entry_label, used, cap, frac in entries
    )
    return (
        '<div class="scell"><div class="sk">Budget remaining</div>'
        f'<div class="sv num{tone_class}">{left * 100:.0f}% left</div>'
        f'<div class="sl">{html.escape(tight_label.lower())} is the tightest budget</div>'
        f'<div class="story-meters">{meters}</div></div>'
    )


def _story_band_html(snapshot: RunSnapshot, official: float, stale_after: int = 1200) -> str:
    return (
        '<div class="story"><div class="story-grid">'
        + _story_health_cell(snapshot, stale_after)
        + _story_claim_cell(snapshot, official)
        + _story_budget_cell(snapshot, stale_after)
        + "</div></div>"
    )


_WAYFINDING = (
    '<div class="wayfinding">'
    "<b>Activity</b> live loop, iterations &amp; every model call · "
    "<b>Evidence</b> gate, provenance, dataset &amp; features · "
    "<b>Compare</b> all runs side by side · "
    "<b>Judge sheet</b> the print-ready one-pager</div>"
)


def _score_trajectory(snapshot: RunSnapshot, official: float) -> None:
    st.subheader(
        "Score trajectory",
        help="Source: iterations.jsonl outcomes, scored on the frozen validation split.",
    )
    chart = _primary_trajectory_chart(snapshot, official)
    if chart is None:
        _empty("No successful validation metrics are recorded for this run yet.")
        return
    st.altair_chart(chart, width="stretch")
    st.caption(
        f"Dashed rule: official published validation baseline {official:.4f}. "
        "Primary score = (GAUC + nDCG@5) / 2."
    )
    rows = []
    for item in snapshot.iterations:
        if item.metrics:
            rows.append(
                {
                    "Iteration": item.iteration,
                    "Experiment": item.experiment_id,
                    "GAUC": item.metrics.get("GAUC"),
                    "nDCG@5": item.metrics.get("nDCG@5"),
                    "Primary": item.metrics.get("primary"),
                    "Baseline": official,
                    "Δ vs official": (
                        item.metrics.get("primary") - official
                        if item.metrics.get("primary") is not None
                        else None
                    ),
                }
            )
    if rows:
        with st.expander("Component metrics per iteration", expanded=False):
            df_res = pd.DataFrame(rows)
            st.line_chart(
                df_res.set_index("Iteration")[["GAUC", "nDCG@5"]],
                color=["#1baf7a", "#e87ba4"],
            )
            st.dataframe(df_res, width="stretch", hide_index=True)


def _story(snapshot: RunSnapshot, official: float, stale_after: int) -> None:
    _source_line(
        "the 60-second story · summary.json + state.json + resources.json + run_config.json"
    )
    st.markdown(_story_band_html(snapshot, official, stale_after), unsafe_allow_html=True)
    st.markdown(_WAYFINDING, unsafe_allow_html=True)
    for warning in snapshot.warnings:
        st.warning(warning)
    _score_trajectory(snapshot, official)
    with st.expander("Full verdict ledger — six cells and the budget rail", expanded=False):
        _verdict_ledger(snapshot, stale_after)
        _metric_cards(snapshot, official)


# ---------------------------------------------------------------------------
# Live overlay & role stream (existing capability, redesigned)
# ---------------------------------------------------------------------------


def _stage_strip(snapshot: RunSnapshot, activity: StageTransition | None) -> str:
    iteration = activity.iteration if activity else max((item.iteration for item in snapshot.iterations), default=0)
    latest: dict[str, str] = {}
    for event in snapshot.transitions:
        if event.iteration == iteration:
            latest[event.stage] = event.status
    pills = []
    run_live = snapshot.status == "running"
    for stage in STAGE_ORDER:
        status = latest.get(stage, "")
        class_name = "stage"
        if run_live and activity and activity.stage == stage and activity.status == "active":
            class_name += " active"
        elif status == "completed":
            class_name += " done"
        elif status == "failed":
            class_name += " failed"
        pills.append(f'<span class="{class_name}">{html.escape(STAGE_LABELS[stage])}</span>')
    return '<div class="stage-row">' + "".join(pills) + "</div>"


def _render_note(note: dict[str, Any]) -> None:
    if not note:
        st.caption("No structured note has been completed for this stage yet.")
        return
    labels = {
        "objective": "Objective",
        "hypothesis": "Hypothesis",
        "rationale": "Rationale",
        "decision": "Decision",
        "action": "Action",
        "family": "Model family",
        "concerns": "Risks / concerns",
        "next_focus": "Next focus",
        "diagnosis": "Diagnosis",
        "parameters": "Parameters",
        "result": "Result",
        "stop_reason": "Stop reason",
        "best_experiment": "Best experiment",
    }
    for key, value in note.items():
        if key == "evidence":
            st.markdown("**Evidence**")
            for item in value:
                title = item.get("title", "Source")
                url = item.get("url")
                st.markdown(f"- [{title}]({url})" if url else f"- {title}")
        elif isinstance(value, (dict, list)):
            st.markdown(f"**{labels.get(key, key.replace('_', ' ').title())}**")
            st.json(value, expanded=False)
        elif value not in (None, ""):
            st.markdown(f"**{labels.get(key, key.replace('_', ' ').title())}:** {value}")


def _render_live_role_stream(snapshot: RunSnapshot) -> None:
    if not snapshot.live_role_passes:
        return
    current_iter = (
        snapshot.activity.iteration
        if snapshot.activity
        else (max((item.iteration for item in snapshot.iterations), default=0) if snapshot.iterations else 1)
    )
    run_live = snapshot.status == "running"
    label = "Live role passes" if run_live else "Final iteration's role passes"
    with st.expander(
        f"{label} — iteration {current_iter:03d} · {len(snapshot.live_role_passes)} calls",
        expanded=run_live,
    ):
        for rp in snapshot.live_role_passes:
            role_label = ROLE_LABELS.get(rp.role, rp.role.replace("_", " ").title())
            st.markdown(
                f"**Pass {rp.sequence + 1}: {role_label}** (`{rp.model}` · `{rp.latency_seconds:.2f}s` · `{rp.usage.get('total_tokens', 0)} tokens`)"
            )
            if rp.role == "eda_researcher":
                st.caption(f"**Objective:** {rp.data.get('objective', 'Plan EDA pass')}")
                if rp.data.get("questions"):
                    st.markdown("*Research questions:*")
                    for q in rp.data["questions"]:
                        st.markdown(f"- {q}")
                if rp.data.get("feature_hypotheses"):
                    st.markdown("*Feature hypotheses:*")
                    for h in rp.data["feature_hypotheses"]:
                        st.markdown(f"- {h}")
                if rp.data.get("leakage_risks"):
                    st.markdown("*Leakage guardrails:*")
                    for r in rp.data["leakage_risks"]:
                        st.markdown(f"- ⚠ {r}")
            elif rp.role == "eda_builder":
                if rp.data.get("summary"):
                    st.info(f"**EDA summary:** {rp.data['summary']}")
                if rp.data.get("findings"):
                    st.markdown("*Empirical findings:*")
                    st.dataframe(rp.data["findings"], width="stretch", hide_index=True)
                if rp.data.get("feature_candidates"):
                    st.markdown("*Feature proposals:*")
                    st.dataframe(rp.data["feature_candidates"], width="stretch", hide_index=True)
                if rp.data.get("recommended_next_focus"):
                    st.success(f"**Recommended focus:** {rp.data['recommended_next_focus']}")
            elif rp.role == "researcher":
                st.markdown(f"**Hypothesis:** {rp.data.get('hypothesis', '')}")
                st.caption(f"**Family:** `{rp.data.get('family', '')}` · **Action:** `{rp.data.get('action', '')}`")
                if rp.data.get("rationale"):
                    st.caption(f"**Rationale:** {rp.data['rationale']}")
                if rp.data.get("parameters"):
                    st.json(rp.data["parameters"], expanded=False)
            elif rp.role == "critic_preflight":
                approved = rp.data.get("approved", False)
                decision = rp.data.get("decision", "approve" if approved else "reject")
                badge = "✅ Approved" if approved else f"❌ {decision}"
                st.markdown(f"**Preflight decision:** {badge}")
                if rp.data.get("rationale"):
                    st.caption(f"**Rationale:** {rp.data['rationale']}")
                if rp.data.get("concerns"):
                    st.markdown("*Concerns / risks:*")
                    for c in rp.data["concerns"]:
                        st.markdown(f"- {c}")
                if rp.data.get("next_focus"):
                    st.caption(f"**Next focus:** {rp.data['next_focus']}")
            elif rp.role == "builder":
                st.markdown(f"**Candidate generated:** `{rp.data.get('candidate_id', '')}`")
                if rp.data.get("code"):
                    with st.expander("Candidate implementation code", expanded=False):
                        st.code(rp.data["code"], language="python", line_numbers=True)
            elif rp.role == "debugger":
                st.warning(f"**Debugger diagnosis:** {rp.data.get('diagnosis', '')}")
                if rp.data.get("replacement_code"):
                    with st.expander("Debugger repaired code", expanded=False):
                        st.code(rp.data["replacement_code"], language="python", line_numbers=True)
            elif rp.role == "critic_postflight":
                st.markdown(f"**Postflight reflection:** {rp.data.get('reflection', '')}")
                st.markdown(f"**Next focus:** {rp.data.get('next_focus', '')}")
            else:
                _render_note(rp.data)
            st.divider()


def _render_live_diagnostics(snapshot: RunSnapshot) -> None:
    activity = snapshot.activity
    current_iter = (
        activity.iteration
        if activity is not None
        else (max((item.iteration for item in snapshot.iterations), default=0) if snapshot.iterations else 1)
    )
    iter_events = [e for e in snapshot.debugger_events if e.iteration == current_iter]
    has_activity_issue = bool(
        activity and (activity.error or activity.repair or activity.attempt > 1 or activity.stage in {"debugger", "safety_tests"})
    )
    if not has_activity_issue and not iter_events:
        return

    st.markdown("### Diagnostics & re-attempt engine")
    if activity and activity.attempt > 1:
        st.warning(
            f"**Active re-attempt in progress:** iteration {activity.iteration} · attempt {activity.attempt} "
            f"for stage `{STAGE_LABELS.get(activity.stage, activity.stage)}`"
        )
    if activity and activity.error:
        st.error(f"**Trigger / failure output:**\n\n```text\n{activity.error}\n```")
    if activity and activity.repair:
        st.info(f"**Repair / recovery strategy:**\n\n{activity.repair}")

    if iter_events:
        with st.expander(
            f"Why it re-attempted — debugger journal ({len(iter_events)} events)",
            expanded=snapshot.status == "running",
        ):
            for event in iter_events:
                st.markdown(f"**Stage:** `{event.stage}` · **Event:** `{event.event_type}`")
                if event.error_type:
                    st.caption(f"**Classification:** `{event.error_type}`")
                if event.candidate_id:
                    st.caption(f"**Candidate:** `{event.candidate_id}`")
                if event.error:
                    st.code(event.error, language="text")
                if event.lesson:
                    st.success(f"**Diagnosis / corrective action:** {event.lesson}")
                st.divider()


def _render_live_overlay(snapshot: RunSnapshot, stale_after: int) -> None:
    activity = snapshot.activity
    if activity is None:
        _empty(
            "No live activity artifact exists for this run. "
            "Completed iteration data remains available below."
        )
        return
    run_live = snapshot.status == "running"
    is_live = activity.status == "active" and run_live
    age = activity_age_seconds(activity)
    stale = is_live and age is not None and age > stale_after
    stage_label = STAGE_LABELS.get(activity.stage, activity.stage.replace("_", " ").title())
    marker = '<span class="pulse"></span>' if is_live and not stale else ""
    if stale:
        state_label = "Possibly stale"
    elif is_live:
        state_label = "Active"
    elif not run_live:
        # The run has ended: whatever activity.json froze on is a recording,
        # not a live stage - even if its own status still says "active".
        state_label = "Recorded"
    else:
        state_label = activity.status.title()
    eyebrow = "Live execution · activity.json" if run_live else "Last recorded activity · activity.json"
    experiment = f" · {html.escape(activity.experiment_id)}" if activity.experiment_id else ""
    elapsed = _elapsed_label(activity) if is_live else "stage recorded"
    st.markdown(
        f'''<div class="live-overlay">
<div class="eyebrow">{html.escape(eyebrow)}</div>
<div class="live-title">{marker}Iteration {activity.iteration} · {html.escape(stage_label)} · {html.escape(state_label)}</div>
<div class="live-meta">{html.escape(elapsed)}{experiment}</div>
<div class="live-meta">{html.escape(activity.objective)}</div>
{_stage_strip(snapshot, activity)}
</div>''',
        unsafe_allow_html=True,
    )
    if stale:
        st.warning(
            f"No activity transition has been written for more than {stale_after // 60} minutes. "
            "The process may still be in a long blocking stage or may have been interrupted."
        )

    _render_live_role_stream(snapshot)
    _render_live_diagnostics(snapshot)

    with st.expander(
        "Agent notes — structured decision trace",
        expanded=snapshot.status == "running",
    ):
        _render_note(activity.agent_note)
        st.caption("Summarized decisions only; raw hidden reasoning and full prompts are not displayed.")
    with st.expander("Recent timeline"):
        recent = list(snapshot.transitions)[-10:][::-1]
        if recent:
            st.dataframe(
                [
                    {
                        "iteration": item.iteration,
                        "stage": STAGE_LABELS.get(item.stage, item.stage),
                        "status": item.status,
                        "attempt": item.attempt,
                        "updated": item.updated_at,
                    }
                    for item in recent
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("No stage transitions recorded yet.")


def _metric_cards(
    snapshot: RunSnapshot, official: float, reference_label: str = "official"
) -> None:
    metrics = snapshot.best_metrics or {}
    columns = st.columns(4)
    columns[0].metric("GAUC", f"{metrics.get('GAUC', float('nan')):.4f}" if "GAUC" in metrics else "—")
    columns[1].metric("nDCG@5", f"{metrics.get('nDCG@5', float('nan')):.4f}" if "nDCG@5" in metrics else "—")
    primary = metrics.get("primary")
    columns[2].metric("Primary score", f"{primary:.4f}" if primary is not None else "—")
    columns[3].metric(
        f"Δ vs {reference_label} {official:.4f}",
        f"{primary - official:+.4f}" if primary is not None else "—",
    )


# ---------------------------------------------------------------------------
# Experiment lineage DAG
# ---------------------------------------------------------------------------


def _dot_escape(value: Any) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", " ")
        .replace("\n", "\\n")
    )


def _render_journal_markdown(markdown_text: str) -> None:
    """journal.md is harness/LLM-authored: render it without unsafe HTML, and
    fold each ```diff fence into its own expander instead of a raw <details>."""
    parts = re.split(r"(```diff\n.*?\n```)", markdown_text, flags=re.DOTALL)
    for part in parts:
        if part.startswith("```diff\n") and part.endswith("\n```"):
            body = part[len("```diff\n") : -len("\n```")]
            with st.expander("View code changes", expanded=False):
                st.code(body, language="diff")
        elif part.strip():
            st.markdown(part)


def _experiment_dag_dot(
    nodes: tuple[dict[str, Any], ...], best_id: str | None
) -> str:
    # The DOT fontcolor is only a fallback both-grounds mid-tone: the CSS
    # rule on [data-testid="stGraphVizChart"] svg text repaints labels from
    # the live theme ink, so the in-menu toggle restyles the DAG instantly.
    fontcolor = "#79898B"
    dot_lines = [
        "digraph experiments {",
        '  graph [rankdir="TB", bgcolor="transparent"];',
        f'  node [shape="box", style="rounded", fontname="Courier", fontsize="10", fontcolor="{fontcolor}"];',
        '  edge [color="#7f958f"];',
    ]
    id_by_experiment: dict[str, str] = {}
    for index, node in enumerate(nodes):
        node_id = f"n{index}"
        eid = str(node.get("experiment_id") or f"unknown_{index}")
        id_by_experiment.setdefault(eid, node_id)
        family = node.get("family", "")
        status = node.get("status", "")
        m = node.get("metrics") or {}
        p = m.get("primary")
        score_str = f"P: {p:.4f}" if isinstance(p, (int, float)) else status
        label = _dot_escape(f"{eid}\n[{family}] · {score_str}")
        # Strokes only, mid-tone: graphviz cannot read the CSS theme tokens,
        # so the DAG stays legible on both the porcelain and the slate ground.
        if eid == best_id:
            stroke, penwidth = "#2E8C99", 2.4
        elif status == "failed":
            stroke, penwidth = "#C75844", 1.4
        else:
            stroke, penwidth = "#7f958f", 1
        dot_lines.append(
            f'  {node_id} [label="{label}", '
            f'color="{stroke}", penwidth="{penwidth}"];'
        )

    for index, node in enumerate(nodes):
        parent_id = id_by_experiment.get(str(node.get("parent_experiment")))
        if parent_id is not None:
            dot_lines.append(f"  {parent_id} -> n{index};")

    dot_lines.append("}")
    return "\n".join(dot_lines)


def _render_experiment_dag(nodes: tuple[dict[str, Any], ...], best_id: str | None) -> None:
    if nodes:
        st.graphviz_chart(_experiment_dag_dot(nodes, best_id), width="stretch")


# ---------------------------------------------------------------------------
# Activity — one destination, three summoned sections: the loop as it moves,
# the audited iteration records, and every model call. One job per screen.
# ---------------------------------------------------------------------------


def _section(key: str, label: str, options: tuple[str, ...]) -> str:
    """A quiet in-page switcher: one section visible, the others summoned."""
    return str(
        st.radio(
            label,
            options,
            horizontal=True,
            key=f"section-{key}",
            label_visibility="collapsed",
        )
    )


def _activity_lineage(snapshot: RunSnapshot) -> None:
    st.subheader(
        "Experiment lineage",
        help="Source: state.json experiment nodes. The teal frame marks the claimed best.",
    )
    if not snapshot.nodes:
        _empty("No experiment tree recorded yet — the first scored iteration creates it.")
        return
    _render_experiment_dag(snapshot.nodes, snapshot.best_experiment_id)
    with st.expander(
        f"Node table — all {len(snapshot.nodes)} experiments behind the picture",
        expanded=False,
    ):
        st.dataframe(
            [
                {
                    "iteration": node.get("iteration"),
                    "experiment": node.get("experiment_id"),
                    "parent": node.get("parent_experiment"),
                    "family": node.get("family"),
                    "status": node.get("status"),
                    "primary": (node.get("metrics") or {}).get("primary"),
                    "GAUC": (node.get("metrics") or {}).get("GAUC"),
                    "nDCG@5": (node.get("metrics") or {}).get("nDCG@5"),
                }
                for node in snapshot.nodes
            ],
            width="stretch",
            hide_index=True,
        )


def _activity(snapshot: RunSnapshot, stale_after: int) -> None:
    section = _section(
        "activity", "Activity section", ("Loop stages", "Iterations", "Model calls")
    )
    if section == "Iterations":
        _activity_lineage(snapshot)
        st.divider()
        _iterations(snapshot)
    elif section == "Model calls":
        _agents(snapshot)
    else:
        _render_live_overlay(snapshot, stale_after)


# ---------------------------------------------------------------------------
# Agent Trace tab (LLM observability)
# ---------------------------------------------------------------------------


def _family_label(family: str) -> str:
    return FAMILY_LABELS.get(family, family.title() or "Other")


def _agents_tiles(snapshot: RunSnapshot) -> None:
    calls = snapshot.llm_calls
    events = snapshot.memory_events
    total_tokens = sum(call.total_tokens for call in calls)
    cached = sum(call.cached_tokens for call in calls)
    latencies = [call.latency_seconds for call in calls if call.latency_seconds > 0]
    median_latency = statistics.median(latencies) if latencies else None
    reprompts = sum(1 for event in events if event.kind == "role_retry")
    failures = sum(1 for event in events if event.kind in {"controller_error", "eda_error"})
    cap = snapshot.token_cap
    # The harness meter (resources.json) is authoritative for the cap; the
    # trace sum counts only what landed in pass files. Show both, say which.
    harness_tokens = _resource_metrics(snapshot)["total_tokens"]
    if cap:
        try:
            token_note = (
                f"harness meter {int(harness_tokens):,} "
                f"({float(harness_tokens) / cap * 100:.0f}% of cap)"
            )
        except (TypeError, ValueError):
            token_note = "harness meter unavailable"
    else:
        token_note = f"harness meter {_fmt_int(harness_tokens)} · cap not recorded"
    tiles = [
        ("Model calls", _fmt_int(len(calls)), "passes/*.json"),
        ("Tokens in trace", _fmt_int(total_tokens), token_note),
        ("Cached tokens", _fmt_int(cached), "per pass files"),
        (
            "Median latency",
            f"{median_latency:.1f}s" if median_latency is not None else "—",
            "per model call",
        ),
        ("Re-prompts", _fmt_int(reprompts), "rejected outputs re-asked"),
        ("Controller faults", _fmt_int(failures), "harness-level errors"),
    ]
    cells = "".join(
        f'<div class="tile"><div class="tk">{html.escape(k)}</div>'
        f'<div class="tv">{html.escape(v)}</div><div class="tn">{html.escape(n)}</div></div>'
        for k, v, n in tiles
    )
    st.markdown(f'<div class="tiles">{cells}</div>', unsafe_allow_html=True)


def _burn_down_chart(snapshot: RunSnapshot) -> alt.LayerChart | alt.Chart | None:
    calls = snapshot.llm_calls
    if not calls:
        return None
    families = [f for f in FAMILY_ORDER if any(c.family == f for c in calls)]
    totals = {family: 0 for family in families}
    rows: list[dict[str, Any]] = []
    for index, call in enumerate(calls, start=1):
        family = call.family if call.family in totals else "other"
        if family not in totals:
            totals[family] = 0
            families.append(family)
        totals[family] += call.total_tokens
        for fam in families:
            rows.append(
                {
                    "call": index,
                    "family": _family_label(fam),
                    "tokens": totals[fam],
                    "order": FAMILY_ORDER.index(fam) if fam in FAMILY_ORDER else 99,
                }
            )
    df = pd.DataFrame(rows)
    domain = [_family_label(f) for f in families]
    color_range = [FAMILY_COLORS.get(f, FAMILY_COLORS["other"]) for f in families]
    area = (
        alt.Chart(df)
        .mark_area(interpolate="step-after", line={"strokeWidth": 1})
        .encode(
            x=alt.X("call:Q", title="model call #", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("tokens:Q", title="cumulative tokens", stack="zero"),
            color=alt.Color(
                "family:N",
                title=None,
                scale=alt.Scale(domain=domain, range=color_range),
                legend=alt.Legend(orient="top"),
            ),
            order=alt.Order("order:Q"),
            tooltip=[
                alt.Tooltip("call:Q", title="call #"),
                alt.Tooltip("family:N", title="role family"),
                alt.Tooltip("tokens:Q", title="cumulative tokens", format=","),
            ],
        )
        .properties(height=260)
    )
    if snapshot.token_cap:
        cap_rule = (
            alt.Chart(pd.DataFrame({"cap": [snapshot.token_cap]}))
            .mark_rule(strokeDash=[5, 4], color=CHART_BASELINE, strokeWidth=1.5)
            .encode(y="cap:Q", tooltip=[alt.Tooltip("cap:Q", title="token cap", format=",")])
        )
        return area + cap_rule
    return area


def _role_summary_rows(snapshot: RunSnapshot) -> list[dict[str, Any]]:
    calls = snapshot.llm_calls
    grand_total = sum(call.total_tokens for call in calls) or 1
    rows = []
    for family in FAMILY_ORDER:
        family_calls = [c for c in calls if c.family == family]
        if not family_calls:
            continue
        tokens = sum(c.total_tokens for c in family_calls)
        latencies = [c.latency_seconds for c in family_calls if c.latency_seconds > 0]
        rows.append(
            {
                "role family": _family_label(family),
                "calls": len(family_calls),
                "tokens": tokens,
                "share": f"{tokens / grand_total * 100:.1f}%",
                "median s": round(statistics.median(latencies), 1) if latencies else None,
                "max s": round(max(latencies), 1) if latencies else None,
            }
        )
    return rows


def _trace_rows(calls: tuple[LLMCall, ...]) -> list[dict[str, Any]]:
    return [
        {
            "time (UTC)": _fmt_clock(call.recorded_at),
            "iter": call.iteration,
            "role": ROLE_LABELS.get(call.role, call.role),
            "model": call.model,
            "lat s": round(call.latency_seconds, 1),
            "in": call.input_tokens,
            "out": call.output_tokens,
            "cached": call.cached_tokens,
            "total": call.total_tokens,
            "retries": call.retries,
            "gist": call.gist,
        }
        for call in calls
    ]


def _render_call_inspector(calls: tuple[LLMCall, ...]) -> None:
    options = {
        f"#{index:03d} · iter {call.iteration:03d} · {ROLE_LABELS.get(call.role, call.role)} · "
        f"{_fmt_clock(call.recorded_at)} · {call.total_tokens:,} tok": call
        for index, call in enumerate(calls, start=1)
    }
    keys = list(options)
    selected = options[
        st.selectbox(
            "Inspect a call — full prompt and parsed reply",
            keys,
            index=len(keys) - 1,
            key="call_inspector",
        )
    ]
    st.caption(
        f"`{selected.model}` · {selected.latency_seconds:.1f}s · "
        f"{selected.input_tokens:,} in / {selected.output_tokens:,} out / "
        f"{selected.cached_tokens:,} cached · retries {selected.retries}"
    )
    with st.expander("Prompt sent to the model", expanded=False):
        prompt_text = selected.prompt or load_call_prompt(selected.path)
        if prompt_text:
            st.code(prompt_text, language="text")
        else:
            st.caption("The pass file holds no prompt text.")
    with st.expander("Parsed structured reply", expanded=True):
        if selected.data:
            st.json(selected.data, expanded=False)
        else:
            st.caption("No structured payload was parsed from this call.")
    if selected.sources:
        st.markdown("**Cited sources**")
        for source in selected.sources:
            title = source.get("title", "Source")
            url = source.get("url")
            st.markdown(f"- [{title}]({url})" if url else f"- {title}")


def _render_memory_chains(snapshot: RunSnapshot) -> None:
    st.subheader(
        "Re-prompts, repairs & failure chains",
        help="Sources: research_memory.jsonl (role_retry / controller_error / "
        "eda_error) and debugger_memory.jsonl.",
    )
    events = snapshot.memory_events
    # Debugger interventions live in their own journal; without them the
    # "clean record" claim below would contradict the spend table beside it.
    repair_events = tuple(
        event
        for event in snapshot.debugger_events
        if event.event_type not in {"role_retry", "controller_error", "eda_error"}
    )
    if not events and not repair_events:
        _note(
            "No re-prompts, repairs, or controller faults on the record — every role "
            "answered validly on the first ask.",
            "ok",
        )
        return
    by_iteration: dict[int, list] = {}
    for event in events:
        by_iteration.setdefault(event.iteration, []).append(("memory", event))
    for event in repair_events:
        by_iteration.setdefault(event.iteration, []).append(("repair", event))
    for iteration in sorted(by_iteration, reverse=True):
        group = by_iteration[iteration]
        st.markdown(f"**Iteration {iteration:03d}** · {len(group)} event(s)")
        for source, event in group:
            if source == "repair":
                head = f"DEBUGGER · {event.event_type}" + (
                    f" — {event.error_type}" if event.error_type else ""
                )
                cls = "chain repair"
                body_text = event.error or "(no error text recorded)"
                if event.lesson:
                    body_text = f"{body_text}\n→ {event.lesson}"
            elif event.kind == "role_retry":
                head = (
                    f"RE-PROMPT #{event.reprompt} · {event.label}"
                    + (f" — {event.error_type}" if event.error_type else "")
                )
                cls = "chain"
                body_text = event.error or "(no error text recorded)"
            elif event.kind == "controller_error":
                head = f"CONTROLLER FAULT · {event.label}" + (
                    f" — {event.error_type}" if event.error_type else ""
                )
                cls = "chain error"
                body_text = event.error or "(no error text recorded)"
            else:
                head = "EDA FAULT" + (f" — {event.error_type}" if event.error_type else "")
                cls = "chain"
                body_text = event.error or "(no error text recorded)"
            st.markdown(
                f'<div class="{cls}"><div class="ck">{html.escape(head)}</div>'
                f'<div class="cbody">{html.escape(body_text)}</div></div>',
                unsafe_allow_html=True,
            )


_TRACE_PAGE_SIZE = 100


def _agents(snapshot: RunSnapshot) -> None:
    st.subheader(
        "Every model call on the record",
        help="Source: passes/*.json — one file per LLM call, written "
        "atomically by the harness.",
    )
    calls = snapshot.llm_calls
    if not calls:
        _empty(
            "No pass artifacts recorded yet. The first role call writes "
            "passes/001_researcher_0.json and this console fills in."
        )
        _render_memory_chains(snapshot)
        return

    _agents_tiles(snapshot)

    st.markdown("#### Token burn-down by role family")
    chart = _burn_down_chart(snapshot)
    if chart is not None:
        st.altair_chart(chart, width="stretch")
        if snapshot.token_cap:
            st.caption(
                f"Dashed rule: configured cap of {snapshot.token_cap:,} total tokens "
                "(llm.max_total_tokens)."
            )
        else:
            st.caption("No token cap recorded in this run's artifacts (run_config.json absent).")

    summary_rows = _role_summary_rows(snapshot)
    if summary_rows:
        with st.expander("Spend & latency by role family", expanded=False):
            st.dataframe(summary_rows, width="stretch", hide_index=True)

    st.markdown("#### Call trace")
    total_calls = len(calls)
    if total_calls > _TRACE_PAGE_SIZE:
        page_count = math.ceil(total_calls / _TRACE_PAGE_SIZE)

        def _page_label(page: int) -> str:
            start = page * _TRACE_PAGE_SIZE
            end = min(start + _TRACE_PAGE_SIZE, total_calls)
            return f"calls {start + 1}–{end} of {total_calls}"

        page_index = st.selectbox(
            "Trace page",
            list(range(page_count)),
            index=page_count - 1,
            format_func=_page_label,
            key="trace_page",
        )
        page_index = min(int(page_index), page_count - 1)
        window = calls[page_index * _TRACE_PAGE_SIZE : page_index * _TRACE_PAGE_SIZE + _TRACE_PAGE_SIZE]
    else:
        window = calls
    st.dataframe(_trace_rows(window), width="stretch", hide_index=True)

    _render_call_inspector(calls)
    st.divider()
    _render_memory_chains(snapshot)


# ---------------------------------------------------------------------------
# EDA tab
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, max_entries=8)
def _profile_cached(path_str: str, fingerprint: int) -> dict[str, Any] | None:
    try:
        value = json.loads(Path(path_str).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _eda(config, snapshot: RunSnapshot) -> None:
    st.subheader("Dataset card & run EDA")
    if snapshot.data_card_markdown:
        _source_line("DATA_CARD.md · written by the harness at baseline adoption")
        with st.expander("Dataset card — splits, label rates, tab breakdown", expanded=False):
            st.markdown(snapshot.data_card_markdown)
    else:
        _empty("No DATA_CARD.md recorded for this run yet — it is written when the baseline is adopted.")

    if snapshot.live_eda:
        live = snapshot.live_eda
        st.markdown("### Live EDA (current run)")
        st.caption(f"Status: **{live.status}**")
        if live.plan:
            with st.expander(f"Active EDA plan (iteration {live.iteration:03d})", expanded=True):
                st.markdown(f"**Objective:** {live.plan.get('objective', '')}")
                if live.plan.get("questions"):
                    st.markdown("*Questions:*")
                    for q in live.plan["questions"]:
                        st.markdown(f"- {q}")
                if live.plan.get("feature_hypotheses"):
                    st.markdown("*Feature hypotheses:*")
                    for h in live.plan["feature_hypotheses"]:
                        st.markdown(f"- {h}")
                if live.plan.get("leakage_risks"):
                    st.markdown("*Leakage risks:*")
                    for r in live.plan["leakage_risks"]:
                        st.markdown(f"- ⚠ {r}")
        if live.report:
            with st.expander(f"Active EDA report findings (iteration {live.iteration:03d})", expanded=True):
                if live.report.get("summary"):
                    st.info(live.report["summary"])
                if live.report.get("findings"):
                    st.markdown("#### Empirical findings")
                    st.dataframe(list(live.report["findings"]), width="stretch", hide_index=True)
                if live.report.get("feature_candidates"):
                    st.markdown("#### Proposed features")
                    st.dataframe(list(live.report["feature_candidates"]), width="stretch", hide_index=True)
                if live.report.get("recommended_next_focus"):
                    st.success(f"**Recommended focus:** {live.report['recommended_next_focus']}")
        st.divider()

    if snapshot.eda_artifacts:
        latest = snapshot.eda_artifacts[-1]
        st.caption(f"Latest finalized EDA artifact: `{latest.path.relative_to(snapshot.path)}`")
        if latest.status != "completed":
            st.warning(f"EDA artifact status: {latest.status}")
        if latest.error:
            st.error(latest.error)
        if latest.summary:
            st.write(latest.summary)
        if latest.findings:
            st.markdown("#### Findings")
            st.dataframe(list(latest.findings), width="stretch", hide_index=True)
        if latest.feature_candidates:
            st.markdown("#### Feature candidates")
            st.dataframe(list(latest.feature_candidates), width="stretch", hide_index=True)
        with st.expander("All EDA artifacts", expanded=False):
            for artifact in snapshot.eda_artifacts:
                st.markdown(f"**Iteration {artifact.iteration:03d}**")
                if artifact.summary:
                    st.caption(artifact.summary)
                st.json(artifact.raw, expanded=False)
    elif not snapshot.live_eda:
        _empty("No autonomous EDA artifacts recorded for this run yet.")

    st.divider()
    st.subheader("Trusted train/validation profile")
    if not config.eda_profile_path.is_file():
        st.markdown(
            '<div class="empty-panel">No aggregate EDA profile exists. Generate it explicitly with '
            '<code>python -m src.ui.profile_data --config configs/ui.json</code>.</div>',
            unsafe_allow_html=True,
        )
        return
    try:
        fingerprint = config.eda_profile_path.stat().st_mtime_ns
    except OSError:
        fingerprint = 0
    profile = _profile_cached(str(config.eda_profile_path), fingerprint)
    if profile is None:
        st.markdown(
            '<div class="note-panel warn">The aggregate EDA profile could not be read this refresh.</div>',
            unsafe_allow_html=True,
        )
        return
    st.caption(profile.get("provenance", "Trusted train/validation aggregates only."))
    splits = profile.get("splits", {})
    st.dataframe(
        [
            {
                "split": name,
                "rows": value.get("rows"),
                "users": value.get("users"),
                "positive_rate": value.get("positive_rate"),
            }
            for name, value in splits.items()
        ],
        width="stretch",
        hide_index=True,
    )

    activity = profile.get("activity_by_date", [])
    if activity:
        st.markdown("#### Temporal interaction & label trends")
        df_act = pd.DataFrame(activity)
        required = {"date", "split", "rows", "positive_rate"}
        if required.issubset(df_act.columns):
            df_act["date_str"] = df_act["date"].astype(str)
            daily_rows = df_act.pivot_table(
                index="date_str", columns="split", values="rows", aggfunc="sum"
            )
            daily_rates = df_act.pivot_table(
                index="date_str", columns="split", values="positive_rate", aggfunc="mean"
            )
            left, right = st.columns(2)
            left.markdown("**Daily interaction rows**")
            left.bar_chart(daily_rows, color=_series_colors(len(daily_rows.columns)))
            right.markdown("**Daily long-view rate**")
            right.line_chart(daily_rates, color=_series_colors(len(daily_rates.columns)))

    durations = profile.get("duration_histogram", [])
    if durations:
        st.markdown("#### Video duration distribution (quantile buckets)")
        df_dur = pd.DataFrame(durations)
        if "seconds" in df_dur.columns and "rows" in df_dur.columns:
            st.bar_chart(df_dur.set_index("seconds")["rows"], color=CHART_PRIMARY)


# ---------------------------------------------------------------------------
# Feature Lab tab
# ---------------------------------------------------------------------------


def _feature_lab(snapshot: RunSnapshot) -> None:
    st.subheader("Leakage-safe feature lineage & catalog")

    live_proposals = []
    if snapshot.live_eda and snapshot.live_eda.feature_candidates:
        for f in snapshot.live_eda.feature_candidates:
            live_proposals.append({"source": f"Live EDA (iter {snapshot.live_eda.iteration:03d})", **f})
    for rp in snapshot.live_role_passes:
        if rp.role == "eda_builder" and rp.data.get("feature_candidates"):
            for f in rp.data["feature_candidates"]:
                live_proposals.append({"source": f"Live builder pass {rp.sequence + 1}", **f})

    if live_proposals:
        st.markdown("### Live feature proposals")
        st.dataframe(live_proposals, width="stretch", hide_index=True)
        st.divider()

    fields = [
        ("user_id", "interaction log", "train-fitted vocabulary", "train only", "Categorical ID", "FM, BPR, group-softmax"),
        ("video_id", "interaction log", "train-fitted vocabulary", "train only", "Categorical ID", "FM, BPR, group-softmax"),
        ("author_id", "video metadata", "train-fitted vocabulary", "train only", "Categorical ID", "FM, BPR, group-softmax"),
        ("tab", "interaction log", "train-fitted vocabulary", "train only", "Feed Context", "FM, BPR, group-softmax"),
        ("dur_bucket", "duration_ms", "train-fitted quantile bucket", "train only", "Continuous Bucketed", "FM, BPR, group-softmax"),
    ]
    st.dataframe(
        [
            {
                "field": a,
                "source": b,
                "transformation": c,
                "fit split": d,
                "domain": e,
                "consumers": f,
            }
            for a, b, c, d, e, f in fields
        ],
        width="stretch",
        hide_index=True,
    )

    generated = []
    for artifact in snapshot.eda_artifacts:
        for feature in artifact.feature_candidates:
            generated.append({"iteration": artifact.iteration, **feature})
    if generated:
        st.markdown("#### EDA-generated feature candidates")
        st.dataframe(generated, width="stretch", hide_index=True)
    elif not live_proposals:
        _empty("Feature families appear here only after trusted run metadata logs them.")


# ---------------------------------------------------------------------------
# Iterations tab
# ---------------------------------------------------------------------------


def _fmt_param(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _iteration_overview_rows(snapshot: RunSnapshot) -> list[dict[str, Any]]:
    rows = []
    for item in snapshot.iterations:
        metrics = item.metrics or {}
        primary = metrics.get("primary")
        params = item.parameters or {}
        rows.append(
            {
                "iter": item.iteration,
                "experiment": item.experiment_id,
                "family": item.family,
                "action": item.action,
                "k": _fmt_param(params.get("k")),
                "l2": _fmt_param(params.get("l2")),
                "lr": _fmt_param(params.get("learning_rate")),
                "primary": _fmt_metric(primary),
                "Δ run baseline": (
                    f"{primary - snapshot.baseline_primary:+.4f}" if primary is not None else "—"
                ),
                "repairs": item.repairs,
                "failure": item.failure_class or "",
                "status": item.status,
                "sec": f"{item.duration_seconds:.1f}" if isinstance(item.duration_seconds, (int, float)) else "—",
            }
        )
    return rows


def _render_role_passes(role_passes: tuple[RolePass, ...]) -> None:
    if not role_passes:
        st.caption("No role passes recorded for this iteration.")
        return
    for rp in role_passes:
        label = ROLE_LABELS.get(rp.role, rp.role.title())
        with st.expander(f"Pass {rp.sequence + 1}: {label} ({rp.model})", expanded=(rp.sequence == 0)):
            c1, c2, c3 = st.columns(3)
            c1.caption(f"**Model:** `{rp.model}`")
            c2.caption(f"**Latency:** `{rp.latency_seconds:.2f}s`")
            tot = rp.usage.get("total_tokens", 0)
            inp = rp.usage.get("input_tokens", 0)
            out = rp.usage.get("output_tokens", 0)
            c3.caption(f"**Tokens:** `{tot}` (`{inp}` in / `{out}` out)")

            if rp.data:
                st.markdown("**Structured decision:**")
                st.json(rp.data, expanded=True)

            if rp.sources:
                st.markdown("**Cited primary sources:**")
                for s in rp.sources:
                    t = s.get("title", "Source")
                    u = s.get("url")
                    st.markdown(f"- [{t}]({u})" if u else f"- {t}")


def _iteration_summary_rows(snapshot: RunSnapshot) -> list[dict[str, Any]]:
    """The glanceable ledger: what ran, what it scored, how it ended.

    Hyperparameters, repair counts, and timings stay one click away in the
    full table — a summary first, every recorded column preserved behind it."""
    keep = ("iter", "experiment", "family", "action", "primary", "Δ run baseline", "status")
    return [
        {key: row[key] for key in keep} for row in _iteration_overview_rows(snapshot)
    ]


def _iterations(snapshot: RunSnapshot) -> None:
    st.subheader(
        "Iteration ledger",
        help="Source: iterations.jsonl — one audited record per completed iteration.",
    )
    if not snapshot.iterations:
        _empty("No completed iteration records yet — the first record lands after train + validate.")
        return

    st.dataframe(_iteration_summary_rows(snapshot), width="stretch", hide_index=True)
    with st.expander("All recorded columns — parameters, repairs, timings", expanded=False):
        st.dataframe(_iteration_overview_rows(snapshot), width="stretch", hide_index=True)

    st.markdown("#### Iteration inspector")
    options = {f"{item.iteration:03d} · {item.experiment_id}": item for item in snapshot.iterations}
    selected = options[
        st.selectbox(
            "Select iteration", list(options), index=len(options) - 1, key="iteration_select"
        )
    ]
    role_passes = load_role_passes(snapshot.path, selected.iteration)
    candidate_code, candidate_tests = load_candidate_files(
        snapshot.path, selected.candidate_dir
    )

    left, right = st.columns([2, 1])
    left.markdown(f"### {selected.experiment_id}")
    left.write(f"**Hypothesis:** {selected.hypothesis or 'No hypothesis recorded.'}")
    right.metric("Status", selected.status.upper())
    if selected.failure_class:
        _note(f"Failure class: {selected.failure_class}", "fail")
        if selected.error:
            with st.expander("Recorded failure output", expanded=False):
                st.code(selected.error, language="text")
    if selected.repairs:
        _note(f"{selected.repairs} debugger repair(s) were needed before this outcome.", "warn")
    if selected.metrics:
        _metric_cards(
            RunSnapshot(
                run_id="",
                path=snapshot.path,
                status="",
                stop_reason=None,
                started_at=None,
                best_experiment_id=selected.experiment_id,
                best_metrics=selected.metrics,
                baseline_primary=snapshot.baseline_primary,
            ),
            snapshot.baseline_primary,
            reference_label="run baseline",
        )

    if role_passes:
        st.subheader("Role-pass sequence for this iteration")
        _render_role_passes(role_passes)

    with st.expander("Candidate source & test implementation", expanded=bool(candidate_code)):
        if candidate_code:
            st.markdown("**`candidate.py`:**")
            st.code(candidate_code, language="python", line_numbers=True)
            if candidate_tests:
                st.markdown("**`test_candidate.py`:**")
                st.code(candidate_tests, language="python", line_numbers=True)
        else:
            st.caption("No candidate source files located.")

    with st.expander("Changes & code diff", expanded=bool(selected.change_summary)):
        if selected.change_summary:
            st.dataframe(list(selected.change_summary.files), width="stretch", hide_index=True)
            patch = load_patch_text(snapshot.path, selected.change_summary.patch_path)
            if patch:
                st.code(patch, language="diff", line_numbers=True)
        else:
            st.caption(selected.raw.get("code_diff", "No generated-code change recorded."))

    with st.expander("Configuration parameters", expanded=False):
        st.json(selected.parameters)

    with st.expander("Agent notes", expanded=False):
        if selected.agent_notes:
            st.json(selected.agent_notes, expanded=False)
        else:
            reflection = selected.raw.get("reflection")
            _render_note(reflection or {})

    with st.expander("Full audited JSON record", expanded=False):
        st.json(selected.raw, expanded=False)


# ---------------------------------------------------------------------------
# Results tab
# ---------------------------------------------------------------------------


def _primary_trajectory_chart(
    snapshot: RunSnapshot, official: float
) -> alt.LayerChart | None:
    rows = [
        {"iteration": item.iteration, "primary": item.metrics.get("primary")}
        for item in snapshot.iterations
        if item.metrics and item.metrics.get("primary") is not None
    ]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    # Keep the official-baseline rule inside the visible domain, with a
    # little air, so the dashed reference can never fall off the chart.
    values = [row["primary"] for row in rows] + [official]
    span = max(values) - min(values)
    pad = max(span * 0.15, 0.0004)
    domain = [min(values) - pad, max(values) + pad]
    line = (
        alt.Chart(df)
        .mark_line(point={"size": 70}, strokeWidth=2, color=CHART_PRIMARY)
        .encode(
            x=alt.X("iteration:O", title="iteration", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "primary:Q",
                scale=alt.Scale(zero=False, domain=domain),
                title="primary = (GAUC + nDCG@5) / 2",
            ),
            tooltip=[
                alt.Tooltip("iteration:O", title="iteration"),
                alt.Tooltip("primary:Q", title="primary", format=".4f"),
            ],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"baseline": [official]}))
        .mark_rule(strokeDash=[5, 4], color=CHART_BASELINE, strokeWidth=1.5)
        .encode(
            y="baseline:Q",
            tooltip=[alt.Tooltip("baseline:Q", title="official baseline", format=".4f")],
        )
    )
    return (line + rule).properties(height=280)


def _render_baseline_provenance(snapshot: RunSnapshot) -> None:
    st.subheader(
        "Baseline provenance",
        help="Source: baseline_selection.json — which prior run seeded the "
        "baseline, and why others were not.",
    )
    selection = snapshot.baseline_selection
    if not selection:
        _empty(
            "No baseline_selection.json recorded — the baseline was measured in-run "
            "or seeded directly from configuration."
        )
        return
    adopted = selection.get("selected")
    skipped = selection.get("skipped") or []
    if adopted:
        _note(f"Adopted baseline artifact: {adopted}", "ok")
    else:
        _note("No prior run was adopted; the baseline was re-measured.", "warn")
    if skipped:
        with st.expander(f"Skipped candidates ({len(skipped)}) — every rejection has a reason", expanded=False):
            st.dataframe(
                [
                    {
                        "skipped run": item.get("path", "?"),
                        "reason": item.get("reason", "?"),
                    }
                    for item in skipped
                    if isinstance(item, dict)
                ],
                width="stretch",
                hide_index=True,
            )
    else:
        st.caption("No candidates were skipped.")


def _render_interventions(snapshot: RunSnapshot) -> None:
    st.subheader(
        "Manual interventions ledger",
        help="Source: interventions.jsonl — the falsifiable autonomy evidence "
        "a judge can read.",
    )
    if not snapshot.interventions_recorded:
        reported = _resource_metrics(snapshot)["manual_interventions"]
        _empty(
            f"No interventions ledger file exists for this run; state reports "
            f"{reported} manual intervention(s)."
        )
        return
    if not snapshot.interventions:
        _note(
            "The ledger exists and is empty — zero manual interventions. "
            "The loop ran untouched.",
            "ok",
        )
        return
    _note(
        f"{len(snapshot.interventions)} manual intervention(s) on the record.",
        "warn",
    )
    for index, entry in enumerate(snapshot.interventions, start=1):
        with st.expander(f"Intervention {index}", expanded=False):
            st.json(entry, expanded=True)


@st.cache_data(show_spinner=False, max_entries=4)
def _validate_submission_cached(payload: bytes):
    return validate_submission(payload)


def _evidence_trust(snapshot: RunSnapshot) -> None:
    """The receipts a judge checks: gate, provenance, interventions, spend,
    and the loop's own journal — strongest evidence first."""
    if snapshot.gate_info:
        st.subheader(
            "Official gate & submission status",
            help="Source: the harness gate record in summary.json.",
        )
        gate = snapshot.gate_info
        details = gate.get("details") or {}
        gate_status = str(gate.get("status", "unknown"))
        if gate_status == "error":
            st.metric("Gate status", gate_status.upper())
            st.error(f"Gate failed: {details.get('reason', 'unknown_reason')}")
            diagnostics = {
                key: details[key]
                for key in ("error", "got_rows", "expected_rows", "searched")
                if key in details
            }
            if diagnostics:
                st.json(diagnostics, expanded=False)
        else:
            g1, g2, g3 = st.columns(3)
            g1.metric("Gate status", gate_status.upper())
            gate_rows = details.get("rows")
            g2.metric("Submission rows", f"{gate_rows:,}" if isinstance(gate_rows, int) else "—")
            checked_with = details.get("checked_with")
            verified_with = (
                str(checked_with).replace("\\", "/").split("/")[-1]
                if checked_with
                else "—"
            )
            g3.metric("Verified with", verified_with)
            if "sha256" in details:
                st.caption(f"**SHA256:** `{details['sha256']}`")
            if "check_stdout" in details:
                st.success(details["check_stdout"])
        st.divider()

    _render_baseline_provenance(snapshot)
    st.divider()
    _render_interventions(snapshot)

    st.divider()
    st.subheader(
        "Telemetry & resource breakdown",
        help="Source: resources.json — tokens, wall clock, attempts.",
    )
    usage = _resource_metrics(snapshot)
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Total tokens", f"{usage['total_tokens']:,}")
    t2.metric("Input / output", f"{usage['input_tokens']:,} / {usage['output_tokens']:,}")
    t3.metric("Wall clock", _fmt_duration(_wall_clock_seconds(snapshot)))
    t4.metric("GPU hours / interventions", f"{usage['gpu_hours']}h / {usage['manual_interventions']}")

    if snapshot.journal_markdown or snapshot.results_markdown:
        st.divider()
        st.subheader("Autonomous research journal")
        with st.expander("View journal.md", expanded=bool(snapshot.journal_markdown)):
            if snapshot.journal_markdown:
                _render_journal_markdown(snapshot.journal_markdown)
            else:
                st.caption("journal.md not rendered yet.")
        with st.expander("View results.md", expanded=False):
            if snapshot.results_markdown:
                st.markdown(snapshot.results_markdown)
            else:
                st.caption("results.md not rendered yet.")

    st.divider()
    with st.expander("Local submission schema check — validate a prediction CSV", expanded=False):
        _render_submission_checker()


def _render_submission_checker() -> None:
    uploaded = st.file_uploader(
        "Preview and validate a prediction CSV", type=["csv"], key="submission_csv"
    )
    if uploaded is not None:
        payload = uploaded.getvalue()
        check = _validate_submission_cached(payload)
        (st.success if check.valid else st.error)(
            f"{'Schema checks passed' if check.valid else 'Schema checks failed'} · {check.row_count:,} rows · "
            f"{check.duplicate_pairs} duplicate user-video pairs"
        )
        for error in check.errors[:10]:
            st.error(error)
        if len(check.errors) > 10:
            st.caption(f"…and {len(check.errors) - 10} further error(s).")
        for warning in check.warnings:
            st.warning(warning)


def _evidence(config, snapshot: RunSnapshot) -> None:
    section = _section(
        "evidence", "Evidence section", ("Trust & audit", "Dataset", "Features")
    )
    if section == "Dataset":
        _eda(config, snapshot)
    elif section == "Features":
        _feature_lab(snapshot)
    else:
        _evidence_trust(snapshot)


# ---------------------------------------------------------------------------
# Compare tab
# ---------------------------------------------------------------------------


def _rule_verdict(snapshot: RunSnapshot) -> str:
    if snapshot.converged_official is True:
        fired_at = snapshot.converged_official_iteration
        return f"fired @ {fired_at}" if fired_at else "fired"
    if snapshot.converged_official is False:
        return "never fired"
    return "pending"


def _compare(runs: list[RunSnapshot], official: float, stale_after: int = 1200) -> None:
    st.subheader(
        "Run comparison",
        help="Sources: summary.json, state.json, and resources.json across "
        "every discovered run.",
    )
    if not runs:
        _empty("No runs discovered under the configured run root.")
        return
    labels = [run.run_id for run in runs]
    chosen = st.multiselect(
        "Runs to compare",
        labels,
        default=labels[: min(len(labels), 6)],
        key="compare_runs",
    )
    selected = [run for run in runs if run.run_id in chosen]
    if not selected:
        _empty("Pick at least one run to compare.")
        return
    rows = []
    for run in selected:
        usage = _resource_metrics(run)
        best_primary = (run.best_metrics or {}).get("primary")
        stale_age = _stale_age(run, stale_after)
        status = run.status
        if stale_age is not None:
            status = f"running? (no activity for {_fmt_duration(stale_age)})"
        rows.append(
            {
                "run": run.run_id,
                "status": status,
                "stop reason": (run.stop_reason or "—").replace("_", " "),
                "official rule": _rule_verdict(run),
                "best primary": _fmt_metric(best_primary),
                "Δ vs official": (
                    f"{best_primary - official:+.4f}" if best_primary is not None else "—"
                ),
                "max scored": _fmt_metric(run.max_scored_primary),
                "replicated": f"n={run.best_replicated['n']}" if run.best_replicated else "no",
                "iterations": usage["iteration_count"],
                "tokens": usage["total_tokens"],
                "wall clock": _fmt_duration(usage["wall_clock_seconds"]),
                "interventions": (
                    str(len(run.interventions)) if run.interventions_recorded else "no ledger"
                ),
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption(
        f"Official published validation baseline: {official:.4f}. “Best primary” is the "
        "margin-gated claim; “max scored” is the raw maximum the run measured; "
        "“no ledger” means the run recorded no interventions.jsonl file."
    )


# ---------------------------------------------------------------------------
# Judge view — the one-page honest story
# ---------------------------------------------------------------------------


def _judge_view(snapshot: RunSnapshot, official: float, stale_after: int = 1200) -> None:
    _verdict_ledger(snapshot, stale_after)

    st.markdown("#### Primary score per iteration")
    chart = _primary_trajectory_chart(snapshot, official)
    if chart is not None:
        st.altair_chart(chart, width="stretch")
        st.caption(f"Dashed rule: official baseline {official:.4f}.")
    else:
        _empty("No scored iterations yet.")

    # Full width: squeezed into a side column the node labels become
    # unreadable, and a judge page that cannot be read is a broken page.
    st.markdown("#### Experiment lineage")
    if snapshot.nodes:
        _render_experiment_dag(snapshot.nodes, snapshot.best_experiment_id)
    else:
        _empty("No experiment tree recorded yet.")

    st.markdown("#### Iteration ledger")
    if snapshot.iterations:
        st.dataframe(_iteration_overview_rows(snapshot), width="stretch", hide_index=True)
    else:
        _empty("No completed iterations yet.")

    st.markdown("#### Agent spend")
    if snapshot.llm_calls:
        _agents_tiles(snapshot)
        summary_rows = _role_summary_rows(snapshot)
        if summary_rows:
            st.dataframe(summary_rows, width="stretch", hide_index=True)
    else:
        _empty("No model calls recorded yet.")

    provenance_bits = []
    selection = snapshot.baseline_selection or {}
    if selection.get("selected"):
        skipped = len(selection.get("skipped") or [])
        provenance_bits.append(
            f"Baseline adopted from {selection['selected']} ({skipped} candidate(s) skipped with recorded reasons)."
        )
    if snapshot.interventions_recorded:
        provenance_bits.append(
            f"Interventions ledger: {len(snapshot.interventions)} entr"
            + ("y" if len(snapshot.interventions) == 1 else "ies")
            + " on the record."
        )
    gate = snapshot.gate_info or {}
    if gate:
        provenance_bits.append(f"Official gate: {str(gate.get('status', 'unknown')).upper()}.")
    if provenance_bits:
        st.markdown("#### Provenance")
        for bit in provenance_bits:
            _note(bit)
    st.caption(
        f"Read-only view generated from runs/{snapshot.run_id} — the dashboard never "
        "launches, resumes, cancels, or edits an experiment."
    )


# ---------------------------------------------------------------------------
# Persistent header & main
#
# One compact masthead carries the whole context — run picker, health chip,
# budget chip — so no destination has to repeat it.
# ---------------------------------------------------------------------------


def _status_chip(snapshot: RunSnapshot, stale_after: int) -> str:
    stale_age = _stale_age(snapshot, stale_after)
    if snapshot.status == "running" and stale_age is not None:
        return (
            f'<span class="chip warn">● STALLED? {html.escape(_fmt_duration(stale_age))} '
            "quiet</span>"
        )
    if snapshot.status == "running":
        return '<span class="chip live">● LIVE</span>'
    if snapshot.status == "failed":
        return '<span class="chip fail">FAILED</span>'
    return f'<span class="chip">{html.escape(snapshot.status.upper() or "UNKNOWN")}</span>'


def _budget_chip(snapshot: RunSnapshot, stale_after: int) -> str:
    entries = _budget_entries(snapshot, stale_after)
    if not entries:
        return '<span class="chip">BUDGET · NO CAPS ON RECORD</span>'
    _, _, _, tight_frac = max(entries, key=lambda entry: entry[3])
    left = min(1.0, max(0.0, 1.0 - tight_frac))
    tone = "" if left > 0.15 else (" warn" if left > 0.0 else " fail")
    return f'<span class="chip{tone}">BUDGET · {left * 100:.0f}% LEFT</span>'


def _mast_title() -> None:
    st.markdown(
        '''<div>
<div class="eyebrow">TechJam 2026 · KuaiRand-Pure · read-only observatory</div>
<h1 class="mast-title">Run Ledger</h1>
<div class="mast-sub">An autonomous research loop under audit — every claim margin-gated,
every model call, repair, and intervention on the record.</div>
</div>''',
        unsafe_allow_html=True,
    )


def _mast_rule() -> None:
    st.markdown('<div class="mast-rule"></div>', unsafe_allow_html=True)


def _header_plain() -> None:
    _mast_title()
    _mast_rule()


def _header(runs: list[RunSnapshot], config) -> RunSnapshot:
    """Masthead row: identity on the left; run picker + live chips on the
    right. Returns the selected run's fresh snapshot."""
    # Option values are bare run ids so a status flip ("running" → "completed")
    # cannot reset the viewer's selection mid-read.
    by_id = {run.run_id: run for run in runs}
    left, right = st.columns([2.9, 1.25], vertical_alignment="bottom")
    with left:
        _mast_title()
    with right:
        selected_id = st.selectbox(
            "Run",
            list(by_id),
            key="run_select",
            label_visibility="collapsed",
            help="Every run directory discovered under the configured run root. "
            "The dashboard never launches, resumes, cancels, or changes an experiment.",
        )
        snapshot = _snapshot(by_id[selected_id].path, config.official_baseline)
        st.markdown(
            '<div class="chip-row">'
            + _status_chip(snapshot, config.stale_after_seconds)
            + _budget_chip(snapshot, config.stale_after_seconds)
            + "</div>",
            unsafe_allow_html=True,
        )
    _mast_rule()
    return snapshot


def main() -> None:
    st.set_page_config(
        page_title="Run Ledger · ML Research Observatory",
        page_icon="◉",
        layout="wide",
    )
    _css()
    try:
        config = load_dashboard_config(CONFIG_PATH)
        runs = discover_runs(config.run_root, config.official_baseline)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _header_plain()
        st.error(f"Dashboard configuration error: {exc}")
        return
    if not runs:
        _header_plain()
        st.info("No run artifacts were found under the configured run root.")
        return

    initial = _header(runs, config)
    selected_path = initial.path
    official = config.official_baseline
    stale_after = config.stale_after_seconds

    def fresh() -> RunSnapshot:
        return _snapshot(selected_path, official)

    panels: list[tuple[str, str, Callable[[], None]]] = [
        ("Story", "story", lambda: _story(fresh(), official, stale_after)),
        ("Activity", "activity", lambda: _activity(fresh(), stale_after)),
        ("Evidence", "evidence", lambda: _evidence(config, fresh())),
        (
            "Compare",
            "compare",
            lambda: _compare(
                discover_runs(config.run_root, official), official, stale_after
            ),
        ),
        ("Judge sheet", "judge sheet", lambda: _judge_view(fresh(), official, stale_after)),
    ]

    tabs = st.tabs([label for label, _, _ in panels])
    running = initial.status == "running"
    if running:

        @st.fragment(run_every=f"{config.active_refresh_seconds}s")
        def _live_panel(panel: str, render: Callable[[], None]) -> None:
            _guard(render, panel)

    for tab, (_, panel, render) in zip(tabs, panels):
        with tab:
            if running:
                _live_panel(panel, render)
            else:
                _guard(render, panel)


if __name__ == "__main__":
    main()
