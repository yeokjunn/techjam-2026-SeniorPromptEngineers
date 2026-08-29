from __future__ import annotations

import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Streamlit adds the entry-point directory to sys.path. When this nested file is
# launched directly, make the repository package root explicit before importing
# project modules. This also makes absolute-path launches independent of cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st

from src.agent.activity import STAGE_ORDER
from src.ui.loaders import (
    activity_age_seconds,
    discover_runs,
    load_dashboard_config,
    load_patch_text,
    load_run_snapshot,
    validate_submission,
)
from src.ui.models import IterationSnapshot, RolePass, RunSnapshot, StageTransition


CONFIG_PATH = REPO_ROOT / "configs" / "ui.json"

STAGE_LABELS = {
    "initializing": "Initialize",
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


def _css() -> None:
    st.markdown(
        """
<style>
:root { --ink:#252827; --muted:#66706b; --blue:#dcecf7; --mint:#dff1e8; --amber:#f7e8bf; --red:#f4d9d7; }
.stApp { background: #fbfaf6; color: var(--ink); }
[data-testid="stSidebar"] { background: #f2f5f1; }
.block-container { max-width: 1220px; padding-top: 2.2rem; }
.eyebrow { color: #62736b; font-size: .78rem; letter-spacing: .12em; text-transform: uppercase; }
.live-overlay {
  background: linear-gradient(135deg, rgba(220,236,247,.80), rgba(255,255,255,.70));
  border: 1px solid rgba(72,105,124,.22); border-radius: 18px; padding: 1.15rem 1.3rem;
  box-shadow: 0 14px 34px rgba(38,62,57,.09); backdrop-filter: blur(12px); margin:.5rem 0 1rem;
}
.live-title { font-size: 1.18rem; font-weight: 650; margin:.2rem 0; }
.live-meta { color:var(--muted); font-size:.92rem; }
.pulse { display:inline-block; width:.62rem; height:.62rem; border-radius:50%; background:#27845e;
  box-shadow:0 0 0 rgba(39,132,94,.45); animation:pulse 1.8s infinite; margin-right:.45rem; }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(39,132,94,.42)} 70%{box-shadow:0 0 0 9px rgba(39,132,94,0)} 100%{box-shadow:0 0 0 0 rgba(39,132,94,0)} }
.stage-row { display:flex; flex-wrap:wrap; gap:.42rem; margin:.8rem 0 1rem; }
.stage { padding:.38rem .66rem; border-radius:999px; border:1px solid #d8ddd8; color:#7b827e; font-size:.77rem; background:rgba(255,255,255,.58); }
.stage.done { color:#276247; background:var(--mint); border-color:#bfdcca; }
.stage.active { color:#264d64; background:#c8e3f2; border-color:#8fbdd6; font-weight:650; }
.stage.failed { color:#843c36; background:var(--red); border-color:#e5b8b4; }
.metric-note { color:var(--muted); font-size:.82rem; }
.empty-panel { border:1px dashed #cbd2cc; border-radius:14px; padding:1.2rem; color:var(--muted); background:rgba(255,255,255,.45); }
.pass-card { border: 1px solid #dce4de; border-radius: 12px; padding: 0.9rem 1.1rem; background: #ffffff; margin-bottom: 0.8rem; }
.badge { display: inline-block; padding: 0.2rem 0.55rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
.badge-success { background: #dff1e8; color: #276247; }
.badge-info { background: #dcecf7; color: #264d64; }
.badge-warning { background: #f7e8bf; color: #73510d; }
.badge-danger { background: #f4d9d7; color: #843c36; }
</style>
""",
        unsafe_allow_html=True,
    )


def _parse_time(value: str) -> datetime | None:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result.replace(tzinfo=result.tzinfo or timezone.utc)
    except (TypeError, ValueError):
        return None


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


def _stage_strip(snapshot: RunSnapshot, activity: StageTransition | None) -> str:
    iteration = activity.iteration if activity else max((item.iteration for item in snapshot.iterations), default=0)
    latest: dict[str, str] = {}
    for event in snapshot.transitions:
        if event.iteration == iteration:
            latest[event.stage] = event.status
    pills = []
    for stage in STAGE_ORDER:
        status = latest.get(stage, "")
        class_name = "stage"
        if activity and activity.stage == stage and activity.status == "active":
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


def _render_live_overlay(snapshot: RunSnapshot, stale_after: int) -> None:
    activity = snapshot.activity
    if activity is None:
        st.markdown(
            '<div class="empty-panel">No live activity artifact exists for this run. '
            "Completed iteration data remains available below.</div>",
            unsafe_allow_html=True,
        )
        return
    is_live = activity.status == "active" and snapshot.status == "running"
    age = activity_age_seconds(activity)
    stale = is_live and age is not None and age > stale_after
    stage_label = STAGE_LABELS.get(activity.stage, activity.stage.replace("_", " ").title())
    marker = '<span class="pulse"></span>' if is_live and not stale else ""
    state_label = "Possibly stale" if stale else ("Active" if is_live else activity.status.title())
    experiment = f" · {html.escape(activity.experiment_id)}" if activity.experiment_id else ""
    elapsed = _elapsed_label(activity) if is_live else "stage recorded"
    st.markdown(
        f'''<div class="live-overlay">
<div class="eyebrow">Live execution overlay</div>
<div class="live-title">{marker}Iteration {activity.iteration} · {html.escape(stage_label)} · {state_label}</div>
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
    with st.expander("Agent Notes — structured decision trace", expanded=True):
        _render_note(activity.agent_note)
        st.caption("Summarized decisions only; raw hidden reasoning and full prompts are not displayed.")
    with st.expander("Changes", expanded=bool(activity.change_summary)):
        changes = activity.change_summary
        if changes is None:
            st.caption("No finalized candidate change is attached to this stage yet.")
        else:
            st.write(f"**+{changes.lines_added} / −{changes.lines_deleted} lines**")
            st.dataframe(list(changes.files), width="stretch", hide_index=True)
            patch = load_patch_text(snapshot.path, changes.patch_path)
            if patch:
                st.code(patch, language="diff", line_numbers=True)
    with st.expander("Errors & repairs", expanded=bool(activity.error)):
        if activity.error:
            st.error(activity.error)
        if activity.repair:
            st.info(activity.repair)
        if not activity.error and not activity.repair:
            st.caption("No error or repair is attached to the current stage.")
    with st.expander("Recent timeline"):
        recent = list(snapshot.transitions)[-10:][::-1]
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


def _metric_cards(snapshot: RunSnapshot, official: float) -> None:
    metrics = snapshot.best_metrics or {}
    columns = st.columns(4)
    columns[0].metric("GAUC", f"{metrics.get('GAUC', float('nan')):.4f}" if "GAUC" in metrics else "—")
    columns[1].metric("nDCG@5", f"{metrics.get('nDCG@5', float('nan')):.4f}" if "nDCG@5" in metrics else "—")
    primary = metrics.get("primary")
    columns[2].metric("Primary Score", f"{primary:.4f}" if primary is not None else "—")
    columns[3].metric(
        "Δ vs official 0.6016",
        f"{primary - official:+.4f}" if primary is not None else "—",
    )


def _render_budget_gauges(snapshot: RunSnapshot) -> None:
    resources = snapshot.resources or {}
    config = snapshot.run_config or {}
    budgets = config.get("budgets", {})
    max_iters = budgets.get("max_iterations", 50)
    max_seconds = budgets.get("max_wall_clock_seconds", 21600)
    max_tokens = (config.get("llm") or {}).get("max_total_tokens", 150000)

    token_usage = resources.get("token_usage") or {}
    total_tokens = token_usage.get("total_tokens", 0)
    wall_seconds = resources.get("wall_clock_seconds", 0.0)
    training_attempts = resources.get("training_attempts", len(snapshot.iterations))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Iterations Budget", f"{len(snapshot.iterations)} / {max_iters}")
    c2.metric("Training Attempts", f"{training_attempts} / {max_iters}")
    c3.metric("Wall Clock Time", f"{wall_seconds:.1f}s / {max_seconds}s")
    c4.metric("LLM Token Budget", f"{total_tokens:,} / {max_tokens:,}")


def _render_experiment_dag(nodes: tuple[dict[str, Any], ...], best_id: str | None) -> None:
    if not nodes:
        return
    mermaid_lines = [
        "graph TD",
        "  classDef best fill:#dff1e8,stroke:#27845e,stroke-width:2px,color:#1e4d38;",
        "  classDef normal fill:#f4f8fb,stroke:#8fbdd6,stroke-width:1px,color:#264d64;",
        "  classDef failed fill:#f4d9d7,stroke:#843c36,stroke-width:1px,color:#662520;",
    ]
    for node in nodes:
        eid = str(node.get("experiment_id", "unknown"))
        safe_id = re.sub(r"[^a-zA-Z0-9_]", "_", eid)
        family = node.get("family", "")
        status = node.get("status", "")
        m = node.get("metrics") or {}
        p = m.get("primary")
        score_str = f"P: {p:.4f}" if isinstance(p, (int, float)) else status
        label = f"<b>{eid}</b><br/>[{family}] · {score_str}"
        parent = node.get("parent_experiment")
        if parent:
            safe_parent = re.sub(r"[^a-zA-Z0-9_]", "_", str(parent))
            mermaid_lines.append(f'  {safe_parent} --> {safe_id}["{label}"]')
        else:
            mermaid_lines.append(f'  {safe_id}["{label}"]')

        if eid == best_id:
            mermaid_lines.append(f"  class {safe_id} best;")
        elif status == "failed":
            mermaid_lines.append(f"  class {safe_id} failed;")
        else:
            mermaid_lines.append(f"  class {safe_id} normal;")

    st.markdown("```mermaid\n" + "\n".join(mermaid_lines) + "\n```")


def _pipeline(snapshot: RunSnapshot, stale_after: int, official: float) -> None:
    _render_live_overlay(snapshot, stale_after)
    _metric_cards(snapshot, official)
    _render_budget_gauges(snapshot)

    st.caption(
        f"Run status: **{snapshot.status}** · stop reason: **{snapshot.stop_reason or 'not set'}** · "
        f"validation best: **{snapshot.best_experiment_id or 'not available'}**"
    )
    if snapshot.warnings:
        for warning in snapshot.warnings:
            st.warning(warning)

    st.subheader("Visual Experiment Lineage (DAG)")
    if snapshot.nodes:
        _render_experiment_dag(snapshot.nodes, snapshot.best_experiment_id)
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
    else:
        st.caption("This run does not contain a research experiment tree.")


def _eda(config) -> None:
    st.subheader("Trusted train/validation profile")
    if not config.eda_profile_path.is_file():
        st.markdown(
            '<div class="empty-panel">No aggregate EDA profile exists. Generate it explicitly with '
            '<code>python -m src.ui.profile_data --config configs/ui.json</code>.</div>',
            unsafe_allow_html=True,
        )
        return
    try:
        profile = json.loads(config.eda_profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not read the aggregate EDA profile: {exc}")
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
        st.markdown("#### Temporal Interaction & Label Trends")
        df_act = pd.DataFrame(activity)
        if "date" in df_act.columns:
            df_act["date_str"] = df_act["date"].astype(str)
            left, right = st.columns(2)
            left.markdown("**Daily Interaction Rows**")
            left.bar_chart(df_act.set_index("date_str")["rows"])
            right.markdown("**Daily Long-View Rate**")
            right.line_chart(df_act.set_index("date_str")["positive_rate"])

    durations = profile.get("duration_histogram", [])
    if durations:
        st.markdown("#### Video Duration Distribution (Quantile Buckets)")
        df_dur = pd.DataFrame(durations)
        if "bucket" in df_dur.columns and "count" in df_dur.columns:
            st.bar_chart(df_dur.set_index("bucket")["count"])


def _feature_lab() -> None:
    st.subheader("Leakage-Safe Feature Lineage & Catalog")
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

    st.markdown("#### Search Surface Extensions & Safety Invariants")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """<div class="pass-card">
<b>History & Sequence Features</b><br/>
<span class="badge badge-info">Planned</span><br/>
<small>Train-only historical long_view rate, user-author affinity, and recency decay. Fitted strictly prior to prediction timestamp.</small>
</div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """<div class="pass-card">
<b>Multi-Task Auxiliary Heads</b><br/>
<span class="badge badge-info">Planned</span><br/>
<small>Shared FM embeddings with joint loss on <code>is_click</code> (44%), <code>is_like</code>, and <code>play_time_ms</code> for dense gradient signals.</small>
</div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """<div class="pass-card">
<b>Tree-Based Ranking (GBDT)</b><br/>
<span class="badge badge-info">Planned</span><br/>
<small>LightGBM LambdaRank grouped by user over dense behavioral features and target encodings.</small>
</div>""",
            unsafe_allow_html=True,
        )


def _render_role_passes(role_passes: tuple[RolePass, ...]) -> None:
    if not role_passes:
        st.caption("No role passes recorded for this iteration.")
        return
    for rp in role_passes:
        with st.expander(f"Pass {rp.sequence + 1}: {rp.role.title()} ({rp.model})", expanded=(rp.sequence == 0)):
            c1, c2, c3 = st.columns(3)
            c1.caption(f"**Model:** `{rp.model}`")
            c2.caption(f"**Latency:** `{rp.latency_seconds:.2f}s`")
            tot = rp.usage.get("total_tokens", 0)
            inp = rp.usage.get("input_tokens", 0)
            out = rp.usage.get("output_tokens", 0)
            c3.caption(f"**Tokens:** `{tot}` (`{inp}` in / `{out}` out)")

            if rp.data:
                st.markdown("**Structured Decision:**")
                st.json(rp.data, expanded=True)

            if rp.sources:
                st.markdown("**Cited Primary Sources:**")
                for s in rp.sources:
                    t = s.get("title", "Source")
                    u = s.get("url")
                    st.markdown(f"- [{t}]({u})" if u else f"- {t}")

            st.caption("Prompt Context:")
            st.text(rp.prompt[:800] + ("…" if len(rp.prompt) > 800 else ""))


def _iterations(snapshot: RunSnapshot) -> None:
    st.subheader("Iteration Inspector")
    if not snapshot.iterations:
        st.markdown('<div class="empty-panel">No completed iteration records yet.</div>', unsafe_allow_html=True)
        return
    options = {f"{item.iteration:03d} · {item.experiment_id}": item for item in snapshot.iterations}
    selected = options[st.selectbox("Select Iteration", list(options), index=len(options) - 1)]

    left, right = st.columns([2, 1])
    left.markdown(f"### {selected.experiment_id}")
    left.write(f"**Hypothesis:** {selected.hypothesis or 'No hypothesis recorded.'}")
    right.metric("Status", selected.status.upper())
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
        )

    if selected.role_passes:
        st.subheader("Autonomous Multi-Role Pass Sequence")
        _render_role_passes(selected.role_passes)

    with st.expander("Candidate Source & Test Implementation", expanded=bool(selected.candidate_code)):
        if selected.candidate_code:
            st.markdown("**`candidate.py`:**")
            st.code(selected.candidate_code, language="python", line_numbers=True)
            if selected.candidate_tests:
                st.markdown("**`test_candidate.py`:**")
                st.code(selected.candidate_tests, language="python", line_numbers=True)
        else:
            st.caption("No candidate source files located.")

    with st.expander("Changes & Code Diff", expanded=bool(selected.change_summary)):
        if selected.change_summary:
            st.dataframe(list(selected.change_summary.files), width="stretch", hide_index=True)
            patch = load_patch_text(snapshot.path, selected.change_summary.patch_path)
            if patch:
                st.code(patch, language="diff", line_numbers=True)
        else:
            st.caption(selected.raw.get("code_diff", "No generated-code change recorded."))

    with st.expander("Configuration Parameters", expanded=False):
        st.json(selected.parameters)

    with st.expander("Agent Notes", expanded=False):
        if selected.agent_notes:
            st.json(selected.agent_notes, expanded=False)
        else:
            reflection = selected.raw.get("reflection")
            _render_note(reflection or {})

    with st.expander("Full Audited JSON Record", expanded=False):
        st.json(selected.raw, expanded=False)


def _results(snapshot: RunSnapshot, official: float) -> None:
    st.subheader("Validation Results & Benchmark Trajectory")
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
                    "Δ vs Baseline": (item.metrics.get("primary", 0) - official),
                }
            )

    if rows:
        df_res = pd.DataFrame(rows)
        st.line_chart(df_res.set_index("Iteration")[["Primary", "GAUC", "nDCG@5", "Baseline"]])
        st.dataframe(df_res, width="stretch", hide_index=True)
    else:
        st.caption("No successful validation metrics are recorded for this run.")

    st.caption(
        "Official published validation baseline: **0.6016**. Primary score = `(GAUC + nDCG@5) / 2`."
    )

    st.divider()
    st.subheader("Telemetry & Resource Breakdown")
    res = snapshot.resources or {}
    tokens = res.get("token_usage") or {}
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Total Tokens", f"{tokens.get('total_tokens', 0):,}")
    t2.metric("Input / Output", f"{tokens.get('input_tokens', 0):,} / {tokens.get('output_tokens', 0):,}")
    t3.metric("Wall Clock", f"{res.get('wall_clock_seconds', 0.0):.1f}s")
    t4.metric("GPU Hours / Interventions", f"{res.get('gpu_hours', 0.0)}h / {res.get('manual_interventions', 0)}")

    if snapshot.gate_info:
        st.divider()
        st.subheader("Official Gate & Submission Status")
        gate = snapshot.gate_info
        details = gate.get("details") or {}
        g1, g2, g3 = st.columns(3)
        g1.metric("Gate Status", str(gate.get("status", "unknown")).upper())
        g2.metric("Submission Rows", f"{details.get('rows', 0):,}")
        g3.metric("Verified With", str(details.get("checked_with", "starter kit")).replace("\\", "/").split("/")[-1])
        if "sha256" in details:
            st.caption(f"**SHA256:** `{details['sha256']}`")
        if "check_stdout" in details:
            st.success(details["check_stdout"])

    if snapshot.journal_markdown or snapshot.results_markdown:
        st.divider()
        st.subheader("Autonomous Research Journal")
        with st.expander("View journal.md", expanded=bool(snapshot.journal_markdown)):
            if snapshot.journal_markdown:
                st.markdown(snapshot.journal_markdown)
            else:
                st.caption("journal.md not rendered yet.")
        with st.expander("View results.md", expanded=False):
            if snapshot.results_markdown:
                st.markdown(snapshot.results_markdown)
            else:
                st.caption("results.md not rendered yet.")

    st.divider()
    st.subheader("Local Submission Schema Check")
    uploaded = st.file_uploader("Preview and validate a prediction CSV", type=["csv"])
    if uploaded is not None:
        payload = uploaded.getvalue()
        check = validate_submission(payload)
        (st.success if check.valid else st.error)(
            f"{'Schema checks passed' if check.valid else 'Schema checks failed'} · {check.row_count:,} rows · "
            f"{check.duplicate_pairs} duplicate user-video pairs"
        )
        for error in check.errors:
            st.error(error)
        for warning in check.warnings:
            st.warning(warning)


def main() -> None:
    st.set_page_config(page_title="ML Research Observatory", page_icon="◉", layout="wide")
    _css()
    st.markdown('<div class="eyebrow">KuaiRand-Pure · read-only observability</div>', unsafe_allow_html=True)
    st.title("ML Research Observatory")
    st.caption("One orchestrated loop, role-based passes, trusted validation, and an inspectable change trail.")
    try:
        config = load_dashboard_config(CONFIG_PATH)
        runs = discover_runs(config.run_root, config.official_baseline)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"Dashboard configuration error: {exc}")
        return
    if not runs:
        st.info("No run artifacts were found under the configured run root.")
        return
    labels = {f"{run.run_id} · {run.status}": run.path for run in runs}
    selected_label = st.sidebar.selectbox("Run", list(labels))
    selected_path = labels[selected_label]
    st.sidebar.caption("The dashboard never launches, resumes, cancels, or changes an experiment.")
    tabs = st.tabs(["Pipeline", "EDA", "Feature Lab", "Iterations", "Results"])
    with tabs[0]:
        initial = load_run_snapshot(selected_path, config.official_baseline)
        if initial.status == "running":
            @st.fragment(run_every=f"{config.active_refresh_seconds}s")
            def live_pipeline() -> None:
                _pipeline(
                    load_run_snapshot(selected_path, config.official_baseline),
                    config.stale_after_seconds,
                    config.official_baseline,
                )
            live_pipeline()
        else:
            _pipeline(initial, config.stale_after_seconds, config.official_baseline)
    snapshot = load_run_snapshot(selected_path, config.official_baseline)
    with tabs[1]:
        _eda(config)
    with tabs[2]:
        _feature_lab()
    with tabs[3]:
        _iterations(snapshot)
    with tabs[4]:
        _results(snapshot, config.official_baseline)


if __name__ == "__main__":
    main()

