from __future__ import annotations

import html
import json
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


def _render_live_role_stream(snapshot: RunSnapshot) -> None:
    if not snapshot.live_role_passes:
        return
    current_iter = (
        snapshot.activity.iteration
        if snapshot.activity
        else (max((item.iteration for item in snapshot.iterations), default=0) if snapshot.iterations else 1)
    )
    with st.expander(
        f"⚡ Live Sub-Iteration Agent Stream (Iteration {current_iter:03d} · {len(snapshot.live_role_passes)} passes)",
        expanded=True,
    ):
        for rp in snapshot.live_role_passes:
            role_label = STAGE_LABELS.get(rp.role, rp.role.replace("_", " ").title())
            st.markdown(
                f"**Pass {rp.sequence + 1}: {role_label}** (`{rp.model}` · `{rp.latency_seconds:.2f}s` · `{rp.usage.get('total_tokens', 0)} tokens`)"
            )
            if rp.role == "eda_researcher":
                st.caption(f"**Objective:** {rp.data.get('objective', 'Plan EDA pass')}")
                if rp.data.get("questions"):
                    st.markdown("*Research Questions:*")
                    for q in rp.data["questions"]:
                        st.markdown(f"- {q}")
                if rp.data.get("feature_hypotheses"):
                    st.markdown("*Feature Hypotheses:*")
                    for h in rp.data["feature_hypotheses"]:
                        st.markdown(f"- {h}")
                if rp.data.get("leakage_risks"):
                    st.markdown("*Leakage Guardrails:*")
                    for r in rp.data["leakage_risks"]:
                        st.markdown(f"- ⚠️ {r}")
            elif rp.role == "eda_builder":
                if rp.data.get("summary"):
                    st.info(f"**EDA Summary:** {rp.data['summary']}")
                if rp.data.get("findings"):
                    st.markdown("*Empirical Findings:*")
                    st.dataframe(rp.data["findings"], width="stretch", hide_index=True)
                if rp.data.get("feature_candidates"):
                    st.markdown("*Feature Proposals:*")
                    st.dataframe(rp.data["feature_candidates"], width="stretch", hide_index=True)
                if rp.data.get("recommended_next_focus"):
                    st.success(f"**Recommended Focus:** {rp.data['recommended_next_focus']}")
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
                st.markdown(f"**Preflight Decision:** {badge}")
                if rp.data.get("rationale"):
                    st.caption(f"**Rationale:** {rp.data['rationale']}")
                if rp.data.get("concerns"):
                    st.markdown("*Concerns / Risks:*")
                    for c in rp.data["concerns"]:
                        st.markdown(f"- {c}")
                if rp.data.get("next_focus"):
                    st.caption(f"**Next Focus:** {rp.data['next_focus']}")
            elif rp.role == "builder":
                st.markdown(f"**Candidate Generated:** `{rp.data.get('candidate_id', '')}`")
                if rp.data.get("code"):
                    with st.expander("Candidate Implementation Code", expanded=False):
                        st.code(rp.data["code"], language="python", line_numbers=True)
            elif rp.role == "debugger":
                st.warning(f"**Debugger Diagnosis:** {rp.data.get('diagnosis', '')}")
                if rp.data.get("replacement_code"):
                    with st.expander("Debugger Repaired Code", expanded=False):
                        st.code(rp.data["replacement_code"], language="python", line_numbers=True)
            elif rp.role == "critic_postflight":
                st.markdown(f"**Postflight Reflection:** {rp.data.get('reflection', '')}")
                st.markdown(f"**Next Focus:** {rp.data.get('next_focus', '')}")
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

    st.markdown("### 🛠️ Diagnostics & Re-attempt Engine")
    if activity and activity.attempt > 1:
        st.warning(
            f"**Active Re-attempt in Progress:** Iteration {activity.iteration} · Attempt {activity.attempt} "
            f"for Stage `{STAGE_LABELS.get(activity.stage, activity.stage)}`"
        )
    if activity and activity.error:
        st.error(f"**Trigger / Failure Output:**\n\n```text\n{activity.error}\n```")
    if activity and activity.repair:
        st.info(f"**Repair / Recovery Strategy:**\n\n{activity.repair}")

    if iter_events:
        with st.expander(f"WHY Re-attempt & Debugger Journal ({len(iter_events)} events)", expanded=True):
            for event in iter_events:
                st.markdown(f"**Stage:** `{event.stage}` · **Event:** `{event.event_type}`")
                if event.error_type:
                    st.caption(f"**Classification:** `{event.error_type}`")
                if event.candidate_id:
                    st.caption(f"**Candidate:** `{event.candidate_id}`")
                if event.error:
                    st.code(event.error, language="text")
                if event.lesson:
                    st.success(f"**Diagnosis / Corrective Action:** {event.lesson}")
                st.divider()


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

    _render_live_role_stream(snapshot)
    _render_live_diagnostics(snapshot)

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


def _resource_metrics(snapshot: RunSnapshot) -> dict[str, Any]:
    resources = snapshot.resources or {}
    tokens = resources.get("token_usage") or {}
    return {
        "total_tokens": tokens.get("total_tokens", 0),
        "input_tokens": tokens.get("input_tokens", 0),
        "output_tokens": tokens.get("output_tokens", 0),
        "wall_clock_seconds": resources.get("wall_clock_seconds", 0.0),
        "training_attempts": resources.get("training_attempts", len(snapshot.iterations)),
        "iteration_count": resources.get("iteration_count", len(snapshot.iterations)),
        "gpu_hours": resources.get("gpu_hours", 0.0),
        "manual_interventions": resources.get("manual_interventions", 0),
    }



def _render_budget_gauges(snapshot: RunSnapshot) -> None:
    config = snapshot.run_config or {}
    budgets = config.get("budgets") or {}
    if not budgets:
        return
    max_iters = budgets.get("max_iterations")
    max_seconds = budgets.get("max_wall_clock_seconds")

    usage = _resource_metrics(snapshot)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Iterations Budget", f"{usage['iteration_count']} / {max_iters}")
    c2.metric("Training Attempts", f"{usage['training_attempts']}")
    c3.metric("Wall Clock Time", f"{usage['wall_clock_seconds']:.1f}s / {max_seconds}s")
    c4.metric("LLM Tokens Used", f"{usage['total_tokens']:,}")


def _dot_escape(value: Any) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", " ")
        .replace("\n", "\\n")
    )


def _experiment_dag_dot(
    nodes: tuple[dict[str, Any], ...], best_id: str | None
) -> str:
    dot_lines = [
        "digraph experiments {",
        '  graph [rankdir="TB", bgcolor="transparent"];',
        '  node [shape="box", style="rounded,filled", fontname="Arial", fontsize="10"];',
        '  edge [color="#8fbdd6"];',
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
        if eid == best_id:
            fill, stroke, penwidth = "#dff1e8", "#27845e", 2
        elif status == "failed":
            fill, stroke, penwidth = "#f4d9d7", "#843c36", 1
        else:
            fill, stroke, penwidth = "#f4f8fb", "#8fbdd6", 1
        dot_lines.append(
            f'  {node_id} [label="{label}", fillcolor="{fill}", '
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


def _eda(config, snapshot: RunSnapshot) -> None:
    st.subheader("Run EDA reports")
    if snapshot.live_eda:
        live = snapshot.live_eda
        st.markdown("### ⚡ Live Active EDA (Current Run)")
        st.caption(f"Status: **{live.status}**")
        if live.plan:
            with st.expander(f"Active EDA Plan (Iteration {live.iteration:03d})", expanded=True):
                st.markdown(f"**Objective:** {live.plan.get('objective', '')}")
                if live.plan.get("questions"):
                    st.markdown("*Questions:*")
                    for q in live.plan["questions"]:
                        st.markdown(f"- {q}")
                if live.plan.get("feature_hypotheses"):
                    st.markdown("*Feature Hypotheses:*")
                    for h in live.plan["feature_hypotheses"]:
                        st.markdown(f"- {h}")
                if live.plan.get("leakage_risks"):
                    st.markdown("*Leakage Risks:*")
                    for r in live.plan["leakage_risks"]:
                        st.markdown(f"- ⚠️ {r}")
        if live.report:
            with st.expander(f"Active EDA Report Findings (Iteration {live.iteration:03d})", expanded=True):
                if live.report.get("summary"):
                    st.info(live.report["summary"])
                if live.report.get("findings"):
                    st.markdown("#### Empirical Findings")
                    st.dataframe(list(live.report["findings"]), width="stretch", hide_index=True)
                if live.report.get("feature_candidates"):
                    st.markdown("#### Proposed Features")
                    st.dataframe(list(live.report["feature_candidates"]), width="stretch", hide_index=True)
                if live.report.get("recommended_next_focus"):
                    st.success(f"**Recommended Focus:** {live.report['recommended_next_focus']}")
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
        st.markdown(
            '<div class="empty-panel">No autonomous EDA artifacts recorded for this run yet.</div>',
            unsafe_allow_html=True,
        )

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
            left.markdown("**Daily Interaction Rows**")
            left.bar_chart(daily_rows)
            right.markdown("**Daily Long-View Rate**")
            right.line_chart(daily_rates)

    durations = profile.get("duration_histogram", [])
    if durations:
        st.markdown("#### Video Duration Distribution (Quantile Buckets)")
        df_dur = pd.DataFrame(durations)
        if "seconds" in df_dur.columns and "rows" in df_dur.columns:
            st.bar_chart(df_dur.set_index("seconds")["rows"])


def _feature_lab(snapshot: RunSnapshot) -> None:
    st.subheader("Leakage-Safe Feature Lineage & Catalog")

    live_proposals = []
    if snapshot.live_eda and snapshot.live_eda.feature_candidates:
        for f in snapshot.live_eda.feature_candidates:
            live_proposals.append({"source": f"⚡ Live EDA (Iter {snapshot.live_eda.iteration:03d})", **f})
    for rp in snapshot.live_role_passes:
        if rp.role == "eda_builder" and rp.data.get("feature_candidates"):
            for f in rp.data["feature_candidates"]:
                live_proposals.append({"source": f"⚡ Live Builder Pass {rp.sequence + 1}", **f})

    if live_proposals:
        st.markdown("### ⚡ Live Feature Proposals")
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
        st.info("Future feature families appear here only after trusted run metadata logs them.")



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

def _iterations(snapshot: RunSnapshot) -> None:
    st.subheader("Iteration Inspector")
    if not snapshot.iterations:
        st.markdown('<div class="empty-panel">No completed iteration records yet.</div>', unsafe_allow_html=True)
        return
    options = {f"{item.iteration:03d} · {item.experiment_id}": item for item in snapshot.iterations}
    selected = options[st.selectbox("Select Iteration", list(options), index=len(options) - 1)]
    role_passes = load_role_passes(snapshot.path, selected.iteration)
    candidate_code, candidate_tests = load_candidate_files(
        snapshot.path, selected.candidate_dir
    )

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

    if role_passes:
        st.subheader("Autonomous Multi-Role Pass Sequence")
        _render_role_passes(role_passes)

    with st.expander("Candidate Source & Test Implementation", expanded=bool(candidate_code)):
        if candidate_code:
            st.markdown("**`candidate.py`:**")
            st.code(candidate_code, language="python", line_numbers=True)
            if candidate_tests:
                st.markdown("**`test_candidate.py`:**")
                st.code(candidate_tests, language="python", line_numbers=True)
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
    usage = _resource_metrics(snapshot)
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Total Tokens", f"{usage['total_tokens']:,}")
    t2.metric("Input / Output", f"{usage['input_tokens']:,} / {usage['output_tokens']:,}")
    t3.metric("Wall Clock", f"{usage['wall_clock_seconds']:.1f}s")
    t4.metric("GPU Hours / Interventions", f"{usage['gpu_hours']}h / {usage['manual_interventions']}")

    if snapshot.gate_info:
        st.divider()
        st.subheader("Official Gate & Submission Status")
        gate = snapshot.gate_info
        details = gate.get("details") or {}
        gate_status = str(gate.get("status", "unknown"))
        if gate_status == "error":
            st.metric("Gate Status", gate_status.upper())
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
            g1.metric("Gate Status", gate_status.upper())
            rows = details.get("rows")
            g2.metric("Submission Rows", f"{rows:,}" if isinstance(rows, int) else "—")
            checked_with = details.get("checked_with")
            verified_with = (
                str(checked_with).replace("\\", "/").split("/")[-1]
                if checked_with
                else "—"
            )
            g3.metric("Verified With", verified_with)
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
    initial = load_run_snapshot(selected_path, config.official_baseline)

    if initial.status == "running":
        with tabs[0]:
            @st.fragment(run_every=f"{config.active_refresh_seconds}s")
            def live_pipeline() -> None:
                _pipeline(
                    load_run_snapshot(selected_path, config.official_baseline),
                    config.stale_after_seconds,
                    config.official_baseline,
                )
            live_pipeline()

        with tabs[1]:
            @st.fragment(run_every=f"{config.active_refresh_seconds}s")
            def live_eda() -> None:
                _eda(config, load_run_snapshot(selected_path, config.official_baseline))
            live_eda()

        with tabs[2]:
            @st.fragment(run_every=f"{config.active_refresh_seconds}s")
            def live_feature_lab() -> None:
                _feature_lab(load_run_snapshot(selected_path, config.official_baseline))
            live_feature_lab()

        with tabs[3]:
            @st.fragment(run_every=f"{config.active_refresh_seconds}s")
            def live_iterations() -> None:
                _iterations(load_run_snapshot(selected_path, config.official_baseline))
            live_iterations()

        with tabs[4]:
            @st.fragment(run_every=f"{config.active_refresh_seconds}s")
            def live_results() -> None:
                _results(load_run_snapshot(selected_path, config.official_baseline), config.official_baseline)
            live_results()
    else:
        with tabs[0]:
            _pipeline(initial, config.stale_after_seconds, config.official_baseline)
        with tabs[1]:
            _eda(config, initial)
        with tabs[2]:
            _feature_lab(initial)
        with tabs[3]:
            _iterations(initial)
        with tabs[4]:
            _results(initial, config.official_baseline)



if __name__ == "__main__":
    main()
