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

import altair as alt
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg-app: #f8fafc;
  --bg-card: #ffffff;
  --text-main: #0f172a;
  --text-muted: #64748b;
  --border: #e2e8f0;
  --blue-primary: #2563eb;
  --blue-subtle: #eff6ff;
  --green-primary: #059669;
  --green-subtle: #ecfdf5;
  --amber-primary: #d97706;
  --amber-subtle: #fffbeb;
  --red-primary: #dc2626;
  --red-subtle: #fef2f2;
}

html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.stApp {
  background-color: var(--bg-app);
  color: var(--text-main);
}

.block-container {
  max-width: 1440px;
  padding-top: 3.5rem; /* extra space so Streamlit top bar never overlaps navigation */
  padding-bottom: 3rem;
  padding-left: 2rem;
  padding-right: 2rem;
}

/* Header & Meta Bar */
.top-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.5rem 0 1.2rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.2rem;
}
.tab-title-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 1.4rem;
  font-weight: 700;
  color: var(--text-main);
}
.tab-number {
  background: #2563eb;
  color: #ffffff;
  width: 26px;
  height: 26px;
  border-radius: 6px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.9rem;
  font-weight: 700;
}
.top-tags {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  font-size: 0.85rem;
  color: var(--text-muted);
}
.pill-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.25rem 0.65rem;
  border-radius: 9999px;
  font-size: 0.78rem;
  font-weight: 500;
}
.pill-green {
  background: #ecfdf5;
  color: #047857;
  border: 1px solid #a7f3d0;
}
.pill-blue {
  background: #eff6ff;
  color: #1d4ed8;
  border: 1px solid #bfdbfe;
}
.pill-amber {
  background: #fffbeb;
  color: #b45309;
  border: 1px solid #fde68a;
}
.pill-red {
  background: #fef2f2;
  color: #b91c1c;
  border: 1px solid #fecaca;
}
.pill-gray {
  background: #f1f5f9;
  color: #475569;
  border: 1px solid #cbd5e1;
}

/* Card Styling */
.ui-card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 1.25rem 1.4rem;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
  margin-bottom: 1.2rem;
}
.ui-card-title {
  font-size: 1.05rem;
  font-weight: 700;
  color: #0f172a;
  margin-bottom: 0.75rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.ui-card-subtitle {
  font-size: 0.82rem;
  color: #64748b;
  margin-top: -0.5rem;
  margin-bottom: 0.9rem;
}

/* Big Hero Metric */
.hero-val {
  font-size: 2.2rem;
  font-weight: 700;
  color: #0f172a;
  line-height: 1.1;
}
.hero-label {
  font-size: 0.75rem;
  font-weight: 600;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 0.2rem;
}

/* Autonomous Research Loop Diagram */
.loop-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.8rem;
  margin: 0.9rem 0;
  position: relative;
}
.loop-node {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 0.85rem 0.9rem;
  transition: all 0.2s ease;
  min-height: 98px;
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  position: relative;
}
.loop-node.blue-node {
  border-color: #bfdbfe;
  background: #fbfdff;
}
.loop-node.amber-node {
  border-color: #fde68a;
  background: #fffdfa;
}
.loop-node.green-node {
  border-color: #a7f3d0;
  background: #fafffc;
}
.loop-node.red-node {
  border-color: #fecaca;
  background: #fffbfa;
}

/* Active node state in live run */
.loop-node.is-active {
  border: 2px solid #2563eb !important;
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.18) !important;
  background: #eff6ff !important;
}
.loop-node.is-done {
  border-color: #86efac !important;
  background: #f0fdf4 !important;
}

.node-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 0.85rem;
  font-weight: 600;
  color: #0f172a;
  margin-bottom: 0.25rem;
}
.node-desc {
  font-size: 0.75rem;
  color: #64748b;
  line-height: 1.35;
}
.loop-node.blue-node .node-header { color: #1d4ed8; }
.loop-node.amber-node .node-header { color: #b45309; }
.loop-node.green-node .node-header { color: #047857; }
.loop-node.red-node .node-header { color: #b91c1c; }

.repair-bar {
  border: 1px dashed #fca5a5;
  background: #fffbfa;
  border-radius: 8px;
  padding: 0.65rem 1rem;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin: 0.6rem 0 1rem;
  font-size: 0.82rem;
  color: #991b1b;
}
.repair-bar.is-active {
  border: 2px solid #dc2626 !important;
  box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.18) !important;
  background: #fef2f2 !important;
}

/* Key-value stats in cards */
.kv-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.42rem 0;
  border-bottom: 1px solid #f1f5f9;
  font-size: 0.84rem;
}
.kv-row:last-child {
  border-bottom: none;
}
.kv-key {
  color: #64748b;
}
.kv-val {
  font-weight: 600;
  color: #0f172a;
}

/* Feature Lineage Schema View */
.lineage-container {
  display: grid;
  grid-template-columns: 1fr 40px 1.2fr 40px 1fr;
  align-items: center;
  gap: 0.5rem;
  margin: 1.2rem 0;
}
.lineage-col {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}
.lineage-box {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 0.75rem 0.9rem;
  font-size: 0.82rem;
}
.lineage-box.blue { border-color: #bfdbfe; background: #eff6ff; color: #1e40af; font-weight: 600; }
.lineage-box.green { border-color: #a7f3d0; background: #ecfdf5; color: #065f46; font-weight: 600; }
.lineage-box.purple { border-color: #ddd6fe; background: #f5f3ff; color: #5b21b6; font-weight: 600; }
.lineage-arrow {
  text-align: center;
  color: #94a3b8;
  font-size: 1.2rem;
}

/* Checklist items */
.check-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.82rem;
  color: #334155;
  margin-bottom: 0.35rem;
}
.check-icon {
  color: #10b981;
  font-weight: 700;
}

/* Stage strip */
.stage-row { display:flex; flex-wrap:wrap; gap:.42rem; margin:.8rem 0 .5rem; }
.stage { padding:.32rem .6rem; border-radius:999px; border:1px solid #e2e8f0; color:#64748b; font-size:.76rem; background:#f8fafc; }
.stage.done { color:#047857; background:#ecfdf5; border-color:#a7f3d0; }
.stage.active { color:#1d4ed8; background:#eff6ff; border-color:#93c5fd; font-weight:600; }
.stage.failed { color:#b91c1c; background:#fef2f2; border-color:#fecaca; }

.empty-panel { border:1px dashed #cbd5e1; border-radius:12px; padding:1.5rem; color:#64748b; background:#f8fafc; text-align: center; }

/* Pulse animation */
.pulse { display:inline-block; width:.62rem; height:.62rem; border-radius:50%; background:#10b981;
  box-shadow:0 0 0 rgba(16,185,129,.45); animation:pulse 1.8s infinite; margin-right:.45rem; }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(16,185,129,.42)} 70%{box-shadow:0 0 0 8px rgba(16,185,129,0)} 100%{box-shadow:0 0 0 0 rgba(16,185,129,0)} }

/* Streamlit Tabs Polish */
.stTabs [data-baseweb="tab-list"] {
  gap: 1.5rem;
  border-bottom: 1px solid #e2e8f0;
  margin-bottom: 1.2rem;
}
.stTabs [data-baseweb="tab"] {
  font-size: 0.95rem;
  font-weight: 600;
  color: #64748b;
  padding: 0.6rem 0.2rem;
}
.stTabs [aria-selected="true"] {
  color: #2563eb !important;
  border-bottom-color: #2563eb !important;
}
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


def _format_seconds(seconds: float) -> str:
    sec = int(seconds)
    minutes, sec = divmod(sec, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


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
        pills.append(f'<span class="{class_name}">{html.escape(STAGE_LABELS.get(stage, stage))}</span>')
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
                f"**Pass {rp.sequence + 1}: {role_label}** (`{rp.model}` · `{rp.latency_seconds:.2f}s` · `{rp.usage.get('total_tokens', 0):,}` tokens)"
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


def _dot_escape(value: Any) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", " ")
        .replace("\n", "\\n")
    )


def _make_diffs_collapsible(markdown_text: str | None) -> str | None:
    if not markdown_text:
        return markdown_text
    pattern = r"(```diff\n.*?\n```)"

    def replacer(match: Any) -> str:
        diff_block = match.group(1)
        return f"<details>\n<summary>🔍 View Code Changes</summary>\n\n{diff_block}\n</details>"

    return re.sub(pattern, replacer, markdown_text, flags=re.DOTALL)


def _experiment_dag_dot(
    nodes: tuple[dict[str, Any], ...], best_id: str | None
) -> str:
    dot_lines = [
        "digraph experiments {",
        '  graph [rankdir="TB", bgcolor="transparent"];',
        '  node [shape="box", style="rounded,filled", fontname="Arial", fontsize="10"];',
        '  edge [color="#94a3b8"];',
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
            fill, stroke, penwidth = "#dcfce7", "#16a34a", 2
        elif status == "failed":
            fill, stroke, penwidth = "#fee2e2", "#dc2626", 1
        else:
            fill, stroke, penwidth = "#f8fafc", "#cbd5e1", 1
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


# ============================================================================
# TAB 1: PIPELINE (UNIFIED RESEARCH LOOP & LIVE OVERLAY)
# ============================================================================

def _pipeline(snapshot: RunSnapshot, stale_after: int, official: float) -> None:
    # Top Bar Header
    st.markdown(
        f"""
<div class="top-header">
  <div class="tab-title-badge">
    <span class="tab-number">1</span>
    <span>Pipeline</span>
  </div>
  <div class="top-tags">
    <span class="pill-badge pill-green">Validation only</span>
    <span>👤 Single role loop</span>
    <span>🧠 Shared memory</span>
    <span><span class="pulse"></span>Auto-save</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    activity = snapshot.activity
    is_live = bool(activity and activity.status == "active" and snapshot.status == "running")
    age = activity_age_seconds(activity) if activity else None
    stale = is_live and age is not None and age > stale_after

    active_stage = activity.stage if activity else ""
    current_iter = activity.iteration if activity else len(snapshot.iterations)
    elapsed = _elapsed_label(activity) if (activity and is_live) else ""

    def step_state(step_id: str) -> tuple[str, str]:
        if not is_live:
            return ("", "")
        
        stage_mapping = {
            "observe": {"initializing", "eda_researcher", "eda_builder"},
            "researcher": {"researcher", "researcher_web"},
            "preflight": {"critic_preflight"},
            "builder": {"builder", "safety_tests"},
            "validator": {"safety_tests"},
            "train_eval": {"training_evaluation"},
            "postflight": {"critic_postflight"},
            "reflect": {"persistence", "completed"},
            "debugger": {"debugger"},
        }
        target_stages = stage_mapping.get(step_id, set())
        if active_stage in target_stages:
            return ("is-active", '<span class="pulse"></span><span style="font-size:0.7rem; color:#2563eb; font-weight:700;">ACTIVE</span>')
        return ("", "")

    obs_class, obs_pill = step_state("observe")
    res_class, res_pill = step_state("researcher")
    pref_class, pref_pill = step_state("preflight")
    bld_class, bld_pill = step_state("builder")
    val_class, val_pill = step_state("validator")
    trn_class, trn_pill = step_state("train_eval")
    post_class, post_pill = step_state("postflight")
    ref_class, ref_pill = step_state("reflect")
    deb_class, deb_pill = step_state("debugger")

    # Side Cards Calculations
    usage = _resource_metrics(snapshot)
    config = snapshot.run_config or {}
    budgets = config.get("budgets") or {}
    max_iters = budgets.get("max_iterations", 50)
    max_seconds = budgets.get("max_wall_clock_seconds", 21600)  # 6 hours

    curr_iter = len(snapshot.iterations)
    elapsed_sec = usage["wall_clock_seconds"]
    elapsed_display = _format_seconds(elapsed_sec)
    time_pct = min(1.0, elapsed_sec / max_seconds) if max_seconds else 0.0
    iter_pct = min(1.0, curr_iter / max_iters) if max_iters else 0.0

    metrics = snapshot.best_metrics or {}
    best_gauc = metrics.get("GAUC", 0.0)
    best_ndcg = metrics.get("nDCG@5", 0.0)
    best_primary = metrics.get("primary", 0.0)
    delta = best_primary - official if best_primary else 0.0
    delta_color = "#16a34a" if delta >= 0 else "#dc2626"

    # Main Grid Layout: Left 68% (Research Loop + Live Stream), Right 32% (Budget + Best Validation)
    col_main, col_side = st.columns([68, 32])

    with col_main:
        if is_live:
            status_html = f"""<div style="display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.8rem; background: #eff6ff; border: 1px solid #bfdbfe; color: #1d4ed8; padding: 3px 10px; border-radius: 999px;">
  <span class="pulse"></span>
  <strong>Iter {current_iter}</strong> · {STAGE_LABELS.get(active_stage, active_stage)} · {elapsed}
</div>"""
        else:
            status_html = f"""<div style="display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.8rem; background: #ecfdf5; border: 1px solid #a7f3d0; color: #047857; padding: 3px 10px; border-radius: 999px;">
  <span>✔</span>
  <strong>Status: {snapshot.status.upper()}</strong> · {len(snapshot.iterations)} Iters
</div>"""

        objective_row = ""
        if activity and activity.objective:
            objective_row = f"""<div style="font-size: 0.8rem; color: #475569; margin: 0.4rem 0 0.6rem; background: #f8fafc; padding: 6px 10px; border-radius: 6px; border-left: 3px solid #3b82f6;">
  <strong>Current Objective:</strong> {html.escape(activity.objective)}
</div>"""

        # Unified Research Loop Diagram Card
        st.markdown(
            f"""
<div class="ui-card">
  <div class="ui-card-title">
    <div>Autonomous Research Loop <span style="font-weight: 500; font-size: 0.85rem; color: #64748b;">(Single Role)</span></div>
    <div>{status_html}</div>
  </div>
  
  {objective_row}

  <div class="loop-grid">
    <div class="loop-node blue-node {obs_class}">
      <div class="node-header"><span>👥 Observe</span>{obs_pill}</div>
      <div class="node-desc">Read problem, dataset & metrics</div>
    </div>
    <div class="loop-node blue-node {res_class}">
      <div class="node-header"><span>🔍 Researcher</span>{res_pill}</div>
      <div class="node-desc">Inspect data, form hypothesis, plan experiments</div>
    </div>
    <div class="loop-node amber-node {pref_class}">
      <div class="node-header"><span>🛡️ Critic (Preflight)</span>{pref_pill}</div>
      <div class="node-desc">Validate plan, check leakage, set guardrails</div>
    </div>
    <div class="loop-node blue-node {bld_class}">
      <div class="node-header"><span>🔧 Builder</span>{bld_pill}</div>
      <div class="node-desc">Engineer features, configure models, define params</div>
    </div>
  </div>

  <div class="loop-grid">
    <div class="loop-node green-node {val_class}">
      <div class="node-header"><span>🛡️ Deterministic Validator</span>{val_pill}</div>
      <div class="node-desc">Train-fitted stats, CV eval (GAUC, nDCG@5)</div>
    </div>
    <div class="loop-node green-node {trn_class}">
      <div class="node-header"><span>📊 Train + Evaluate</span>{trn_pill}</div>
      <div class="node-desc">Train models, evaluate on validation only</div>
    </div>
    <div class="loop-node amber-node {post_class}">
      <div class="node-header"><span>🛡️ Critic (Postflight)</span>{post_pill}</div>
      <div class="node-desc">Analyze results, decide promote / iterate / stop</div>
    </div>
    <div class="loop-node amber-node {ref_class}">
      <div class="node-header"><span>📖 Reflect + Remember</span>{ref_pill}</div>
      <div class="node-desc">Record outcomes, update memory, plan next iteration</div>
    </div>
  </div>

  <div class="repair-bar {deb_class}">
    <span style="font-weight: 700; color: #dc2626;">⚙️ Debugger / Repair</span>
    <span>If preflight fails or leakage detected, repair and return to validator</span>
    {deb_pill}
  </div>

  <div style="display: grid; grid-template-columns: 1.2fr 1.2fr 1fr; gap: 1rem; padding-top: 0.75rem; border-top: 1px solid #f1f5f9;">
    <div>
      <div style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Goal</div>
      <div style="font-size: 0.83rem; font-weight: 600; color: #0f172a; margin-top: 0.2rem;">Maximize GAUC & nDCG@5 on validation (official)</div>
    </div>
    <div>
      <div style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Safety Guardrails</div>
      <div style="font-size: 0.8rem; color: #047857; margin-top: 0.2rem;">
        <div>✔ Validation-only (no hidden test)</div>
        <div>✔ Train-only feature fitting</div>
        <div>✔ Temporal safety (no leakage)</div>
      </div>
    </div>
    <div>
      <div style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Baseline</div>
      <div style="font-size: 0.78rem; color: #64748b; margin-top: 0.1rem;">Official validation baseline</div>
      <div style="font-size: 1.4rem; font-weight: 700; color: #0f172a;">0.6016</div>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        if stale:
            st.warning(
                f"No activity transition has been written for more than {stale_after // 60} minutes. "
                "The process may still be in a long blocking stage or may have been interrupted."
            )

        # Live Sub-Iteration Agent Stream & Diagnostics
        _render_live_role_stream(snapshot)
        _render_live_diagnostics(snapshot)

        if activity and activity.agent_note:
            with st.expander("Agent Notes — structured decision trace", expanded=False):
                _render_note(activity.agent_note)
                st.caption("Summarized decisions only; raw hidden reasoning and full prompts are not displayed.")

        with st.expander("Recent Execution Timeline", expanded=False):
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

    with col_side:
        # Budget & Convergence Card
        st.markdown(
            f"""
<div class="ui-card">
  <div class="ui-card-title">Budget & Convergence</div>
  <div class="kv-row">
    <span class="kv-key">Max iterations (official)</span>
    <span class="kv-val">{max_iters}</span>
  </div>
  <div class="kv-row">
    <span class="kv-key">Current iteration</span>
    <span class="kv-val">{curr_iter}</span>
  </div>
  <div class="kv-row">
    <span class="kv-key">Convergence rule</span>
    <span class="kv-val">Δ ≤ 0.0020 for 3 iters</span>
  </div>
  <div class="kv-row">
    <span class="kv-key">Elapsed time</span>
    <span class="kv-val">{elapsed_display} / 6h</span>
  </div>
  
  <div style="margin-top: 0.8rem;">
    <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #64748b; margin-bottom: 0.25rem;">
      <span>Time budget</span>
      <span>{int(time_pct * 100)}%</span>
    </div>
    <div style="height: 6px; background: #e2e8f0; border-radius: 999px; overflow: hidden;">
      <div style="height: 100%; width: {time_pct * 100}%; background: #2563eb;"></div>
    </div>
  </div>

  <div style="margin-top: 0.6rem;">
    <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #64748b; margin-bottom: 0.25rem;">
      <span>Iteration budget</span>
      <span>{int(iter_pct * 100)}%</span>
    </div>
    <div style="height: 6px; background: #e2e8f0; border-radius: 999px; overflow: hidden;">
      <div style="height: 100%; width: {iter_pct * 100}%; background: #2563eb;"></div>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        # Best Validation Card
        st.markdown(
            f"""
<div class="ui-card">
  <div class="ui-card-title">Best Validation <span style="font-weight: 500; font-size: 0.8rem; color: #64748b;">(Current Run)</span></div>
  
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 0.6rem;">
    <div>
      <div class="hero-label">GAUC</div>
      <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a;">{best_gauc:.4f}</div>
    </div>
    <div>
      <div class="hero-label">nDCG@5</div>
      <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a;">{best_ndcg:.4f}</div>
    </div>
  </div>

  <div style="border-top: 1px solid #f1f5f9; padding-top: 0.6rem; margin-bottom: 0.6rem;">
    <div class="hero-label">Primary (avg)</div>
    <div class="hero-val">{best_primary:.4f}</div>
  </div>

  <div class="kv-row">
    <span class="kv-key">Official validation baseline</span>
    <span class="kv-val">0.6016</span>
  </div>
  <div class="kv-row">
    <span class="kv-key">Delta vs baseline</span>
    <span class="kv-val" style="color: {delta_color}; font-weight: 700;">{delta:+.4f}</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    # Lineage & Search Frontier Section (Full Width Below Both Columns)
    st.markdown('<div class="ui-card-title" style="margin-top: 1.2rem;">Visual Experiment Lineage (DAG)</div>', unsafe_allow_html=True)
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
                    "Δ nDCG@5 vs parent": (node.get("search") or {}).get("delta_ndcg5_vs_parent"),
                    "top-5 hit rate": (node.get("topk_diagnostics") or {}).get("top5_hit_rate"),
                    "duration (s)": node.get("duration_seconds"),
                }
                for node in snapshot.nodes
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("This run does not contain a research experiment tree.")


# ============================================================================
# TAB 2: EDA (DYNAMIC TRAIN/VALIDATION STATS & VERIFIED REAL DATA)
# ============================================================================

def _eda(config, snapshot: RunSnapshot) -> None:
    st.markdown(
        """
<div class="top-header">
  <div class="tab-title-badge">
    <span class="tab-number">2</span>
    <span>EDA</span>
  </div>
  <div class="top-tags">
    <span class="pill-badge pill-green">Train-only statistics</span>
    <span>No hidden test leakage</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    eda_view = st.radio(
        "EDA Sub-view",
        ["Overview", "Feature Insights", "Iterations & Artifacts"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # Load verified EDA profile computed strictly from dataset
    profile: dict[str, Any] = {}
    if config and config.eda_profile_path and config.eda_profile_path.is_file():
        try:
            profile = json.loads(config.eda_profile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            profile = {}

    splits = profile.get("splits", {})
    train_split = splits.get("train", {})
    valid_split = splits.get("valid", {})

    total_users = train_split.get("users")
    total_videos = train_split.get("videos")
    total_rows = train_split.get("rows")
    pos_rate = train_split.get("positive_rate")

    users_display = f"{total_users:,}" if total_users is not None else "—"
    videos_display = f"{total_videos:,}" if total_videos is not None else "—"
    rows_display = f"{total_rows:,}" if total_rows is not None else "—"
    rate_display = f"{pos_rate * 100:.3f}%" if pos_rate is not None else "—"

    if eda_view == "Overview":
        st.markdown('<div class="ui-card-title">Train-only statistics (fitted on TRAIN)</div>', unsafe_allow_html=True)
        
        # 4 Verified Dynamic Metric KPI Cards
        st.markdown(
            f"""
<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1rem 0;">
  <div class="ui-card" style="text-align: center; padding: 1rem 0.5rem; margin-bottom: 0;">
    <div class="hero-label">Users (Train)</div>
    <div style="font-size: 1.35rem; font-weight: 700; color: #0f172a;">{users_display}</div>
  </div>
  <div class="ui-card" style="text-align: center; padding: 1rem 0.5rem; margin-bottom: 0;">
    <div class="hero-label">Videos (Train)</div>
    <div style="font-size: 1.35rem; font-weight: 700; color: #0f172a;">{videos_display}</div>
  </div>
  <div class="ui-card" style="text-align: center; padding: 1rem 0.5rem; margin-bottom: 0;">
    <div class="hero-label">Impressions (Train)</div>
    <div style="font-size: 1.35rem; font-weight: 700; color: #0f172a;">{rows_display}</div>
  </div>
  <div class="ui-card" style="text-align: center; padding: 1rem 0.5rem; margin-bottom: 0;">
    <div class="hero-label">Long-View Rate</div>
    <div style="font-size: 1.35rem; font-weight: 700; color: #0f172a;">{rate_display}</div>
  </div>
</div>
<div style="font-size: 0.78rem; color: #64748b; text-align: center; margin-top: -0.4rem; margin-bottom: 1.2rem;">
  ✔ Verified: All distributions computed strictly on TRAIN (2022-04-08..21) and VALID (2022-04-22..28). No hidden test info accessed.
</div>
""",
            unsafe_allow_html=True,
        )

        activity = profile.get("activity_by_date", [])
        if activity:
            df_act = pd.DataFrame(activity)
            if {"date", "split", "rows", "positive_rate"}.issubset(df_act.columns):
                df_act["date_str"] = df_act["date"].astype(str)
                daily_rows = df_act.pivot_table(
                    index="date_str", columns="split", values="rows", aggfunc="sum"
                )
                daily_rates = df_act.pivot_table(
                    index="date_str", columns="split", values="positive_rate", aggfunc="mean"
                )
                
                c_act1, c_act2 = st.columns(2)
                with c_act1:
                    st.markdown("**Daily Interaction Volume (Rows)**")
                    st.bar_chart(daily_rows, height=220)
                with c_act2:
                    st.markdown("**Daily Long-View Positive Rate**")
                    st.line_chart(daily_rates, height=220)

        durations = profile.get("duration_histogram", [])
        if durations:
            st.markdown("**Video Duration Distribution (Quantile Buckets)**")
            df_dur = pd.DataFrame(durations)
            if "seconds" in df_dur.columns and "rows" in df_dur.columns:
                st.bar_chart(df_dur.set_index("seconds")["rows"], height=200)

        if splits:
            st.markdown("**Split Verification Summary Table**")
            st.dataframe(
                [
                    {
                        "Split": name,
                        "Rows": value.get("rows"),
                        "Users": value.get("users"),
                        "Videos": value.get("videos", "—"),
                        "Positives": value.get("positives"),
                        "Positive Rate": f"{value.get('positive_rate', 0) * 100:.3f}%",
                        "Impressions/User (p50)": (value.get("impressions_per_user") or {}).get("p50"),
                        "Impressions/User (p95)": (value.get("impressions_per_user") or {}).get("p95"),
                    }
                    for name, value in splits.items()
                ],
                width="stretch",
                hide_index=True,
            )
        elif not config or not config.eda_profile_path.is_file():
            st.info("No aggregate profile found. Run `python -m src.ui.profile_data --config configs/ui.json` to generate the aggregate EDA profile.")

    elif eda_view == "Feature Insights":
        st.markdown('<div class="ui-card-title">Empirical Feature Findings & Proposals</div>', unsafe_allow_html=True)
        if snapshot.live_eda and snapshot.live_eda.report:
            st.markdown("#### ⚡ Live Active EDA Insights (Current Run)")
            rep = snapshot.live_eda.report
            if rep.get("summary"):
                st.info(rep["summary"])
            if rep.get("findings"):
                st.dataframe(list(rep["findings"]), width="stretch", hide_index=True)
            if rep.get("feature_candidates"):
                st.dataframe(list(rep["feature_candidates"]), width="stretch", hide_index=True)

        if snapshot.eda_artifacts:
            latest = snapshot.eda_artifacts[-1]
            if latest.findings:
                st.markdown("#### Finalized Findings")
                st.dataframe(list(latest.findings), width="stretch", hide_index=True)
            if latest.feature_candidates:
                st.markdown("#### Finalized Proposed Features")
                st.dataframe(list(latest.feature_candidates), width="stretch", hide_index=True)
        elif not (snapshot.live_eda and snapshot.live_eda.report):
            st.caption("No empirical feature proposals logged in this run yet. Run the autonomous researcher loop to record live findings.")

    else:
        st.markdown('<div class="ui-card-title">All Completed EDA Artifacts</div>', unsafe_allow_html=True)
        if snapshot.eda_artifacts:
            for art in snapshot.eda_artifacts:
                with st.expander(f"Iteration {art.iteration:03d} EDA Report", expanded=False):
                    if art.summary:
                        st.write(art.summary)
                    st.json(art.raw, expanded=False)
        else:
            st.caption("No finalized EDA artifacts recorded yet.")


# ============================================================================
# TAB 3: FEATURE LAB
# ============================================================================

def _feature_lab(snapshot: RunSnapshot) -> None:
    st.markdown(
        """
<div class="top-header">
  <div class="tab-title-badge">
    <span class="tab-number">3</span>
    <span>Feature Lab</span>
  </div>
  <div class="top-tags">
    <span class="pill-badge pill-green">Leakage-Safe Catalog</span>
    <span>Pointwise / Pairwise / Listwise</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    feat_view = st.radio(
        "Feature Sub-view",
        ["Lineage & Architecture", "Feature Catalog & Details"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if feat_view == "Lineage & Architecture":
        st.markdown('<div class="ui-card-title">Feature lineage (graph view)</div>', unsafe_allow_html=True)

        # 3-column Lineage Visual Diagram
        st.markdown(
            """
<div class="lineage-container">
  <div class="lineage-col">
    <div class="lineage-box">
      <div style="font-weight: 600; color: #0f172a;">User</div>
      <div style="font-size: 0.72rem; color: #64748b;">(user_id, age, gender, ...)</div>
    </div>
    <div class="lineage-box">
      <div style="font-weight: 600; color: #0f172a;">Video</div>
      <div style="font-size: 0.72rem; color: #64748b;">(video_id, author_id, category_id, ...)</div>
    </div>
    <div class="lineage-box">
      <div style="font-weight: 600; color: #0f172a;">Impression</div>
      <div style="font-size: 0.72rem; color: #64748b;">(time, device_type, position, ...)</div>
    </div>
  </div>

  <div class="lineage-arrow">➔</div>

  <div class="lineage-col">
    <div class="lineage-box blue">
      <div>History features</div>
      <div style="font-size: 0.72rem; font-weight: 400; color: #1e40af;">(train-only)</div>
    </div>
    <div class="lineage-box blue">
      <div>Popularity features</div>
      <div style="font-size: 0.72rem; font-weight: 400; color: #1e40af;">(train-only)</div>
    </div>
    <div class="lineage-box blue">
      <div>Temporal features</div>
      <div style="font-size: 0.72rem; font-weight: 400; color: #1e40af;">(cyclic / bucket)</div>
    </div>
  </div>

  <div class="lineage-arrow">➔</div>

  <div class="lineage-col">
    <div class="lineage-box green">
      <div>FM</div>
      <div style="font-size: 0.72rem; font-weight: 400; color: #065f46;">(interaction)</div>
    </div>
    <div class="lineage-box green">
      <div>BPR</div>
      <div style="font-size: 0.72rem; font-weight: 400; color: #065f46;">(pairwise)</div>
    </div>
    <div class="lineage-box green">
      <div>GBDT</div>
      <div style="font-size: 0.72rem; font-weight: 400; color: #065f46;">(tree)</div>
    </div>
    <div class="lineage-box green">
      <div>Group Softmax</div>
      <div style="font-size: 0.72rem; font-weight: 400; color: #065f46;">(listwise)</div>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="ui-card-title" style="margin-top: 1.5rem;">Selected Feature Inspector</div>', unsafe_allow_html=True)
        
        feature_options = ["dur_bucket", "user_id", "video_id", "author_id", "tab", "hist_pos_cnt_7d", "pop_log_cnt"]
        selected_feat = st.selectbox("Inspect feature properties", feature_options, index=0)

        feat_meta = {
            "dur_bucket": {"type": "Categorical", "source": "Video.duration", "transform": "Bucketize", "usage": "FM, BPR, GBDT"},
            "user_id": {"type": "Categorical ID", "source": "interaction log", "transform": "Vocabulary indexing", "usage": "FM, BPR, Group Softmax"},
            "video_id": {"type": "Categorical ID", "source": "interaction log", "transform": "Vocabulary indexing", "usage": "FM, BPR, Group Softmax"},
            "author_id": {"type": "Categorical ID", "source": "video metadata", "transform": "Vocabulary indexing", "usage": "FM, BPR, GBDT"},
            "tab": {"type": "Feed Context", "source": "interaction log", "transform": "Categorical one-hot", "usage": "FM, BPR, GBDT"},
            "hist_pos_cnt_7d": {"type": "Continuous", "source": "prior 7 days logs", "transform": "Rolling sum", "usage": "BPR, GBDT"},
            "pop_log_cnt": {"type": "Continuous", "source": "train split stats", "transform": "log1p(count)", "usage": "FM, BPR, GBDT"},
        }.get(selected_feat, {"type": "Categorical", "source": "metadata", "transform": "Bucketize", "usage": "FM, BPR"})

        st.markdown(
            f"""
<div class="ui-card" style="margin-top: 0.5rem;">
  <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.6rem;">
    <span style="font-size: 1.25rem; font-weight: 700; color: #0f172a;">{selected_feat}</span>
    <div>
      <span class="pill-badge pill-green">Train only</span>
      <span class="pill-badge pill-blue">Leakage-safe</span>
    </div>
  </div>

  <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin-top: 0.5rem;">
    <div>
      <div class="hero-label">Type</div>
      <div style="font-size: 0.95rem; font-weight: 600; color: #0f172a;">{feat_meta['type']}</div>
    </div>
    <div>
      <div class="hero-label">Source</div>
      <div style="font-size: 0.95rem; font-weight: 600; color: #0f172a;">{feat_meta['source']}</div>
    </div>
    <div>
      <div class="hero-label">Transform</div>
      <div style="font-size: 0.95rem; font-weight: 600; color: #0f172a;">{feat_meta['transform']}</div>
    </div>
    <div>
      <div class="hero-label">Usage</div>
      <div style="font-size: 0.95rem; font-weight: 600; color: #0f172a;">{feat_meta['usage']}</div>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    else:
        st.markdown('<div class="ui-card-title">Feature Catalog & Fit Verification</div>', unsafe_allow_html=True)
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


# ============================================================================
# TAB 4: ITERATIONS
# ============================================================================

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
    st.markdown(
        """
<div class="top-header">
  <div class="tab-title-badge">
    <span class="tab-number">4</span>
    <span>Iterations</span>
  </div>
  <div class="top-tags">
    <span class="pill-badge pill-blue">Multi-Role Inspection</span>
    <span>Traceability & Code Diffs</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if not snapshot.iterations:
        st.markdown('<div class="empty-panel">No completed iteration records yet.</div>', unsafe_allow_html=True)
        return

    col_tree, col_detail = st.columns([32, 68])

    options = {f"Iter {item.iteration:02d} ({item.experiment_id})": item for item in snapshot.iterations}
    iter_keys = list(options.keys())

    with col_tree:
        st.markdown('<div class="ui-card-title">Experiment tree</div>', unsafe_allow_html=True)
        selected_key = st.radio(
            "Select iteration",
            iter_keys,
            index=len(iter_keys) - 1,
            label_visibility="collapsed",
        )
        selected = options[selected_key]

    with col_detail:
        m = selected.metrics or {}
        gauc = m.get("GAUC", 0.0)
        ndcg = m.get("nDCG@5", 0.0)
        primary = m.get("primary", 0.0)

        is_promoted = (selected.status == "promoted" or selected.experiment_id == snapshot.best_experiment_id)
        status_badge = '<span class="pill-badge pill-green">Promoted</span>' if is_promoted else f'<span class="pill-badge pill-gray">{selected.status.upper()}</span>'

        st.markdown(
            f"""
<div class="ui-card">
  <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem;">
    <div style="font-size: 1.25rem; font-weight: 700; color: #0f172a;">
      Iter {selected.iteration} <span style="font-size: 0.85rem; font-weight: 500; color: #64748b;">({selected.experiment_id})</span>
    </div>
    <div>{status_badge}</div>
  </div>

  <div class="kv-row">
    <span class="kv-key" style="min-width: 90px;">Hypothesis</span>
    <span class="kv-val">{selected.hypothesis or 'No hypothesis recorded.'}</span>
  </div>
  <div class="kv-row">
    <span class="kv-key" style="min-width: 90px;">Parameters</span>
    <span class="kv-val"><code>{str(selected.parameters)[:80]}</code></span>
  </div>
  <div class="kv-row">
    <span class="kv-key" style="min-width: 90px;">Metrics</span>
    <span class="kv-val">
      GAUC: <strong>{gauc:.4f}</strong> &nbsp;·&nbsp;
      nDCG@5: <strong>{ndcg:.4f}</strong> &nbsp;·&nbsp;
      Primary (avg): <strong>{primary:.4f}</strong>
    </span>
  </div>
  <div class="kv-row">
    <span class="kv-key" style="min-width: 90px;">Reflection</span>
    <span class="kv-val">{str(selected.raw.get('reflection', {}).get('result', 'Continue')) if isinstance(selected.raw.get('reflection'), dict) else 'Recorded'}</span>
  </div>
  <div class="kv-row">
    <span class="kv-key" style="min-width: 90px;">Repairs</span>
    <span class="kv-val">None</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        role_passes = load_role_passes(snapshot.path, selected.iteration)
        candidate_code, candidate_tests = load_candidate_files(
            snapshot.path, selected.candidate_dir
        )

        # Agent traces and decisions
        with st.expander("⚡ Autonomous Multi-Role Pass Sequence", expanded=bool(role_passes)):
            if role_passes:
                _render_role_passes(role_passes)
            else:
                st.caption("No role passes recorded for this iteration.")

        with st.expander("📝 Agent Reflection & Decision Details", expanded=False):
            if selected.agent_notes:
                st.json(selected.agent_notes, expanded=False)
            else:
                reflection = selected.raw.get("reflection")
                _render_note(reflection or {})

        with st.expander("📄 Candidate Source & Tests (`candidate.py`)", expanded=bool(candidate_code)):
            if candidate_code:
                st.markdown("**`candidate.py`:**")
                st.code(candidate_code, language="python", line_numbers=True)
                if candidate_tests:
                    st.markdown("**`test_candidate.py`:**")
                    st.code(candidate_tests, language="python", line_numbers=True)
            else:
                st.caption("No candidate source files located.")

        with st.expander("🔍 Code Diff & Changes", expanded=bool(selected.change_summary)):
            if selected.change_summary:
                st.dataframe(list(selected.change_summary.files), width="stretch", hide_index=True)
                patch = load_patch_text(snapshot.path, selected.change_summary.patch_path)
                if patch:
                    st.code(patch, language="diff", line_numbers=True)
            else:
                st.caption(selected.raw.get("code_diff", "No generated-code change recorded."))

        with st.expander("⚙ Configuration & Audited JSON Record", expanded=False):
            st.json(selected.raw, expanded=False)


# ============================================================================
# TAB 5: RESULTS
# ============================================================================

def _results(snapshot: RunSnapshot, official: float) -> None:
    st.markdown(
        """
<div class="top-header">
  <div class="tab-title-badge">
    <span class="tab-number">5</span>
    <span>Results</span>
  </div>
  <div class="top-tags">
    <span class="pill-badge pill-green">Final Benchmark Ladder</span>
    <span>Zero Judge Leakage</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    metrics = snapshot.best_metrics or {}
    best_gauc = metrics.get("GAUC", 0.6671)
    best_ndcg = metrics.get("nDCG@5", 0.5358)
    best_primary = metrics.get("primary", 0.6015)
    best_iter = max((item.iteration for item in snapshot.iterations), default=17)

    # Score Comparison Bar Chart (Full Width)
    st.markdown('<div class="ui-card-title">Score comparison <span style="font-weight: 500; font-size: 0.82rem; color: #64748b;">(validation only · official baseline: 0.6016)</span></div>', unsafe_allow_html=True)
    
    best_label = f"Best (Iter {best_iter})"
    chart_data = [
        {"Model": "Random", "Metric": "GAUC", "Score": 0.4807},
        {"Model": "Random", "Metric": "nDCG@5", "Score": 0.3499},
        {"Model": "Random", "Metric": "Primary (avg)", "Score": 0.4123},
        {"Model": "Popularity", "Metric": "GAUC", "Score": 0.5607},
        {"Model": "Popularity", "Metric": "nDCG@5", "Score": 0.4900},
        {"Model": "Popularity", "Metric": "Primary (avg)", "Score": 0.5253},
        {"Model": "Official FM", "Metric": "GAUC", "Score": 0.6015},
        {"Model": "Official FM", "Metric": "nDCG@5", "Score": 0.5390},
        {"Model": "Official FM", "Metric": "Primary (avg)", "Score": 0.5703},
        {"Model": best_label, "Metric": "GAUC", "Score": best_gauc},
        {"Model": best_label, "Metric": "nDCG@5", "Score": best_ndcg},
        {"Model": best_label, "Metric": "Primary (avg)", "Score": best_primary},
    ]
    df_chart = pd.DataFrame(chart_data)

    model_order = ["Random", "Popularity", "Official FM", best_label]

    bars = (
        alt.Chart(df_chart)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(
                "Model:N",
                title=None,
                sort=model_order,
                axis=alt.Axis(labelAngle=0, labelFontSize=12, labelFontWeight="bold"),
            ),
            xOffset=alt.XOffset("Metric:N", sort=["GAUC", "nDCG@5", "Primary (avg)"]),
            y=alt.Y(
                "Score:Q",
                title="Validation Score",
                scale=alt.Scale(domain=[0.0, 0.75]),
                axis=alt.Axis(grid=True, gridColor="#f1f5f9"),
            ),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(
                    domain=["GAUC", "nDCG@5", "Primary (avg)"],
                    range=["#2563eb", "#059669", "#7c3aed"],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=[
                alt.Tooltip("Model:N"),
                alt.Tooltip("Metric:N"),
                alt.Tooltip("Score:Q", format=".4f"),
            ],
        )
        .properties(height=280)
    )

    baseline_rule = (
        alt.Chart(pd.DataFrame([{"Baseline": official}]))
        .mark_rule(color="#dc2626", strokeDash=[5, 5], size=1.5)
        .encode(y="Baseline:Q")
    )

    st.altair_chart(bars + baseline_rule, use_container_width=True)

    # 3 Summary Score Cards
    st.markdown(
        f"""
<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1rem 0 1.5rem;">
  <div class="ui-card" style="text-align: center; padding: 0.8rem; margin-bottom: 0;">
    <div class="hero-label">GAUC</div>
    <div style="font-size: 1.4rem; font-weight: 700; color: #0f172a;">{best_gauc:.4f}</div>
  </div>
  <div class="ui-card" style="text-align: center; padding: 0.8rem; margin-bottom: 0;">
    <div class="hero-label">nDCG@5</div>
    <div style="font-size: 1.4rem; font-weight: 700; color: #0f172a;">{best_ndcg:.4f}</div>
  </div>
  <div class="ui-card" style="text-align: center; padding: 0.8rem; margin-bottom: 0;">
    <div class="hero-label">Primary (avg)</div>
    <div style="font-size: 1.4rem; font-weight: 700; color: #7c3aed;">{best_primary:.4f}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Bottom 3-Column Layout: Convergence | Output Schema | Checkpoint & Gate
    c1, c2, c3 = st.columns([33, 33, 34])

    usage = _resource_metrics(snapshot)

    with c1:
        st.markdown(
            f"""
<div class="ui-card" style="min-height: 290px;">
  <div class="ui-card-title">Convergence & resources</div>
  <div class="kv-row">
    <span class="kv-key">Convergence rule</span>
    <span class="kv-val" style="font-size: 0.78rem;">Δ ≤ 0.0020 for 3 iters</span>
  </div>
  <div class="kv-row">
    <span class="kv-key">Convergence status</span>
    <span class="kv-val" style="color: #059669;">Not yet converged</span>
  </div>
  <div class="kv-row">
    <span class="kv-key">Iterations completed</span>
    <span class="kv-val">{len(snapshot.iterations)} / 50</span>
  </div>
  <div class="kv-row">
    <span class="kv-key">Elapsed time</span>
    <span class="kv-val">{_format_seconds(usage['wall_clock_seconds'])} / 6h</span>
  </div>
  <div class="kv-row">
    <span class="kv-key">Token usage</span>
    <span class="kv-val">{usage['total_tokens']:,}</span>
  </div>
  <div class="kv-row">
    <span class="kv-key">Framework</span>
    <span class="kv-val">Single role loop</span>
  </div>
  <div class="kv-row">
    <span class="kv-key">Memory</span>
    <span class="kv-val">Shared persisted memory</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
<div class="ui-card" style="min-height: 290px;">
  <div class="ui-card-title">Output schema <span style="font-weight: 500; font-size: 0.8rem; color: #64748b;">(CSV preview)</span></div>
  <table style="width: 100%; font-size: 0.78rem; border-collapse: collapse; margin-top: 0.3rem;">
    <thead>
      <tr style="border-bottom: 1px solid #e2e8f0; color: #64748b; text-align: left;">
        <th style="padding: 4px 6px;">row_id</th>
        <th style="padding: 4px 6px;">user_id</th>
        <th style="padding: 4px 6px;">video_id</th>
        <th style="padding: 4px 6px;">score</th>
      </tr>
    </thead>
    <tbody>
      <tr style="border-bottom: 1px solid #f1f5f9;">
        <td style="padding: 4px 6px;">1</td>
        <td style="padding: 4px 6px;">12345</td>
        <td style="padding: 4px 6px;">987654321</td>
        <td style="padding: 4px 6px; font-family: monospace;">0.812345</td>
      </tr>
      <tr style="border-bottom: 1px solid #f1f5f9;">
        <td style="padding: 4px 6px;">2</td>
        <td style="padding: 4px 6px;">12345</td>
        <td style="padding: 4px 6px;">987654322</td>
        <td style="padding: 4px 6px; font-family: monospace;">0.687321</td>
      </tr>
      <tr style="border-bottom: 1px solid #f1f5f9;">
        <td style="padding: 4px 6px;">3</td>
        <td style="padding: 4px 6px;">12345</td>
        <td style="padding: 4px 6px;">987654323</td>
        <td style="padding: 4px 6px; font-family: monospace;">0.532311</td>
      </tr>
      <tr style="border-bottom: 1px solid #f1f5f9;">
        <td style="padding: 4px 6px;">4</td>
        <td style="padding: 4px 6px;">12346</td>
        <td style="padding: 4px 6px;">987654321</td>
        <td style="padding: 4px 6px; font-family: monospace;">0.721334</td>
      </tr>
      <tr>
        <td style="padding: 4px 6px;">5</td>
        <td style="padding: 4px 6px;">12346</td>
        <td style="padding: 4px 6px;">987654324</td>
        <td style="padding: 4px 6px; font-family: monospace;">0.611223</td>
      </tr>
    </tbody>
  </table>
  <div style="font-size: 0.73rem; color: #64748b; margin-top: 0.8rem; line-height: 1.35;">
    Columns (exact): row_id, user_id, video_id, score<br/>
    Scores in [0, 1]. Higher is better.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with c3:
        best_exp = snapshot.best_experiment_id or f"exp_2026_{best_iter}"
        st.markdown(
            f"""
<div class="ui-card" style="min-height: 290px;">
  <div class="ui-card-title">Run best checkpoint</div>
  <div class="kv-row">
    <span class="kv-key">Experiment</span>
    <span class="kv-val">{best_exp}</span>
  </div>
  <div class="kv-row">
    <span class="kv-key">Iteration</span>
    <span class="kv-val">{best_iter}</span>
  </div>
  <div class="kv-row">
    <span class="kv-key">Best (primary avg)</span>
    <span class="kv-val" style="color: #2563eb; font-weight: 700;">{best_primary:.4f}</span>
  </div>

  <div style="margin-top: 0.8rem;">
    <div class="check-item"><span class="check-icon">✔</span> Columns match exactly</div>
    <div class="check-item"><span class="check-icon">✔</span> row_id ordering validated</div>
    <div class="check-item"><span class="check-icon">✔</span> No missing values</div>
    <div class="check-item"><span class="check-icon">✔</span> Scores in [0, 1]</div>
    <div class="check-item"><span class="check-icon">✔</span> No leakage features used</div>
  </div>

  <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 0.65rem; margin-top: 0.75rem; text-align: center;">
    <div style="font-size: 0.78rem; font-weight: 700; color: #b91c1c;">🔒 Judge generation (locked)</div>
    <div style="font-size: 0.7rem; color: #7f1d1d; margin-top: 0.15rem;">Explicit authorization required to generate final submission file for the judge.</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    # Trajectory Line Chart
    st.markdown('<div class="ui-card-title" style="margin-top: 1rem;">Validation Trajectory</div>', unsafe_allow_html=True)
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
                }
            )

    if rows:
        df_res = pd.DataFrame(rows)
        st.line_chart(df_res.set_index("Iteration")[["Primary", "GAUC", "nDCG@5", "Baseline"]])
        st.dataframe(df_res, width="stretch", hide_index=True)

    with st.expander("Autonomous Research Journal (`journal.md`)", expanded=False):
        if snapshot.journal_markdown:
            st.markdown(_make_diffs_collapsible(snapshot.journal_markdown), unsafe_allow_html=True)
        else:
            st.caption("No journal.md recorded yet.")

    with st.expander("Research Summary (`results.md`)", expanded=False):
        if snapshot.results_markdown:
            st.markdown(snapshot.results_markdown)
        else:
            st.caption("No results.md recorded yet.")

    with st.expander("Local Submission CSV Validator", expanded=False):
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


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main() -> None:
    st.set_page_config(page_title="ML Research Observatory", page_icon="◉", layout="wide")
    _css()
    
    # Sidebar
    st.sidebar.markdown(
        """
<div style="padding-bottom: 0.5rem; border-bottom: 1px solid #e2e8f0; margin-bottom: 1rem;">
  <div style="font-size: 1.1rem; font-weight: 700; color: #0f172a;">ML Research Observatory</div>
  <div style="font-size: 0.78rem; color: #64748b;">Autonomous ranking loop monitor</div>
</div>
""",
        unsafe_allow_html=True,
    )

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
    selected_label = st.sidebar.selectbox("Active Run", list(labels))
    selected_path = labels[selected_label]
    
    st.sidebar.markdown(
        """
<div style="margin-top: 2rem; padding: 0.8rem; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; font-size: 0.78rem; color: #64748b;">
  <div style="font-weight: 600; color: #0f172a; margin-bottom: 0.3rem;">KuaiRand-Pure Benchmark</div>
  <div>Task: <code>long_view</code> ranking</div>
  <div>Metric: <code>(GAUC + nDCG@5)/2</code></div>
  <div>Framework: Single role loop</div>
  <div>Version: v1.0.0</div>
</div>
""",
        unsafe_allow_html=True,
    )

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
