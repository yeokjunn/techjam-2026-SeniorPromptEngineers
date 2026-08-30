"""Run report renderer (review item I16).

Writes journal.md and results.md from the run's audit trail.
"""

from __future__ import annotations

import difflib
import json
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_NR = "not reported"

# Official baseline, per-metric, from the problem statement (docs/problem_statement.md, the
# "Official baseline" bullet) and echoed in kuairand-starter-kit/README.en.md. Judging scores
# `mean over m of (score_agent(m) - score_baseline(m))`, so a primary-only baseline cannot
# produce the graded number. `official_validation_metrics` in the run config overrides these.
OFFICIAL_VALIDATION_BASELINE = {"GAUC": 0.6674, "nDCG@5": 0.5357, "primary": 0.6016}
OFFICIAL_TEST_BASELINE = {"GAUC": 0.6610, "nDCG@5": 0.5282, "primary": 0.5946}
#: The attainable ceiling on the hidden test: a perfect ranking scores this, not 1.0, because
#: 27.1% of users have no positive and 9.2% are all-positive. Random scoring sits at 0.4753.
ORACLE_TEST_PRIMARY = 0.8645
RANDOM_TEST_PRIMARY = 0.4753
_SEED_SUFFIX = re.compile(r"_seed\d+$")


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _load_jsonl(path: Path) -> list[dict]:
    entries: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass
    return entries


def _resolve_generated_root(run_config: dict | None) -> Path:
    if run_config is None:
        return REPO_ROOT / "generated_experiments"
    raw = run_config.get("generated_root", "generated_experiments")
    p = Path(raw)
    return p if p.is_absolute() else REPO_ROOT / p


def _candidate_path(generated_root: Path, run_id: str, iteration: int,
                     candidate_id: str) -> Path:
    return generated_root / run_id / f"{iteration:03d}_{candidate_id}" / "candidate.py"


def _read_candidate_source(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None


def _builder_pass_code(run_dir: Path, iteration: int) -> str | None:
    pattern = f"{iteration:03d}_builder_*.json"
    matches = sorted(run_dir.joinpath("passes").glob(pattern))
    for match in matches:
        data = _load_json(match)
        if data is None:
            continue
        result = data.get("result", {})
        code = result.get("data", {}).get("code")
        if code:
            return code
    return None


def _make_diff(parent_src: str | None, child_src: str, child_path: str,
               max_lines: int = 200) -> list[str]:
    parent_lines = (parent_src or "").splitlines(keepends=True)
    child_lines = child_src.splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        parent_lines, child_lines,
        fromfile="parent", tofile="candidate",
    ))
    lines: list[str] = ["```diff"]
    if not diff:
        lines.append("(no differences)")
    elif len(diff) > max_lines:
        lines.extend(line.rstrip("\n") for line in diff[:max_lines])
        lines.append(f"… truncated, full source at {child_path}")
    else:
        lines.extend(line.rstrip("\n") for line in diff)
    lines.append("```")
    return lines


def _display_path(raw: str, run_dir_name: str) -> str:
    # The writers normalise to repo-relative (run_candidate.py:144 and the
    # relative_to calls around it), so the ordinary case passes through whole —
    # shortening it would hide whether an artefact sits under runs/ or
    # generated_experiments/. Runs recorded before that normalisation carry
    # absolute paths (the retired baseline run held Windows OneDrive ones), so
    # anchor only those, rather than printing someone's home directory.
    text = str(raw)
    parts = [p for p in re.split(r"[\\/]+", text) if p]
    if not parts:
        return text
    if not (text.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", text)):
        return "/".join(parts)
    if run_dir_name in parts:
        return "/".join(parts[parts.index(run_dir_name):])
    return "/".join(parts[-3:])


def _fmt_delta(val: float | None) -> str:
    if val is None:
        return _NR
    return f"{val:+.4f}"


def _build_delta_map(results: list[dict] | None) -> dict[str, float | None]:
    if results is None:
        return {}
    return {
        r.get("experiment_id", ""): r.get("delta_vs_baseline")
        for r in results
    }


def _tokens_by_iteration(memory: list[dict]) -> dict[int, int]:
    totals: dict[int, int] = defaultdict(int)
    for rec in memory:
        it = rec.get("iteration")
        usage = rec.get("usage", {})
        if it is not None:
            totals[int(it)] += int(usage.get("total_tokens", 0))
    return totals


def _tokens_by_role(memory: list[dict]) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for rec in memory:
        role = rec.get("role", "unknown")
        usage = rec.get("usage", {})
        totals[role] += int(usage.get("total_tokens", 0))
    return totals


def _render_journal(records: list[dict], run_dir: Path, run_config: dict | None,
                    results: list[dict] | None, memory: list[dict]) -> str:
    generated_root = _resolve_generated_root(run_config)
    run_id = (run_config or {}).get("name", run_dir.name)
    if run_config and "budgets" in run_config:
        run_id = run_dir.name
    baseline_primary = float((run_config or {}).get("official_validation_baseline", 0.6016))
    delta_map = _build_delta_map(results)
    iter_tokens = _tokens_by_iteration(memory)

    id_to_record: dict[str, dict] = {}
    for rec in records:
        manifest = rec.get("manifest", {})
        cid = manifest.get("candidate_id") if manifest else None
        eid = rec.get("experiment_id")
        if cid:
            id_to_record[cid] = rec
        if eid:
            id_to_record[eid] = rec

    all_ids: set[str] = set()
    for rec in records:
        manifest = rec.get("manifest") or {}
        cid = manifest.get("candidate_id") or rec.get("experiment_id", "unknown")
        all_ids.add(cid)

    lines: list[str] = ["# Experiment Journal", ""]
    best_so_far = 0.0
    replications: dict[str, list[dict]] = defaultdict(list)

    for rec in records:
        manifest = rec.get("manifest") or {}
        candidate_id = manifest.get("candidate_id") or rec.get("experiment_id", "unknown")
        m = _SEED_SUFFIX.search(candidate_id)
        if m:
            source_id = candidate_id[:m.start()]
            if source_id in all_ids:
                replications[source_id].append(rec)

    replication_ids = set()
    for lst in replications.values():
        for rec in lst:
            manifest = rec.get("manifest") or {}
            replication_ids.add(manifest.get("candidate_id") or rec.get("experiment_id", "unknown"))

    for rec in records:
        iteration = rec.get("iteration", "?")
        status = rec.get("status", _NR)

        manifest = rec.get("manifest") or {}
        candidate_id = manifest.get("candidate_id") or rec.get("experiment_id", "unknown")

        if candidate_id in replication_ids:
            continue

        proposal = rec.get("proposal") or {}
        preflight = rec.get("preflight") or {}
        outcome = rec.get("outcome") or {}
        postflight = rec.get("postflight") or {}

        hypothesis = proposal.get("hypothesis") or rec.get("hypothesis", _NR)

        lines.append(f"## Iteration {iteration} — {candidate_id}")
        lines.append("")

        if status == "critic_rejected":
            rationale = proposal.get("rationale", _NR)
            reject_reason = preflight.get("rationale", _NR)
            lines.append(f"**Hypothesis:** {hypothesis}")
            lines.append("")
            lines.append(f"**Rationale:** {rationale}")
            lines.append("")
            lines.append("**Status:** Rejected before code generation")
            lines.append("")
            lines.append(f"**Rejection reason:** {reject_reason}")
            lines.append("")
            continue

        is_baseline = "kind" in rec and "proposal" not in rec
        if is_baseline:
            _render_baseline_record(lines, rec, outcome, best_so_far, baseline_primary)
            metrics = outcome.get("metrics") or {}
            primary = float(metrics.get("primary", 0))
            if primary > best_so_far:
                best_so_far = primary
            continue

        rationale = proposal.get("rationale", _NR)
        evidence = proposal.get("evidence", [])
        family = manifest.get("family") or proposal.get("family", _NR)
        parameters = manifest.get("parameters") or proposal.get("parameters", {})
        repairs = rec.get("repairs", 0)

        lines.append(f"**Hypothesis:** {hypothesis}")
        lines.append("")
        lines.append(f"**Rationale:** {rationale}")
        lines.append("")

        if evidence:
            lines.append("**Evidence:**")
            for src in evidence:
                title = src.get("title", "untitled")
                url = src.get("url", "")
                lines.append(f"- [{title}]({url})" if url else f"- {title}")
            lines.append("")

        lines.append(f"**Family:** {family}  ")
        lines.append(f"**Parameters:** `{json.dumps(parameters, sort_keys=True)}`")
        lines.append("")

        child_src = None
        source_label = ""
        child_path_str = ""
        if candidate_id != "unknown":
            cp = _candidate_path(generated_root, run_dir.name, int(iteration), candidate_id)
            child_src = _read_candidate_source(cp)
            child_path_str = str(_display_path(cp, run_dir.name))
            if child_src is not None:
                source_label = "file"
        if child_src is None:
            child_src = _builder_pass_code(run_dir, int(iteration)) if iteration != "?" else None
            source_label = "builder pass" if child_src else ""
            child_path_str = f"passes/{int(iteration):03d}_builder_*.json" if iteration != "?" else ""

        if child_src:
            parent_src = None
            parent_id = proposal.get("parent_experiment")
            if parent_id and parent_id in id_to_record:
                prec = id_to_record[parent_id]
                pm = prec.get("manifest") or {}
                pid = pm.get("candidate_id") or prec.get("experiment_id", "")
                pit = prec.get("iteration", 0)
                pp = _candidate_path(generated_root, run_dir.name, int(pit), pid)
                parent_src = _read_candidate_source(pp)

            diff_lines = _make_diff(parent_src, child_src, child_path_str)
            if source_label == "builder pass":
                lines.append(f"*Code source: {source_label} (generated directory absent)*")
                lines.append("")
            lines.extend(diff_lines)
            lines.append("")

        metrics = outcome.get("metrics") or {}
        primary = float(metrics.get("primary", 0))
        delta_baseline = delta_map.get(candidate_id)
        if delta_baseline is None and primary > 0:
            delta_baseline = primary - baseline_primary

        if metrics:
            best_before = best_so_far
            if primary > best_so_far:
                best_so_far = primary
            lines.append("**Metrics:**")
            lines.append("")
            lines.append("| Metric | Value | Δ vs best-so-far | Δ vs baseline |")
            lines.append("|---|---|---|---|")
            for key in ("GAUC", "nDCG@5", "primary"):
                val = metrics.get(key)
                val_str = f"{val:.4f}" if val is not None else _NR
                d_best = _fmt_delta(val - best_before if val is not None and key == "primary" else None)
                d_base = _fmt_delta(delta_baseline if key == "primary" else None)
                lines.append(f"| {key} | {val_str} | {d_best} | {d_base} |")
            lines.append("")

        test_scores_path = outcome.get("test_scores_path")
        if test_scores_path:
            lines.append(
                f"**Test scores:** `{_display_path(test_scores_path, run_dir.name)}`"
            )
            lines.append("")

        failure_class = outcome.get("failure_class")
        error = outcome.get("error")
        recovery = outcome.get("recovery")
        if failure_class or error:
            lines.append("**Errors:**")
            lines.append("")
            if failure_class:
                lines.append(f"- Failure class: {failure_class}")
            if error:
                err_short = error[:500] + "…" if len(error) > 500 else error
                lines.append(f"- Error: {err_short}")
            if recovery:
                lines.append(f"- Recovery: {recovery}")
            if repairs:
                lines.append(f"- Repairs attempted: {repairs}")
            lines.append("")

        if preflight:
            pf_decision = preflight.get("decision", _NR)
            pf_rationale = preflight.get("rationale", _NR)
            lines.append(f"**Critic (preflight):** {pf_decision} — {pf_rationale}")
            lines.append("")

        if postflight:
            po_decision = postflight.get("decision", _NR)
            po_rationale = postflight.get("rationale", _NR)
            lines.append(f"**Critic (postflight):** {po_decision} — {po_rationale}")
            lines.append("")

        duration = outcome.get("duration_seconds")
        tokens = iter_tokens.get(int(iteration), 0) if iteration != "?" else 0
        parts = []
        if duration is not None:
            parts.append(f"{duration:.1f} s")
        if tokens:
            parts.append(f"{tokens:,} tokens")
        if parts:
            lines.append(f"**Resources:** {' · '.join(parts)}")
            lines.append("")

        reps = replications.get(candidate_id, [])
        if reps:
            lines.append("**Replications:**")
            lines.append("")
            lines.append("| Seed | Primary | Status |")
            lines.append("|---|---|---|")
            primaries = []
            for rrec in reps:
                rom = rrec.get("outcome") or {}
                rm = rom.get("metrics") or {}
                rp = rm.get("primary")
                rcid = (rrec.get("manifest") or {}).get("candidate_id", "?")
                m = _SEED_SUFFIX.search(rcid)
                seed_part = m.group(0)[5:] if m else "?"
                rstatus = rrec.get("status", _NR)
                if rp is not None:
                    primaries.append(rp)
                    lines.append(f"| {seed_part} | {rp:.4f} | {rstatus} |")
                else:
                    lines.append(f"| {seed_part} | {_NR} | {rstatus} |")
            if primaries:
                mean_p = sum(primaries) / len(primaries)
                spread = max(primaries) - min(primaries) if len(primaries) > 1 else 0
                lines.append(f"| **mean** | **{mean_p:.4f}** | spread {spread:.4f} |")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines) + "\n"


def _render_baseline_record(lines: list[str], rec: dict, outcome: dict,
                            best_so_far: float, baseline_primary: float) -> None:
    hypothesis = rec.get("hypothesis", _NR)
    configuration = rec.get("configuration", {})
    code_diff = rec.get("code_diff", _NR)
    reflection = rec.get("reflection") or {}
    metrics = outcome.get("metrics") or {}
    primary = float(metrics.get("primary", 0))

    lines.append(f"**Hypothesis:** {hypothesis}")
    lines.append("")
    lines.append(f"**Configuration:** `{json.dumps(configuration, sort_keys=True)}`")
    lines.append("")
    lines.append(f"**Code change:** {code_diff}")
    lines.append("")

    if metrics:
        delta_baseline = primary - baseline_primary
        lines.append("**Metrics:**")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for key in ("GAUC", "nDCG@5", "primary"):
            val = metrics.get(key)
            val_str = f"{val:.4f}" if val is not None else _NR
            lines.append(f"| {key} | {val_str} |")
        lines.append(f"| Δ vs baseline | {delta_baseline:+.4f} |")
        lines.append("")

    ref_summary = reflection.get("summary", "")
    ref_decision = reflection.get("decision", "")
    if ref_summary or ref_decision:
        lines.append(f"**Reflection:** {ref_decision} — {ref_summary}")
        lines.append("")

    duration = outcome.get("duration_seconds")
    if duration is not None:
        lines.append(f"**Duration:** {duration:.1f} s")
        lines.append("")

    lines.append("---")
    lines.append("")


def _load_interventions(run_dir: Path | None, summary_dir: Path | None = None) -> list[str]:
    if run_dir is None:
        return []
    reasons: list[str] = []
    for name in ("interventions.json", "interventions.jsonl"):
        path = run_dir / name
        if not path.is_file():
            continue
        if name.endswith(".jsonl"):
            for entry in _load_jsonl(path):
                reasons.append(str(entry.get("reason", entry)))
        else:
            data = _load_json(path)
            if isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict):
                        reasons.append(str(entry.get("reason", entry)))
                    else:
                        reasons.append(str(entry))
    return reasons


def render_reports(run_dir: Path) -> None:
    """Write journal.md and results.md from the run's audit trail."""
    run_dir = Path(run_dir)
    has_iterations = (run_dir / "iterations.jsonl").is_file()
    has_summary = (run_dir / "summary.json").is_file()
    if not has_iterations and not has_summary:
        return None

    records = _load_jsonl(run_dir / "iterations.jsonl")
    summary = _load_json(run_dir / "summary.json")
    results = _load_json(run_dir / "results.json")
    run_config = _load_json(run_dir / "run_config.json")
    resources = _load_json(run_dir / "resources.json")
    memory = _load_jsonl(run_dir / "research_memory.jsonl")

    journal = _render_journal(records, run_dir, run_config, results, memory)
    (run_dir / "journal.md").write_text(journal, encoding="utf-8")

    results_md = _render_results_full(
        summary, run_config, resources, memory, records, run_dir,
    )
    (run_dir / "results.md").write_text(results_md, encoding="utf-8")

    return None


def _render_results_full(summary: dict | None, run_config: dict | None,
                         resources: dict | None, memory: list[dict],
                         records: list[dict], run_dir: Path) -> str:
    summary = summary or {}
    run_config = run_config or {}
    resources = resources or {}
    best = summary.get("best") or {}
    best_metrics = best.get("metrics") or {}
    gate = summary.get("gate") or {}
    token_usage = summary.get("token_usage") or {}
    baseline_primary = float(run_config.get("official_validation_baseline", 0.6016))
    max_iterations = int(run_config.get("budgets", {}).get("max_iterations", 50))

    lines: list[str] = ["# Results Summary", ""]

    lines.append("## Validation Performance")
    lines.append("")
    # Per-metric baselines: the judged score is the mean of per-metric deltas, so a
    # primary-only baseline cannot produce it. Config may override the published numbers.
    baselines = dict(OFFICIAL_VALIDATION_BASELINE)
    baselines.update(run_config.get("official_validation_metrics") or {})
    baselines["primary"] = baseline_primary

    lines.append("| Metric | Official baseline | Best | Δ |")
    lines.append("|---|---|---|---|")
    scored_deltas: list[float] = []
    for key in ("GAUC", "nDCG@5", "primary"):
        bval = best_metrics.get(key)
        base = baselines.get(key)
        bval_str = f"{bval:.4f}" if bval is not None else _NR
        base_str = f"{base:.4f}" if base is not None else "—"
        delta_value = bval - base if (bval is not None and base is not None) else None
        if delta_value is not None and key in ("GAUC", "nDCG@5"):
            scored_deltas.append(delta_value)
        lines.append(f"| {key} | {base_str} | {bval_str} | {_fmt_delta(delta_value)} |")
    lines.append("")
    if scored_deltas:
        # score_dataset = mean over m of delta(m), over the two scored metrics.
        mean_delta = sum(scored_deltas) / len(scored_deltas)
        lines.append(
            f"**Validation score_dataset (mean of GAUC and nDCG@5 deltas): {mean_delta:+.4f}**"
        )
        lines.append("")
        lines.append(
            "This applies the judging formula to validation. The ranked score uses the same "
            "formula on the hidden test, which is scored once and is not computable here; the "
            f"official test baseline is primary {OFFICIAL_TEST_BASELINE['primary']:.4f} "
            f"(GAUC {OFFICIAL_TEST_BASELINE['GAUC']:.4f} / nDCG@5 "
            f"{OFFICIAL_TEST_BASELINE['nDCG@5']:.4f})."
        )
        lines.append("")
        primary_best = best_metrics.get("primary")
        if primary_best is not None:
            # Progress is judged against the 0.8645 attainable ceiling, not against 1.0.
            span = ORACLE_TEST_PRIMARY - OFFICIAL_TEST_BASELINE["primary"]
            captured = (primary_best - baselines["primary"]) / span * 100.0
            lines.append(
                f"- Headroom context: the attainable ceiling is primary "
                f"{ORACLE_TEST_PRIMARY:.4f} (random {RANDOM_TEST_PRIMARY:.4f}); this run's "
                f"validation gain covers {captured:.1f}% of the baseline-to-ceiling span."
            )
            lines.append("")

    select = best_metrics.get("select_primary")
    report = best_metrics.get("report_primary")
    if select is not None and report is not None:
        lines.append("### Selection vs reporting half")
        lines.append("")
        lines.append(
            "Validation is split by user into a selection half (early stopping and candidate "
            "choice) and a reporting half (never consulted during training). The reporting "
            "number is the honest estimate; a large gap between them is selection noise, not a "
            "gain."
        )
        lines.append("")
        lines.append("| Half | primary | used for |")
        lines.append("|---|---|---|")
        lines.append(f"| selection | {select:.4f} | early stopping, best-candidate choice |")
        lines.append(f"| reporting | {report:.4f} | none — held out |")
        lines.append(f"| **gap** | **{select - report:+.4f}** | selection effect |")
        lines.append("")

    lines.append("## Test Submission")
    lines.append("")
    gate_status = gate.get("status", _NR)
    submission_path = gate.get("submission_path", _NR)
    lines.append(f"- Gate status: {gate_status}")
    lines.append(f"- Submission path: {submission_path}")
    lines.append("")

    lines.append("## Token Usage")
    lines.append("")
    role_tokens = _tokens_by_role(memory)
    if role_tokens:
        lines.append("| Role | Tokens |")
        lines.append("|---|---|")
        for role in sorted(role_tokens):
            lines.append(f"| {role} | {role_tokens[role]:,} |")
        total = int(token_usage.get("total_tokens", sum(role_tokens.values())))
        lines.append(f"| **total** | **{total:,}** |")
    else:
        total = int(token_usage.get("total_tokens", 0))
        lines.append(f"- Total tokens: {total:,}")
    lines.append("")

    lines.append("## Compute")
    lines.append("")
    wall_clock = float(summary.get("wall_clock_seconds", resources.get("wall_clock_seconds", 0)))
    lines.append(f"- Wall-clock: {wall_clock:.0f} s ({wall_clock / 3600:.2f} h)")
    gpu_hours = float(resources.get("gpu_hours", 0.0))
    lines.append(f"- GPU-hours: {gpu_hours:.1f}")
    lines.append("")

    lines.append("## Iterations")
    lines.append("")
    total_iters = int(summary.get("iterations", len(records)))
    failed = sum(1 for r in records if r.get("status") == "failed")
    rejected = sum(1 for r in records if r.get("status") == "critic_rejected")
    lines.append(f"- Iterations used: {total_iters} of {max_iterations}")
    lines.append(f"- Failed: {failed}")
    lines.append(f"- Rejected before code: {rejected}")
    lines.append("")

    lines.append("## Convergence")
    lines.append("")
    stop_reason = summary.get("stop_reason", _NR)
    converged_official = summary.get("converged_official", _NR)
    lines.append(f"- Stop reason: {stop_reason}")
    lines.append(f"- Converged (official rule): {converged_official}")
    lines.append("")

    lines.append("## Interventions")
    lines.append("")
    intervention_count = int(summary.get("manual_interventions", 0))
    lines.append(f"- Count: {intervention_count}")
    reasons = _load_interventions(run_dir)
    if reasons:
        lines.append("- Reasons:")
        for reason in reasons:
            lines.append(f"  - {reason}")
    else:
        lines.append("- Reasons: none recorded")
    lines.append("")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.agent.report <run_dir>", file=sys.stderr)
        sys.exit(1)
    render_reports(Path(sys.argv[1]))
