# Lightweight ML Research Dashboard

## Summary

Build a local, read-only Streamlit dashboard following the refined minimal mockup. Its primary job is to show which coarse stage the autonomous agent is executing, what controlled changes it made, and the structured evidence/decision trace behind each action. It also visualizes completed run artifacts and trusted train/validation summaries without launching experiments or accessing `data/judge/**`.

Use five tabs: Pipeline, EDA, Feature Lab, Iterations, and Results. Represent the system accurately as one role-based loop with shared memory—not separate autonomous agents.

## Implementation Changes

- Add an optional `requirements-ui.txt` with Streamlit; keep the core agent dependencies and Python 3.9 support unchanged.
- Add `configs/ui.json` for run root, EDA-profile location, official baseline, and five-second active-run refresh interval.
- Add a small observability contract under each research run:
  - `activity.json` is an atomically replaced snapshot of the latest stage transition.
  - `activity.jsonl` is the append-only stage timeline.
  - `changes/<iteration>_<candidate>.json` records changed files and added/deleted line counts.
  - `changes/<iteration>_<candidate>.patch` preserves the reproducible candidate diff.
- Emit `active` and terminal events around Researcher, Critic preflight, Builder, source safety/tests, Debugger, training plus trusted validation, Critic postflight, persistence, and completion. Do not emit token-by-token updates. The UI derives elapsed time from `started_at`; an active stage older than the configured threshold is marked possibly stale.
- Record only a structured Agent Note: objective, hypothesis, rationale, evidence, decision, concerns, diagnosis, result, and next focus. Never present raw hidden reasoning, secrets, full prompts, or unrestricted model output.
- Implement normalized read-only loaders that discover `runs/*` and reconcile baseline and research schemas into a common run/iteration representation. Tolerate missing files, incomplete active runs, failed iterations, and a partially written final JSONL line.
- Use `st.fragment(run_every="5s")` only while the selected run is active; completed runs remain static.
- Add a separate profiling command:
  `python -m src.ui.profile_data --config configs/ui.json`
  It must call the trusted `load_train_valid`, skip rows after 2022-04-28 before reading labels, and atomically write only aggregate statistics to `artifacts/ui/kuairand_pure_eda.json`.
- Keep the first experiment-tree representation deterministic and component-free; a table is preferred until the run schema supports a stable graph layout.

### Interface behavior

- **Pipeline:** a translucent live execution overlay, role-pass workflow, Debugger repair loop, selected run status, budgets, convergence, best validation metrics, and experiment-family tree. Dim inactive stages, highlight the active stage, and show iteration, stage, elapsed time, experiment, and stale status.
- The live overlay contains collapsible **Agent Notes**, **Changes**, **Errors & repairs**, and **Recent timeline** sections. Changes are provisional after the candidate passes safety/tests and authoritative after the iteration is persisted.
- **EDA:** aggregate train/validation counts, long-view rate, temporal activity, impressions/positives per user, duration distribution, and explicit train-fitted/validation-only provenance. No row-level identifiers or test information enter the profile.
- **Feature Lab:** read-only lineage for the current fields `user_id`, `video_id`, `author_id`, `tab`, and train-fitted `dur_bucket`; show raw source, transformation, fitting split, temporal/leakage status, consuming model families, and available ablation evidence. Future features appear only when logged by trusted run metadata.
- **Iterations:** experiment tree plus selected-iteration detail for hypothesis, evidence, preflight, parameters, code/test hashes, metrics, repairs, reflection, parent, elapsed time, tokens, and checkpoint promotion.
- **Results:** Random/Popularity/FM/research comparison, GAUC, nDCG@5, primary score, baseline delta, best checkpoint, convergence/resources, and a CSV preview.
- Add an in-memory CSV uploader that checks exact column order, contiguous zero-based `row_id`, finite scores, and duplicate preservation. It must clearly mark judge row-count/alignment checks as unavailable without a separate explicit authorization and must never open `data/judge/test.csv`.

### Visual system

- Minimal warm-white layout with charcoal text, pale blue navigation/data accents, muted mint success states, amber reflection states, and restrained red failures.
- Favor one dominant visualization and one compact inspector per tab; use flat surfaces, light dividers, short labels, and generous whitespace.
- Do not include launch, resume, cancel, configuration-editing, or “authorize judge” controls in v1.

## Public Interfaces

Add UI-local immutable models such as `DashboardConfig`, `RunSnapshot`, `IterationSnapshot`, `EdaProfile`, `FeatureDefinition`, and `SubmissionCheck`, with functions:

- `discover_runs(run_root) -> list[RunSnapshot]`
- `load_run_snapshot(run_dir) -> RunSnapshot`
- `build_eda_profile(data_dir) -> EdaProfile`
- `validate_submission(file_like) -> SubmissionCheck`

Also add `ActivitySnapshot`, `StageTransition`, `ChangeSummary`, and `FileChange`, with functions:

- `load_current_activity(run_dir) -> StageTransition | None`
- `load_activity_timeline(run_dir) -> tuple[tuple[StageTransition, ...], tuple[str, ...]]`
- `load_change_summary(run_dir, relative_path) -> ChangeSummary | None`

Do not change `CandidateManifest`, `CandidateOutput`, `RunState`, evaluator decisions, search policy, metric conventions, or stopping behavior. Additive observability writes from the controller are allowed; the UI remains strictly read-only.

## Test Plan

- Verify normalization of both the existing baseline run and research-run schemas.
- Verify active/incomplete/malformed runs render warnings without crashing.
- Verify every instrumented coarse stage writes an `active`/terminal event pair, snapshots are atomic, and partial JSONL tails are tolerated.
- Verify change patches are written only after candidate source passes existing safety checks, remain under the run directory, and match their line-count summary.
- Verify Agent Notes use an allowlist, omit raw prompts/hidden reasoning, and redact secrets and restricted paths.
- Assert all resolved paths stay inside configured `runs/`, public data, or the EDA-profile location; reject every `data/judge/**` path.
- Verify the profiler uses only trusted train/validation dates, emits aggregates without raw IDs, and produces deterministic histogram/time-series bins.
- Assert the Feature Lab matches the current five starter fields and identifies `dur_bucket` as train-fitted.
- Test submission validation for column order, missing/duplicate `row_id`, NaN/Inf scores, duplicate user-video pairs, and truthful “alignment not checked” status.
- Use Streamlit `AppTest` when UI dependencies are installed; keep loader, activity, path-safety, malformed-artifact, baseline-normalization, and submission checks runnable with the core test suite.
- Acceptance check: the existing baseline run displays GAUC `0.6671`, nDCG@5 `0.5358`, primary `0.6015`, and baseline delta near `-0.0001`, while the dashboard performs no writes to run directories.

## Assumptions

- The refined ImageGen concept shown above is the visual reference; it remains a preview artifact and is not added to the repository.
- V1 is local, read-only, and intended for the team rather than public deployment.
- Live capture uses coarse stage transitions (roughly 10–20 small events per iteration), while the finalized iteration record is the permanent source of truth.
- The visual draft's “Authorize & Generate” control is intentionally not implemented; final judge work remains a separate explicit workflow.
- EDA is generated explicitly before launch and reused across runs.
- Final judge prediction generation and full official alignment checking remain separate, explicitly authorized workflows.
- Update the root README with UI installation, profiling, launch commands, artifact sources, limitations, and safety guarantees.
