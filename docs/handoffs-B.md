# Hand-offs from Owner B (gate & contracts)

Status: **T1–T8 complete on `feat/b-gate-contracts`** (task-paired commits, suite 84-green under `-W error` with no API key). Everything below is B's side of plan §6. Evidence and rationale per task: `docs/worklog-B.md`.

---

## To A — loop & robustness

1. **`run_gate` call site is yours** (`research_controller.py:429-438`): convert to keyword arguments — `run_gate(run_dir=…, node_dir=…, data_dir=…, kit_dir=…)` — and wrap it so an exception becomes `GateResult(status="error", details={"error": …})` in `summary["gate"]` (your T10 step 1). The gate is already exception-free by construction (thin wrapper, `_abs` path resolution, reason-coded failures), so this is containment-in-depth, not a fix. **Tell me when converted** and I flip the signature to I-1's exact keyword-only form in a two-line follow-up — today it accepts both styles, so nothing breaks in between.
2. **≤20-line PR for the two-sided baseline gate** (I11) — apply in your file:
   ```diff
   +from src.evaluation.official import within_baseline_tolerance
    # in _latest_valid_baseline's loop:
   -    ... and primary >= threshold:
   +    ... and within_baseline_tolerance(primary, threshold):
    # in _ensure_baseline:
   -    baseline = _latest_valid_baseline(run_root, official - 0.002)
   +    baseline = _latest_valid_baseline(run_root, official)
   -    if primary < official - 0.002:
   -        raise RuntimeError(f"... {primary:.4f} < {official - 0.002:.4f}")
   +    if not within_baseline_tolerance(primary, official):
   +        raise RuntimeError(f"... {primary:.4f} outside [{official - 0.003:.4f}, {official + 0.003:.4f}]")
   ```
   (`threshold` keeps its name; it now carries the official centre. The predicate itself is merged: `official.py::within_baseline_tolerance`, two-sided ±0.003 with a float-boundary cushion.)
3. **`failure_class` is live** (I-3): every failed `ExperimentOutcome` carries one of `timeout` / `crash` / `bad_output` / `low_score` / `leak` / `missing_test_scores`. Branch your Debugger brief and retry-vs-skip on it; per the review, `"leak"` and `"low_score"` are **not** worth a repair round. This unblocks your C4 branch.
4. **Regenerated baseline run** for your `_latest_valid_baseline` testing: `runs/20260829T041834051989Z_baseline` (primary 0.60147, 3/3 success, `source_manifest.json` + `code_revision` present, `experiment_id == "official_fm_seed0"`).

## To C — LLM layer & docs

1. **The Builder prompt must name `test_scores`** (I-2, blocking real value). Exact wording:
   - context list (`roles.py:160-161`): append `… valid_users, field_dimension, test_x, and evaluate_validation(scores)`
   - return description (`:163-164`): `… Return finite validation scores, finite test scores for every row of test_x in the same order, a dict of numpy checkpoint arrays, …`
   - mention `test_scores=` in the `candidate_manifest` schema instructions.
2. **Until that lands, nothing crashes**: my worker returns `failure_class="missing_test_scores"` with the contract sentence as `error` — the bounded repair loop you wired fixes the candidate on the next attempt.
3. **Request:** a scripted fixture (I-6) whose `candidate.py` returns `test_scores`, so the T2→T3 path can be exercised end-to-end offline.
4. FYI for the README/Devpost claims: the gate's real-data acceptance is pasted in the worklog T3 entry (`✓ … 170,588 行，split=test`), and the baseline reproduction evidence is in the T8 entry.

## To D — data card, journal, hygiene

1. `ExperimentOutcome.failure_class` and `test_scores_path` now appear in `iterations.jsonl` automatically (`research_controller.py:297` writes `outcome.to_dict()`) — `report.py` can link both without touching A's file.
2. New committed baseline run id for your `test_committed_baseline_run_renders`: **`20260829T041834051989Z_baseline`** (the only tracked run dir; five files, no absolute paths).

## To E — search surface & safety

1. **`load_test_meta` is merged** (T1) — what `build_features(spec={"split": "test"})` needs:
   `official.py::load_test_meta(data_dir: Path, *, expected_rows: int | None = None) -> TestSplit`
   - `.meta` — `(row_id, user_id, video_id)` per row, exactly `data.load()['test']` order (0-based).
   - `.rows` — kit-shaped 7-tuples `(date, user_id, video_id, author_id, tab, duration_ms, LABEL_PLACEHOLDER)`; **no label column is read anywhere** (source-tested).
2. `KUAIRAND_DATA_DIR` is in the minimal candidate environment (T4) — your `features.py` fallback entry.
3. **I still owe you** the `restricted_builtins()` wire-in: two lines in `_load_candidate` (`run_candidate.py:42-52`), landing the day you announce your T2. Untouched until then, per plan.
4. After your `safety.py` `FORBIDDEN_TEXT` merge, I re-run `pytest` (nothing I wrote is `validate_source`-validated, so no conflict expected).

---

## B's open consumption list (what I'm waiting on)

- **C**: the Builder wording above + scripted `test_scores` fixture.
- **A**: the C4 wrap + `_ensure_baseline` change; the gate is already exception-free either way.
- **E**: `restricted_builtins()` announcement → I wire the two lines.
