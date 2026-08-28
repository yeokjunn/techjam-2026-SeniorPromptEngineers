# Owner B — Gate & contracts
Branch: `feat/gate-contracts`   ·   Base: `main` after the Step 0 merge (`cbf8330` step 0 + `553095d` step 0b, the branch head)   ·   Estimated effort: ~5.5 h

## 1. Mission

Build the half of the pipeline that does not exist: load the test split's identifiers and features without ever reading its labels, require every candidate to return `test_scores`, persist them, and turn the best node into a `submission.csv` that passes the organizers' own `submit.py --check --split test`. Also close the two isolation holes on B's side (the API key in the candidate environment; no sanity bounds on reported scores) and commit a baseline run produced by the committed code. This moves **Technical — hidden-test delta (~25%)** from "no artifact" to "artifact", **Robustness (~10%)** via typed failure classes and a gate that cannot crash the run, and **Requirement 1 evidence** via a reproducible baseline.

## 2. Files you own (exclusive) / files you must not touch

**Own:** `src/evaluation/official.py` · `src/evaluation/gate.py` · `src/experiments/contracts.py` · `run_candidate.py` · `run_baseline.py` · `src/agent/candidate_runner.py` · `src/agent/types.py` (post-freeze, per team-split.md) · `tests/test_candidate_output.py` · `tests/test_official_evaluation.py` · new `tests/test_gate.py` · new `tests/test_isolation.py` · the regenerated `runs/<id>_baseline/`.

`run_baseline.py` is on the list because it prints `test rows were not loaded` (`run_baseline.py:28-32`). Keep it validation-only — the empirical report cites that line as evidence; do not add test scoring to the ladder.

**Must not touch:** `research_controller.py`, `policy.py`, `convergence.py`, `controller.py`, `configs/*` (A) · `llm.py`, `roles.py`, `README.md`, `AGENTS.md`, `PLAN.md` (C) · `datacard.py`, `report.py`, `audit.py`, `logger.py`, `.gitignore` (D) · `safety.py`, `families.py`, `src/models/*`, `research/methods/*` (E) · every other owner's test file. To change one: ask the owner, or send a ≤20-line PR they merge. B needs exactly three such requests (§6). `tests/test_interfaces.py` is shared and pins the Step 0 stubs; the one sanctioned edit is the owner who fills a stub updating that stub's assertion in the same PR, ≤ 5 lines — for you that is `test_gate_stub_reports_not_implemented` in the T3 PR, and nothing else in that file.

## 3. Setup (15 minutes)

```bash
cd /Users/Ke_Jun_YEO_from.TP/Desktop/personal/techjam-2026-SeniorPromptEngineers
python3 -m venv .venv && source .venv/bin/activate        # .venv/ is already gitignored (.gitignore:4)
python -m pip install -r requirements.txt pytest
env -u OPENAI_API_KEY python -m pytest -q -W error        # expect: 47 passed (28 existing + 19 from Step 0/0b)
git checkout main && git pull && git checkout -b feat/gate-contracts
```

B never needs `OPENAI_API_KEY`. Check `/usr/bin/python3 -c "import numpy"` succeeds (here Python 3.9.6 + numpy 2.0.2) — the gate runs the kit with it, outside the venv. Check `ls data/KuaiRand-Pure/data` shows six CSVs. Runtime stays Python 3.9 + numpy; no new dependencies.

## 4. Tasks, in order

### T1 · C1, I-2 · Test-split loader and `CandidateContext.test_x`   (M, ~1.0 h)

- **Why:** nothing loads the test split, so no test features exist to score (`correctness-safety.md` C1; `grep -rn "row_id" src` → 0 hits).
- **Where:** `official.py` — constants `:12-15`, `load_train_valid` `:33-69`, date filter `:50-57`, label read `:66`. `run_candidate.py:110-114, 124-132`. `contracts.py:18` (`test_x`, already frozen).
- **Do:**
  1. Add `TEST_START = 20220429`, `TEST_END = 20220508`, `TEST_ROWS = 170_588`, `LABEL_PLACEHOLDER = -1`.
  2. Add a frozen `TestSplit` dataclass with `meta: tuple[tuple[int, str, str], ...]` (`row_id, user_id, video_id`, in `data.load()['test']` order) and `rows: tuple[tuple, ...]` (kit-shaped 7-tuples whose label slot is `LABEL_PLACEHOLDER`), returned by `load_test_meta(data_dir: Path, *, expected_rows: int | None = None) -> TestSplit`.
  3. Implement it by copying `load_train_valid`'s structure and inverting the filter: read `date`, then `if not (TEST_START <= date <= TEST_END): continue` **before** any other column — the same skip-before-label pattern as `official.py:50-57`. Append `(date, user_id, video_id, video_to_author.get(video_id, "UNK"), tab, float(duration_ms), LABEL_PLACEHOLDER)`; `row["long_view"]` must not appear anywhere in the function. `row_id` is the 0-based index within the split, which *is* `data.load()['test']` order because both loaders read `log_standard_4_08_to_4_21_pure.csv` then `log_standard_4_22_to_5_08_pure.csv` and filter by date, preserving file order (`data.py:19-30`). Raise `ValueError` when `expected_rows` is given and differs.
  4. Rewrite `run_candidate.py:110-114` as: load train/valid as today, then `splits["test"] = list(load_test_meta(args.data_dir, expected_rows=TEST_ROWS).rows)`, then `encoded, dimension = data_module.encode(splits)`, then `test_x = encoded["test"][0]` — features only; the placeholder `y` is dropped there and never returned. This is correct by construction: the kit derives duration-bucket edges and every vocab from `splits['train']` alone (`data.py:39, 44-51`), so a third key changes nothing about train/valid and reproduces the kit's own test encoding.
  5. Pass `test_x=test_x` into `CandidateContext` (`run_candidate.py:124-132`).
- **Interface (verbatim, I-2):** *`CandidateOutput.test_scores: np.ndarray | None` (frozen in Step 0). **B** gives the candidate `CandidateContext.test_x` (kit-encoded test features, same row order as `data.load()['test']`), requires `test_scores` (float, length 170,588, finite) in the trusted worker, and persists it as `generated_experiments/<run>/<iter>_<id>/test_scores.npy` plus a pointer in the iteration record. **C** updates the Builder prompt/schema instructions in `roles.py` so generated `candidate.py` returns `test_scores` (until C's change lands, B's worker treats a missing `test_scores` as `failure_class="missing_test_scores"`, not a crash).*
- **Tests** (`tests/test_isolation.py`, new): `test_test_split_row_count_and_date_window` (170,588 rows, every date in 20220429–20220508) · `test_test_rows_match_the_kit_loader_element_for_element` (fields 0–5 of all rows equal `data.load()['test']`) · `test_test_loader_source_never_names_the_label_column` (`"long_view" not in inspect.getsource(load_test_meta)`) · `test_synthetic_poisoned_label_is_never_read` (tiny synthetic data dir whose test-period `long_view` is the string `"LEAK"`; rows still load and every label slot is `LABEL_PLACEHOLDER`). Guard the real-data cases with `@unittest.skipUnless(<csv>.is_file(), "KuaiRand-Pure not present")` — D untracks `data/` later.
- **Acceptance:** `python -m pytest tests/test_isolation.py -q -W error` all pass, none skipped with data present. `grep -n "long_view" src/evaluation/official.py` → only `:66`, inside `load_train_valid`. Paste this one-off encoding proof (~40 s) into the PR:
  ```bash
  python3 -c "
  import sys; sys.path.insert(0,'kuairand-starter-kit')
  from pathlib import Path; import numpy as np, data
  from src.evaluation.official import load_train_valid, load_test_meta
  d=Path('data/KuaiRand-Pure/data'); s=load_train_valid(d); s['test']=list(load_test_meta(d).rows)
  ours,dm=data.encode(s); kit,dk=data.encode(data.load(str(d)))
  print(dm==dk, all(np.array_equal(ours[k][0],kit[k][0]) for k in ('train','valid','test')))"   # True True
  ```
- **Depends on / blocks:** nothing; blocks T2, T3.

### T2 · C1, I-2 · Trusted worker requires and persists `test_scores`   (M, ~0.75 h)

- **Why:** the gate needs a per-node array of test predictions, and only the trusted worker may produce it — candidates cannot self-report anything (`run_candidate.py:86-90`).
- **Where:** `run_candidate.py:55-97` and `:135-137`; `candidate_runner.py:83-175` (artifact dir `:90`, result parsing `:141-144`); `types.py:39-55`.
- **Do:**
  1. Widen to `validate_and_persist_output(output, valid_users, valid_y, artifact_dir, *, expected_test_rows: int | None = None)`; the keyword default keeps every existing call and test working.
  2. After the checkpoint block (`:70-84`) validate `output.test_scores`: `None` → `"missing"`; not 1-D, wrong length, or non-finite → `"invalid"`; else `"ok"` and `np.save(artifact_dir / "test_scores.npy", np.asarray(output.test_scores, dtype=np.float64))`. Save **float64** — `%.9g` in the CSV needs more than float32's ~7 digits, or the formatting itself creates ties. `expected_test_rows is None` → `"not_required"`, nothing written.
  3. Add to the returned payload: `"test_scores_status"`, `"test_scores_path"` (repo-relative POSIX or `None`), `"sanity_class"` (T6). `main()` calls it with `expected_test_rows=int(test_x.shape[0])`.
  4. Add `test_scores_path: str | None = None` to `ExperimentOutcome` (`types.py:39-55`). Since `research_controller.py:297` already writes `outcome.to_dict()` into `iterations.jsonl`, this *is* I-2's "pointer in the iteration record" — no edit in A's file.
  5. In `candidate_runner.train()` after `:141-144`, a status other than `"ok"`/`"not_required"` returns `ExperimentOutcome(status="failed", metrics=metrics, failure_class="missing_test_scores", error="CandidateOutput.test_scores must be a finite 1-D float array of length 170588 (one score per data.load()['test'] row, same order).", recovery="Rejected before promotion; the previous best is intact.")`. Keeping `metrics` preserves the ledger; `status="failed"` means `policy.observe_success` (`policy.py:83-87`) never promotes it. That exact error text is what the Debugger sees, so candidates self-repair once C's prompt change lands.
- **Path note:** `test_scores.npy` sits next to `model.npz` because `--artifact-dir` is the only directory the worker gets (`candidate_runner.py:90`). The workspace and artifact directories always share the basename `f"{iteration:03d}_{candidate_id}"` (`candidate_runner.py:25-27` vs `:90`), so the gate resolves I-2's `generated_experiments/<run>/<iter>_<id>/` spelling *and* the artifact one (T3 step 3).
- **Tests** (`tests/test_candidate_output.py`, B's file): `test_test_scores_are_persisted_as_float64` · `test_missing_test_scores_is_reported_not_raised` (no exception, status `"missing"`) · `test_wrong_length_test_scores_is_invalid` · `test_nonfinite_test_scores_is_invalid` · `test_not_required_when_expected_rows_is_none`.
- **Acceptance:** `python -m pytest tests/test_candidate_output.py -q -W error` passes; `python -c "from src.agent.types import ExperimentOutcome as O; print(O('failed',None,0.).to_dict()['test_scores_path'])"` prints `None`.
- **Depends on / blocks:** T1; blocks T3, T5, T6.

### T3 · C1, I-1 · `gate.py` writes `submission.csv` and runs the kit check   (M, ~1.25 h)

- **Why:** the CSV is the only input to the hidden-test delta, and the kit asks teams to run `--check` themselves (`kuairand-starter-kit/README.en.md:118-121`).
- **Where:** `src/evaluation/gate.py:9-18` (Step 0 stub); called at `research_controller.py:429-438`.
- **Do:**
  1. Keep `GateResult` as frozen. Make `run_gate` a thin wrapper that can never raise: it calls a private `_run_gate(...)` inside `try/except Exception`, returning `GateResult(status="error", details={"reason": "unexpected", "error": f"{type(exc).__name__}: {exc}"})`. Resolve all four arguments through `_abs(p) = p if p.is_absolute() else REPO_ROOT / p` — defence in depth: A's T10 step 1 fixes the repo-relative `Path(state.best_candidate_dir)` bug at `research_controller.py:431` and passes an absolute `node_dir`, but the gate must also work when called by hand with a repo-relative path (the acceptance snippet below does exactly that). Keep the four parameters positional-or-keyword until A confirms the call site is converted (§6), then flip to I-1's exact `run_gate(*, run_dir, node_dir, data_dir, kit_dir)` keyword-only form in a two-line follow-up; either form accepts the four keywords A now passes.
  2. `gate_done` marker: if `<run_dir>/gate_done.json` exists, return the stored result with `details["reused"] = True` and do nothing else. Write the marker **only** on `status="ok"`, so a failed gate retries next run.
  3. Resolve the scores: first existing of `node_dir/"test_scores.npy"`, then `run_dir/"artifacts"/node_dir.name/"test_scores.npy"`. Neither → `reason="missing_test_scores"` (list the paths searched). Not 1-D, non-finite, or length != `len(load_test_meta(data_dir).meta)` → `reason="bad_test_scores"` with both counts.
  4. Write `<run_dir>/submission.csv` with `csv.writer(..., newline="")`: header exactly `row_id,user_id,video_id,score` (`submit.py:25`), then `[row_id, user_id, video_id, f"{float(s):.9g}"]` per meta entry. `user_id`/`video_id` are the **original strings**, never re-encoded ints. Write `.tmp` then `os.replace`.
  5. Run the check as a subprocess: `[_kit_python(), str(kit_dir/"submit.py"), "--check", "--split", "test", "--data_dir", str(data_dir), str(csv_path)]`, `cwd=kit_dir`, `capture_output=True`, `text=True`, `timeout=600`, `check=False`, and the minimal env from T4. `_kit_python()` (cached) returns `/usr/bin/python3` when it exists and `import numpy` succeeds there, else `sys.executable`. Keep the reason in the docstring: the kit's `load()` materialises test **labels** (`data.py:23-25`), so the check runs in a throwaway numpy-only process whose only output is a pass/fail line — labels never enter the harness process. Missing `submit.py` → `reason="kit_unavailable"`; non-zero exit → `reason="check_failed"` with the last 2,000 chars of stdout/stderr.
  6. Success → `GateResult(status="ok", submission_path=<repo-relative POSIX, else absolute POSIX>, details={"rows": N, "sha256": <csv digest>, "check_stdout": <tail>, "checked_with": <interpreter>, "scored": False})`. **`details` carries no score of any kind** — the results-table delta is the *validation* delta vs 0.6016 (`README.md` item C1), already computed by A at `research_controller.py:452-454`.
  7. *Optional, only if T1–T8 are done:* a separate entry point `python -m src.evaluation.gate score --run <dir> --data-dir <dir> --confirm-frozen` that requires `gate_done.json`, re-hashes the CSV against the marker, runs `submit.py --score --split test`, and writes `<run_dir>/self_computed_test_score.json` with `{"self_computed": true, …}`. Never called from `run_gate`; never merged into `summary.json`, `best.json` or `results.json`. The kit frames `--score` as "available locally for valid" (`kuairand-starter-kit/README.en.md:115`), so anything computed on test is ours, labelled, and post-freeze.
- **Interface (verbatim, I-1):** *`src/evaluation/gate.py::run_gate(*, run_dir: Path, node_dir: Path, data_dir: Path, kit_dir: Path) -> GateResult` (fields `status: str`, `submission_path: str | None`, `details: dict`). Provided by **B**. Called by the controller at the end of `run()` (already wired in Step 0 at `research_controller.py:429-438`, positional today). **A** converts the call to keyword arguments and wraps it so an exception becomes `GateResult(status="error", details={"error": ...})` written into `summary["gate"]` instead of losing `summary.json`. `status` values: `"ok"`, `"error"`, `"not_implemented"`. `submission_path` is repo-relative POSIX.*
- **Tests** (`tests/test_gate.py`, new) — all on a tiny synthetic data dir built in `tempfile` (three CSVs, a handful of rows across the three date windows) plus the real tracked `kuairand-starter-kit/`; no dependency on the 240 MB dataset, ~1 s total: `test_gate_writes_a_submission_that_passes_the_kit_check` (status `ok`; exact header; N rows; `row_id` 0..N-1; ids are the original strings; `gate_done.json` written) · `test_gate_is_idempotent` (second call → `details["reused"] is True`, digest unchanged) · `test_missing_test_scores_returns_error` (no exception, no CSV) · `test_wrong_length_scores_return_error` · `test_nonfinite_scores_are_rejected_before_the_kit_runs` · `test_missing_kit_returns_error` (empty `kit_dir` → `kit_unavailable`) · `test_unexpected_exception_becomes_status_error` · `test_gate_result_carries_no_test_metric` (`json.dumps(asdict(result))` matches none of `GAUC|nDCG|primary`) · `test_submission_path_is_repo_relative_when_under_repo_root`.
- **Acceptance:** `python -m pytest tests/test_gate.py -q -W error` all pass. Then once against real data (drop a `np.random.default_rng(0).random(170588)` array at `runs/<id>/artifacts/<node>/test_scores.npy` if no candidate exists yet), expecting `status='ok'` and `✓ … 170,588 行，split=test`:
  ```bash
  python -c "from pathlib import Path; from src.evaluation.gate import run_gate; print(run_gate(Path('runs/<id>'), Path('generated_experiments/<id>/<node>'), Path('data/KuaiRand-Pure/data'), Path('kuairand-starter-kit')))"
  /usr/bin/python3 kuairand-starter-kit/submit.py --check --split test --data_dir "$PWD/data/KuaiRand-Pure/data" "$PWD/runs/<id>/submission.csv"
  head -2 runs/<id>/submission.csv     # row_id,user_id,video_id,score  /  0,<uid>,<vid>,<9 sig figs>
  ```
- **Depends on / blocks:** T1, T2. Wants A's C4 wrap before the live run; until then the gate already cannot raise.

### T4 · C3 · Minimal candidate environment   (S, ~0.5 h)

- **Why:** `candidate_runner.py:58-62` copies `os.environ` — into which `.env` was already loaded by `llm.py:195` — so every LLM-written candidate receives `OPENAI_API_KEY`.
- **Where:** `candidate_runner.py:58-62`; call sites `:71` and `:121`.
- **Do:**
  1. Replace `_environment(self)` with `_environment(self, workspace: CandidateWorkspace) -> dict[str, str]` built from scratch: carry over only `PATH`, `LANG`, `LC_ALL`, `TZ` and, when present, `SYSTEMROOT`/`SystemRoot`/`COMSPEC`/`WINDIR`; then set `PYTHONPATH=str(self.repo_root)`, `PYTHONDONTWRITEBYTECODE="1"`, `HOME`/`TMPDIR`/`TEMP`/`TMP` = `str(workspace.directory)`, `KUAIRAND_DATA_DIR=str(self.data_dir)` (`candidate_runner.py:53`; E's one entry — `src/models/features.py` falls back to the repo default without it), and `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` = `"1"`. Never copy `os.environ`; assert in code that no key starts with `OPENAI_` or `ANTHROPIC_`.
  2. Take E's one-line hand-off (C2 d, their T2) into `_load_candidate` (`run_candidate.py:42-52`): extend `run_candidate.py:14` to `from src.agent.safety import restricted_builtins, validate_source` and add, between `module = importlib.util.module_from_spec(spec)` and `spec.loader.exec_module(module)`, `module.__dict__["__builtins__"] = restricted_builtins()`. `exec` injects the real builtins only when the key is absent, so pre-setting it wins. Two lines in your file, E's mapping; land it once E announces T2, and skip it until then.
  3. `train()`'s `cwd=self.repo_root` (`:120`) → `cwd=workspace.directory`. Safe: every path in the command at `:99-113` is already absolute (`data_dir`/`run_root` via `controller.py:21-23 _resolve_repo_path`; `workspace.code_path` via `contained_path`, which resolves), and `src.experiments.run_candidate` still imports through `PYTHONPATH`. `test()` already uses the workspace as cwd (`:70`). Update both call sites to pass the workspace.
- **Tests** (`tests/test_isolation.py`): `test_candidate_environment_drops_provider_keys` — with `patch.dict(os.environ, {"OPENAI_API_KEY": "sentinel", "ANTHROPIC_API_KEY": "sentinel"})`, actually run `subprocess.run([sys.executable, "-c", "import os,json;print(json.dumps(dict(os.environ)))"], env=executor._environment(ws), cwd=ws.directory, …)` and assert no key starts with `OPENAI_`/`ANTHROPIC_`, `"sentinel"` appears nowhere, `PYTHONDONTWRITEBYTECODE == "1"`, every thread cap is `"1"`, `PYTHONPATH` is the repo root, `KUAIRAND_DATA_DIR` is the data directory. `test_candidate_subprocess_cwd_is_the_workspace` — same shape with `-c "import os;print(os.getcwd())"`.
- **Acceptance:** both tests pass; `grep -n "dict(os.environ)\|cwd=self.repo_root" src/agent/candidate_runner.py` → no hits.
- **Depends on / blocks:** nothing — do it first if T1 stalls.

### T5 · I-3 · `failure_class` on every failure path   (S, ~0.25 h)

- **Why:** A picks the Debugger brief and retry-vs-skip from it; D prints it in the journal. Today every failure is an untyped string.
- **Where:** `candidate_runner.py:130-140` (non-zero exit), `:143-144` (non-finite metric), `:156-166` (`TimeoutExpired`), `:167-175` (`OSError/ValueError/KeyError/JSONDecodeError`).
- **Do:** set `failure_class=` on every returned failed `ExperimentOutcome`: `"crash"` at `:130-140`, `"timeout"` at `:156-166`, `"bad_output"` at `:167-175` (which also catches the non-finite-metric `ValueError` raised at `:144`), `"missing_test_scores"` from T2, `"low_score"`/`"leak"` from T6. Never key control flow on message text (I13).
- **Interface (verbatim, I-3):** *`ExperimentOutcome.failure_class: str | None` (frozen). **B** sets it in `candidate_runner.py` to one of `"timeout"`, `"crash"`, `"bad_output"`, `"low_score"`, `"leak"`, `"missing_test_scores"`. **A** uses it to pick the Debugger brief and to decide retry vs. skip; **D** prints it in the journal.*
- **Tests** (`tests/test_isolation.py`): `test_failure_classes_cover_every_return_path` — `ast`-parse `candidate_runner.py`, collect every `ExperimentOutcome(...)` call with `status="failed"`, assert each passes a `failure_class` keyword from the six-member set. `test_timeout_outcome_is_classified` — a `CandidateExecutor` with `experiment_timeout_seconds=0` → `outcome.failure_class == "timeout"`.
- **Acceptance:** both pass; `grep -c "failure_class=" src/agent/candidate_runner.py` ≥ 5.
- **Depends on / blocks:** T2, T6. Blocks A's C4 branch on `failure_class` — tell A the day it lands.

### T6 · I11 · Sanity floor/ceiling and the two-sided baseline predicate   (S, ~0.5 h)

- **Why:** `runner.py:69` and `candidate_runner.py:143` check finiteness only, so a leaked 0.99 would be promoted to best; `research_controller.py:53, 60` accept any primary ≥ 0.5996, so 0.85 passes the baseline gate.
- **Where:** `official.py` (new helpers); `run_candidate.py:68`; `candidate_runner.py:141-155`.
- **Do:**
  1. `official.py` gets `SANITY_FLOOR = 0.47`, `SANITY_CEILING = 0.80` and two pure functions: `classify_primary(primary: float) -> str | None` (`"low_score"` below the floor, `"leak"` above the ceiling, else `None`) and `within_baseline_tolerance(primary: float, official: float = 0.6016, tolerance: float = 0.003) -> bool` (`abs(primary - official) <= tolerance`).
  2. In `validate_and_persist_output`, after `metrics` is computed (`run_candidate.py:68`), set `payload["sanity_class"] = classify_primary(float(metrics["primary"]))`. Do **not** raise — `result.json` is still written, so the ledger keeps the number.
  3. In `candidate_runner.train()` after `:141-144`, a non-`None` `sanity_class` returns `ExperimentOutcome(status="failed", metrics=metrics, failure_class=<that class>, error=f"Validation primary {…:.6f} is outside the sanity band [0.47, 0.80].", recovery="Rejected without promotion; previous best retained.")`. Because `observe_success` runs only for `status == "success"` (`research_controller.py:280-281`), a ceiling hit can never become best.
  4. **Ownership of the baseline gate:** B owns only the predicate. **A** owns both call sites — `research_controller.py:55` (`_latest_valid_baseline(run_root, official - 0.002)` → a two-sided filter) and `research_controller.py:62` (`if primary < official - 0.002` → `if not within_baseline_tolerance(primary, official)`). Send A a ≤20-line PR with those two lines plus the import; do not edit that file yourself.
- **Tests:** `tests/test_isolation.py` — `test_classify_primary_bounds` (0.4699→`low_score`; 0.47 and 0.80→`None`; 0.8001→`leak`; 0.6015→`None`) and `test_within_baseline_tolerance_is_two_sided` (0.5986, 0.6046 True; 0.5985, 0.6047, 0.85 False). `tests/test_candidate_output.py` — `test_ceiling_hit_is_marked_as_leak` (the existing `test_trusted_metrics_override_candidate_diagnostics` fixture scores primary 1.0; assert `sanity_class == "leak"` there, with a comment saying why) and `test_floor_miss_is_marked_low_score`.
- **Acceptance:** the four tests pass; the three existing cases in `tests/test_candidate_output.py` still pass.
- **Depends on / blocks:** T2. Blocks A's `_ensure_baseline` change.

### T7 · I1, I2 · Isolation and evaluator-convention tests   (M, ~0.75 h)

- **Why:** I1 — nothing asserts split sizes, the 20220428 cut-off, or equality with `data.load()`, so a "helpful" `test` key added to `load_train_valid` would break no test. I2 — `tests/test_official_evaluation.py` is one trivial 2-row case; the conventions that diverge between implementations are untested and will matter the moment anyone adds a fast-path evaluator.
- **Where:** `tests/test_isolation.py` (alongside the T1/T4/T5/T6 cases) and `tests/test_official_evaluation.py`.
- **Do (I1):** `test_train_valid_split_sizes` — exactly the keys `{"train","valid"}` with **1,141,112** and **124,909** rows · `test_no_train_or_valid_row_is_dated_after_20220428` — max date 20220428, min 20220408 · `test_train_and_valid_match_the_kit_loader_row_for_row` — the first 5,000 rows of each split equal `data.load()`'s, and the lengths match (row equality implies encoding equality, since the kit's `encode` is a pure function of the rows). Keep the `skipUnless` dataset guard on all of them.
- **Do (I2)** — six cases against `kuairand-starter-kit/evaluate.py:43-61`, each with a hand-computed expectation:
  - `test_zero_positive_user_is_counted_in_ndcg_and_excluded_from_gauc`: user `a` = [(1.0,1),(0.0,0)], `b` = [(1.0,0),(0.0,0)] → GAUC 1.0, nDCG@5 0.5, primary 0.75, `users` 2, `rows` 4.
  - `test_all_positive_user_is_excluded_from_gauc`: adding an all-positive user, ranked arbitrarily, leaves GAUC at 1.0 (`evaluate.py:54` requires `0 < npos < len(labs)`).
  - `test_gauc_is_weighted_by_positive_count`: user `p` (3 rows, labels [1,1,0], perfect order → AUC 1.0, weight 2) and `q` (2 rows, positive last → AUC 0.0, weight 1) → GAUC = 2/3, not the unweighted 0.5.
  - `test_ties_are_broken_by_row_order`: one user, two rows, identical scores. Labels in row order [0,1] → nDCG@5 = 1/log2(3) ≈ 0.6309297535714574; the same rows as [1,0] → 1.0; GAUC 0.5 both ways (tie-corrected AUC).
  - `test_ndcg_truncates_at_k_5`: one user, six impressions, the single positive at rank 5 → nDCG@5 = 1/log2(6) ≈ 0.3868528072345416; at rank 6 → 0.0.
  - `test_users_and_rows_are_reported`.
- **Acceptance:** `env -u OPENAI_API_KEY python -m pytest -q -W error` green; `python -m pytest tests/test_official_evaluation.py -q` runs in under 0.2 s.
- **Depends on / blocks:** T1, T4, T5, T6 (it collects their cases). Blocks nothing.

### T8 · C5 · Regenerate and commit the baseline run   (S, ~0.5 h)

- **Why:** `runs/20260828T141646Z_baseline/` has no `source_manifest.json` and no `code_revision`, which the code at that commit writes unconditionally (`controller.py:57-58, 103`), so it cannot be that code's output — and it embeds `C:\Users\Admin\OneDrive - Nanyang Technological University\…` in `best.json`, `summary.json` and `iterations.jsonl`, in a repo that will be public. The "select by recorded revision, verify the artifact exists" half of C5 is **A's** (`research_controller.py:38-49`) — do not touch it.
- **Do:**
  1. After T1–T7 have merged, from the repo root: `env -u OPENAI_API_KEY python -m src.agent.controller --config configs/baseline.json` (~21 s on an M4).
  2. Check: `primary` within 0.0008 of **0.6016**, GAUC ≈ 0.6671, nDCG@5 ≈ 0.5358, `stop_reason: iteration_budget_reached`, 3/3 successful.
  3. Scrub check, must print nothing: `grep -rn "C:\\\\\|OneDrive\|/Users/" runs/<new_id>/`. Current code already writes relative POSIX paths (`src/models/baselines.py:114`); if anything appears, rewrite it by hand.
  4. `git rm -r --cached runs/20260828T141646Z_baseline`, delete it, then stage only five files — never `git add -A`, never the run directory as a whole:
     ```bash
     git add runs/<new_id>/{summary,best,run_config,source_manifest}.json runs/<new_id>/iterations.jsonl
     git status --porcelain   # nothing else staged; artifacts/ and stdout/ stay ignored (.gitignore:16-17)
     ```
  5. Re-run it once more as the last step before the final submission if `src/` changed after step 1 — ~25 s.
- **Acceptance:** `git ls-files runs/ | cut -d/ -f2 | sort -u` shows exactly one directory (everyone's personal `runs/*_research/` dirs are untracked under rule 5 and D's `.gitignore`); `jq -r .revision runs/<new_id>/source_manifest.json` equals `python -c "from src.agent.controller import _source_manifest; print(_source_manifest()['revision'])"`; `jq 'has("code_revision")' runs/<new_id>/iterations.jsonl | sort -u` → `true`; `jq -r .best.artifact_path runs/<new_id>/best.json` starts with `runs/` and has no drive letter.
- **Depends on / blocks:** T1–T7 merged. Blocks A's `_latest_valid_baseline` change and C's reproduction claim.

## 5. Definition of done (whole plan)

- [ ] `env -u OPENAI_API_KEY python -m pytest -q -W error` green, with `tests/test_gate.py` and `tests/test_isolation.py` present and nothing skipped when the dataset is available.
- [ ] Every task's acceptance criteria met, including T1's `encode` proof and T3's real `submit.py --check` run, both pasted into the PR description.
- [ ] `grep -rn "GAUC\|nDCG\|primary" src/evaluation/gate.py` → no metric ever enters `GateResult`.
- [ ] PRs merged (≤300 lines each; suggested split T1+T2 / T3 / T4+T5+T6 / T7 / T8).
- [ ] Hand-offs delivered (§6): A has the two-line `_ensure_baseline` PR and knows the `run_gate` call site must move to keywords; the `test_gate_stub_reports_not_implemented` assertion is updated inside your own T3 PR (≤ 5 lines); C has the exact Builder wording; D knows `failure_class` and `test_scores_path` are in the iteration record; E's `restricted_builtins()` line is in `_load_candidate` and `KUAIRAND_DATA_DIR` is in the minimal env.
- [ ] The regenerated baseline run is the only directory in `runs/` — five JSON files, no absolute paths.

## 6. Hand-offs

**You provide.** *I-1 `run_gate`* — tell **A** the day T3 merges: `research_controller.py:429-438` must move to keyword arguments (`run_dir=`, `node_dir=`, `data_dir=`, `kit_dir=`) and be wrapped so an exception becomes `GateResult(status="error", details={"error": ...})` in `summary["gate"]` — A's T10 step 1. Separately, `tests/test_interfaces.py:131-137` (`StubTests.test_gate_stub_reports_not_implemented`) calls `run_gate` positionally with four identical temp dirs and asserts `not_implemented`; you fill that stub, so **you** update that one test in the same T3 PR (≤ 5 lines, the only sanctioned edit to that file — do not ask A for it): the new expectation is `status == "error"` with `details["reason"] == "missing_test_scores"`, and its positional call becomes the four keywords when you flip the signature. B keeps the parameters positional-or-keyword until A confirms the call site, so nothing breaks in between. *I-3 `failure_class`* — six values land with T5; A branches on them for the Debugger brief and retry-vs-skip (`"leak"` and `"low_score"` are not worth a repair round), D prints them. *I-2 `test_scores_path`* — a new `ExperimentOutcome` field, so it appears in `iterations.jsonl` automatically via `research_controller.py:297`; tell **D** so `report.py` can link it. *I11 predicates* — `official.py::within_baseline_tolerance` / `classify_primary` plus the ≤20-line PR for `research_controller.py:55` and `:62`. *C1 `load_test_meta`* — tell **E** when T1 merges: `official.py::load_test_meta(data_dir: Path, *, expected_rows: int | None = None) -> TestSplit`, whose `.meta` is `(row_id, user_id, video_id)` per row and whose `.rows` are the kit-shaped 7-tuples `(date, user_id, video_id, author_id, tab, duration_ms, LABEL_PLACEHOLDER)` — **no label column anywhere**, which is what `build_features(spec={"split": "test"})` needs. *C5* — the regenerated baseline run, which A's `_latest_valid_baseline` change should be tested against, and which D's `test_committed_baseline_run_renders` renders — give **D** the new run id when T8 lands.

**You consume.** *From C (I-2, blocking for real value):* the Builder prompt must name the new field — append to the context list at `roles.py:160-161` (`… valid_users, field_dimension, test_x, and evaluate_validation(scores)`) and to the return description at `:163-164` (`… Return finite validation scores, finite test scores for every row of test_x in the same order, a dict of numpy checkpoint arrays, …`), and mention `test_scores=` in the `candidate_manifest` schema instructions. **Until it lands**, B's worker returns `failure_class="missing_test_scores"` with the contract sentence as `error`, so the existing repair loop (`research_controller.py:233-240`) fixes the candidate on the next attempt instead of crashing. Also ask C for a scripted fixture (I-6) whose `candidate.py` returns `test_scores`, so T2/T3 can be exercised end to end offline. *From A:* the C4 wrap and the `_ensure_baseline` change; if A is late the gate is already exception-free, so the only cost is a still-one-sided baseline gate. *From E (C2):* `safety.py` gains `log_standard`, `log_random`, `KuaiRand`, `.csv`, `/data/` in `FORBIDDEN_TEXT` (`safety.py:60-72`); nothing B writes is validated by `validate_source` and `context.test_x` is a plain attribute access, so no conflict — but re-run `pytest` after E merges. Also from E: `safety.py::restricted_builtins(*, test_file: bool = False) -> dict[str, object]`, which you wire in two lines in `_load_candidate` (T4 step 2) once E's T2 merges — until then leave `run_candidate.py:42-52` as it is. E asks you for two things in return, both already in the tasks above: `KUAIRAND_DATA_DIR` in the minimal env (T4 step 1) and `load_test_meta()`'s label-free fields (T1).

**Note from the Step 0 review assigned to B:** containing the `run_gate` call is A's (the call site is A's file); B's half is that `run_gate` never raises on an expected failure and returns `status="error"` with a `details["reason"]` — T3 steps 1, 3 and 5.

**Known limitation to document, not build:** if no candidate ever succeeds, the best node is the baseline FM, there is no `test_scores.npy`, and the gate returns `status="error"` / `reason="missing_test_scores"` with no submission. The zero-code fallback is the kit's own generator — `/usr/bin/python3 kuairand-starter-kit/submit.py --make --split test --data_dir "$PWD/data/KuaiRand-Pure/data" runs/<id>/submission.csv` — which writes the official FM baseline submission. Raise it with A if that outcome starts to look likely.

## 7. Rules

1. **One owner per file.** Need a change in someone else's file? Ask the owner, or send them a ≤20-line PR they merge.
2. **New tests in new files.** Never edit another owner's test file.
3. **Rebase on `main` twice a day; PRs ≤ 300 lines; `pytest` green before merge.** Small, frequent merges beat one big one.
4. **Config: add files, don't edit shared ones** (`configs/offline_smoke.json`, `configs/features_run.json`).
5. **Run directories are personal** (`runs/<initials>_…`, gitignored) until the final run, which A commits.
6. **Shared surfaces only move in the freeze PR** — `types.py`, `contracts.py`, `configs/ranking_losses.json`. A second freeze PR (by A) is the way to change them again, not five drive-by edits.

Plus: never run `git add -A`; never commit `runs/` except the final run (A) and the regenerated baseline (B, T8); never commit `.env`; PR ≤ 300 lines; rebase on `main` twice a day. `types.py` and `contracts.py` are B's post-freeze exception to rule 6 — keep the edit to the one additive field in T2 and announce it in the PR title.

## 8. Daily checkpoints

**Day 1 end.** T1–T4 merged: `load_test_meta` returns 170,588 rows and never names `long_view`; `test_x` reaches the candidate; the worker persists `test_scores.npy` and reports a status; `run_gate` writes a `submission.csv` that passes `submit.py --check --split test` and returns `status="error"` instead of raising on every expected failure; candidate subprocesses no longer see `OPENAI_API_KEY`. C has the Builder wording. With C's scripted provider and A's loop, this is what the day-1 offline end-to-end run needs.

**Day 2 end.** T5–T7 merged: six `failure_class` values set, sanity floor/ceiling live in the trusted worker, the two-sided predicate handed to A, and the tests pinning 1,141,112 / 124,909 / 170,588, the 20220428 cut-off, row-for-row equality with `data.load()`, and the five evaluator conventions. T8 done: the regenerated baseline run committed, the Windows paths gone. Full suite green under `-W error` with no API key.

**Day 3.** Support A's live run: watch for mis-assigned `failure_class` values, confirm the final run directory has `submission.csv` + `gate_done.json`, re-run T8 if `src/` moved. Optionally add the labelled self-computed test score (T3 step 7) — only after the CSV is frozen, and never in `summary.json`.
