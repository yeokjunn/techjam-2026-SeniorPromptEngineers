# Worklog — Owner B (Gate & Contracts)

Branch: `feat/b-gate-contracts` · Base: `main` @ `9dd9d29` (post Step-0 freeze)
Scope: `docs/reviews/2026-08-28-harness-review/plans/B-gate-contracts.md` (T1–T8 + hand-offs). This log is B's own file; one entry per completed task, with the review finding it closes and the verification evidence. Append-only.

---

## 2026-08-29 · Pre-work: repo state check + reference restore

**Done**
- Explored the working repo and `kuairand-starter-kit` against each other; confirmed the project is on-track against the Track 2 problem statement (baseline reproduces 0.6015 vs published 0.6016; evaluator imported not copied; 50-iter/6h/ε=0.002-N=3 knobs all match the kit).
- Restored `docs/problem_statement.md` from git (`git restore`) — it had been deleted in the working tree while `AGENTS.md:28` still cites it as the top authority.

**Justification**
- The plan's authority chain (`AGENTS.md` → problem statement → review docs) breaks if the referenced file is missing; the deletion was in no owner's plan, i.e. accidental, and the file is declared read-only reference material.
- Known open item: `docs/kuairand.md` remains deleted in the working tree (user chose to keep it out for now; it duplicates the upstream KuaiRand README).

---

## 2026-08-29 · Setup (plan §3)

**Done**
- Created `.venv` (Python 3.12.8) and installed `requirements.txt` + pytest → numpy 2.5.2, openai 3.5.0, python-dotenv, pytest 9.1.1.
- Verified all six KuaiRand-Pure CSVs present under `data/KuaiRand-Pure/data/`.
- Baseline suite: `env -u OPENAI_API_KEY python -m pytest -q -W error` → **47 passed, 16 subtests, 2.44s, 0 skipped** — exactly the plan's §3 expectation.

**Justification**
- B never needs `OPENAI_API_KEY`; running the suite with it unset proves the offline contract before any change lands.
- The 47-green baseline is the reference point every later "no regression" claim is measured against.

**Deviation from plan (recorded for T3)**
- Plan §3 expects `/usr/bin/python3 -c "import numpy"` to succeed (the author's machine: 3.9.6 + numpy 2.0.2). This machine: system Python 3.9.6 has **no numpy**. No action taken — `gate.py`'s planned `_kit_python()` already falls back to `sys.executable` (the venv, which has numpy). T3's acceptance line "check runs with `/usr/bin/python3`" therefore becomes "runs with the venv interpreter" here. System Python was deliberately not modified.

---

## 2026-08-29 · T1 — Test-split loader + `CandidateContext.test_x` (C1, I-2)

**Done** (TDD: test file written first, confirmed RED on import, then implementation → GREEN)
- `src/evaluation/official.py`
  - New constants: `TEST_START = 20220429`, `TEST_END = 20220508`, `TEST_ROWS = 170_588`, `LABEL_PLACEHOLDER = -1`.
  - New frozen `TestSplit` dataclass: `meta: tuple[(row_id, user_id, video_id), ...]`, `rows: tuple[7-tuples]`.
  - New `load_test_meta(data_dir, *, expected_rows=None) -> TestSplit`: mirrors `load_train_valid` with the date filter inverted (`20220429–20220508`), skips **before reading any other column**, and fills the label slot with `LABEL_PLACEHOLDER`. `expected_rows` mismatch → `ValueError`.
- `src/experiments/run_candidate.py`
  - Worker now builds `splits["test"]` from `load_test_meta(..., expected_rows=TEST_ROWS).rows`, encodes all three splits via the kit's `encode`, extracts `test_x = encoded["test"][0]` (features only — placeholder `y` dropped, never returned), and passes `test_x=` into `CandidateContext` (field frozen in Step 0; `contracts.py` untouched).
- `tests/test_isolation.py` (new, 5 tests)
  - `test_synthetic_poisoned_label_is_never_read` — synthetic dir whose test-window label column is the string `"LEAK"`; rows still load, every label slot is the placeholder.
  - `test_expected_rows_mismatch_raises` (exact-match accepted).
  - `test_test_loader_source_never_names_the_label_column` — `"long_view" not in inspect.getsource(load_test_meta)`.
  - `test_test_split_row_count_and_date_window` — 170,588 rows, all dates in window, `meta` = 0-based ids aligned with rows.
  - `test_test_rows_match_the_kit_loader_element_for_element` — fields 0–5 equal `data.load()['test']` row-for-row.

**Justification**
- Closes the first half of review Critical **C1**: nothing previously loaded the test split, so no submission could ever be scored ("the deliverable that determines 35% of the score"). T1 supplies the features; T2/T3 supply the scores and the CSV.
- Label isolation is a hard rule (problem statement: no hidden-test access during development). The loader cannot leak what it never reads: the date filter runs before any other column is touched, the label slot is a constant, and both properties are pinned by tests — including a poisoned-input test that would fail if anyone "helpfully" re-adds a label read.
- Kit-parity is proven, not assumed: both loaders read the same two files in the same order and filter by date, so split index == kit row order; the one-off encode proof (`True True`) shows identical dimension and byte-identical `X` for train/valid/test vs `data.load()`, and the kit derives bucket edges + all vocabs from `splits['train']` alone, so the third key cannot perturb train/valid.
- `expected_rows=TEST_ROWS` in the worker makes dataset drift a loud `ValueError` at run time instead of a silent misalignment between `test_x` and the submission's `row_id`s.

**Verification evidence**
- `pytest tests/test_isolation.py -q -W error` → **5 passed in 7.61s, 0 skipped** (real dataset present).
- `grep -n "long_view" src/evaluation/official.py` → single hit, `official.py:71`, inside `load_train_valid` (plan cites :66; shifted by the additions above it — same line).
- Encode proof output: **`True True`** (dimension equality + `np.array_equal` on train/valid/test `X` vs kit loader).
- Full suite after T1: **52 passed, 16 subtests, 8.37s** (47 → 52; no regressions), still `-W error`, still no API key.

**Not committed** — per plan §5 the PR split is T1+T2 as one ≤300-line PR; commit when T2 lands.

---

## 2026-08-29 · T2 — Trusted worker requires + persists `test_scores` (C1, I-2)

**Done** (TDD: 5 new tests written first, confirmed RED, then implementation → GREEN)
- `src/experiments/run_candidate.py`
  - `validate_and_persist_output` widened with keyword-only `expected_test_rows: int | None = None` — every existing call site and test works unchanged (they get `"not_required"`).
  - After the checkpoint block: `None` → `"missing"`; not 1-D / wrong length / non-finite (or uncastable) → `"invalid"`; else `"ok"` and saved as **float64** `test_scores.npy` in the artifact dir. `expected_test_rows is None` → `"not_required"`, nothing written. Nothing raises — the ledger keeps the metrics either way.
  - Payload gains `test_scores_status`, `test_scores_path` (repo-relative POSIX via new `_repo_relative` helper, else absolute), and `sanity_class` (placeholder `None`; T6's `classify_primary` fills it).
  - `main()` now passes `expected_test_rows=int(test_x.shape[0])` — sourced from the T1-loaded, kit-encoded `test_x`, not a hardcoded constant.
- `src/agent/types.py` — one additive field `ExperimentOutcome.test_scores_path: str | None = None` (B's post-freeze exception to rule 6; announced in PR title). `to_dict()` is `asdict`, so the pointer lands in `iterations.jsonl` via `research_controller.py:297` with **zero edits in A's file**.
- `src/agent/candidate_runner.py` — in `train()`, a `test_scores_status` other than `ok`/`not_required` returns `ExperimentOutcome(status="failed", metrics=..., failure_class="missing_test_scores", error=<plan's exact contract sentence>, recovery="Rejected before promotion; the previous best is intact.")`; the success return now carries `test_scores_path` from the payload. Legacy `result.json` without the key reads as `"not_required"` (backward compatible).

**Justification**
- Closes the second half of C1's data path: T1 supplies `test_x`; T2 makes the trusted worker the only producer of test predictions — candidates cannot self-report scores (mirrors the existing trusted-metrics override), and the persisted `.npy` is exactly what T3's gate consumes.
- float64 because `%.9g` in the eventual CSV needs ~9 significant digits; float32's ~7 would let the *formatting itself* create ties in the within-user ranking.
- Reported-not-raised because C's Builder prompt doesn't name `test_scores` yet: until it lands, a missing array becomes a `failure_class="missing_test_scores"` failure whose `error` text is the contract sentence the Debugger sees — the existing bounded repair loop fixes the candidate on the next attempt instead of crashing the run (C4 defence).
- `status="failed"` + kept `metrics` means `policy.observe_success` never promotes it, but the iteration ledger retains the number.

**Verification evidence**
- `pytest tests/test_candidate_output.py -q -W error` → **8 passed** (3 existing + 5 new: persisted-as-float64, missing-reported-not-raised, wrong-length-invalid, non-finite-invalid, not_required-when-None), 0.06s.
- Acceptance one-liner `O('failed',None,0.).to_dict()['test_scores_path']` → **`None`**.
- Full suite: **57 passed, 16 subtests, 8.73s** (52 → 57; no regressions), `-W error`, no API key.
- Self-review fix during implementation: removed a stray invalid kwarg accidentally introduced in the runner edit before running tests.

**Not committed** — completes the T1+T2 PR per plan §5's split (≤300 lines).

---

## 2026-08-29 · T3 — `gate.py` writes `submission.csv` + runs the kit check (C1, I-1)

**Done** (TDD: 10 tests written first, all RED against the stub, then implementation → GREEN)
- `src/evaluation/gate.py` (stub replaced; `GateResult` unchanged from the Step-0 freeze)
  - `run_gate(run_dir, node_dir, data_dir, kit_dir)` — **positional-or-keyword** until A converts the controller call site; a thin wrapper that can never raise: `_run_gate` inside `try/except Exception` → `status="error"`, `details={"reason": "unexpected", ...}`. All four args resolved via `_abs()` (repo-relative accepted; defence in depth alongside A's T10 fix).
  - `gate_done.json` idempotency: existing marker returns the stored result with `details["reused"]=True`; the marker is written **only on `status="ok"`**, so a failed gate retries next run.
  - Score resolution: first of `node_dir/test_scores.npy`, then `run_dir/artifacts/<node name>/test_scores.npy` (the T2 spelling). Neither → `missing_test_scores` with the paths searched. Not 1-D / non-finite / wrong length vs `len(load_test_meta(data_dir).meta)` → `bad_test_scores` with both counts — rejected before any CSV or kit run.
  - CSV: header exactly `row_id,user_id,video_id,score`; original id strings; `%.9g` scores; written `.tmp` → `os.replace`; `run_dir` created if absent (found during verification — the hand-run acceptance points at fresh run dirs).
  - Kit check as a subprocess (`cwd=kit_dir`, minimal env, `timeout=600`, `check=False`): missing `submit.py` → `kit_unavailable`; non-zero exit (or timeout) → `check_failed` with the last 2,000 chars. `_kit_python()` returns `/usr/bin/python3` only when numpy imports there, else `sys.executable`.
  - Success → `status="ok"`, repo-relative `submission_path`, `details={rows, sha256, check_stdout, checked_with, scored: False}` — **no metric of any kind** in `GateResult` (the scored delta is the organizers', against this artifact).
- `tests/test_gate.py` (new, 10 tests) — synthetic 3-window data dir + the real tracked kit, ~0.5 s, no 240 MB dependency: happy path (exact header, 4 rows, ids as original strings, gate_done written), idempotency (digest unchanged), artifacts-dir fallback, missing/wrong-length/non-finite scores, missing kit, unexpected-exception → `status="error"`, no-metric regex on `asdict`, repo-relative path under repo root.
- `tests/test_interfaces.py` — the one sanctioned edit (≤5 lines, in this PR): `test_gate_stub_reports_not_implemented` → `test_gate_reports_error_when_scores_are_missing`, expecting `status="error"` + `details["reason"]=="missing_test_scores"` from the four-identical-temp-dir positional call.

**Justification**
- Closes review **C1** end to end: T1 loaded features → T2 persisted trusted scores → T3 now materialises the only artifact the hidden-test delta (35% rubric weight) is computed from, and validates it with the organizers' own `--check` as the kit README instructs.
- Never-raises + reason-coded errors mean a gate fault costs at most a missing `submission.csv`, never the run's `summary.json` (B's half of the C4 containment; A's call-site wrap lands separately).
- Labels stay out of the harness process: the kit's `load()` materialises test labels only inside the throwaway check subprocess (docstring records why), and no score value enters `GateResult`.
- `%.9g` because float64 carries ~9 significant digits; fewer would let formatting create ties in the within-user ranking that the model never made.

**Verification evidence**
- `pytest tests/test_gate.py tests/test_interfaces.py -q -W error` → **25 passed** (10 new gate + 15 interfaces incl. the updated assertion), 0.48 s.
- Full suite: **67 passed, 16 subtests, 8.86s** (57 → 67), still `-W error`, no API key.
- `grep -n "GAUC\|nDCG\|primary" src/evaluation/gate.py` → **no hits** (DoD item).
- Real-data acceptance (random 170,588 scores at the artifacts spelling, repo-relative data/kit paths): `status: ok`, `rows: 170588`, `check_stdout: "✓ 格式与对齐校验通过：170,588 行，split=test"`, `checked_with: .venv/bin/python` (expected — this machine's `/usr/bin/python3` lacks numpy, per the setup deviation), CSV head `row_id,user_id,video_id,score` / `0,0,3978,0.636961687`.
- Fixed during verification: fresh `run_dir` lacked `mkdir`, surfacing as `unexpected/FileNotFoundError` — now created before the CSV write.

**Not committed** — standalone PR per plan §5's split. Optional step 7 (labelled self-computed test score CLI) deliberately deferred until after T1–T8, per plan.

---

## 2026-08-29 · T4 — Minimal candidate environment (C3)

**Done** (TDD: 2 tests written first, RED, then implementation → GREEN)
- `src/agent/candidate_runner.py`
  - New module constants `PASSTHROUGH_KEYS` (PATH/LANG/LC_ALL/TZ + Windows system vars when present) and `THREAD_CAP_KEYS` (OMP/OPENBLAS/MKL/NUMEXPR/VECLIB = "1").
  - `_environment(self, workspace)` rebuilt **from scratch** — never `dict(os.environ)`: sets `PYTHONPATH=repo_root`, `PYTHONDONTWRITEBYTECODE=1`, `HOME`/`TMPDIR`/`TEMP`/`TMP` = workspace directory, `KUAIRAND_DATA_DIR=data_dir` (E's `build_features` entry), thread caps. In-code assert that no key starts with `OPENAI_`/`ANTHROPIC_`.
  - Both call sites updated (`test()` and `train()`); `train()`'s cwd moved from `self.repo_root` to `workspace.directory` (safe: every command path is absolute by construction via the controller's `_resolve_repo_path`).
  - E's `restricted_builtins()` wire-in in `_load_candidate` **deliberately skipped** — plan says land it only once E announces their T2. `run_candidate.py:42-52` untouched.
- `tests/test_isolation.py` — `test_candidate_environment_drops_provider_keys` (real subprocess with sentinel `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` patched into the parent env; asserts the child saw neither, `PYTHONDONTWRITEBYTECODE=1`, all five thread caps, `PYTHONPATH`=repo, `KUAIRAND_DATA_DIR`, `HOME`=workspace) and `test_candidate_subprocess_cwd_is_the_workspace`.

**Justification**
- Closes **C3**: `.env` is loaded into `os.environ` by the LLM layer before candidates run, and the old `dict(os.environ)` copy handed `OPENAI_API_KEY` to every LLM-written candidate. An allowlist built from scratch is leak-proof by construction; the assert is the belt to that braces.
- The subprocess-based test (rather than inspecting the dict) proves what the *child* actually inherits, not what we think we built.
- Single-thread caps keep candidate subprocesses from oversubscribing cores; `HOME` in the workspace keeps any stray dotfile writes contained.

**Verification evidence**
- `pytest tests/test_isolation.py -q -W error` → **7 passed** (5 from T1 + 2 new).
- Acceptance grep `grep -n "dict(os.environ)\|cwd=self.repo_root" src/agent/candidate_runner.py` → **no hits**.
- Full suite: **69 passed, 16 subtests, 8.71s** (67 → 69), `-W error`, no API key.
- **Real end-to-end smoke** (valid `bpr`-family candidate, real dataset, real `executor.train()`): `status: success`, `metrics.primary 0.4837` for all-zero scores — matching the published random-baseline validation ≈ 0.4834, so the official evaluator path composes with the new env/cwd; `test_scores_path` = repo-relative `runs/.../artifacts/001_cand01/test_scores.npy`; saved array **float64 (170588,)**. Scratch run dir deleted.
- Two smoke lessons (both smoke-harness errors, not code bugs, but worth recording): (1) a *relative* `data_dir` now fails under the workspace cwd — confirming the plan's "every path absolute by construction" invariant is load-bearing; (2) a `run_dir` outside `repo_root` trips the pre-existing `stdout_path.relative_to(repo_root)` in `train()` (caught as a `bad_output`-style failure, no crash). The controller guarantees both invariants; left unchanged, noted for A.

---

## 2026-08-29 · T5 + T6 — `failure_class` everywhere + sanity band/two-sided predicate (I-3, I11)

**Done** (TDD: 6 new tests first, RED, then implementation → GREEN)
- `src/evaluation/official.py` — new constants `SANITY_FLOOR = 0.47`, `SANITY_CEILING = 0.80`, `OFFICIAL_VALIDATION_BASELINE = 0.6016`, `BASELINE_TOLERANCE = 0.003`, and two pure functions:
  - `classify_primary(primary)` → `"low_score"` below the floor, `"leak"` above the ceiling, else `None`. Classifier only — callers decide.
  - `within_baseline_tolerance(primary, official=0.6016, tolerance=0.003)` → `abs(primary - official) <= tolerance + 1e-12` (cushion keeps the boundary inclusive under binary floats — `|0.5986 − 0.6016|` computes as `0.0030000000000000027`, so the naive `<= 0.003` rejects the plan's own True cases).
- `src/experiments/run_candidate.py` — the T2 placeholder `"sanity_class": None` is now `classify_primary(float(metrics["primary"]))`. Classified, never raised: `result.json` still written, ledger keeps the number.
- `src/agent/candidate_runner.py` — every failed `ExperimentOutcome` now carries a `failure_class` from the frozen six-value set: `"crash"` (non-zero exit), `"timeout"` (TimeoutExpired), `"bad_output"` (OSError/ValueError/KeyError/JSONDecodeError, which also catches the non-finite-metric raise), `"missing_test_scores"` (T2), `"low_score"`/`"leak"` (new sanity branch: non-`None` `sanity_class` → failed with the band in the error text, metrics kept). No control flow keys on message text (I13).
- `tests/test_isolation.py` — `test_failure_classes_cover_every_return_path` (AST-parses `candidate_runner.py`; every `status="failed"` call has the `failure_class` keyword; literal values ∈ six-set; the sanity branch's runtime value is domain-pinned by `classify_primary`'s bounds test), `test_timeout_outcome_is_classified` (real executor with `experiment_timeout_seconds=0`), `test_classify_primary_bounds`, `test_within_baseline_tolerance_is_two_sided`.
- `tests/test_candidate_output.py` — `test_ceiling_hit_is_marked_as_leak` (reuses the canonical perfect-ranking fixture, primary 1.0 = above ceiling, with a comment saying why), `test_floor_miss_is_marked_low_score` (inverted scores → primary 0.0).

**Justification**
- I-3: A's Debugger picks retry-vs-skip from `failure_class` and D's journal prints it — untyped strings would have forced exactly the message-text matching I13 forbids. `"leak"`/`"low_score"` tell A *not* to spend a repair round.
- I11: finiteness-only checks let a leaked 0.99 be promoted to best and let 0.85 pass the baseline gate. The band [0.47, 0.80] brackets random (0.475) to well past the official FM (0.5946) while sitting under the oracle ceiling (0.8484 valid); `observe_success` only runs for `status == "success"`, so a ceiling hit can never become best.
- Ownership respected: B shipped only the predicate; the two `_ensure_baseline` call sites are A's — PR draft below, not applied.

**Draft PR for A (≤20 lines, `research_controller.py` — B does not apply this):**
```diff
+from src.evaluation.official import within_baseline_tolerance
 # in _latest_valid_baseline's loop (currently :45):
-            if best.get("experiment_id") == "official_fm_seed0" and primary >= threshold:
+            if best.get("experiment_id") == "official_fm_seed0" and within_baseline_tolerance(primary, threshold):
 # in _ensure_baseline (currently :55, :62):
-    baseline = _latest_valid_baseline(run_root, official - 0.002)
+    baseline = _latest_valid_baseline(run_root, official)
-    if primary < official - 0.002:
-        raise RuntimeError(f"Official FM baseline gate failed: {primary:.4f} < {official - 0.002:.4f}")
+    if not within_baseline_tolerance(primary, official):
+        raise RuntimeError(f"Official FM baseline gate failed: {primary:.4f} outside [{official - 0.003:.4f}, {official + 0.003:.4f}]")
```
(`threshold` keeps its parameter name; it now carries the official centre instead of a lower bound.)

**Verification evidence**
- `pytest tests/test_isolation.py tests/test_candidate_output.py -q -W error` → **21 passed** (11 + 10), including the plan's exact bound cases (0.4699/0.47/0.80/0.8001/0.6015; 0.5986/0.6046 True, 0.5985/0.6047/0.85 False).
- Acceptance grep `grep -c "failure_class=" src/agent/candidate_runner.py` → **5** (≥ 5).
- The three pre-existing `test_candidate_output.py` cases still pass.
- Full suite: **75 passed, 16 subtests, 8.72s** (69 → 75), `-W error`, no API key.
- Two RED-phase fixes: the float-boundary cushion above (implementation nuance), and my timeout test originally used an out-of-repo run dir — tripping the same documented `relative_to(repo_root)` invariant as the T4 smoke; the test now uses an in-repo scratch dir with cleanup.
- Consistency: the T4 smoke candidate's primary (0.4837) sits inside the band, so in-band candidates still succeed — no behavior regression from the sanity branch.

---

## 2026-08-29 · T7 — Isolation + evaluator-convention tests (I1, I2)

**Done**
- `tests/test_official_evaluation.py` (I2) — extended from 1 to 7 tests, each with a hand-computed expectation pinning the kit evaluator's divergent conventions: zero-positive user counted in nDCG (0.0) and excluded from GAUC (GAUC 1.0 / nDCG 0.5 / primary 0.75 / users 2 / rows 4); all-positive user excluded from GAUC; GAUC weighted by positive count (2/3, not 0.5); ties broken by row order via stable sort (labels [0,1] → 1/log2(3), [1,0] → 1.0, GAUC 0.5 both ways); nDCG truncation at k=5 (rank 5 → 1/log2(6), rank 6 → 0.0); users/rows reporting.
- `tests/test_isolation.py` (I1) — `TrainValidSplitTests`: exactly the keys `{train, valid}` with **1,141,112** / **124,909** rows; every date within 20220408–20220428 with the actual bounds pinned; first 5,000 rows of each split equal `data.load()`'s with matching lengths (row equality ⇒ encoding equality — the kit's `encode` is a pure function of the rows). All under the `skipUnless(REAL_DATA)` guard.

**Justification**
- I1: a "helpful" `test` key added to `load_train_valid` (or a date-filter regression) previously broke no test; now it breaks three. These invariants are the isolation contract — the moment they drift, train/valid stop matching the official evaluator's world.
- I2: these six conventions are exactly where a future fast-path evaluator would silently diverge from the kit (tie handling, zero-positive inclusion, positive-count weighting, k-truncation). Hand-computed expectations, not golden-file snapshots of the kit's own output.

**Verification evidence**
- `pytest tests/test_official_evaluation.py -q -W error` → **7 passed in 0.04s** (plan requires < 0.2s). All six hand-computed values matched the kit's behavior on the first run.
- **Deviation from plan found and documented:** the plan's I1 expectation "min date 20220408" is factually off — the dataset's earliest standard-log row is **20220409** (verified against the kit's own loader: train 20220409–20220421, valid 20220422–20220428; "4_08" is the nominal window start, not a row date). Test asserts containment plus the actual bounds, with a comment.
- Full suite: **84 passed, 16 subtests, 19.96s** (75 → 84), `-W error`, no API key. The +11s is the three real-data loads, as expected.

---

## Queue (next up)

- **T8** — regenerate the baseline run with committed code (`env -u OPENAI_API_KEY python -m src.agent.controller --config configs/baseline.json`), verify 0.6016±0.0008 / stop_reason / scrub greps, replace `runs/20260828T141646Z_baseline`, stage exactly five files.
- Then hand-offs per plan §6 and the DoD sweep. Commit/PR sequence per plan §5: T1+T2 / T3 / T4+T5+T6 / T7 / T8.

*Append new entries above this line.*
