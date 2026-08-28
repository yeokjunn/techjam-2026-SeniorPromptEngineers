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

## Queue (next up)

- **T3** — `gate.py`: `submission.csv` + kit `submit.py --check --split test` (note: this machine's `/usr/bin/python3` lacks numpy → `_kit_python()` falls back to the venv interpreter).
- Then T4 (minimal env — dependency-free), T5/T6, T7, T8; hand-offs per plan §6. T1+T2 PR ready to push.

*Append new entries above this line.*
