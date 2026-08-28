# Team split — five owners, zero shared files

Principle: every file has exactly one owner. Shared interfaces are frozen first in one small PR; then five branches
fan out on disjoint files. Cross-cutting one-liners (wiring a new module into the loop) go through the loop owner.
Issue codes (C1…, I1…) refer to `README.md` in this directory.

## Step 0 — interface freeze (first 1–2 hours, one PR, merged before anyone branches)

Adds hooks and stubs, no behaviour. Owner: A (or Claude via SDD, ~30 min).

| File | Change |
|---|---|
| `src/agent/types.py` | `CandidateOutput.test_scores: np.ndarray \| None`; `ExperimentOutcome.failure_class: str \| None`; `RunState.data_card_path`; family validated through a registry instead of a literal |
| `src/experiments/contracts.py` | `CandidateContext.test_x` |
| `src/evaluation/gate.py` (new) | `run_gate(run_dir, node_dir, data_dir, kit_dir) -> GateResult` stub |
| `src/evaluation/datacard.py` (new) | `render_data_card(data_dir) -> str` stub |
| `src/agent/report.py` (new) | `render_reports(run_dir) -> None` stub |
| `src/agent/families.py` (new) | `FAMILIES` registry with the two existing families (name → method card path, trusted sampler, parameter grid) |
| `src/models/features.py` (new) | `build_features(rows, spec) -> np.ndarray` stub |
| `src/agent/llm.py` | `build_provider(config) -> LLMProvider` factory stub (`"openai"` \| `"scripted"`) |
| `configs/ranking_losses.json` | `max_iterations: 50`, `max_training_attempts`, `llm.provider`, `data_card_path` |
| `src/agent/research_controller.py` | one call each to the stubs (data card at start, gate + report at end, provider factory) |

After this merges, everyone branches from `main`.

### Step 0 status — DONE, awaiting merge (2026-08-29)

Branch `feat/step0-interface-freeze`, three commits on top of `main` @ `424238f` (unsigned, not pushed):
`1cc5c4b` design spec · `cbf8330` step 0 (the freeze below) · `553095d` step 0b (`src/agent/errors.py` with
`LLMError`, `TokenBudgetExceeded`, `RoleOutputInvalid`, `IncompleteResponse`; `pytest.ini` registering the `slow`
marker; `tests/test_errors.py`). Reviewed and approved: 0 Critical / 0 Important / 7 Minor. Verified: 47 tests pass
(28 existing + 19 new) under `pytest -W error` with no API key. To land it:

```
git push -u origin feat/step0-interface-freeze   # then open the PR and merge before anyone branches
```

What it contains, exactly: `src/agent/families.py` registry (`types.py` validates against it); optional fields
`CandidateContext.test_x`, `CandidateOutput.test_scores`, `ExperimentOutcome.failure_class`, `RunState.data_card_path`
(old run states still load); no-op stubs `src/evaluation/gate.py` (B), `src/evaluation/datacard.py` (D),
`src/agent/report.py` (D), `src/models/features.py` (E); `build_provider()` in `src/agent/llm.py` (C) — `"scripted"` is
reachable from config; controller wiring (provider factory; `summary["gate"]` before `summary.json`; `render_reports()`
after `results.json`); config `max_iterations: 50`, `max_training_attempts: 50`, `data_card_path: null`.

**Hand-offs from the Step 0 review** (minor findings, deferred to the owner of the file):

| Owner | Note |
|---|---|
| A | `policy.py` still has its own literal family set (`FAMILIES = {"bpr", "group_softmax"}`) — point it at `families.family_names()`; trim the `families.py` docstring to match. |
| A | `max_iterations: 50` currently also raises the training-attempt cap to 50 and `max_proposals` to 100 (`research_controller.py:226, 353, 359, 362`) — I5 splits the knobs; until then the only other bounds are 6 h wall-clock and the 150k-token budget. |
| B | `run_gate(...)` is called uncontained *before* `summary.json` is written (`research_controller.py:429-438`). Once it does real work, an exception there loses the run's summary — wrap it, or write a failure status into `summary["gate"]`. Also pass its four `Path` arguments by keyword. |
| C | `build_provider`'s scripted branch is tested with an empty `[]` payload and the repo-relative `script_path` branch has no test — add a non-empty round-trip and a relative-path case when wiring I14. |

**Two rulings from writing the per-owner plans (`plans/A-…E-*.md`):**

- Typed exceptions live in the frozen `src/agent/errors.py` — nobody owns it. **C** raises them (`llm.py`,
  `roles.py`); **A** catches them (`research_controller.py`). Neither adds exception classes elsewhere.
- `tests/test_interfaces.py` pins the Step 0 stubs (e.g. `run_gate` returns `not_implemented`). The owner who fills a
  stub updates that one assertion in the same PR (≤ 5 lines) — the only sanctioned edit to that file. B and E will
  each do this once; different assertions, so rebase before merge and there is no conflict.
- E's `multi_task` family (T6) is a stretch item: only after T1–T5 are merged and only if Day 3 has room. Core E
  scope is ~6.5 h.
- The offline smoke run's expected `stop_reason` is whatever A's I5 split defines; until A lands, C's e2e test accepts
  either `iteration_budget_reached` or the new name.

**Caveat for everyone during the parallel phase:** the harness refuses to resume a run whose `src/**/*.py` source
manifest changed (`controller.py:26-35`). Every PR that adds or edits a source file changes the revision, so runs started
before a merge cannot be resumed after it. Expected — don't rely on `--resume` across merges until the tree settles;
the committed baseline run is being regenerated by B anyway (C5).

## Owners

### A — Loop & robustness (harness author)
Files: `src/agent/research_controller.py`, `policy.py`, `convergence.py`, `controller.py`, `configs/*`,
`tests/test_research_loop.py`, `tests/test_agent.py`, new `tests/test_controller_robustness.py`.
Tasks: C4 unkillable loop (classify → re-prompt ≤2 → `continue`; breaker for harness errors; injected-bad-response
test) · C5 baseline selection by recorded revision, verify artifact exists · I5 split the three `max_iterations`
semantics, default 50 · I6 report the official convergence verdict separately from the coverage-gated stop · I7 one
convergence implementation, tested against the reference formula · I9 wall-clock from before the baseline gate ·
I10 `intervene --reason` command + counter · I12/I13 log skipped summaries, typed `TokenBudgetExceeded` · I3 `_save()`
before `record_iteration()` · wiring calls for B/C/D/E modules as they land. Also: run operator for the live runs.
≈ 5–6 h.

### B — Gate & contracts
Files: `src/evaluation/official.py`, new `src/evaluation/gate.py`, `src/experiments/contracts.py`, `run_candidate.py`,
`run_baseline.py`, `src/agent/candidate_runner.py`, `src/agent/types.py` (post-freeze), `tests/test_candidate_output.py`,
`tests/test_official_evaluation.py`, new `tests/test_gate.py`, `tests/test_isolation.py`, `runs/` (regenerated baseline).
Tasks: C1 end-to-end — `load_test_meta()` (row_id/user_id/video_id + features, never labels), `test_x` in the context,
`test_scores` required from candidates and persisted per node, `gate.py` writing `submission.csv` (`%.9g`), running
`submit.py --check --split test`, refusing to run twice · C3 minimal candidate env (no `OPENAI_*`, cwd = workspace) ·
I11 sanity floor 0.47 / ceiling 0.80, two-sided baseline gate `|p − 0.6016| ≤ 0.003` · I1/I2 isolation and evaluator
convention tests · C5 regenerate the baseline run with current code, scrub absolute paths, commit. ≈ 5 h.

### C — LLM layer, offline mode, docs
Files: `src/agent/llm.py`, `src/agent/roles.py`, `tests/test_openai_runtime.py`, `tests/test_research_runtime.py`,
new `tests/test_llm_retry.py`, new `configs/offline_smoke.json`, `README.md`, `AGENTS.md`, `PLAN.md`, new `docs/devpost.md`.
Tasks: I4 typed SDK exceptions, 5 attempts 2 s → 60 s, honour `Retry-After`; check `response.status == "incomplete"`
· I14 `build_provider` → `ScriptedProvider` reachable from config; one e2e test through the real training subprocess
· prompt-caching: stable prefix (task text, contract, method card, data card) before volatile state · `{data_card}`
slot in the Researcher prompt read from `RunState.data_card_path` · prompts read family details from E's registry
(so E never edits `roles.py`) · I17 docs: POSIX commands, limitations, contributions, iteration/convergence
definitions, architecture diagram, `docs/devpost.md`; optional video. ≈ 4–5 h.

### D — Data card, journal, repo hygiene
Files: new `src/evaluation/datacard.py`, new `src/agent/report.py`, `src/agent/audit.py`, `logger.py`, `.gitignore`,
new `tests/test_datacard.py`, `tests/test_report.py`.
Tasks: I15 numpy-only data card (rows per split, label rates by tab, duration buckets, day-to-day drift, duplicate
rate, feature-table coverage, the `video_features_statistic` leakage flag, metric conventions) rendered to
`runs/<id>/DATA_CARD.md` · I16 `journal.md` + `results.md` renderer from `iterations.jsonl` + `passes/` with a unified
diff per iteration (`python -m src.agent.report <run_dir>`) · commit `stdout/` and `generated_experiments/` for the
final run · untrack `data/` (`git rm -r --cached data/` + download script), `.DS_Store`, `.pyc`. ≈ 4.5 h.

### E — Search surface, safety, method cards
Files: `src/agent/safety.py`, new `src/agent/families.py` (post-freeze), new `src/models/features.py`,
`src/models/sampling.py`, `src/models/fm_core.py`, `src/models/baselines.py`, `research/methods/*.md`,
`tests/test_safety.py`, `tests/test_sampling.py`, new `tests/test_features.py`.
Tasks: C2 first — reject dunder `Name`s and `Subscript`s on them, forbid attribute *access* not just calls, add
`log_standard`/`log_random`/`KuaiRand`/`.csv`/`/data/` to the text blocklist, exec with restricted `__builtins__`,
tests for each bypass · I8 a `history_features` family: trusted `build_features` (train-only user history rates,
author affinity, recency, tab crosses, temporal/drift), allowlist entries, parameter grid, method card; then a
`multi_task` family (auxiliary `is_click`/`is_like`/`play_time_ms`); keep `k == 16` pinned (kit dead end) · method
cards for both, in the existing card format. ≈ 5–7 h (the stretch item; give it to the strongest ML person).

## Rules that keep it conflict-free

1. **One owner per file.** Need a change in someone else's file? Ask the owner, or send them a ≤20-line PR they merge.
2. **New tests in new files.** Never edit another owner's test file.
3. **Rebase on `main` twice a day; PRs ≤ 300 lines; `pytest` green before merge.** Small, frequent merges beat one big one.
4. **Config: add files, don't edit shared ones.** A new config file belongs to whoever creates it
   (`configs/offline_smoke.json` → C, `configs/features_run.json` → E); `configs/ranking_losses.json` and
   `configs/baseline.json` stay A's.
5. **Run directories are personal** (`runs/<initials>_…`, gitignored) until the final run. The final run is copied
   to `runs/final/` by A; D's `.gitignore` carries the `runs/final/**` exception (including its `stdout/` and
   `generated_experiments/final/`), so no `git add -f` is needed.
6. **Shared surfaces only move in the freeze PR** — `types.py`, `contracts.py`, `configs/ranking_losses.json`. A second
   freeze PR (by A) is the way to change them again, not five drive-by edits.

## Sequencing

| When | What |
|---|---|
| Day 1, hour 0–2 | Step 0 freeze PR merged; everyone branches |
| Day 1 | A: C4 → I5/I6/I9/I10 · B: C1 → C3/I11 · C: I14 → I4 · D: I15 · E: C2 → start I8 |
| Day 1 end | Offline e2e run green on `main` (C's scripted provider + B's gate + A's loop) |
| Day 2 | B: tests + regenerated run · C: docs · D: I16 journal · E: I8 families · A: integration, first **live run** overnight |
| Day 3 | Fixes from the live run (A), final run, README/Devpost (C), commit final run (A) |

If one of the five is not a coder: give them C's docs half (README, Devpost, video, contributions) plus running and
watching the live runs with A; C keeps only the LLM-layer code.
