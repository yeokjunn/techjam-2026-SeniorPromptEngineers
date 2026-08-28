# Owner D — Data card, journal, repo hygiene
Branch: feat/datacard-journal   ·   Base: main after the Step 0 merge (`cbf8330` + `553095d`)   ·   Effort: 4.5 h

## 1. Mission

You give the agent evidence about the data and the judges prose about the run: a computed data card for the
Researcher prompt (I15/I-4), and `journal.md` + `results.md` rendered from the run ledger (I16/I-5). Plus the hygiene
that makes the final run readable in git. Rubric: Innovation 20 % (the card grounds hypotheses in facts, not prior
metrics), Robustness 35 %, Deliverables (a judge reads the reports, not `iterations.jsonl`).

## 2. Files you own (exclusive) / files you must not touch

**Yours:** `src/evaluation/datacard.py` and `src/agent/report.py` (stubs today), `src/agent/audit.py`,
`src/agent/logger.py`, `.gitignore`, new `scripts/download_data.sh`, `tests/test_datacard.py`, `tests/test_report.py`.

**Not yours:** everything else — `research_controller.py`, `policy.py`, `controller.py`, `configs/*` (A);
`gate.py`, `official.py`, `candidate_runner.py` (B); `roles.py`, `llm.py`, `README.md` (C); `safety.py`,
`families.py`, `models/*` (E) — plus the frozen `types.py`, `contracts.py`, `configs/ranking_losses.json` and
`src/agent/errors.py` (nobody owns it). `tests/test_interfaces.py` pins the Step 0 stubs; team-split lets the owner
filling a stub update that one assertion in the same PR (≤ 5 lines), but T1 step 2 and T2 step 1 keep both assertions
true anyway, so you should need no edit. Otherwise: ask the owner, or send a ≤20-line PR.

## 3. Setup (15 minutes)

```sh
cd /Users/Ke_Jun_YEO_from.TP/Desktop/personal/techjam-2026-SeniorPromptEngineers
git checkout main && git pull                 # must contain cbf8330 + 553095d
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -r requirements.txt     # numpy, openai, python-dotenv
python -m pip install pytest
unset OPENAI_API_KEY
pytest -q -W error                            # expect: 47 passed
git checkout -b feat/datacard-journal
ls data/KuaiRand-Pure/data                    # 6 CSVs; T3 step 2 downloads them if missing
```

You do not need `OPENAI_API_KEY`: neither renderer calls an LLM, and both new test files must pass with it unset.

## 4. Tasks, in order

### T1 · I15 / I-4 · Computed data card   (effort L, ~2 h)

- **Why:** the Researcher prompt carries prior metrics only (`roles.py:66-87`) and three of six CSVs are never opened
  (`official.py:36-46`); spec-compliance §5 marks "Data card / EDA artifact" **missing** — a direct hit on Innovation.
- **Where:** `src/evaluation/datacard.py:8-10` (stub returns `""`); split constants `official.py:12-15` /
  `evaluate.py:11` (train 20220408-20220421, valid 20220422-20220428, test 20220429-20220508); skip-before-read
  shape `official.py:50-57`.
- **Do:**
  1. Implement `render_data_card(data_dir: Path) -> str` with `csv`, `hashlib`, `pathlib`, `numpy` only — no pandas.
  2. **Guard first:** if any of the five CSVs is missing under `data_dir`, `return ""`. That makes the card optional
     (I-4: A skips silently on an empty string) *and* keeps `test_data_card_stub_renders_nothing_yet` true.
  3. Read logs with `csv.reader` + a header→index map, not `csv.DictReader` (measured 0.76 s vs 1.70 s on the
     1.14 M-row file). Per row read `date` first and classify exactly as `official.py:51-57`; if the row is
     test-dated, increment a counter and `continue` **before indexing any outcome column** (`is_click` … `long_view`,
     `play_time_ms`, `profile_stay_time`, `comment_stay_time`).
  4. Accumulate per split: rows; user-id and video-id sets; per-outcome positive counts (train and valid only);
     `tab → [rows, clicks, long_views]`; `user_id → count`; `duration_ms == 0` rows and the videos they cover; exact
     duplicates via `hashlib.blake2b(line.encode(), digest_size=16).digest()` in a set (a 32-bit CRC would collide
     ~140× here). Quantiles: `np.percentile(np.fromiter(counts.values(), np.int64), q)`.
  5. Read the three feature tables fully: coverage against the log id sets (expect 100.000 %), the
     `is_live_streamer == -124` count, `'UNKNOWN'` counts in `user_active_degree` / `video_type` / `upload_type`, the
     constant columns, and `sum(show_cnt)` plus `counts` min/max from the statistics table for the leakage paragraph.
  6. Render ≤ 200 lines of Markdown in this order: splits (rows / users / videos; test = row count only) · label
     rates on train and valid · `tab` table · rows-per-user quantiles · data quality (`duration_ms == 0`, duplicates,
     sentinels, constant columns, `hourmin` = hour × 100 in UTC+8, the log labelled 4/08 starts on 20220409) ·
     feature coverage · leakage flag · metric conventions · measured dead ends.
  7. **Determinism:** no timestamp, no absolute path, no echo of `data_dir`, sorted keys — C puts this in the *stable*
     prompt prefix, and a byte that changes between runs costs a cache hit.
  8. **Vocabulary constraint.** The Builder may quote the card in a generated docstring, where `validate_source`
     runs. Use no string from `safety.py:60-72` `FORBIDDEN_TEXT` (`kuairand-starter-kit`, `src.evaluation`,
     `data/judge`, `ground_truth`, `test_truth`, `subprocess`, `socket`, `requests`, `urllib`) and none E adds under
     C2 (`log_standard`, `log_random`, `KuaiRand`, `.csv`, `/data/`) — never a filename or a path.
  9. Static sections. **Metric conventions:** within-user ranking; nDCG sorts by score with a stable sort so ties
     fall back to row order (`evaluate.py:51`); AUC averages ranks over ties (`:22-27`); GAUC counts only users with
     `0 < positives < impressions`, weighted by positive count (`:54-56`); zero-positive users score nDCG 0.0 and
     **are** in the mean (`:8-9, 57-59`); primary is the mean. **Leakage flag:** the video statistics table is a
     period aggregate over full-platform traffic (implied total shows ≈ 12.83 B vs 2.62 M logged exposures, ≈ 4,892×;
     `counts` 45-181 exceeds the log span), so it encodes the future — legal, but any use must be caveated.
     **Measured dead ends** (`kuairand-starter-kit/README.en.md:133-139`): all 13 static feature fields scored 0.5940
     vs 0.5950 for the 5; k = 8/16/32 gave 0.5895/0.5902/0.5887 ("the bottleneck is not features or capacity");
     first-order user-only terms contribute exactly 0 to a within-user ranking.
  10. Add a `__main__` block so `python -m src.evaluation.datacard data/KuaiRand-Pure/data` prints the card.
- **Interface (verbatim, I-4):** *`src/evaluation/datacard.py::render_data_card(data_dir: Path) -> str` (Markdown).
  Provided by **D**. **A** wires it at run start: write the string to `<run_dir>/DATA_CARD.md` and set
  `RunState.data_card_path` to that path (skip silently if the string is empty). **C** reads `state.data_card_path`
  in `roles.py` and places the text in the stable prompt prefix (before volatile state).*
- **Tests:** new `tests/test_datacard.py`, on a synthetic ~30-row fixture written to `tmp_path` (fast, no dataset):
  - `test_missing_files_render_an_empty_card` — empty dir → `""`.
  - `test_test_dated_rows_are_counted_but_never_parsed` — fixture rows dated 20220429 carry `long_view="BOOM"` and
    `duration_ms="BOOM"`; rendering must succeed and report the test row count. The skip-before-read proof.
  - `test_train_and_valid_rates_match_hand_computed_values`; `test_card_is_under_two_hundred_lines`;
    `test_card_is_deterministic`; `test_card_avoids_the_generated_source_blocklist` (import `FORBIDDEN_TEXT` from
    `src.agent.safety`; assert neither it nor `{"log_standard", "KuaiRand", ".csv", "/data/"}` appears).
  - `test_full_dataset_card_matches_the_profiler` — `skipUnless(os.environ.get("DATACARD_FULL"))`; asserts the
    acceptance numbers below and a < 60 s render.
- **Acceptance criteria:**
  - [ ] `pytest -q -W error tests/test_datacard.py` green with `OPENAI_API_KEY` unset; same with `DATACARD_FULL=1`.
  - [ ] `time python -m src.evaluation.datacard data/KuaiRand-Pure/data | tee /tmp/card.md` < 60 s; `wc -l` ≤ 200;
        `grep -c 'BOOM\|/Users/\|20260' /tmp/card.md` → 0 (no absolute paths, no timestamps).
  - [ ] Splits: train 1,141,112 rows / 26,210 users / 7,538 videos; valid 124,909; test 170,588 (count only). Train
        rates `long_view` 33.6620 %, `is_click` 46.3447 %, `is_like` 1.8677 %, `is_follow` 0.1007 %, `is_comment`
        0.2568 %, `is_forward` 0.0996 %, `is_hate` 0.0421 %, `is_profile_enter` 2.5391 %.
  - [ ] `tab`: tab 1 = 73.16 % of train rows, click 52.97 %, long_view 38.61 %; tab 0 = 13.15 %, long_view 4.22 %;
        tab 4 = 6.62 %, long_view 48.93 %.
  - [ ] Quality: 239 videos with `duration_ms == 0` (24,076 train rows); 15,609 exact duplicate rows in the 4/08-4/21
        log; `is_live_streamer == -124` on 21,127 of 27,285 user rows; `'UNKNOWN'` in `user_active_degree` (6) /
        `video_type` (1) / `upload_type` (80); constant columns `is_lowactive_period`, `visible_status`, `is_rand`;
        `hourmin` = hour × 100 UTC+8; coverage 100.000 % both ways; leakage flag, metric conventions, dead ends.
- **Depends on / blocks:** nothing. Blocks A's I-4 wiring and C's `{data_card}` slot — tell both when it merges.

### T2 · I16 / I-5 · `journal.md` + `results.md` renderer   (effort L, ~1.75 h)

- **Why:** spec-compliance §11 marks "Rendered `journal.md` / `results.md`" **missing** and per-iteration code a bare
  sha256 (`research_controller.py:291-292`), with the source only in gitignored `generated_experiments/`. Judges read
  prose; Innovation and Robustness are graded from it.
- **Where:** `src/agent/report.py:8-10` (stub returns `None`), already called at `research_controller.py:459` after
  `results.json`. Inputs in `<run_dir>`: `iterations.jsonl` (`audit.py:45-46`; research record at
  `research_controller.py:283-302`, rejection at `:163-171`), `summary.json` (`:413-438`, `gate` at `:437`),
  `results.json` (`:442-458`), `resources.json` (`audit.py:51-60`), `research_memory.jsonl` (`audit.py:43`),
  `passes/NNN_<role>_<seq>.json` (`audit.py:39-42`) and `run_config.json` (`:103`).
- **Do:**
  1. `render_reports(run_dir: Path) -> None`. If neither `iterations.jsonl` nor `summary.json` exists, return `None`
     and write nothing (keeps `test_report_stub_returns_none` true). Every field access is `.get(...)` with a
     `not reported` fallback; the renderer must never raise on a partial run. Add a `__main__` block taking the run
     directory as its one argument, so I-5's `python -m src.agent.report <run_dir>` is re-runnable.
  2. Define `REPO_ROOT = Path(__file__).resolve().parents[2]` locally (as `official.py:10`) — do not import
     `controller.py`. Resolve `generated_root` from `run_config.json`: absolute stays absolute, else
     `REPO_ROOT / value` (mirrors `controller.py:21-23`); the candidate directory is then deterministic
     (`candidate_runner.py:25-30`) as `<generated_root>/<run_id>/<NNN>_<manifest.candidate_id>/candidate.py`.
  3. **Diff:** parent = the record whose `manifest.candidate_id == proposal.parent_experiment`, path derived the same
     way. Emit `difflib.unified_diff(parent_src, child_src, ...)` in a fenced ```diff block, truncated to 200 lines
     with a `… truncated, full source at <path>` line. No parent → diff against the empty string.
  4. **Fallback when the directory is absent** (cleaned or gitignored): read the code from
     `passes/<iteration:03d>_builder_*.json` → `["result"]["data"]["code"]` (shape: `audit.py:39-42` wrapping
     `LLMCallResult.to_record()`, `llm.py:146-149`), and say in the journal which source was used.
  5. `journal.md` — one `## Iteration N — <experiment_id>` section per record, in file order: title, hypothesis,
     rationale and evidence (`proposal.hypothesis`, `.rationale`, `.evidence[*].title/url`); family and parameters
     (`manifest.family`, `.parameters`); the diff; metrics with **Δ vs best-so-far** (running max of
     `outcome.metrics.primary`) and **Δ vs baseline** (`results.json[*].delta_vs_baseline`, else
     `primary − run_config.official_validation_baseline`); `outcome.failure_class` / `.error` / `.recovery` /
     `repairs`; both critic verdicts (`preflight`, `postflight`: `decision` + rationale); seconds
     (`outcome.duration_seconds`) and tokens (summed over `research_memory.jsonl` records whose `iteration` matches —
     step 7); replications (`candidate_id` ending `_seed<N>`, grouped under their source with mean and spread of
     `primary`). `critic_rejected` records have no manifest or outcome — render a short "rejected before code" block,
     the "what we chose not to try and why" material judges reward.
  6. `results.md` — one page: baseline vs best on validation (GAUC / nDCG@5 / primary) with deltas against
     `official_validation_baseline` (0.6016, from `run_config.json`); `summary["gate"]["status"]` and
     `["submission_path"]`; tokens by role (aggregate `research_memory.jsonl` on `role` → `usage.total_tokens`) and
     total from `summary["token_usage"]`; wall-clock in seconds and hours; iterations used of
     `budgets.max_iterations` (50) and how many failed; `summary["converged_official"]` beside `["stop_reason"]`;
     interventions (count from `summary["manual_interventions"]`, reasons from `interventions.jsonl` if present —
     A's I-8 — else "none recorded"); `resources.json["gpu_hours"]` (0.0).
  7. `src/agent/audit.py` — one additive line in `record_pass` (`audit.py:43`): include `"iteration": iteration` in
     the record appended to `research_memory.jsonl`, enabling per-iteration token attribution. Nothing else reads
     that file (`grep -rn research_memory src tests` → the writer and `research_controller.py:399` only). `logger.py`
     needs no change; you own it so the ledger's writer and reader have one owner.
- **Interface (verbatim, I-5):** *`src/agent/report.py::render_reports(run_dir: Path) -> None` writes
  `<run_dir>/journal.md` and `<run_dir>/results.md` from `iterations.jsonl`, `summary.json`, `results.json`,
  `resources.json`, `passes/*.json` and `generated_experiments/`. Provided by **D**; already wired after
  `results.json` (Step 0). Must be re-runnable: `python -m src.agent.report <run_dir>`.*
  Also consumed verbatim: **I-3** *… **D** prints it [`failure_class`] in the journal.* **I-8** *… **D** prints them
  [interventions] in `results.md`.* **I-9** *… **A** reports both `converged_official` and the stop reason; **D** prints both.*
- **Tests:** new `tests/test_report.py`.
  - `test_empty_run_directory_is_a_no_op` — empty `tmp_path` → returns `None`, writes nothing.
  - `test_fixture_run_renders_both_reports` — build a fixture run dir in `tmp_path`: `run_config.json` (**absolute**
    `generated_root` inside `tmp_path`, `official_validation_baseline: 0.6016`, `budgets.max_iterations: 50`),
    `iterations.jsonl` with four records (`critic_rejected`, a success, a `failed` with `failure_class: "timeout"`,
    a `_seed1` replication), `summary.json` (`gate`, `token_usage`, `stop_reason`, `converged_official`,
    `manual_interventions: 1`), `results.json`, `resources.json`, `research_memory.jsonl`,
    `passes/002_builder_0.json`, `interventions.jsonl`, two `candidate.py` files.
  - `test_journal_contains_a_unified_diff_between_parent_and_child` (a ```diff fence plus a `+` line unique to the
    child); `test_journal_falls_back_to_the_builder_pass_when_the_directory_is_gone` (delete the generated tree,
    re-render, code still present and the fallback named); `test_journal_prints_failure_class_and_repairs`.
  - `test_results_reports_gate_deltas_tokens_and_interventions` (gate status, `0.6016`, per-role token table,
    `of 50`, stop reason, `converged_official`, intervention reason, `GPU-hours: 0`);
    `test_partial_summary_does_not_raise` (drop `gate`, `converged_official`, `interventions.jsonl` → `not reported`
    / `none recorded`, no exception).
  - `test_committed_baseline_run_renders` — against a **copy** in `tmp_path` of whichever baseline run is committed
    at the time: B's T8 deletes `runs/20260828T141646Z_baseline` (Windows paths, no `source_manifest.json`) and
    commits a regenerated `runs/<new_id>/` in its place, so glob `runs/*/summary.json` rather than hard-coding the
    id, and ask B for the new id when T8 lands. Its records use the ladder shape (`controller.py:95-110`:
    `hypothesis`, `configuration`, `code_diff`, `reflection`; no `proposal`/`manifest`) and it lacks `results.json`,
    `resources.json`, `passes/` and `research_memory.jsonl` — B stages only the five JSON/JSONL files. Both reports
    must render, naming the three rungs and the run's own `best.metrics.primary` read from `best.json` (0.6014695 in
    the old run; ≈ 0.6016 ± 0.0008 in B's regenerated one) — assert against the file, never a literal.
- **Acceptance criteria:**
  - [ ] `pytest -q -W error tests/test_report.py` green, `OPENAI_API_KEY` unset; full suite still green.
  - [ ] `python -m src.agent.report <copy of the committed baseline run>` exits 0, writes both files, a second render
        is byte-identical (`cmp`), and `git status --porcelain runs/` is empty.
- **Depends on / blocks:** consumes A's `converged_official` / `interventions.jsonl` (I-9, I-8) and B's
  `failure_class` (I-3) and real `summary["gate"]` — none blocks you; re-check against real values once they land.

### T3 · I17 share · Repo hygiene   (effort S, ~0.75 h)

- **Why:** the dataset is tracked — 9 files, ~240 MB including the tar.gz (`git ls-files data/`) — against
  `.gitignore:13` and `AGENTS.md:170`, as are `.DS_Store` and the kit's `__pycache__/evaluate.cpython-311.pyc`; and
  `.gitignore:16-19` hides exactly the artefacts judges need for the final run (spec-compliance §11, §15).
- **Where:** `.gitignore` (19 lines: `data/` :13, `artifacts/` :15, `runs/*/artifacts/` :16, `runs/*/stdout/` :17,
  `*.npz` :18, `generated_experiments/` :19), new `scripts/download_data.sh`.
- **Do** (steps 1-2 are safe and land as their own PR, so A gets the exception early; step 3 is disruptive):
  1. Open the final-run hole. Git cannot re-include a path under an excluded *directory*, so line 19 becomes a child
     glob:
     ```
     runs/*/stdout/
     !runs/final/stdout/
     generated_experiments/*
     !generated_experiments/final/
     runs/*_research/
     ```
     Keep `artifacts/` and `*.npz` ignored — checkpoints stay out (`AGENTS.md:170`); also ignore `.DS_Store`. The last
     line is A's ask (rule 5): every research run directory, personal ones included
     (`runs/<initials>_<timestamp>_research/`, A's `run_id_prefix`), stays out of git; `runs/*_baseline/` is
     deliberately *not* matched, so B can `git add` the five files of the regenerated baseline run (their T8) without
     `-f`. Comment both routes for the final run: this `final/` convention, and
     `git add -f runs/<id> runs/<id>/stdout generated_experiments/<id>`.
  2. `scripts/download_data.sh` — POSIX `sh`, `set -eu`, `chmod +x`; skip if the training log already exists under
     `data/KuaiRand-Pure/data`; else fetch `https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz`
     (`curl -fL -o`, falling back to `wget`), verify md5 `0820331067a3784d9691136f772b35a7` (`md5 -q` on macOS,
     `md5sum` on Linux — both branches), extract into `data/`, delete the archive (`docs/kuairand.md:54-66`).
  3. **Untrack, with coordination.** Post in the team channel first: *"Untracking `data/` in ~30 min. `git rm
     --cached` keeps my local copy, but your next `git pull` removes `data/` from your working tree because the
     commit deletes the path — `mv data ../data-backup`, pull, `mv ../data-backup data`, or run
     `bash scripts/download_data.sh`."* Then:
     ```sh
     git rm -r --cached data/
     git rm --cached .DS_Store
     git rm --cached kuairand-starter-kit/__pycache__/evaluate.cpython-311.pyc
     ```
     `--cached` only — never plain `git rm`. This does not shrink `.git` (the blobs stay in history; a rewrite would
     invalidate everyone's clone) — note that as a known limitation in the PR.
- **Interface:** none; this gives A the mechanism for the final-run commit (rule 5). **Tests:** none.
- **Acceptance criteria:**
  - [ ] `git ls-files data/ | wc -l` → 0; `git ls-files | grep -cE 'DS_Store|\.pyc$'` → 0;
        `ls data/KuaiRand-Pure/data | wc -l` → 6 (your working copy survived).
  - [ ] `git check-ignore -v` exits 1 for `runs/final/stdout/x.log`, `generated_experiments/final/003/candidate.py`
        and `runs/20260829T093000Z_baseline/summary.json` (B's run stays committable); exits 0 for
        `runs/other/stdout/x.log`, `generated_experiments/20260829T/003/candidate.py`,
        `runs/final/artifacts/model.npz`, `runs/kj_20260829T093000Z_research/state.json`.
  - [ ] `sh -n scripts/download_data.sh` clean; run with data present it prints "already present" and exits 0.
- **Depends on / blocks:** step 3 needs the team told first; A needs step 1 before the Day 3 final-run commit.

## 5. Definition of done (whole plan)

- [ ] `pytest -q -W error` green with `OPENAI_API_KEY` unset (47 + your new tests); `tests/test_interfaces.py`
      untouched and still passing.
- [ ] Every acceptance box in T1, T2, T3 ticked; three PRs merged, each ≤ 300 lines.
- [ ] A confirms `<run_dir>/DATA_CARD.md` is written and `RunState.data_card_path` set on a real run; C confirms the
      card sits in the stable prompt prefix; `journal.md` and `results.md` are in the final run directory, with its
      `stdout/` and generated code in git.

## 6. Hand-offs

- **You provide:**
  - **A** — `render_data_card(data_dir) -> str` (I-4) to wire at run start; `render_reports(run_dir)` (I-5), already
    called at `research_controller.py:459` — keep it after `results.json`; and, from T3 step 1, the `.gitignore`
    lines A asked for: `runs/*_research/` so personal run directories stay out of git (rule 5), the `final/`
    re-inclusions, and the `git add -f` recipe for a final run whose id is not literally `final`. Tell A when T1
    merges and again when T3 step 1 merges.
  - **C** — the same string for the `{data_card}` prompt slot, reached through `RunState.data_card_path` (A writes
    `<run_dir>/DATA_CARD.md` and sets the field; C reads the field in `roles.py`), deterministic (no timestamp, no
    path, sorted keys) so prompt caching works, and free of the generated-source blocklist vocabulary.
  - **E** — `tests/test_datacard.py` reads `safety.FORBIDDEN_TEXT` dynamically, so adding `log_standard` /
    `KuaiRand` / `.csv` under C2 fails D's test if the card ever names a file. Intentional: D fixes the card.
- **You consume:** A's `converged_official` / `converged_official_iteration` and `interventions.jsonl` (I-9, I-8);
  B's `failure_class` (I-3), their new `ExperimentOutcome.test_scores_path` (I-2 — it rides `outcome.to_dict()` into
  `iterations.jsonl`, so link it from the journal), and real gate statuses (I-1); B's regenerated baseline run id
  (C5) for `test_committed_baseline_run_renders`. If unready, every field is `.get()`-guarded and the fixture
  supplies all of them.
- **Notes from the Step 0 review assigned to you:** none — the hand-off table lists A, B and C only. The stubs you
  inherit are pinned by `tests/test_interfaces.py` (`test_data_card_stub_renders_nothing_yet`,
  `test_report_stub_returns_none`); T1 step 2 and T2 step 1 keep both true without the sanctioned ≤5-line edit.

## 7. Rules

1. **One owner per file.** Need a change in someone else's file? Ask the owner, or send them a ≤20-line PR they merge.
2. **New tests in new files.** Never edit another owner's test file.
3. **Rebase on `main` twice a day; PRs ≤ 300 lines; `pytest` green before merge.** Small, frequent merges beat one big one.
4. **Config: add files, don't edit shared ones** (`configs/offline_smoke.json`, `configs/features_run.json`).
5. **Run directories are personal** (`runs/<initials>_…`, gitignored) until the final run, which A commits.
6. **Shared surfaces only move in the freeze PR** — `types.py`, `contracts.py`, `configs/ranking_losses.json`. A second
   freeze PR (by A) is the way to change them again, not five drive-by edits.

Plus: never run `git add -A`; never commit `runs/` except the final run (A only); never commit `.env`; PR ≤ 300 lines;
rebase on `main` twice a day.

## 8. Daily checkpoints

**Day 1 end** — T1 merged: the card renders in < 60 s, ≤ 200 lines and hits every reference number, so A can wire
I-4 and C the prompt slot. T3 steps 1-2 pushed (the safe half).

**Day 2 end** — T2 merged: both reports render for the fixture run and for a copy of the committed baseline run; the
untracking commit (T3 step 3) is in and the team told. Re-render against A's first live run and report what breaks.

**Day 3** — re-render the final run (`python -m src.agent.report runs/<final>`), check the journal reads as prose a
judge can follow, hand A the `git add -f` command, confirm both reports are in the final commit.

