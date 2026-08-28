# Review of the agent harness (`8239f90..424238f`) against the design spec

Date: 2026-08-28 · Reviewed: `src/`, `tests/`, `configs/`, `research/`, `runs/20260828T141646Z_baseline/`, README/AGENTS/PLAN
Against: `docs/superpowers/specs/2026-08-28-autonomous-mle-agent-design.md` and, above it, `docs/problem_statement.md`
Method: three independent read-only reviews — spec compliance, correctness & safety, and an empirical run of the
test suite and baseline in an isolated worktree. The three full reports with `file:line` evidence sit beside this file:
`spec-compliance.md` (347 lines), `correctness-safety.md` (410 lines), `empirical.md`.

## Verdict

**Build on this harness. Do not rebuild to the spec.** The parts that are expensive to get right are right:

- The metric is the organizers' own `evaluate.py`, imported by path (`src/evaluation/official.py:18-30, 72-75`), never copied.
- The split loader was verified **row-for-row identical** to the kit's `data.load()` — 1,141,112 / 124,909 rows, identical
  encoding, dim 40260.
- Test-period rows are skipped **before** the label is read (`official.py:50-57`) — stronger than the spec's own rule.
- Candidates cannot self-report metrics (`run_candidate.py:86-90`, tested), must call trusted samplers, and pass an AST allowlist.
- Token and wall-clock accounting, source-gated resume, an experiment tree, seed-1/2 replication of any >ε gain, a Critic
  preflight that rejects weak hypotheses before code tokens are spent, and `research/methods/*.md` method cards — all real,
  and several are better than what the spec described.
- Empirically: **28/28 tests pass offline in 2 s**; `python -m src.agent.controller --config configs/baseline.json`
  reproduces the published baseline within seed noise (primary **0.60147** vs 0.6016; GAUC 0.66713; nDCG@5 0.53581) in
  **21 s** on this Mac; the fresh run is bit-identical to the committed one; no test metric appears in any artifact.

What is missing is additive, not architectural — but two of the gaps are worth more than half the rubric.

## Scorecard against the judging rubric, as the code stands today

| Criterion | Weight | State today | Blocking issue |
|---|---|---|---|
| Technical — hidden-test delta | ~25% | **No artifact.** Nothing writes `submission.csv`, loads test rows, or produces test predictions. | C1 |
| Technical — robustness | ~10% | Loop dies on the first non-conforming LLM output (`break` on any exception). | C4 |
| Innovation | 20% | Real code generation, but for one stage only (loss), two families, ~12-point grid, no data facts in the prompt. | I8, I15 |
| Autonomy | 20% | `manual_intervention: False` is hardcoded; final prediction is documented as a manual step. | I10, C1 |
| Feasibility | 15% | Accounting is good; wall-clock excludes the baseline gate; no smoke mode; 8-iteration cap. | I9, I5 |

## Must fix (Critical)

**C1 — No submission path, no test scoring, no Gate.** `grep -rn "submission\|row_id" src configs tests` → 0 hits.
`CandidateContext` (`src/experiments/contracts.py:10-17`) has no test fields; candidates return `validation_scores` only;
the saved `model.npz` is an opaque dict nothing can re-score. *Fix:* add `load_test_meta()` to `official.py`
(`row_id, user_id, video_id` + features, never label columns), add `test_x` to the context, require `test_scores` from
candidates, persist them per node, and add a one-shot `gate.py`: write `submission.csv` (`row_id,user_id,video_id,score`,
`%.9g`), run `kuairand-starter-kit/submit.py --check --split test` (the kit's README asks teams to run `--check`
themselves), refuse to run twice. The results table's delta is the **validation** delta vs 0.6016 — that is the number the
deliverables ask for and it touches no test data. A self-computed hidden-test score is optional: the kit frames `--score` as
"available locally for valid", so if you compute it at all, do it once, only after `submission.csv` is frozen, and label it
self-computed.

**C2 — The AST validator is bypassable.** `src/agent/safety.py:105-111` inspects `ast.Call`/`ast.Attribute` only;
`__builtins__['open']` is an `ast.Subscript` and was **verified accepted and reachable at runtime** (`open` and `__import__`
callable inside a loaded candidate). `FORBIDDEN_TEXT` (`safety.py:60-72`) does not include `log_standard`, `KuaiRand`, or
`data/`. Aliasing (`f = np.load; f(...)`) also defeats `FORBIDDEN_ATTRIBUTES`. *Fix:* reject any `ast.Name` starting with
`__` and any `Subscript` on it; reject forbidden attribute *access*, not just calls; add the dataset paths to the text
blocklist; exec candidates with an explicit restricted `__builtins__`.

**C3 — Generated code inherits the API key.** `src/agent/candidate_runner.py:58-62` copies `os.environ` (into which
`.env` was already loaded) into every LLM-written candidate and test subprocess. No cwd restriction, no network deny.
*Fix:* build a minimal env (`PATH`, `PYTHONPATH`, `HOME`/`TMPDIR` → workspace, `PYTHONDONTWRITEBYTECODE`), pop every
`OPENAI_*`/`ANTHROPIC_*`, set `cwd` to the candidate workspace.

**C4 — One bad LLM response ends the whole run.** `src/agent/research_controller.py:395-407` catches `Exception` and
`break`s with `stop_reason="controller_error"`. Reachable from: any off-grid parameter (`policy.py:37-63` — `epochs`/`patience`
bounds are not even in the method cards), family/hypothesis mismatches (`roles.py:111, 175, 207`), missing schema fields
(`types.py:9`), and an empty `output_text` when reasoning eats the token budget (`llm.py:276`; `response.status ==
"incomplete"` is never checked). *Fix:* classify exceptions; re-prompt the same role with the validation error (≤2 attempts);
record `PROPOSAL_FAILED` and `continue`; keep `break` only for disk/config errors behind a `consecutive_harness_errors`
breaker. Then add a test that injects a malformed response and asserts the run survives.

**C5 — The committed "verified baseline" run was not produced by the committed code.** `runs/20260828T141646Z_baseline/`
lacks `code_revision` and `source_manifest.json`, which the code at that same commit writes unconditionally, and embeds
`C:\Users\Admin\OneDrive - Nanyang Technological University\...` (a real name and organisation, in a repo that will be
public). `_latest_valid_baseline` (`research_controller.py:36-47`) adopts it **by mtime**, so a research run today inherits a
dangling artifact path. *Fix:* regenerate with current code, scrub absolute paths, select baselines by recorded revision
and verify the artifact exists.

## Should fix (Important), in suggested order

1. **I5 — `max_iterations` is 8, not 50** (`configs/ranking_losses.json:23`), and the same knob is overloaded three ways in
   `research_controller.py:224, 357, 360`. The organizers score the *converged* result; at 8 the run ends on budget every time.
   Split the knobs and default to 50.
2. **I6 — Non-official conjuncts in the stop rule.** `policy.py:98-99` also requires `coverage_complete` (both families
   succeeded) and no pending replications. Report the official rule's verdict separately from the harness's stop.
3. **I10 — Intervention count is a constant.** `manual_intervention: False` hardcoded (`controller.py:107`,
   `research_controller.py:167, 298`); `interventions.json` written once as `[]`. Add an `intervene --reason` command that
   appends to it and increments the counter. Autonomy is 20% and this number is its only evidence.
4. **I9 — Wall-clock under-reported.** The baseline gate runs before `session_started` is set (`research_controller.py:85`
   vs `:137`), so ~20 s–3.5 min of work is excluded from `resources.json`.
5. **I11 — Baseline gate is one-sided, no sanity bounds.** `research_controller.py:53, 60` accepts any primary ≥ 0.5996;
   `runner.py:69` / `candidate_runner.py:143` check finiteness only. A leaked 0.99 would be promoted to best. Add
   `|primary − 0.6016| ≤ 0.003` for the gate, floor 0.47 and ceiling 0.80 for candidates.
6. **I4 — Retries too thin for a 6-hour run.** `llm.py:211` sets `max_retries=0`; own loop allows 2 attempts, 0.5 s → 1 s,
   ignores `Retry-After`, matches exceptions by class-name string (`:214-221`). Use the SDK's typed exceptions, 5 attempts,
   2 s → 60 s.
7. **I1 — No test guards the isolation property.** Nothing in `tests/` asserts split sizes, max date 20220428, or
   equality with `data.load()`. **I2** — `test_official_evaluation.py` is one trivial 2-row case. **I3** — the "persists
   resume state" test never constructs a resumed loop; and `record_iteration()` writes `iterations.jsonl` before `_save()`
   (`research_controller.py:281-301`), so a crash in between duplicates a line on resume.
8. **I14 — No offline mode from the CLI.** `ScriptedProvider` exists (`llm.py:301-322`) but no flag reaches it; the loop
   raises at `llm.py:198` without a key. The real generate→train→evaluate path (`run_candidate.py`) has zero end-to-end
   coverage — the loop test fakes both the LLM and the executor. Add `--provider scripted` and one e2e test through the
   real subprocess.
9. **I15 — Nothing tells the Researcher about the data.** Prompts carry prior metrics only (`roles.py:66-87`); three of six
   CSVs (`user_features`, `video_features_statistic`, `log_random`) are never opened (`official.py:36-46`). Add a computed
   data card (label rates by tab, duration buckets, drift, duplicates, leakage flag on `video_stat`).
10. **I8 — Search surface is one stage.** Trusted samplers mandatory (`safety.py:114-133`), `k == 16` pinned
    (`policy.py:36-37`), families `{bpr, group_softmax}` (`types.py:110`), imports limited to numpy + 3 project modules
    (`safety.py:8-18`), features frozen at the kit's 5 fields (`run_candidate.py:110-114`). Against
    `problem_statement.md:71-72, 116-118`. Widen **after** C4, and widen in the directions the kit itself says are
    untested (`README.en.md:150-175`): the two existing families are the kit's #1 pick (pairwise/listwise loss); next are
    **user behaviour sequences** (#2), **multi-objective** on `is_click`/`is_like`/`play_time_ms` (#3), **watch time** (#4),
    and **temporal/drift** (#6). So the second axis should be a `history_features` family — train-only user history
    statistics, author affinity, recency, tab crosses — with a trusted `build_features(rows) -> X` contract, plus a
    multi-task family. **Not** more static categorical fields and **not** a `k` sweep: the kit measured both as dead ends
    (`README.en.md:133-139`, 0.5940 vs 0.5950; k=8/16/32 flat — "the bottleneck is not features or capacity"), so keeping
    `k == 16` pinned is correct. "Unclamp the grid" means off-grid proposals must not be fatal (C4), not that `k` should vary.
11. **I16 — Judges can't read the run.** No rendered `journal.md`/`results.md`; per-iteration code is a sha256 in the
    record (`research_controller.py:291-292`) with full source only in gitignored `generated_experiments/` and
    `runs/*/stdout/` (`.gitignore:16-19`). Render a journal with a unified diff per iteration and commit the final run's
    logs in full.
12. **I17 — Repo hygiene.** The dataset is tracked in git (9 of 71 files, ~240 MB, including the tar.gz) despite
    `.gitignore` and AGENTS.md forbidding it — `git rm -r --cached data/` plus a download script. `.DS_Store` and a `.pyc`
    are tracked. README commands are PowerShell-only (`README.md:29, 46, 83`); README lacks limitations, contributions, the
    iteration/convergence definitions, and an architecture diagram; no `docs/devpost.md`.
13. **I7 / I12 / I13** — two independent convergence implementations (`convergence.py`, `policy.py`), only one tested;
    silent `continue` on corrupt summaries (`research_controller.py:45-46`); control flow keyed on exception message text
    (`:400`). Small, mechanical.

Minor items (13) are listed in `correctness-safety.md` §Minor.

## Things the harness does better than the spec — the spec will adopt these

| Their design | Spec section to amend |
|---|---|
| Skip test rows before reading the label; never materialise a test copy (only test *metadata* is needed for the Gate) | §5 |
| Candidates cannot self-report metrics; trusted worker computes everything | §8 |
| Mandatory trusted samplers for the invariant LLMs most often get wrong (same-user negatives) | §6 |
| Critic preflight: reject a hypothesis before spending code tokens; ledger the rejection | §9.2 |
| `research/methods/*.md` method cards (citation, closed-form gradient, safe search space, known failure modes) instead of a flat idea bank | Appendix A |
| Source-manifest-gated resume | §10 |
| Automatic seed-1/2 replication of any >ε gain (add the mean/std aggregation it lacks) | §8 |
| Experiment tree with `parent_experiment` / `replicated_from` | §10 |
| OpenAI Responses API behind an `LLMProvider` protocol — provider is a non-issue; keep the spec's retry/caching requirements, which are provider-independent | §9.1 |
| numpy-only, Python 3.9-compatible runtime — keep until a candidate family needs torch/LightGBM | §4 |

The spec's `sandbox-exec` layer is downgraded to optional; the AST allowlist plus a minimal env (C3) plus the C2 fixes give
most of the value.

## Suggested order of work

| # | Work | Effort | Rubric moved |
|---|---|---|---|
| 1 | Gate: test meta loader, `test_scores` contract, `submission.csv`, `submit.py --check`, one-shot test score (C1) | M | Technical 25% |
| 2 | Unkillable loop: classify → re-prompt → `continue`; breaker for harness errors; injected-failure test (C4) | S | Robustness, Autonomy |
| 3 | Close `__builtins__`/alias/path holes; minimal candidate env (C2, C3) | S | Credibility of isolation |
| 4 | Regenerate + re-commit the baseline run with current code; scrub paths; isolation + evaluator tests (C5, I1, I2) | S | Requirement 1 evidence |
| 5 | `max_iterations` 50 with split knobs; report official convergence verdict; wall-clock from run start; real `intervene` (I5, I6, I9, I10) | S | Feasibility, Autonomy |
| 6 | Sanity floor/ceiling + two-sided baseline gate; typed retries (I11, I4) | S | Robustness |
| 7 | `--provider scripted` CLI path + one e2e test through the real subprocess (I14) | S–M | Dev velocity |
| 8 | Rendered `journal.md`/`results.md` with per-iteration diffs; commit final run logs (I16) | M | Innovation, Robustness |
| 9 | Data card in the Researcher prompt; open the other three CSVs (I15) | M | Innovation |
| 10 | `features` family with a trusted contract, then unclamp the grid (I8) | L | Technical delta, Innovation |
| 11 | README (POSIX commands, limitations, contributions, definitions, diagram), `docs/devpost.md`, untrack `data/` (I17) | S | Deliverables |

Items 1–6 are small-to-medium and together cover roughly 55% of the rubric.

## Questions for the author

1. Is "final judge prediction generation remains a separate, explicit user-authorized step" (`README.md:110-111`,
   `AGENTS.md:39-41`) intentional? As written it is a manual intervention on the path to the scored deliverable.
2. Why `max_iterations: 8` and the `coverage_complete` requirement — a deliberate research agenda, or a dev-time setting?
3. Is the 150,000-token budget (`roles.py:51-52`) meant to hold for a 50-iteration run? Five `gpt-5.5` passes per iteration
   will exceed it well before convergence.
4. Are you open to a `features` family (trusted `build_features` contract) as the second search axis? It is where the
   dataset's headroom is (baseline 0.5946 vs oracle 0.8645).
5. Was committing `data/` intentional (README says to download it)?
