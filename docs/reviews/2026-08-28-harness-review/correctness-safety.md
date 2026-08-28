# Code review — agent harness (correctness & safety)

**Range:** `8239f90..424238f` (merge of `px-agent-harness`, 50 files, +4255 lines)
**Reviewer scope:** hidden-test isolation, metric fidelity, split fidelity, loop/convergence,
LLM layer, safety modules, error handling, tests, committed run, docs.
**Method:** every file in the range read in full; claims below verified by execution where noted
(test suite run offline on Python 3.9.6; kit-vs-harness data diff run against the real
`data/KuaiRand-Pure`). No repo files, index, HEAD or branch state were modified.

---

## Strengths

These are verified, not assumed.

1. **Hidden-test isolation is designed in, not bolted on.** `src/evaluation/official.py:47-57` is the
   only data-loading function in `src/`, and it filters on `date` *before* touching the label:
   ```python
   date = int(row["date"])
   if TRAIN_START <= date <= TRAIN_END:      split = "train"
   elif VALID_START <= date <= VALID_END:    split = "valid"
   else:
       # Crucially skip before reading the relevance label.
       continue
   ```
   `TEST_START/TEST_END` constants do not exist. `grep -rn "20220429\|20220508\|log_random\|'test'"
   src configs tests runs research` returns **nothing**. Nothing in the loop — proposer, roles,
   policy, candidate runner, audit — can reach test rows through the intended path.

2. **Split and row-order fidelity is exact.** I diffed `load_train_valid()` against the kit's
   `data.load()` on the real dataset:
   ```
   kit sizes: {'train': 1141112, 'valid': 124909, 'test': 170588}
   our sizes: {'train': 1141112, 'valid': 124909}
   train identical row-for-row: True
   valid identical row-for-row: True
   ```
   Feeding the two-split dict to the kit's `encode()` also reproduces the kit's encoding bit-for-bit
   (`field dim 40260 == 40260`; valid `X`, `y`, `users` all `array_equal`/`==` True), because the kit
   derives vocabs and duration-bucket edges from `train` only. Row order → `row_id` semantics is
   therefore preserved for whoever writes the submission later (see C1).

3. **Metric fidelity is total — the evaluator is not reimplemented.** `src/evaluation/official.py:18-30`
   imports the untouched kit modules by path and `official_evaluate` (`:72-75`) is a one-line
   delegation: `evaluate_module.evaluate(user_ids, labels, scores)`. Tie handling, GAUC weighting,
   zero-positive-user inclusion and `k=5` are the kit's, by construction. Stability of ties by row
   order is preserved because users/labels/scores are passed in `data.load()` order.

4. **Convergence maths is correct.** `ConvergenceTracker.observe` (`src/agent/convergence.py:14-29`)
   is equivalent to the organizers' rule, despite looking different. `meaningful_best` only ever
   ratchets up (assigned only when `score > meaningful_best + ε`), so `mb ≤ best ≤ mb + ε` always;
   three consecutive non-meaningful observations therefore imply `best_k − best_{k−3} ≤ ε`, and
   conversely a flat `best` over 3 iterations implies three non-meaningful observations. `patience=3`,
   `epsilon=0.002` match. The first observation correctly seeds rather than counts. Only successful
   iterations are observed (`controller.py:113-124`), matching `count_failed_as_nonimproving=False`.

5. **Candidates never receive validation labels.** `CandidateContext`
   (`src/experiments/contracts.py:10-17`) exposes `train_y` but only a callback for validation —
   `evaluate_validation`. `valid_y` stays in the trusted worker. Candidate-reported metrics are
   discarded: `RESERVED_DIAGNOSTIC_KEYS` (`run_candidate.py:17,86-90`) strips `primary`/`gauc`/
   `ndcg` from diagnostics, and the worker recomputes metrics itself (`:68`). This is tested
   (`tests/test_candidate_output.py:14-29`).

6. **Baseline ladder reproduces the published numbers.** The committed run reports random 0.4827
   (published 0.4834, mean over seeds 0–4), popularity **0.5807219** (published 0.5807, exact), FM
   **0.60147** (published 0.6016, Δ −0.00013). `src/models/baselines.py:50-115` mirrors the kit's
   `run_fm` including the `best + 1e-5` early-stop margin and `patience=4`, and improves on it by
   raising instead of crashing when `best_state is None` (`:105-106`).

7. **Trusted-worker architecture.** Generated code runs in a subprocess (`candidate_runner.py:99-126`)
   with timeouts, output truncation, atomic result writes (`run_candidate.py:36-39`), finiteness and
   shape validation, and a 50M-element checkpoint cap (`:80-81`). Defense in depth: `validate_source`
   runs both at write time (`candidate_runner.py:33`) and again inside the worker before
   `exec_module` (`run_candidate.py:44`).

8. **The run ledger covers most of Starter-Kit §5.** `iterations.jsonl` carries iteration, parent,
   hypothesis, configuration, code revision, metrics, error/recovery, `manual_intervention`, tokens;
   research runs add `passes/`, `research_memory.jsonl`, `resources.json`, `experiment_tree.json`.
   `AGENTS.md` is a genuinely good guardrail document.

---

## Issues

### Critical (Must Fix)

**C1. There is no submission path, no test scoring, and no Gate — the primary deliverable cannot be produced.**
`grep -rn "submission\|row_id\|write_submission" src configs tests` returns **zero hits** (only
`AGENTS.md:149-162` describes the format, as prose). Design spec §12 (`gate.py`) is unimplemented.
Concretely:
- Nothing loads the test split, so no test features exist to score.
- `CandidateContext` (`contracts.py:10-17`) has no test fields, so even re-running a candidate is impossible.
- The saved checkpoint is an opaque `dict[str, np.ndarray]` chosen by the LLM
  (`run_candidate.py:70-84`); nothing maps it back to a scoring function. `FMRanker.load_state_dict`
  (`fm_core.py:86-89`) expects `V/W/b`, but nothing enforces those keys on candidates.
- *Why it matters:* Technical Execution (35%) is scored on `score_agent − score_baseline` on the
  hidden test set, computed from the submitted CSV. Right now the run ends with a validation number
  and an unusable `model.npz`. This is the single largest gap.
- *Fix:* add `src/evaluation/official.py::load_test_meta()` (row_id/user_id/video_id only, no label
  columns), extend `CandidateContext` with `test_x`, require candidates to return `test_scores`, and
  add a `gate.py` that writes `submission.csv` with `%.9g` scores and shells out to
  `kuairand-starter-kit/submit.py --check --split test`. Keep it a *separate entry point* run once at
  the end, exactly as spec §12 says — do not put it in the loop.

**C2. `validate_source` is bypassable via `__builtins__`, which re-opens the leakage channel.**
`src/agent/safety.py:105-111` inspects only `ast.Call` with `func` of type `Name` or `Attribute`, and
`ast.Attribute` nodes whose `attr` starts with `__`. `__builtins__` is an `ast.Name`, and
`__builtins__['open']` is an `ast.Subscript`. Verified by execution:
```
ACCEPTED  builtins_open      # __builtins__['open']('data/KuaiRand-Pure/data/log_standard_4_22_to_5_08_pure.csv')
ACCEPTED  builtins_import    # __builtins__['__import__']('os')
ACCEPTED  np_load_via_alias  # f = np.load ; f('x.npz')      (alias defeats FORBIDDEN_ATTRIBUTES)
ACCEPTED  literal_test_path  # 'data/KuaiRand-Pure/data/log_standard_4_22_to_5_08_pure.csv'
```
and confirmed at runtime through the real loader:
```
$ _load_candidate(evil.py)
module loaded OK; validate_source did NOT reject the __builtins__ subscript
type(__builtins__) inside candidate module: dict
open reachable: True | __import__ reachable: True
```
`FORBIDDEN_TEXT` (`safety.py:60-72`) guards `data/judge`, `test_truth`, `ground_truth`,
`kuairand-starter-kit` — but **not** `log_standard`, `KuaiRand`, or `data/`, so the raw CSV path is
not even a string-level tripwire. Combined with C3 this means LLM-written code can read test labels.
- *Why it matters:* the "no hidden-test access during development" rule is the one hard rule that
  keeps the ranking fair; a judge reading `safety.py` will find this in five minutes.
- *Fix:* (a) reject `ast.Name` whose `id` starts with `__` and any `ast.Subscript` on such a name;
  (b) reject *any* attribute access (not just calls) whose `attr` is in `FORBIDDEN_ATTRIBUTES`, so
  aliasing fails; (c) add `log_standard`, `log_random`, `KuaiRand`, `.csv`, `/data/` to
  `FORBIDDEN_TEXT`; (d) exec candidates with an explicit restricted `__builtins__` mapping rather
  than the default globals.

**C3. The candidate subprocess inherits the full parent environment, including `OPENAI_API_KEY`.**
`src/agent/candidate_runner.py:58-62`:
```python
def _environment(self) -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(self.repo_root) + ...
```
`load_project_environment()` (`llm.py:16-25`) has already pushed `.env` into `os.environ` in the
parent, so the API key is handed to every LLM-generated candidate and every LLM-generated unit test.
There is no OS sandbox (spec §7's `sandbox-exec` / MINIMAL env is unimplemented), no network deny, no
CWD restriction, and no memory/CPU cap. `AGENTS.md:172` says "Never put API keys ... in logs, or
prompts" — this violates the spirit of that rule.
- *Fix:* build a minimal env explicitly (`PATH`, `PYTHONPATH`, `HOME`/`TMPDIR` pointed at the
  workspace, `PYTHONDONTWRITEBYTECODE`, thread caps) and never copy `os.environ`. Pop every
  `OPENAI_*`/`ANTHROPIC_*` key. Set `cwd` to the candidate workspace, not `REPO_ROOT`.

**C4. Any single non-conforming LLM output terminates the entire autonomous run.**
`src/agent/research_controller.py:395-407` catches `Exception` from the whole
research→critic→build→execute block and **breaks** with `stop_reason = "controller_error"`. There is
no retry, no reprompt, no skip-and-continue. The set of exceptions reachable from one bad model
response is large:
- `policy.py:37,39,41,43,45,50,52,57,59,61,63` — `sanitize_parameters` raises on any out-of-grid
  value. `epochs` must be 1–40 and `patience` 1–6, and **neither bound appears in the method cards**
  (`research/methods/bpr.md:26-33`, `group_softmax.md:30-37` list only lr / batch / K / temperature),
  so a plausible `epochs=50` kills the run.
- `roles.py:111` (family mismatch), `:175` (builder changed family/hypothesis_id), `:207` (debugger
  set `preserve_hypothesis=false`).
- `types.py:110,113` (unsupported family/action), `types.py:9` (any missing field).
- `llm.py:276` — empty `output_text`. With `reasoning.effort: medium` and
  `max_output_tokens: 24000`, a response that spends its budget on reasoning returns empty text and
  is *not* retried (`response.status == "incomplete"` is never checked).
- *Why it matters:* Robustness is explicitly scored ("when a step fails … the agent can recover,
  retry, or route around it, and … long iterative runs neither crash, stall, nor diverge") and
  Autonomy is measured by manual interventions. A run that dies on iteration 1 and needs a human
  restart scores badly on both.
- *Fix:* classify the exception. Schema/policy violations → re-prompt the same role with the
  validation error appended (bounded, e.g. 2 attempts), then record the iteration as
  `PROPOSAL_FAILED` and `continue`. Only unrecoverable harness errors (disk, config) should break.
  Add a `consecutive_harness_errors` counter as the real circuit breaker.

**C5. The committed run was not produced by the committed code, and the research loop bootstraps off it.**
`runs/20260828T141646Z_baseline/` was added in `cb91b96`, but that same commit's
`src/models/baselines.py:114` already returned `checkpoint.relative_to(REPO_ROOT).as_posix()` and
`controller.py:103` already wrote a `code_revision` field. The committed artifacts contain neither:
```json
"artifact_path": "C:\\Users\\Admin\\OneDrive - Nanyang Technological University\\...\\model.npz"
```
and `iterations.jsonl` keys are `[code_diff, command_owner, configuration, experiment_id, hypothesis,
iteration, kind, llm_tokens, manual_intervention, outcome, parent_experiment, reflection]` — no
`code_revision`. `source_manifest.json` is absent from the run directory although
`controller.py:57-58` always writes it. So the run predates the code in every commit in this range.
Downstream, `_ensure_baseline` → `_latest_valid_baseline` (`research_controller.py:36-47`) picks the
**most recent by `st_mtime`** summary whose `best.experiment_id == "official_fm_seed0"` and
`primary >= 0.5996`, and adopts it wholesale — so a research run started today silently inherits this
stale summary and a `best_artifact_path` pointing at a nonexistent Windows path.
- *Why it matters:* "reproduce the official baseline" is Task Requirement 1, and the only evidence in
  the repo is unreproducible. It also leaks the author's real name/organisation into a public repo.
- *Fix:* regenerate the run with current code and re-commit; scrub the absolute paths. Make
  `_latest_valid_baseline` select by run-id/recorded `source_manifest.revision` rather than mtime,
  and verify the referenced `artifact_path` exists before adopting a summary.

---

### Important (Should Fix)

**I1. No test covers the single most important safety property.**
`grep -rn "load_train_valid\|data.load\|1141112\|124909\|170588\|starter_modules" tests/` → **no
hits**. Nothing asserts that test dates are excluded, that split sizes match, or that row order
equals `data.load()`. If someone "helpfully" adds a `test` key to `load_train_valid`, every test
still passes. Add: (a) `load_train_valid` returns exactly `{train, valid}` with 1,141,112 / 124,909
rows; (b) max `date` in the returned rows is `20220428`; (c) row-for-row equality with the kit for a
sampled prefix (a fast version of the check I ran).

**I2. `tests/test_official_evaluation.py` is 22 lines and tests one trivial case.**
`:9-17` asserts a 2-row single-user perfect ranking scores 1.0. None of the conventions that actually
diverge between implementations are covered: zero-positive users included in nDCG with 0; GAUC
excluding all-positive/all-negative users; positive-count weighting; ties broken by row order;
`k=5` truncation with >5 impressions. Since `official.py` delegates to the kit these can't currently
break — but they will the moment anyone adds a fast-path evaluator, which is a likely optimisation.

**I3. `test_mocked_loop_covers_both_families_and_persists_resume_state` does not test resume.**
`tests/test_research_loop.py:108-170` never constructs `ResearchLoop(..., resume_dir=...)`. The test
name asserts a behaviour it does not exercise. `PLAN.md:170` lists "Resume continues without
duplicating completed iterations" in the test plan; that test does not exist. Resume also has a real
gap: `audit.record_iteration()` appends to `iterations.jsonl` *before* `_save()`
(`research_controller.py:281-301`), so a crash between them replays the iteration and duplicates the
JSONL line.

**I4. Retry/backoff on the OpenAI call is too thin for a 6-hour unattended run.**
`llm.py:211` disables SDK retries (`max_retries=0`), `:269-272` allows 2 attempts with
`time.sleep(min(4.0, 0.5 * 2**retries))` → 0.5 s then 1.0 s, and ignores `Retry-After`. A 429 burst or
a 30-second 503 window ends the run via C4. Spec §9.1 asked for 5 attempts, 2 s → 60 s, honouring
`retry-after`. Also `_retryable` (`:214-221`) matches on `status_code` and class *name* strings —
brittle; import the SDK exception types.

**I5. The research config caps the run at 8 iterations, not 50.**
`configs/ranking_losses.json:23` `"max_iterations": 8`. The problem statement's cap is 50 iterations
/ 6 h, with convergence normally firing first. Worse, `max_iterations` is overloaded three ways in
`research_controller.py:357,360` and `:224` (training-attempt cap, candidate cap, and in-`_execute`
retry cap), so raising it to 50 changes three semantics at once. Split into
`max_iterations` / `max_training_attempts`, and default to 50.

**I6. `SearchPolicy.should_stop` adds non-official conjuncts to the convergence rule.**
`policy.py:98-99`: `coverage_complete(state) and stagnant >= patience and not pending_replications`.
A run that has genuinely converged by the official rule keeps burning iterations until both BPR and
group-softmax have a *successful* node. If one family repeatedly fails to build, the loop can never
report `converged` and will exit as `iteration_budget_reached`. That is defensible as a research
agenda but it is not the organizers' rule; report both ("official rule fired at iteration k;
harness continued for coverage") so the ledger stays honest.

**I7. Two independent implementations of the convergence rule.**
`convergence.py:14-29` (baseline ladder) and `policy.py:73-96` (research loop) duplicate the ratchet.
Only the former is tested (`tests/test_agent.py:12-24`), and its two test cases happen to be ones
where the ratchet and the reference rule agree, so they would not catch a divergence. Extract one
implementation, test it against the reference formula over randomised score sequences.

**I8. The agent's search space is narrower than the challenge asks for.**
`policy.py:36-37` hard-pins `k == 16`; `types.py:110` restricts `family` to `{bpr, group_softmax}`;
`safety.py:114-133` requires the candidate to call one of two trusted samplers; `ALLOWED_IMPORTS`
(`safety.py:8-18`) permits only numpy plus three project modules. So the agent can vary a loss inside
two pre-approved families over a ~20-point grid. Judging explicitly rewards changes "not just the
model architecture, but every upstream and downstream module" and "originality in drawing on
published methods". Feature engineering, multi-task heads, and blending — all listed in
`AGENTS.md:131-140` as priorities 3-5 — are unreachable. Consider allowing a `features` family with a
trusted feature-builder contract before widening anything else.

**I9. Wall-clock under-reporting.** `_ensure_baseline` runs at `research_controller.py:85`, but
`self.session_started` is only set at `:137`. The ~3.5 minutes of baseline reproduction (and any
retries) are excluded from `state.wall_clock_seconds`, which is what `resources.json` reports for
Feasibility scoring. Start the clock before the baseline gate.

**I10. Manual-intervention accounting is a hardcoded constant.** `manual_intervention: False` is
written unconditionally (`controller.py:107`, `research_controller.py:167,298`);
`summary["manual_interventions"]` is `0` (`controller.py:141`) or `state.manual_interventions`, which
nothing ever increments. `interventions.json` is written once as `[]` (`research_controller.py:105`)
and never appended to; spec §11's `labrat intervene` CLI does not exist. Deliverable §3 requires the
intervention count. Add a CLI/marker so a restart after a `controller_error` is recorded honestly —
otherwise the number is unfalsifiable rather than impressive.

**I11. `_ensure_baseline`'s reproduction gate is one-sided and loose.** `research_controller.py:53,60`
accept any `primary >= 0.6016 − 0.002 = 0.5996`. A run scoring 0.85 — which would indicate a leak or
a bug — passes silently. Spec §8 asked for `|primary − 0.6016| ≤ 0.003` plus sanity floor 0.47 and
sanity ceiling 0.80. The floor/ceiling checks are absent everywhere: `runner.py:69` and
`candidate_runner.py:143` only check finiteness, so a candidate returning `primary = 0.99` would be
promoted to best.

**I12. Silent skips in baseline discovery.** `research_controller.py:45-46` catches
`(OSError, ValueError, TypeError, JSONDecodeError)` and `continue`s with no log line, so a corrupt
`summary.json` looks identical to "no baseline exists" and triggers a surprise 3.5-minute retrain.
Log the skipped path and reason.

**I13. Control flow keyed on exception message text.** `research_controller.py:400`:
`if "token budget" in str(exc).lower()`. Both `roles.py:52,63` messages happen to contain that
phrase, so it works today and breaks the moment anyone rewords a string. Raise a dedicated
`TokenBudgetExceeded` exception.

---

### Minor (Nice to Have)

**M1.** `README.md:39,110` and `AGENTS.md:30-43` build a whole guardrail section around
`data/judge/**`, a directory that **does not exist in this repo** (`find . -name judge` → nothing).
The real hidden-test surface is the 20220429–20220508 date range inside
`log_standard_4_22_to_5_08_pure.csv`. `safety.py:61-62` guards `data/judge` but not `log_standard`
(see C2). This looks like carry-over from a different project and misdirects a reader — and a judge —
about where the actual risk lives.

**M2.** `README.md:73` gives `python -m unittest discover -s tests -v`; it errors with
`ModuleNotFoundError: No module named 'src'` unless run from the repo root with the root on
`sys.path`. `python -m unittest discover -s tests -t .` works. Also `README.md:29,46,72,83,104` use
PowerShell fencing and backtick line continuations — fine for the author's Windows box, but the
graders will likely be on macOS/Linux; add bash equivalents.

**M3.** Three tests error without optional deps installed
(`tests/test_openai_runtime.py:42,50,66` need `python-dotenv` and `openai`). None need network or an
API key — `FakeResponses` (`:13-38`) is used for the request test and
`inspect.signature(provider.client.responses.create)` for the schema check. Worth a `skipUnless` so
`unittest discover` is green on a bare checkout. Suite result here: **25 passed, 3 errored (missing
deps), 0 failed**.

**M4.** `runs/.../best.json:2` and `summary.json:3` embed
`C:\Users\Admin\OneDrive - Nanyang Technological University\...` — the author's account name and
institution, in a repo intended to be public. Scrub when regenerating for C5.

**M5.** `official.py:22-29` inserts the starter-kit dir at `sys.path[0]` and imports modules named
`data`, `evaluate`, `baseline`. I verified this resolves correctly today (the repo's `data/`
namespace package loses to the kit's regular `data.py`), but the names are maximally collision-prone.
`importlib.util.spec_from_file_location` with unique names would be safer. The `finally` block
(`:28-29`) also only pops `sys.path[0]` if it is still the starter dir, so an import side effect can
leave the entry in place permanently.

**M6.** `validate_family_contract` (`safety.py:114-133`) is enforced at write time
(`candidate_runner.py:34`) but **not** in the worker's `_load_candidate` (`run_candidate.py:44` calls
only `validate_source`). The two validation points should be symmetric.

**M7.** `run_agent`'s `for … else` (`controller.py:130-131`) reports
`stop_reason: "iteration_budget_reached"` whenever the loop runs to `range` exhaustion — which is
exactly what happens when the experiment list length equals `max_iterations`. The committed run says
`iteration_budget_reached` when the true reason is "ladder complete". Cosmetic but it misreads.

**M8.** `_execute` records `"Training-attempt budget reached before execution."` as a *node failure*
(`research_controller.py:224-226`) rather than a budget stop, inflating the failure count in
`results.json`.

**M9.** `ScriptedProvider.complete` (`llm.py:313`) does `payload.pop("_usage", …)`, mutating the
caller's dict. Harmless in the current tests but a trap for reused fixtures.

**M10.** `_json_safe` (`run_candidate.py:20-33`) caps arrays at 1,000 elements but places no cap on
`training_trace` list length, so a runaway candidate can write an arbitrarily large `result.json`.

**M11.** `official_evaluate` casts the kit's integer `users`/`rows` to `float`
(`official.py:75`), so `best.json` carries `"rows": 124909.0`. Cosmetic.

**M12.** Progress output is `print()` (`controller.py:126,147`, `baselines.py:90`,
`run_baseline.py:28`, `research_controller.py:448`) rather than `logging`. The structured JSONL
ledgers carry the substance, so this is low-impact, but a long unattended run benefits from
levelled, timestamped output.

**M13.** `.gitignore:16-17` excludes `runs/*/artifacts/` and `runs/*/stdout/`, so the stdout/stderr
paths recorded in `iterations.jsonl` point at files that are not in the repo. Deliverable §3 asks for
error/recovery events; consider committing a truncated stderr tail inside the JSONL record itself.

---

### Notes on the spec (not the implementation)

- Spec §5's sealed-zone/parquet Steward is a heavier design than what was built, and the
  implementation's "never parse test rows at all" is arguably *stronger* than "materialise a
  test parquet with outcome columns stripped" — there is no stripped copy to get wrong. I would keep
  the implementation's approach and amend the spec, adding only the test-side metadata
  (`row_id, user_id, video_id` + features) needed for C1.
- Spec §10 defines convergence as `best_k − best_{k−N} ≤ ε`; the implementation's ratchet is provably
  equivalent (see Strength 4). No change needed, but the spec should note the equivalence so a future
  reader does not "fix" a non-bug.
- Spec §9.1 describes an Anthropic client; the implementation uses OpenAI Responses. That is a
  deliberate, documented divergence (`PLAN.md:21-34`) and fine — but §9.1's retry/streaming/caching
  requirements were dropped along with it (see I4), and those were provider-independent.

---

## Recommendations

Ordered by return on effort for the hackathon score.

1. **Build the Gate (C1).** Nothing else in the repo matters to the Primary metric until a
   `submission.csv` exists and passes `submit.py --check --split test`. Keep it a single, one-shot,
   end-of-run code path that is the *only* place test data is touched, and assert `gate_done` so it
   cannot run twice. This also forces the checkpoint format question to be answered.
2. **Make the loop unkillable (C4).** Wrap each role call in a bounded re-prompt-on-validation-error
   helper and downgrade proposal failures to `continue`. This is maybe 60 lines and it converts the
   two most heavily weighted soft criteria (Robustness, Autonomy) from "fragile" to "demonstrated".
   Then deliberately inject a bad LLM response in a test and assert the run survives.
3. **Close the `__builtins__` hole and stop leaking the API key (C2, C3).** Both are small,
   mechanical fixes with outsized credibility value — a reviewer who finds them stops trusting the
   isolation story, which is otherwise excellent.
4. **Regenerate and re-commit the baseline run with current code (C5), and add the isolation
   tests (I1).** The reproduction claim needs evidence produced by the code being judged.
5. **Widen the search space (I8) only after 1–4.** Two loss families over a fixed grid will plateau
   quickly; a `features` family with its own trusted contract is the highest-value second axis, and
   it is where the dataset actually has headroom (baseline 0.5946 vs 0.8645 ceiling).
6. **Small ledger honesty items:** start the wall clock before the baseline gate (I9), add a real
   intervention counter (I10), and report the official convergence verdict separately from the
   harness's coverage-gated stop (I6).

---

## Assessment

**Ready to build on? — With fixes.**

The foundation is genuinely sound where it is hardest to get right: hidden-test isolation is
structural rather than policy-based, the evaluator is the organizers' own file rather than a
reimplementation, and I verified row-for-row equality with `data.load()` on the real dataset — so the
parts that would silently invalidate every number are correct. What is missing is the other half of
the pipeline: there is no submission/test-scoring path at all (C1), the loop dies on the first
malformed LLM response (C4), and the AST sandbox has a demonstrable `__builtins__` bypass that
re-opens the leakage channel it exists to close (C2). Fix those four Criticals and this becomes a
credible Track 2 submission; ship it as-is and it produces a validation number with nothing to submit.
