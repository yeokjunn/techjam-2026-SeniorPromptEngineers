# Spec-compliance review — teammate's agent harness (`8239f90..424238f`)

Reviewer: senior code review pass, read-only. No training run, no LLM call, no mutation of tree/index/HEAD.
Authorities: `docs/problem_statement.md` (wins) > `docs/superpowers/specs/2026-08-28-autonomous-mle-agent-design.md` (our spec).
Tiers: **M** = hackathon-mandated · **D** = our design choice · **C** = cosmetic.
Verified empirically: unit suite (25/28 pass; 3 errors are `openai`/`dotenv` not installed on this Mac, not defects)
and split fidelity (below).

---

## 1. What the harness is

Two loops share one `runs/<id>/` logging convention and one metric adapter.

- **Baseline loop** (`src/agent/controller.py:39-148`): deterministic, no LLM. A `ConfigProposer`
  (`src/agent/proposer.py:8-26`) walks a fixed list of three experiments from `configs/baseline.json:15-45`
  (random → popularity → official FM). Each runs in a subprocess (`src/agent/runner.py:19-104`) via
  `src/experiments/run_baseline.py`, is scored by the kit's evaluator, reflected on by a rule-based
  `reflect()` (`src/agent/reflector.py:6-33`), and logged to `iterations.jsonl` / `best.json` / `summary.json`.
- **Research loop** (`src/agent/research_controller.py:348-449`): the autonomous one. Roles are four
  *structured LLM passes* over one shared `RunState`, not separate agents (`PLAN.md:17`):
  **Researcher** proposes (`roles.py:89-136`) → **Critic-preflight** approves/rejects (`roles.py:138-150`)
  → **Builder** emits a full `candidate.py` + `test_candidate.py` as strings (`roles.py:152-177`) →
  deterministic AST validator (`safety.py:85-133`) → generated unit tests in a subprocess → training
  subprocess (`candidate_runner.py:83-175`) → trusted scorer computes metrics → **Critic-postflight**
  interprets (`roles.py:210-228`); **Debugger** repairs up to twice on any validation/runtime error
  (`roles.py:179-208`, `research_controller.py:172-203`). A deterministic `SearchPolicy`
  (`policy.py:67-99`) — not the LLM — promotes best-so-far, counts stagnation, and queues seed-1/2
  replications.
- **LLM**: OpenAI Responses API, `gpt-5.5`, medium reasoning, `store=false`, JSON-Schema strict
  structured outputs, `prompt_cache_key` per role, 2 bounded retries, optional `web_search` tool
  (`llm.py:193-298`; config `configs/ranking_losses.json:10-21`). `ScriptedProvider` (`llm.py:301-322`)
  is the offline fake.
- **A candidate experiment** = `CandidateManifest{candidate_id, hypothesis_id, family, code, tests,
  parameters}` (`types.py:153-174`), written to `generated_experiments/<run>/<iter>_<id>/`.
- **Logging**: `ResearchAudit` (`audit.py:11-61`) writes `state.json`, `experiment_tree.json`,
  `iterations.jsonl`, `research_memory.jsonl`, `resources.json`, `passes/NNN_<role>_<seq>.json`
  (full prompt + response + usage), `summary.json`, `results.json`, `baseline_gate.json`.

```
configs/baseline.json ─▶ [deterministic ladder: random → pop → official FM] ─▶ baseline_gate.json (≥0.5996)
                                                                                       │
      ┌────────────────────────────────────────────────────────────────────────────────┘
      ▼
  Researcher ──ResearchDecision──▶ Critic-pre ──▶ Builder ──code+tests──▶ AST validator ──▶ unittest subproc
   (LLM)          (sanitize_        (LLM)          (LLM)                  (safety.py)          │ fail
                   parameters                                                                  ▼
                   grid-clamps)                                              Debugger (LLM) ≤2 repairs
                                                                                               │ pass
   SearchPolicy ◀── Critic-post (LLM) ◀── trusted metrics ◀── kit evaluate.py ◀── train subprocess
   (best / stagnation / replication seeds 1,2)          │
      └──▶ stop: converged | iteration | wall-clock | token budget ──▶ summary.json + results.json
                                                   ⚠ NO submission.csv, NO test-split path, NO gate
```

---

## 2. Compliance matrix

### §2 Roles / architecture

| item | tier | verdict | evidence | action | note |
|---|---|---|---|---|---|
| LLM roles drive the loop | M | implemented | `roles.py:89-228`; `research_controller.py:381-394` | keep theirs | 5 passes (researcher, critic×2, builder, debugger) vs spec's 3. |
| Scientist / Engineer / Medic naming + split | D | diverges | `roles.py:28-39` | keep theirs | Researcher/Critic/Builder/Debugger is equivalent; the added Critic preflight is a genuine improvement (blocks bad hypotheses before tokens are spent on code). |
| Deterministic non-LLM everything else | D | implemented | `policy.py:67-99`, `convergence.py:14-29` | keep theirs | Search policy is deterministic, as spec wanted. |
| Fake LLM for zero-cost E2E | D | implemented | `llm.py:301-322`; `tests/test_research_loop.py:107-227` | keep theirs | Equivalent to spec's `FakeLLM`. |
| Loop covers all 5 MLE stages | M | **missing** | `run_candidate.py:110-132` hands candidates a *fixed* kit encoding; `safety.py:8-18` forbids all other libs | adopt spec | Only *train* is agent-controlled. Inspect / feature-engineering / evaluate stages are not reachable. See §3(a). |

### §5 Data & sealing

| item | tier | verdict | evidence | action | note |
|---|---|---|---|---|---|
| Official splits + row order | M | implemented | `official.py:33-69` | keep theirs | **Verified**: `load_train_valid` output is `==` to kit `data.load()` for train (1,141,112) and valid (124,909), and `encode()` gives an identical valid `X` and dim 40260. |
| Test rows never materialised | M | implemented (stronger than spec) | `official.py:50-57` — `continue` fires *before* `row['long_view']` is read | keep theirs | Better than the spec's "strip columns" approach: the label is never touched in the process. |
| Sealed zone readable only by harness | D | partial | no `sealed`/`harness/` concept; `safety.py:60-72` bans `data/judge`, `kuairand-starter-kit` in generated source | merge both | The banned `data/judge/**` path no longer exists (deleted in `8239f90`); the guard is stale but harmless. |
| Data card / EDA artifact | D | **missing** | grep: no `DATA_CARD`, no datacard module | adopt spec | Nothing tells the Researcher anything about the data — prompts carry only prior metrics (`roles.py:66-87`). Direct hit on Innovation (20%). |
| `user_features`, `video_stat`, `log_random` handled | M/D | **missing** | `official.py:36-46` reads only `video_features_basic_pure.csv` + the two standard logs | adopt spec | Three of six CSVs are never opened. |
| Workspace parquet | D | diverges | in-memory tuples + kit `encode()` | keep theirs | Avoids a pandas/pyarrow dependency; runs on `/usr/bin/python3` 3.9. Simpler and adequate at this scale. |

### §6 Pipeline contract + leak check

| item | tier | verdict | evidence | action | note |
|---|---|---|---|---|---|
| Single-file `pipeline.py` with 4 stage fns | D | diverges | `contracts.py:9-25`; `run(context, parameters) -> CandidateOutput` | merge both | Their contract is narrower (one function, no I/O, trusted context) which makes candidates easier to validate — but it is *why* only one stage is agent-writable. Keep the safety idea, widen the surface. |
| Row-ordered score output, length-checked | M | implemented | `run_candidate.py:63-67`, `116-122` | keep theirs | Shape + finiteness enforced by the trusted worker, not the candidate. |
| Candidate cannot supply its own metrics | M | implemented | `run_candidate.py:68`, `86-90` strips reserved diagnostic keys; test at `tests/test_candidate_output.py:14-29` | keep theirs | Excellent; the spec did not think of the fake-metric attack. |
| Static leak/AST check | D | implemented | `safety.py:85-111` (import allowlist, forbidden calls/attrs, dunder ban), `:114-133` (must call trusted sampler) | keep theirs | Stricter than the spec's blacklist. But it is text+AST on the *whole* source: `FORBIDDEN_TEXT` (`safety.py:60-72`) substring-matches `subprocess`/`requests`/`urllib`, and `FORBIDDEN_ATTRIBUTES` bans `.replace`/`.load` — false positives will burn Debugger repairs. |
| `--smoke` fast path | D | **missing** | grep: no smoke mode | merge both | Every candidate is a full run bounded only by `experiment_timeout_seconds: 900`. A smoke gate would cut wall-clock (15% of score). |
| Determinism from seed | M | implemented | `policy.py:29`, seeds threaded into parameters; replication seeds `policy.py:90-96` | keep theirs | |

### §7 Sandbox

| item | tier | verdict | evidence | action | note |
|---|---|---|---|---|---|
| Subprocess isolation + timeout | D | implemented | `candidate_runner.py:117-166` | keep theirs | |
| Minimal env, no API keys leaked in | D | **diverges (defect)** | `candidate_runner.py:58-62`: `environment = dict(os.environ)` — the candidate process inherits `OPENAI_API_KEY` | adopt spec | LLM-written code runs with the project's API key in its environment. Cheap fix. |
| Process-group kill | D | partial | `subprocess.run(..., timeout=)` kills only the direct child | merge both | Sub-process spawning is AST-blocked, so exposure is low. |
| macOS `sandbox-exec` layer | D | **missing** | grep: none | drop from spec | The import allowlist + no-`open` rule gives most of the value at a fraction of the cost. Their choice is defensible. |
| Failure classification (TIMEOUT/CRASH/BAD_OUTPUT/LEAK) | D | partial | `candidate_runner.py:130-175` distinguishes non-zero exit / timeout / invalid result, all as `status="failed"` with a distinct `error` string | merge both | Class is embedded in prose, not a field; the Debugger sees the string. Adding a class field is a one-liner and improves the robustness narrative. |

### §8 Scorekeeper

| item | tier | verdict | evidence | action | note |
|---|---|---|---|---|---|
| Metric = the kit's `evaluate.py`, not a reimplementation | M | implemented | `official.py:18-30` imports by path via `importlib`; `:72-75` calls `evaluate_module.evaluate` | keep theirs | Kit files untouched (`git diff --stat 8239f90..424238f -- kuairand-starter-kit` is empty). |
| Artifact validation (length / finite / dtype) | M | implemented | `run_candidate.py:63-67`; `tests/test_candidate_output.py:31-53` | keep theirs | |
| Sanity floor / ceiling | D | **missing** | grep `sanity`: only a config hypothesis string | adopt spec | Nothing catches a 0.85 "win" from an accidental label leak, or a 0.30 misalignment bug. Cheap and high-value. |
| Significance vs seed noise | D | partial | `policy.py:89-96` queues seed-1/2 replications when improvement > ε | merge both | Replication is *better* than the spec's `significant` flag — but no field records the flag, and nothing aggregates the 3 seeds into a mean/std for the results table. |
| Baseline reproduction check | M | implemented | `research_controller.py:50-63`: reuse a prior run or run the ladder; hard-fail if primary < 0.6016−0.002 | keep theirs | One-sided (a too-high score passes). |
| Best-so-far preserved across regressions | M | implemented | `policy.py:83-88`, strict `>` | keep theirs | Ties keep the earlier node, as spec. |

### §9 LLM layer

| item | tier | verdict | evidence | action | note |
|---|---|---|---|---|---|
| Structured outputs | M/D | implemented | `llm.py:47-127` JSON Schema, `strict: true`; `:240-248` | keep theirs | 4 schemas. Dataclass re-validation on top (`types.py:106-191`). |
| Provider = Anthropic / `claude-opus-5` | D | diverges | `llm.py:193-298`, `configs/ranking_losses.json:12` (`gpt-5.5`) | keep theirs | Provider is behind an `LLMProvider` Protocol (`llm.py:152-162`); swapping is contained. Non-defect. |
| Token accounting | M | implemented | `llm.py:281-287` (incl. `cached_tokens`, `web_search_calls`), summed at `roles.py:60`, persisted `audit.py:51-60` | keep theirs | |
| Retries / backoff | D | implemented | `llm.py:264-272`, retryable set at `:213-221` | keep theirs | 2 retries, cap 4 s — thinner than spec's 5/60 s but adequate. |
| Prompt caching | D | partial | `llm.py:251` `prompt_cache_key` per role; but the whole state summary is inlined in every prompt (`roles.py:66-87`) so the prefix changes each call | merge both | Move the immutable task/method-card text into a stable prefix. |
| Tool loop with turn cap | D | partial | only `web_search` with `max_tool_calls: 2` (`llm.py:234-260`) | keep theirs | Builder emits whole files rather than editing via tools — fewer turns, fewer tokens. |
| Fake LLM | D | implemented | `llm.py:301-322` | keep theirs | |
| Evidence / citation capture | — | **implemented (spec lacks)** | `types.py:79-91`, `roles.py:113-134` attaches web-search source URLs when curated evidence is thin | fold into spec | Directly serves Innovation (20%). |

### §10 Conductor

| item | tier | verdict | evidence | action | note |
|---|---|---|---|---|---|
| Iteration defined + counted | M | implemented | `research_controller.py:381-394`; `iteration_count` (proposals, incl. critic-rejected), `training_attempts`, `proposal_attempts`; documented `PLAN.md:118` | keep theirs | Three counters is arguably clearer than the spec's one. |
| Convergence ε=0.002 / N=3 | M | implemented | `policy.py:73-81` + `:98-99`; `convergence.py:14-29` | keep theirs | Reference is `meaningful_best`, seeded at the baseline primary (`research_controller.py:97`); stagnation counts only *scored* iterations. Matches the spec's default `count_failed_as_nonimproving=false`. See §3(d) for the one caveat. |
| 50-iteration cap | M | **diverges** | `configs/ranking_losses.json:23` `max_iterations: 8`; `configs/baseline.json:7` `3`. `50` appears only in `AGENTS.md:22` prose | adopt spec | Parameterised correctly but shipped at 8. A run will almost always stop at `iteration_budget_reached`, not `converged` — which is the state the organizers score. |
| 6 h wall-clock | M | implemented | `configs/ranking_losses.json:24` = 21600; checked `research_controller.py:354-356` | keep theirs | |
| Resume | D | implemented (stronger) | `research_controller.py:106-117`: config equality + source-manifest revision equality before resuming | keep theirs | Refusing to resume after a source change is a good idea the spec lacks. |
| Iteration-0 baseline reproduction inside the loop | M | partial | `research_controller.py:50-64` runs it as a *pre-loop gate* with harness-written FM code | merge both | Satisfies "reproduce the official baseline"; does not satisfy "the agent writes the code for each stage". |
| Never exits mid-run on a step failure | M | **missing (highest-impact defect)** | `research_controller.py:395-407`: `except Exception` → `stop_reason="controller_error"` → `break` | adopt spec | Single-strike kill. `sanitize_parameters` raising on an out-of-grid `learning_rate` (`policy.py:38-39`), a family mismatch (`roles.py:111`, `:175`), or `preserve_hypothesis=false` (`roles.py:207`) all end the whole run. See §5 rec 1. |
| Parent policy / experiment tree | D | implemented | `types.py:194-207` `parent_experiment` + `replicated_from`; `experiment_tree.json` (`audit.py:50`) | keep theirs | Richer than the spec's linear best-parent. |

### §11 Ledger

| item | tier | verdict | evidence | action | note |
|---|---|---|---|---|---|
| Per-iteration hypothesis | M | implemented | `research_controller.py:281-300` stores full `proposal` incl. hypothesis, rationale, evidence | keep theirs | |
| Per-iteration code diff | M | partial | only `code_sha256`/`tests_sha256` in the record (`research_controller.py:291-292`); full source lives in `generated_experiments/`, which is **gitignored** (`.gitignore:19`) | adopt spec | Recoverable from `runs/<id>/passes/NNN_builder_0.json` (`audit.py:39-42`), which *is* committed — so the code survives, but no unified diff and no obvious place a judge would look. |
| Per-iteration metrics | M | implemented | `outcome.metrics` in each record; `results.json` at `research_controller.py:431-447` | keep theirs | |
| Error / recovery events | M | partial | `outcome.error` + `outcome.recovery` (`candidate_runner.py:135-136`, `162`), `repairs` count, `research_memory.jsonl` `controller_error` | merge both | stdout/stderr paths are recorded but `runs/*/stdout/` is gitignored (`.gitignore:17`), so judges get the pointer and not the log. |
| Reflection per node | M | implemented | Critic-postflight stored as `postflight` (`research_controller.py:242-249, 288`) | keep theirs | |
| Event stream (`events.jsonl`) | D | diverges | split across `iterations.jsonl` + `research_memory.jsonl` + `passes/` | keep theirs | Equivalent coverage. |
| Rendered `journal.md` / `results.md` | D | **missing** | grep: only `results.json` | adopt spec | Judges read prose. A rendered journal is where Innovation (20%) and Robustness are actually assessed. |
| Intervention count | M | partial | `interventions.json` written once as `[]` (`research_controller.py:105`); `manual_interventions` is a `RunState` field (`types.py:235`) never incremented; every record hardcodes `"manual_intervention": False` (`:167, :298`) | adopt spec | The number reported is an assumption, not a measurement. No `intervene` command exists. |
| Wall-clock accounting | M | implemented | `research_controller.py:139-143`, `resources.json` (`audit.py:51-60`) incl. `gpu_hours: 0.0` | keep theirs | Correctly accumulates across resume. |
| Committed run dir is portable | C | **diverges (defect)** | `runs/20260828T141646Z_baseline/best.json:3` contains `C:\Users\Admin\OneDrive - Nanyang Technological University\...`; `iterations.jsonl` has `runs\...\stdout\...` backslash paths | adopt spec | Absolute Windows paths leak a personal directory and are unusable to a judge. Store repo-relative POSIX paths. |
| `source_manifest.json` committed | C | partial | written by `controller.py:58`, absent from the committed run (`git ls-files runs/`) | keep theirs | Just re-commit it. |

### §12 Gate

| item | tier | verdict | evidence | action | note |
|---|---|---|---|---|---|
| `submission.csv` (`row_id,user_id,video_id,score`) | M | **missing** | grep for `submission`/`row_id` across `src tests configs runs`: **zero hits** outside `AGENTS.md:149-162` prose | adopt spec | No code anywhere writes a submission file. |
| `submit.py --check` invoked | M | **missing** | same grep | adopt spec | |
| One-shot hidden-test scoring | M | **missing** | `official.py:33-69` never loads test; no test code path exists | adopt spec | |
| Test predictions produced at all | M | **missing** | `contracts.py:9-17` `CandidateContext` has no `test_x`; candidates return `validation_scores` only (`contracts.py:20-25`) | adopt spec | **This is the deliverable that determines 35% of the score.** A model can win on validation and still have nothing to submit. |
| Sealed-label discipline | M | implemented | test rows are never parsed; `AGENTS.md:39-41` reserves final test prediction as an explicit human-authorized step | merge both | Safe, but the "explicit user-authorized step" is by construction a manual intervention (Autonomy, 20%). |

### §13 CLI / config

| item | tier | verdict | evidence | action | note |
|---|---|---|---|---|---|
| `run` / `resume` | D | implemented | `controller.py:151-178` (`--config`, `--resume`) | keep theirs | |
| `status` / `intervene` / `report` | D | **missing** | `controller.py:152-165` | merge both | `intervene` is the mechanism that makes the M-tier intervention count honest. |
| Frozen run config | M | implemented | `research_controller.py:102`, `controller.py:56` | keep theirs | |
| Config defaults match the rules | M | partial | ε/patience/6 h correct; `max_iterations` 8 not 50 | adopt spec | |

### §14 Tests

| item | tier | verdict | evidence | action | note |
|---|---|---|---|---|---|
| Unit suite, no API, no network | D | implemented | 28 tests; **verified 25 pass**, 3 error only because `openai`/`dotenv` are not installed here | keep theirs | |
| Safety / leak-check tests | D | implemented | `tests/test_safety.py:12-64` | keep theirs | |
| Convergence maths tests | M | implemented | `tests/test_agent.py:11-24` | keep theirs | Only the happy path; no failed-iteration policy test. |
| Scorekeeper equals the kit | M | partial | `tests/test_official_evaluation.py:8-18` is a 2-row smoke test | merge both | No test asserting split sizes or kit-equality of `load_train_valid` (I verified this by hand — it holds; it should be a test). |
| Sandbox tests | D | partial | `tests/test_safety.py:38-64` runs a candidate's unit test | merge both | No timeout-kill test, no env-isolation test. |
| Gate tests | M | **missing** | no gate | adopt spec | |
| E2E dry run with fake LLM | D | implemented | `tests/test_research_loop.py:108-170` (full loop, both families, resume state, 8 scripted calls) | keep theirs | Uses a `FakeExecutor`, so it does not exercise the real training subprocess. |

### §15 Deliverables

| item | tier | verdict | evidence | action | note |
|---|---|---|---|---|---|
| README: overview, setup, reproduce, limits | M | partial | `README.md:1-169` — good on setup/commands/results ladder | merge both | Missing: limitations & what-we'd-do-with-more-time, team contributions, convergence/iteration definitions, architecture diagram. PowerShell-only commands (`README.md:29,46,83`) on a POSIX-graded repo. |
| Results table with deltas | M | partial | `README.md:158-166` validation ladder; `results.json` has `delta_vs_baseline` (validation) | adopt spec | No hidden-test row, because there is no hidden-test path. |
| Resource accounting reported | M | partial | `resources.json` complete; `README.md:165-166` reports tokens/wall-clock for the baseline run | keep theirs | |
| Devpost description | M | **missing** | no `docs/devpost.md` | adopt spec | |
| Per-iteration logs committed | M | partial | run dir committed, but `artifacts/` and `stdout/` gitignored (`.gitignore:16-17`) | adopt spec | |

### §16 Rulings

| item | tier | verdict | evidence | action | note |
|---|---|---|---|---|---|
| `log_random` not training data | M | n/a (moot) | never read (`official.py:43-46`) | merge both | Compliant by omission; the *analysis* use the spec allows is also unreachable. |
| `video_stat` legal but caveated | M | n/a (moot) | never read | adopt spec | |
| Eval rows never deduped/reordered | M | implemented | order preserved; verified `==` against kit | keep theirs | |
| Kit never modified | M | implemented | `git diff 8239f90..424238f -- kuairand-starter-kit` empty; `AGENTS.md:45-58` codifies it | keep theirs | |
| No external data | M | implemented | import allowlist `safety.py:8-18` makes it impossible | keep theirs | |

---

## 3. The big questions

**(a) Does the agent write code for each stage, or pick from pre-written configs?** — *It writes real code, for
one stage, inside a narrow grid.* The Builder returns whole files as strings: `llm.py:98-110` declares
`"code": {"type": "string"}, "tests": {"type": "string"}`, and `roles.py:156-158` instructs
*"Generate a self-contained candidate.py and test_candidate.py … candidate.py must define
`run(context, parameters) -> CandidateOutput`."* That source is written to disk (`candidate_runner.py:32-40`)
and imported and executed (`run_candidate.py:42-52`). So this is genuine LLM code generation, not config
selection — and the trusted-context design is a real strength.

But the *reachable* surface is one stage. Three clamps:
1. `roles.py:166-167` — *"For BPR you must call `src.models.sampling.sample_bpr_pairs`. For group-softmax
   you must call `sample_softmax_groups`. These trusted samplers are mandatory."* — enforced by AST at
   `safety.py:114-133`.
2. `policy.py:36-61` hard-rejects anything off-grid: `if parameters["k"] != 16: raise ValueError(...)`,
   `if parameters["learning_rate"] not in {0.0003, 0.0005, 0.001}: raise`, batch/temperature/negatives
   from fixed sets. `types.py:109-110` restricts `family` to `{"bpr", "group_softmax"}`.
3. `safety.py:8-18` allows only `numpy, collections, math, time, typing, dataclasses` + three project
   modules — no sklearn, LightGBM, torch, pandas.

And features are fixed before the agent sees anything: `run_candidate.py:110-114` calls the kit's
`encode(splits)` and hands the candidate the frozen 5-field matrix. There is no way for generated code to
build a feature, run EDA, or touch the evaluate stage. Against the problem statement's *"Improvements may
target any part of the algorithmic stack"* (`problem_statement.md:116-118`) and *"writing the code for each
stage is part of the agent's job"* (`:71-72`), this is **partial**: one stage of five, two families, ~12
grid points.

**(b) Can any code path read hidden-test labels or the test-period standard log?** — *No.* Traced:
`run_candidate.py:110` → `load_train_valid` (`official.py:33-69`), whose loop `continue`s on any date
outside 20220408–20220428 **before** `row['long_view']` is read (`official.py:50-57`, comment: *"Crucially
skip before reading the relevance label"*). The `CandidateContext` (`contracts.py:9-17`) carries
`train_x/train_y/train_users/valid_x/valid_users/field_dimension/evaluate_validation` — no test field, and
`valid_y` is never handed over; the candidate can only reach validation labels through the trusted
`evaluate_validation` closure (`run_candidate.py:116-122`). `safety.py:60-72` additionally text-blocks
`data/judge`, `ground_truth`, `test_truth`, `src.evaluation`, `kuairand-starter-kit` in generated source, and
the import allowlist blocks `open`/`os`/`csv`. The only residual: candidates are `exec_module`'d **in the
same process** as the trusted worker (`run_candidate.py:49`), so isolation rests entirely on the AST
validator rather than an OS boundary — but since test rows are never loaded into that process, there is
nothing to steal. Verdict: architecturally unreachable, for the stronger reason that it is never read.

**(c) Kit metric or reimplementation?** — *The kit's, imported by path.* `official.py:18-30` inserts
`kuairand-starter-kit` on `sys.path`, `importlib.import_module("evaluate")`, pops the path, and
`official.py:72-75` calls `evaluate_module.evaluate(...)` directly. Nothing is copied. Tie handling by row
order, GAUC positive-count weighting, and zero-positive users in nDCG therefore come from the frozen file by
construction. The split builder *is* a reimplementation (`official.py:33-69`) — I verified it: train and
valid row tuples compare `==` to `data.load()`'s, sizes 1,141,112 / 124,909, and `encode()` yields an
identical valid feature matrix and dim 40260. The FM baseline reuses the kit's own `FM` class
(`baselines.py:51-57` → `baseline_module.FM`) with the kit's `p > best + 1e-5` / patience-4 rule
(`baselines.py:96-103`, cf. `kuairand-starter-kit/baseline.py:88-95`).

**(d) Convergence rule as written?** — *Yes in substance, with two caveats.* `policy.py:73-81`: a scored
iteration that fails to clear `meaningful_best + ε` increments `stagnant_iterations`; `policy.py:98-99`
stops at `stagnant >= patience` (3) — ε and N come from `configs/ranking_losses.json:29-32` (0.002 / 3).
Caveat 1: `should_stop` also requires `coverage_complete(state)` — both BPR *and* group-softmax must have
produced a success — so if one family never succeeds, convergence can never fire and the run ends on a
budget instead. Caveat 2: `max_iterations: 8` (`configs/ranking_losses.json:23`) means the cap almost
certainly fires before convergence does; the organizers score *the converged result*
(`problem_statement.md:311-317`), so shipping at 8 changes what gets scored.

**(e) Per-iteration artifacts vs the deliverables list** (`problem_statement.md:273-280`): hypothesis ✅
(full `proposal` object with rationale + cited evidence); code diff ⚠️ (sha256 in the record, full source in
`passes/` and in gitignored `generated_experiments/`, no diff); metrics ✅ (GAUC/nDCG@5/primary + trace +
diagnostics); error/recovery events ⚠️ (`error`, `recovery`, `repairs`, `controller_error` recorded; the
stdout/stderr they point at are gitignored); intervention summary ⚠️ (a hardcoded `false` per record and a
never-incremented counter). **Verdict: would partly satisfy a judge; the missing halves are all cheap.**

**(f) Token and wall-clock accounting** — *Present and good.* Per-call `input/output/total/cached_tokens`
and `web_search_calls` (`llm.py:281-287`), aggregated into `RunState.token_usage` (`roles.py:60`), per-call
records in `research_memory.jsonl` and `passes/` (`audit.py:39-43`), rolled into `resources.json` with
`gpu_hours: 0.0` (`audit.py:51-60`) and into `summary.json` (`research_controller.py:418-419`). Wall-clock
accumulates correctly across resume (`research_controller.py:139-143`). A 150,000-token budget is enforced
as a stop condition (`research_controller.py:366-368`, `roles.py:51-52`).

**(g) Autonomy — what manual steps do the docs assume?** Four: (1) copy `.env.example` → `.env` and add a
key (`README.md:79-86`) — one-time setup, not an in-run intervention; (2) *"Final judge prediction
generation remains a separate, explicit user-authorized step"* (`README.md:110-111`) and
`AGENTS.md:39-41` — this is a per-run human step on the critical path to the scored deliverable;
(3) resume after a `controller_error`, which any off-grid proposal triggers (`research_controller.py:395-407`);
(4) authoring the results table / README numbers by hand, since no `journal.md`/`results.md` is rendered.
(2) and (3) are the ones that will show up as interventions.

---

## 4. Things the harness has that the spec lacks — fold these in

1. **`research/methods/*.md` method cards** (`research/methods/bpr.md`, `group_softmax.md`) — primary
   citation, objective + gradient in closed form, a *safe search space*, and *known failure modes*. Far
   better than the spec's flat idea bank: the gradient formula makes the Builder's code more likely to be
   right first time, and "known failure modes" pre-empts Debugger rounds. **Adopt, and expand to the spec's
   18 ideas.**
2. **Trusted samplers as a mandatory dependency** (`src/models/sampling.py`, enforced `safety.py:114-133`)
   — the same-user-negatives invariant (the one thing a BPR implementation most often gets wrong) is
   guaranteed by trusted code rather than hoped for from the LLM. **Adopt.**
3. **Candidates cannot report their own metrics** (`run_candidate.py:86-90`, tested at
   `tests/test_candidate_output.py:14-29`). Closes a reward-hacking hole the spec left open. **Adopt.**
4. **Critic preflight** (`roles.py:138-150`) — rejects a hypothesis before code tokens are spent, and the
   rejection is ledgered as an iteration (`research_controller.py:147-170`). Cheap, and it produces the
   "what we chose *not* to try and why" material judges reward. **Adopt.**
5. **Source-manifest-gated resume** (`research_controller.py:112-114`) — refuses to resume a run whose
   harness source changed. **Adopt.**
6. **Skip-before-read data loading** (`official.py:50-57`) — strictly stronger than the spec's
   strip-the-columns approach. **Adopt as the spec's §5 rule.**
7. **Automatic replication at seeds 1 and 2 on any >ε gain** (`policy.py:89-96`) — better than the spec's
   static significance threshold, and it produces the mean/std the README should report. **Adopt** (and add
   the aggregation step, which is missing).
8. **`AGENTS.md`** — a genuinely good repo-level contract for coding agents; keep as-is.
9. **Python 3.9-compatible, numpy-only runtime** (`requirements.txt`) — runs on the kit's own interpreter;
   avoids the spec's two-venv `scripts/setup_env.sh` complexity. Keep unless a candidate needs torch.

---

## 5. Top 10 recommendations, ordered by judging impact

| # | Recommendation | Tier | Effort | Why |
|---|---|---|---|---|
| 1 | **Build the Gate**: give `CandidateContext` a `test_x`, have the trusted worker persist `test_scores`, write `submission.csv` with `row_id,user_id,video_id,score`, run `submit.py --check`, then score once via the kit. | M | M | 35% of the score has no artifact today (`grep submission src tests` → 0 hits). Without this there is nothing to submit. |
| 2 | **Stop killing the run on one exception.** Replace `research_controller.py:395-407`'s `break` with: ledger a `harness_error`, retry the step once, else mark the node failed and `continue`. Convert `roles.py:111/175/207` and `policy.py:36-61` raises into critic-style rejections. | M | S | An LLM proposing `lr=0.002` currently ends the run. Hits Robustness (35%) and Autonomy (20%) simultaneously. |
| 3 | **Widen the agent's code surface to features and inspection.** Let the Builder write a `build_features(rows) -> X` alongside the loss, expose `user_features_pure.csv` / `video_features_statistic_pure.csv` / raw log columns, and drop the `k==16` / lr-grid clamps once #2 makes off-grid proposals non-fatal. | M | L | The problem statement demands the full stack (`:71-72`, `:116-118`). Also the largest realistic source of hidden-test delta. |
| 4 | **Raise `max_iterations` to 50** and let convergence be the stop reason; remove `coverage_complete` from `should_stop` (or make it a *minimum-coverage* precondition that expires). | M | S | Organizers score the converged result; `configs/ranking_losses.json:23` currently guarantees `iteration_budget_reached`. |
| 5 | **Render `journal.md` + `results.md`** from `iterations.jsonl` at run end, and commit `stdout/` and `generated_experiments/` (drop `.gitignore:16-19` for the final run). Include a unified diff per iteration. | M | M | Innovation (20%) and Robustness are graded from prose the judges can actually read. |
| 6 | **Make the intervention count real**: `agent intervene --run <id> --reason …` appending to `interventions.jsonl`, incrementing `RunState.manual_interventions`; stop hardcoding `"manual_intervention": False` (`research_controller.py:167, 298`). | M | S | Autonomy is 20% and is measured by this number; today it is an assertion. |
| 7 | **Add a data card** computed from train+valid (label rates by tab/user, duration buckets, drift, duplicate rate) into the Researcher prompt, which today sees only prior metrics (`roles.py:66-87`). | D | M | Grounds hypotheses in evidence — exactly what Innovation is scored on. |
| 8 | **Sanity floor/ceiling** in the trusted worker (reject primary < 0.47 as a bug, flag > 0.80 as suspicious, never promote it). | D | S | Cheap insurance against a silent leak or misalignment becoming the submitted checkpoint. |
| 9 | **Fix candidate env + portability**: build a minimal env in `candidate_runner.py:58-62` (drop `OPENAI_API_KEY`), add a smoke mode, and store repo-relative POSIX paths — `runs/.../best.json:3` currently embeds `C:\Users\Admin\OneDrive - Nanyang Technological University\…`. | D/C | S | Secret hygiene, wall-clock (15%), and a committed run a judge can read. |
| 10 | **Round out README/Devpost**: limitations, team contributions, iteration/convergence definitions, architecture diagram, POSIX commands beside the PowerShell ones (`README.md:29,46,83`), plus `docs/devpost.md`. | M | S | Explicitly required by `problem_statement.md:252-269`. |

---

## 6. Verdict — **(iii) merge, with this harness as the base**

Build on their harness and port the spec's missing organs into it. Their foundation is sound in exactly the
places that are expensive to redo: metric fidelity is the kit's own file and I verified their splits are
byte-identical to `data.load()`; the trusted-context / AST-validator / no-self-reported-metrics design is
stricter and better thought out than the spec's; the audit trail, token accounting and resume are real.
The gaps are additive, not architectural — a Gate, a wider code surface, a non-fatal error path, a rendered
journal, a real intervention counter. Rebuilding to the spec would throw away working, tested code to regain
things the spec only describes. The two things the spec must impose regardless: **an end-to-end submission
path** (nothing scores without it) and **an agent whose code reaches more than one stage** (the problem
statement's central requirement). Do #1 and #2 first — they are S/M effort and together move ~55% of the
rubric.
