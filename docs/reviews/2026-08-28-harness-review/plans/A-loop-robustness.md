# Owner A — Loop & robustness (harness author)
Branch: `feat/loop-robustness`   ·   Base: `main` after the Step 0 merge (`cbf8330` step 0 + `553095d` step 0b, the branch head)   ·   Estimated effort: ≈ 6 h code + ~1 h attended run-operator time

## 1. Mission

Make the research loop survive everything an LLM can throw at it, report the organizers' convergence verdict honestly, and be the single place where every other owner's module is wired in. This moves **Technical — robustness (~10%)** (the loop dies on the first non-conforming response today) and **Autonomy (20%)** (`manual_intervention: False` is hardcoded, so the number that scores autonomy is unfalsifiable). It also moves **Feasibility (15%)**: real caps, honest wall-clock. A is also the run operator — the live runs and the one committed final run.

## 2. Files you own (exclusive) / files you must not touch

**Own:** `src/agent/research_controller.py`, `policy.py`, `convergence.py`, `controller.py`, `configs/*` — meaning the shared configs (`ranking_losses.json`, `baseline.json`) and your own `configs/run_<initials>.json`, *not* the per-owner files rule 4 lets others add (C's `configs/offline_smoke.json`, E's `configs/features_run.json`) — `tests/test_research_loop.py`, `tests/test_agent.py`, new `tests/test_controller_robustness.py`.

**Must not touch:** `llm.py`, `roles.py`, `catalog.py`, `tests/test_openai_runtime.py`, `tests/test_research_runtime.py`, README/AGENTS/PLAN (C) · `src/evaluation/*`, `contracts.py`, `run_candidate.py`, `run_baseline.py`, `candidate_runner.py`, the `runs/` baseline (B) · `audit.py`, `logger.py`, `report.py`, `datacard.py`, `.gitignore` (D) · `safety.py`, `families.py`, `src/models/*`, `research/methods/*` (E). Plus the frozen shared surfaces `types.py`, `contracts.py`, `configs/ranking_losses.json`, and `src/agent/errors.py` — frozen in Step 0b, nobody owns it: C raises those exceptions, you import and catch them.

To change someone else's file: ask the owner, or send a ≤20-line PR they merge — never edit it yourself. **One exception, and it is yours:** `configs/ranking_losses.json` is frozen but `configs/*` is your area, and rule 6 says a second freeze PR, by A, is how a frozen surface moves. T3 is that PR: three new `budgets` keys plus C's one-line `llm.max_retries` bump, nothing else, opened first thing on Day 1 and announced in chat.

## 3. Setup (15 minutes)

```bash
cd /Users/Ke_Jun_YEO_from.TP/Desktop/personal/techjam-2026-SeniorPromptEngineers
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt pytest      # pytest is not in requirements.txt
unset OPENAI_API_KEY
python -m pytest -q -W error                          # expect: 47 passed (28 existing + 19 from Step 0/0b)
git checkout main && git pull && git checkout -b feat/loop-robustness
```

`ModuleNotFoundError: No module named 'dotenv'` plus 3 errors in `tests/test_openai_runtime.py` means you are on the system interpreter, not the venv. You are the only owner who needs `OPENAI_API_KEY`, and only for T11; keep it in `.env` and never export it into a shell you run tests in.

## 4. Tasks, in order

### T1 · C4 · Unkillable loop   (L, ~1.5 h)

- **Why:** `research_controller.py:397-409` catches `Exception` from the whole research→critic→build→execute block and `break`s with `stop_reason="controller_error"`. One bad model response ends a 6-hour run. Robustness and Autonomy are both scored on exactly this.
- **Where:** `research_controller.py:372-409` (the `try`/`except` in `run()`), importing from the frozen `src/agent/errors.py` (Step 0b). Every exception source the correctness report names, and how each classifies:

  | Source | Raise site | Exception | Kind |
  |---|---|---|---|
  | `sanitize_parameters` off-grid | `policy.py:37,39,41,43,45,50,52,57,59,61,63` | `ValueError` | proposal |
  | Researcher ignored the required family | `roles.py:111` | `ValueError` | proposal |
  | Builder changed family / hypothesis_id | `roles.py:175` | `ValueError` | proposal |
  | Debugger dropped the hypothesis | `roles.py:207` | `ValueError` | proposal |
  | Missing / wrong-typed schema field | `types.py:9` (`_required`) | `ValueError` | proposal |
  | Unsupported family, unsupported action | `types.py:110`, `types.py:113` | `ValueError` | proposal |
  | Empty `output_text` (reasoning ate the budget) | `llm.py:276` | `ValueError` | proposal |
  | Malformed JSON body | `llm.py:277` `json.loads` | `JSONDecodeError` ⊂ `ValueError` | proposal |
  | Token budget | `roles.py:52`, `roles.py:63` | `RuntimeError` → `TokenBudgetExceeded` | budget |
  | No API key / SDK missing / script exhausted | `llm.py:198,202,310` | `RuntimeError` | harness |
  | Disk, permissions, corrupt run dir | anywhere | `OSError` | harness |

  You cannot edit `roles.py`, `types.py` or `llm.py`, so classification is by **exception type**, with one message check for the budget case until C's typed exception lands (T8).
- **Do:**
  1. `from .errors import LLMError, TokenBudgetExceeded` — the classes are already frozen in `src/agent/errors.py` (Step 0b: `LLMError(RuntimeError)` and subclasses `TokenBudgetExceeded`, `RoleOutputInvalid`, `IncompleteResponse`). Nobody owns that file: **C raises** them from `llm.py`/`roles.py`, **you catch** them. Define no exception class of your own anywhere. They subclass `RuntimeError`, so the message fallback and existing `except RuntimeError` sites keep working.
  2. Two module functions in `research_controller.py`:
     ```python
     def _is_budget_error(exc: BaseException) -> bool:
         return isinstance(exc, TokenBudgetExceeded) or (
             isinstance(exc, RuntimeError) and "token budget" in str(exc).lower())

     def _error_kind(exc: BaseException) -> str:      # 'budget' | 'proposal' | 'harness'
         if _is_budget_error(exc): return "budget"
         return "proposal" if isinstance(exc, (ValueError, TypeError, KeyError)) else "harness"
     ```
     `KeyboardInterrupt` is a `BaseException` and is deliberately not caught: Ctrl-C still exits, and the operator records it with `intervene` (T7).
  3. Add `ResearchLoop._role_call(self, label: str, iteration: int, call) -> Any`, where `call` takes a feedback string or `None`. It retries while `_error_kind(exc) == "proposal"`, at most `budgets.max_role_reprompts` (default 2) times, setting `feedback = f"Your previous {label} response was rejected: {exc}"` and appending a `{"type": "role_retry", …}` line to `research_memory.jsonl` each time; anything else re-raises. Wrap the three role calls at `:385`, `:388`, `:395`. Until C adds the `feedback` keyword the lambda ignores it (`lambda fb: self.roles.research(self.state, iteration, required_family(self.state))`); after C's change, pass `feedback=fb`. Both forms compile today.
  4. Replace the handler at `:397-409` with a branch on `_error_kind`, each branch first appending `{"type": "controller_error", "kind": …, "iteration": …, "error": …}` to `research_memory.jsonl`. **budget** → `stop_reason = "llm_token_budget_reached"`, `break` (as today). **proposal** → reset `self.consecutive_harness_errors`, `_record_failed_proposal(iteration, exc, "proposal_failed")`, `continue`. **harness** → increment the counter; at `budgets.max_consecutive_harness_errors` (default 3) write `error.json`, set `stop_reason = "harness_error_breaker"` and `break`, otherwise record `"harness_error"` and `continue`.
  5. `self.consecutive_harness_errors = 0` lives on the loop, not on `RunState` — `types.py` is frozen, and resetting the breaker on resume is the behaviour you want anyway. `_record_failed_proposal(iteration, exc, status)` calls `self._save()` then `audit.record_iteration({…})`; it appends **no** `ExperimentNode` (a failed proposal has no family) and does not advance `iteration_count`. `proposal_attempts` was already incremented at `:384`, so `max_proposals` (T3) bounds this path — the breaker is only for harness errors.
- **Interface:** provides the non-fatal path I-7 relies on — *"Until A lands that, a new family's parameters pass through A's C4 non-fatal path (rejection, not a crash)."*
- **Tests** (`tests/test_controller_robustness.py`, new): `test_malformed_research_response_is_reprompted_and_the_run_survives` (first researcher payload has `"family": "nope"` → `types.py:113`, second valid; `stop_reason != "controller_error"`) · `test_three_bad_responses_record_proposal_failed_and_continue` (an `iterations.jsonl` line with `"status": "proposal_failed"`, and a successful node afterwards) · `test_off_grid_parameters_do_not_kill_the_run` (manifest with `"epochs": 99`, `policy.py:41`) · `test_empty_output_text_is_a_proposal_failure` (provider raising `ValueError("… contained no output text.")`) · `test_consecutive_harness_errors_trip_the_breaker` (`OSError` every call → `stop_reason == "harness_error_breaker"`, `error.json` written, exactly 3 attempts) · `test_harness_error_counter_resets_after_a_good_iteration` (OSError, good iteration, 2× OSError → no breaker).
- **Acceptance:** `pytest -q -W error tests/test_controller_robustness.py` green with no API key; `grep -n 'stop_reason = "controller_error"' src/agent/research_controller.py` → no hits; full suite green.
- **Depends on / blocks:** nothing, but every owner's Day-1 offline e2e run needs it — land it first.

### T2 · C5, I12 · Baseline selection by recorded revision, and log every skip   (M, ~0.75 h)

- **Why:** `_latest_valid_baseline` picks by **mtime** and adopts the summary wholesale, so a run today inherits `runs/20260828T141646Z_baseline/` — produced by code in no commit in this range, with `best.artifact_path` pointing at `C:\Users\Admin\OneDrive - Nanyang Technological University\…`. The silent `except … continue` makes a corrupt summary look identical to "no baseline exists" (I12).
- **Where:** `research_controller.py:38-49` (`_latest_valid_baseline`), `:52-66` (`_ensure_baseline`), `:87` (call site); `controller.py:26-36` (`_source_manifest`).
- **Do:**
  1. `_latest_valid_baseline(run_root: Path, threshold: float, revision: str) -> tuple[dict | None, list[dict]]` — the chosen summary plus `{"path": str, "reason": str}` skip records.
  2. Accept only when all hold, recording the first failure as the reason: `best.experiment_id == "official_fm_seed0"` (`experiment_id_mismatch`) · `primary >= threshold` (`below_threshold`) · `<run>/source_manifest.json`'s `revision` equals the current one (`revision_mismatch` / `no_source_manifest`) · `best.artifact_path` resolves through `_resolve_repo_path` to an existing file (`artifact_missing`). Keep the existing `except (OSError, ValueError, TypeError, JSONDecodeError)` but record `unreadable_summary` instead of a bare `continue`.
  3. Order by run-id (`path.parent.name`), not `st_mtime` — run ids are UTC timestamps, so lexicographic order is chronological and does not depend on the filesystem.
  4. `_ensure_baseline` passes `_source_manifest()["revision"]`, returns `(summary, skips)`, and prints one line per skip so a live operator sees it. `__init__` writes `<run_dir>/baseline_selection.json` = `{"selected": …, "skipped": […]}` once the audit exists.
- **Interface:** none. Coordinate with B: B regenerates and commits the baseline run (C5's other half), and B's I11 two-sided gate lands in **your** file at `:55, :62` as a sanctioned ≤20-line PR from B — `official.py::within_baseline_tolerance(primary: float, official: float = 0.6016, tolerance: float = 0.003) -> bool` replacing `if primary < official - 0.002`, plus the same two-sided filter for `_latest_valid_baseline`'s threshold argument. Merge it after this task so B rebases once.
- **Tests:** `test_baseline_is_rejected_when_the_source_revision_differs` (reason `revision_mismatch`) · `test_baseline_is_rejected_when_the_artifact_is_missing` (Windows path → `artifact_missing`) · `test_baseline_selection_logs_every_skipped_summary` (both reasons present in `baseline_selection.json` — this is I12) · `test_baseline_prefers_the_newest_matching_run_id` (older mtime on the newer id).
- **Acceptance:** `grep -n 'st_mtime' src/agent/research_controller.py` → no hits; with only the stale `runs/20260828T141646Z_baseline/` present, constructing a `ResearchLoop` re-runs the baseline instead of adopting it and `baseline_selection.json` names the reason.
- **Depends on / blocks:** none. While waiting on B, test against fixture directories built in `tmp_path`.

### T3 · I5 · Split `max_iterations` / `max_training_attempts` / `max_proposals`   (S, ~0.3 h)

- **Why:** one knob means three things — the candidate cap (`:362`), the training-attempt cap (`:226`, `:359`) and, doubled, the proposal cap (`:353`). Step 0 raised it to 50 and silently raised all three. The organizers score the **converged** result; at 8 the run ended on budget every time.
- **Where:** `research_controller.py:226, 351, 353, 359, 362`; `configs/ranking_losses.json:23-30`.
- **Do:** in `run()`, `max_training_attempts = int(self.budgets.get("max_training_attempts", max_iterations))` and `max_proposals = int(self.budgets.get("max_proposals", max_iterations * 2))` — the `.get` defaults keep the inline configs at `tests/test_research_loop.py:138-147, 202-208` working unedited. Point `:226` at `max_training_attempts` (store it on `self` in `__init__` so `_execute` can read it) and `:359` too, with `stop_reason = "training_attempt_budget_reached"`; `:362` stays `max_iterations` / `"candidate_budget_reached"`. In `configs/ranking_losses.json` keep `max_iterations: 50` and `max_training_attempts: 50`, and add `"max_proposals": 100`, `"max_consecutive_harness_errors": 3`, `"max_role_reprompts": 2`. Carry C's one-line ask in the same PR: `configs/ranking_losses.json:21` `"max_retries": 2` → `"max_retries": 5` (C's I4 retry policy, §6). This is the second freeze PR — announce it, keep it to these four lines.
- **Tests:** `test_iteration_training_and_proposal_caps_are_independent` — `max_iterations: 5`, `max_training_attempts: 1` → `stop_reason == "training_attempt_budget_reached"` and `training_attempts == 1`.
- **Acceptance:** `grep -n 'max_iterations' src/agent/research_controller.py` → only the candidate-cap sites; `pytest -q -W error tests/test_research_loop.py` green without editing its fixtures.

### T4 · I6, I-9 · Official convergence verdict, separate from the coverage-gated stop   (S, ~0.4 h)

- **Why:** `policy.py:98-99` is `coverage_complete(state) and stagnant >= patience and not pending_replications`. A run that converged by the organizers' rule keeps burning iterations until both families have a *successful* node — and once E adds `history_features`/`multi_task`, `coverage_complete` may never fire at all, so `stop_reason` reads `iteration_budget_reached` on a genuinely converged run. Flag that interaction to D and E when you merge.
- **Where:** `policy.py:98-99`; `research_controller.py:365-367` and the summary at `:413-428`.
- **Do:** leave `should_stop` as the harness agenda — do not weaken it. After the loop, with `seq = [state.baseline_primary] + scored_primaries(state)` (T5), set `summary["converged_official"] = official_converged(seq, epsilon, patience)` and `summary["converged_official_iteration"]` = the smallest k whose prefix fires, else `None`. The first time it fires mid-run, append `{"type": "convergence", "iteration": k, "official": true}` to `research_memory.jsonl`, so the journal can say "official rule fired at iteration k; harness continued for coverage". `stop_reason` semantics are unchanged.
- **Interface (verbatim):** **I-9 Scored-iteration convergence** — **A** reports both the official verdict (`converged_official: bool`, per ε=0.002 / N=3 over scored iterations) and the harness stop reason in `summary.json`; **D** prints both.
- **Tests:** `test_official_convergence_is_reported_when_the_harness_keeps_going` — only `bpr` ever succeeds, 4+ stagnant scored iterations → `converged_official is True` and `stop_reason != "converged"`.
- **Acceptance:** `summary.json` from the mocked run in `tests/test_research_loop.py` carries both keys; tell D the key names the moment it merges. **Depends on:** T5.

### T5 · I7 · One convergence implementation, tested against the reference formula   (M, ~0.6 h)

- **Why:** `convergence.py:14-29` and `policy.py:73-96` duplicate the ratchet; only the former is tested (`tests/test_agent.py:12-24`) and both of its cases are ones where ratchet and reference agree.
- **Where:** all of `src/agent/convergence.py`; `policy.py:73-99`; `controller.py:62-65, 122`.
- **Do:**
  1. `convergence.py` becomes the only implementation, with two pure functions: `official_converged(scores: Sequence[float], epsilon: float = 0.002, patience: int = 3) -> bool` — with `best_k = max(scores[:k])`, converged when `k > patience` and `best_k − best_{k−patience} <= epsilon`; and `stagnation(scores: Sequence[float], epsilon: float) -> tuple[float | None, int]` returning `(meaningful_best, stagnant_iterations)`. The spec writes the rule as `k ≥ N`, which needs an undefined `best_0`; resolve it to `k > N` (N+1 scores = N consecutive deltas), which matches the shipped behaviour at `tests/test_agent.py:12-17` where the tracker converges on the 4th observation.
  2. `ConvergenceTracker.observe` keeps its signature (`controller.py:122` calls it) but stores `self.scores` and delegates. Keep the `stagnant_iterations` field — `tests/test_agent.py:24` asserts on it.
  3. `policy.py`: add `scored_primaries(state) -> list[float]` = `[float(n.metrics["primary"]) for n in state.nodes if n.status == "success" and n.metrics]`, and have `observe_success` set `state.meaningful_best, state.stagnant_iterations = stagnation([state.baseline_primary] + scored_primaries(state), self.epsilon)`. Seeding with the baseline preserves today's behaviour (`research_controller.py:99`) and needs no new `RunState` field — `types.py` is frozen. Delete the inline ratchet at `policy.py:76-81`.
- **Tests** (`tests/test_agent.py`, yours): `test_official_rule_matches_the_literal_reference_over_random_sequences` — 2,000 sequences from `random.Random(0)`, lengths 1–20, scores in [0.45, 0.75]; assert `official_converged` equals a separately written literal `max(s[:k]) - max(s[:k-N]) <= eps` loop at every k. If ratchet and reference ever disagree, **the reference wins** — it is the organizers' rule. Plus `test_stagnation_is_the_only_ratchet` (`ConvergenceTracker` and `SearchPolicy` agree on one sequence); the two existing `ConvergenceTests` cases stay green.
- **Acceptance:** `grep -n 'stagnant_iterations +=' src/agent/policy.py` → no hits; `pytest -q -W error tests/test_agent.py` green. **Blocks:** T4.

### T6 · I9 · Wall-clock starts before the baseline gate   (S, ~0.15 h)

- **Why:** `_ensure_baseline` runs at `research_controller.py:87`; `self.session_started` is only set at `:139`, so 20 s–3.5 min of baseline reproduction is missing from `resources.json` — the file Feasibility is scored on.
- **Where / Do:** move `self.session_started = time.monotonic()` to the first statement of `__init__` (`:78`) and delete it from `:139`. `_elapsed()` (`:141-142`) already adds `state.wall_clock_seconds`, loaded on resume at `:117`, so resume accounting stays correct.
- **Tests:** `test_wall_clock_includes_the_baseline_gate` — monkeypatch `research_controller._ensure_baseline` to `time.sleep(0.05)`, build the loop with `baseline_summary=None`, call `loop._save()`, assert `state.wall_clock_seconds >= 0.05`.
- **Acceptance:** `resources.json` from a real run reports at least the baseline duration.

### T7 · I10, I-8 · `intervene` command and a real intervention counter   (M, ~0.75 h)

- **Why:** `manual_intervention: False` is hardcoded at `controller.py:107`, `research_controller.py:169` and `:300`; `interventions.json` is written once as `[]` at `:107` and never appended to. Autonomy is 20% and this number is its only evidence — today it is an assumption, not a measurement.
- **Where:** `controller.py:151-178` (`main`), `:107`, `:141`; `research_controller.py:107, 144-147, 169, 300, 419`.
- **Do:**
  1. `controller.py::main` gains `parser.add_argument("command", nargs="?", choices=["run","intervene"], default="run")` plus `--run` and `--reason`. `python -m src.agent.controller --config configs/baseline.json` keeps parsing unchanged (the README and B's baseline regeneration depend on it).
  2. `intervene` appends one line to `<run_dir>/interventions.jsonl` (`{"ts": <ISO-8601 UTC>, "run_id": …, "reason": …}`), then, if `state.json` exists, reloads it with `RunState.from_dict`, sets `manual_interventions` to the file's **line count**, and writes it back via `ResearchAudit(run_dir, resume=True).save_state(…)` — one call refreshes `state.json`, `experiment_tree.json` and `resources.json`. Exit 2 with a clear message if the run dir is missing.
  3. Make the count **derived, never incremented**: `_count_interventions(run_dir: Path) -> int`, called at the top of `ResearchLoop._save()` (`:144-147`). A live loop holds `self.state` in memory and rewrites `state.json` on every save, so an incremented counter would be clobbered by a concurrent `intervene`; a derived count cannot be, and it is falsifiable against a file a judge can read.
  4. Replace `interventions.json` at `:107` with an empty `interventions.jsonl` — tell D, the journal reads it. Then `research_controller.py:169` and `:300` become `"manual_intervention": self.state.manual_interventions > self._interventions_at_iteration_start`, captured at the top of each loop pass; `controller.py:107` and `:141` use `_count_interventions(logger.run_dir)` the same way.
- **Interface (verbatim):** **I-8 Interventions** — **A** provides `python -m src.agent.controller intervene --run <run_dir> --reason "…"` appending to `<run_dir>/interventions.jsonl` and incrementing `RunState.manual_interventions`; the summary/results report the count; **D** prints them in `results.md`.
- **Tests:** `test_intervene_appends_and_increments` (one line; `state.json` and `resources.json` show 1) · `test_intervention_count_is_derived_from_the_file` (3 hand-written lines, `_save()`, count 3) · `test_intervene_on_a_missing_run_dir_exits_nonzero` · `test_baseline_cli_still_accepts_the_documented_flags`.
- **Acceptance:** `python -m src.agent.controller intervene --run runs/<id> --reason "restarted after API outage"` writes the line and `resources.json` shows 1; `grep -rn '"manual_intervention": False' src/` → no hits.

### T8 · I13 · Typed `TokenBudgetExceeded`   (S, ~0.25 h)

- **Why:** `research_controller.py:402` keys control flow on `"token budget" in str(exc).lower()`. It works only because `roles.py:52` and `:63` happen to contain the phrase.
- **Where / Do:** **you raise it nowhere, and you define it nowhere.** The class is frozen in `src/agent/errors.py` (Step 0b, unowned); the only two raise sites are in `roles.py`, C's file. You own the catch only (T1's `_is_budget_error`, replacing `:402`). C converts those two `raise RuntimeError(...)` at `roles.py:52, 63` to `TokenBudgetExceeded` in their own T2 step 7, importing from `src.agent.errors` with the messages unchanged — chase them for it; only if C is late send the sanctioned ≤20-line PR yourself. Because it subclasses `RuntimeError`, `_is_budget_error` is correct both before and after that lands — no ordering dependency, and no window where a budget stop is misclassified as a harness error.
- **Tests:** `test_token_budget_message_stops_the_run_cleanly` (provider raising `RuntimeError("LLM token budget reached before the next role pass.")`) and `test_token_budget_exceeded_type_stops_the_run` (raising the class) — both assert `stop_reason == "llm_token_budget_reached"`, and both pass before and after C's PR.
- **Acceptance:** `grep -n '"token budget"' src/agent/research_controller.py` → exactly one hit, inside `_is_budget_error`, never in `run()`.

### T9 · I3 · `_save()` before `record_iteration()`   (S, ~0.15 h)

- **Why:** `record_iteration()` appends to `iterations.jsonl` at `:163` and `:283`, and `_save()` follows at `:172` and `:303`. A crash in between replays the iteration on resume and duplicates the JSONL line.
- **Where / Do:** swap the order in `_record_rejection` (`:162-172`) and `_execute` (`:279-303`). The node is already appended to `state.nodes` at `:162` and `:279`, so `_save()` persists a complete state and the JSONL append follows. A crash now loses at most one ledger line instead of duplicating one; ask D to de-duplicate by `iteration` when rendering, so either failure mode reads correctly.
- **Tests:** `test_state_is_saved_before_the_iteration_is_recorded` — monkeypatch `ResearchAudit.record_iteration` to raise; assert the node is already in `state.json`.
- **Acceptance:** in both methods `self._save()` precedes `self.audit.record_iteration(`.

### T10 · I-1, I-4, I-3, I-7 · The wiring you owe B, C, D and E   (M, ~1.0 h)

- **Why:** cross-cutting one-liners go through the loop owner. Four of them, all in your files.
- **Where:** `research_controller.py:429-441` (gate), `:459` (reports), `:104-107` (run start), `:233-240` (debug briefs); `policy.py:8, 15-20, 27-64`.
- **Do:**
  1. **I-1 gate.** Convert `:429-436` to keyword arguments (`run_dir=`, `node_dir=`, `data_dir=`, `kit_dir=`) and wrap it in `try/except Exception` writing `summary["gate"] = {"status": "error", "submission_path": None, "details": {"error": str(exc)}}`. Fix a live bug while you are there: `:431` passes `Path(self.state.best_candidate_dir)`, but `candidate_dir` is stored **repo-relative** at `:262`, so it resolves against the process cwd — `_replication` already gets this right at `:309`. Use `node_dir = REPO_ROOT / self.state.best_candidate_dir if self.state.best_candidate_dir else self.run_dir`. Wrap `render_reports(self.run_dir)` at `:459` the same way, logging to `research_memory.jsonl`.
  2. **I-4 data card.** In `__init__`, inside the new-run branch after `:107`: `card = render_data_card(self.data_dir)`; if `card.strip()`, write `<run_dir>/DATA_CARD.md` and set `state.data_card_path`; if empty, skip silently. If `config.get("data_card_path")` is already set (`configs/ranking_losses.json:10`), use it and do not call the renderer.
  3. **I-3 failure class.** Add `DEBUG_BRIEFS: dict[str, str]` keyed by B's six values (`"timeout"`, `"crash"`, `"bad_output"`, `"low_score"`, `"leak"`, `"missing_test_scores"`) — e.g. `"timeout"` → *"The run exceeded its time budget. Reduce epochs or batch work; do not change the hypothesis."*, `"missing_test_scores"` → *"CandidateOutput.test_scores was absent. Return test scores for context.test_x in the same row order."* At `:238` pass `starting_error=f"{DEBUG_BRIEFS.get(outcome.failure_class, '')}\n{outcome.error}".strip()`; it reaches the Debugger through `roles.debug(..., error=…)` (`roles.py:180-207`) with **no edit to roles.py**. Retry vs skip: `"leak"` skips repair entirely (record `status="failed"`), `"timeout"` gets at most one repair, `failure_class is None` keeps today's behaviour.
  4. **I-7 registry.** `policy.py:8` → `FAMILIES = families.family_names()`. In `sanitize_parameters`, look the family up in `families.FAMILIES`, raise `ValueError(f"Unsupported family: {family}")` if absent, then fill missing keys from `defaults = dict(getattr(entry, "defaults", {}) or {})` and read `grid = dict(getattr(entry, "grid", {}) or {})`: for each key the grid names, require membership (`in` works for E's `tuple` and `range` values); for keys it does not name, keep today's bound checks (`k == 16` stays pinned — the kit measured a k-sweep as a dead end, `README.en.md:133-139`). Raw keys that are neither shared nor in the grid are dropped, not fatal. Point `coverage_complete` and `required_family` (`policy.py:11-24`, used at `:98-99`) at `families.coverage_families()` — the minimum coverage set E keeps at `{"bpr", "group_softmax"}` — so every family E adds does not make the harness stop rule unsatisfiable (the interaction T4 flags). Also make `required_family` deterministic: `sorted(missing)[0]` at `:20`.
- **Interface (verbatim):** **I-1** … **A** converts the call to keyword arguments and wraps it so an exception becomes `GateResult(status="error", details={"error": ...})` written into `summary["gate"]` instead of losing `summary.json`. · **I-4** … **A** wires it at run start: write the string to `<run_dir>/DATA_CARD.md` and set `RunState.data_card_path` to that path (skip silently if the string is empty). · **I-3** … **A** uses it to pick the Debugger brief and to decide retry vs. skip. · **I-7** … **A** makes `policy.py::sanitize_parameters` read the family's `grid` from the registry instead of its hard-coded checks, and **A** replaces `policy.py`'s literal family set with `families.family_names()`.
- **Tests:** `test_gate_failure_does_not_lose_the_summary` · `test_gate_is_called_with_keyword_arguments` (recorder taking `**kwargs` only; asserts the four names and an absolute `node_dir`) · `test_data_card_is_written_and_skipped_when_empty` · `test_debug_brief_follows_the_failure_class` (fake executor with `failure_class="timeout"`; the recorded `passes/*_debugger_*.json` prompt contains the brief) · `test_sanitize_parameters_uses_the_registry_grid` (monkeypatched `Family` with `grid={"batch_size": [256]}`; 256 accepted, 512 rejected; `policy.FAMILIES == families.family_names()`; `coverage_complete` satisfied by `families.coverage_families()` even with a third family registered).
- **Acceptance:** all five pass against the **current stubs** (`run_gate` → `not_implemented`, `render_data_card` → `""`), so nothing here waits on another owner. **Blocks:** B (gate wired), D (data card, journal), E (new families accepted without a `policy.py` edit).

### T11 · Run operator — personal run dirs, live runs, the committed final run   (~1 h attended)

- **Do:** (1) Personal run dirs (rule 5): add one optional config key `run_id_prefix` (default `""`) used at `research_controller.py:91` and `controller.py:53`, so `run_id = f"{prefix}{timestamp}_research"`. Each operator copies `configs/ranking_losses.json` to `configs/run_<initials>.json` (rule 4: add files, don't edit shared ones). Ask D for the `.gitignore` lines that ignore personal run directories (`runs/*_research/`, which covers `runs/<initials>_<timestamp>_research/`) while leaving the final run committable — D's T3 step 1 also opens `runs/final/stdout/` and `generated_experiments/final/`, and documents the `git add -f` route for a run id that is not literally `final`. (2) Before each live run: full suite green · `OPENAI_API_KEY` set · disk free · `git rev-parse HEAD` posted in the team thread · the baseline gate passes without adopting a stale summary (`baseline_selection.json`). (3) During: watch `research_memory.jsonl` for `role_retry` and `controller_error` lines; record any restart, Ctrl-C or config touch immediately with `intervene --reason "…"` — an unrecorded intervention is worse than a recorded one. (4) Final run: `git add` the run directory **by name** (never `git add -A`), including `stdout/` and `generated_experiments/` per D's I16 (`git add -f` for paths D's `.gitignore` still excludes); before committing, `grep -rn '/Users/\|C:\\\\' runs/<id>` must return nothing, and no `.env`.
- **Acceptance:** one committed final run with `source_manifest.json`, `summary.json` (both convergence verdicts), `interventions.jsonl`, `resources.json`, `journal.md`, `results.md`, and no absolute paths.

## 5. Definition of done (whole plan)

- [ ] `python -m pytest -q -W error` green with `OPENAI_API_KEY` unset (47 baseline tests + the new ones).
- [ ] Every task's acceptance criteria ticked; each PR ≤ 300 lines and merged; the config freeze PR (T3) announced separately.
- [ ] Hand-offs delivered: C's `llm.max_retries: 5` line merged in the T3 config PR, the two summary keys to D (T4), the `interventions.jsonl` rename to D (T7), the wiring note to B and E (T10).
- [ ] The final run committed with its journal, results, resources and interventions artefacts.

## 6. Hand-offs

**You provide:** I-8 the `intervene` command, `interventions.jsonl` and the derived count → D (T7) · I-9 `summary["converged_official"]` + `["converged_official_iteration"]` alongside `stop_reason` → D (T4) · `failure_class` and `test_scores_path` in every `iterations.jsonl` record (they ride `outcome.to_dict()` at `research_controller.py:297`, no extra work) → D · I-1 the contained keyword-argument `run_gate` call (`run_dir=`, `node_dir=`, `data_dir=`, `kit_dir=`) with an absolute `node_dir` → B (T10) · I-4 `RunState.data_card_path` and `<run_dir>/DATA_CARD.md`, which C reads in `roles.py` (T10) · I-7 `policy.py` reading `families.family_names()`, `families.coverage_families()` and the family `grid`/`defaults` → E (T10) · the one config line C asks for, `llm.max_retries: 5` in `configs/ranking_losses.json:21` → C (T3), plus the two answers C's README needs: the I5 iteration definition and the I6/I-9 convergence definition, and confirmation that `configs/offline_smoke.json` still stops after one scored iteration once T3 splits the knobs (`max_proposals` defaults to `max_iterations * 2`, so 1 → 2 there) · the C4 non-fatal path, so any owner's new failure mode is a rejection, not a dead run (T1).

**You consume:**
- **C** — `TokenBudgetExceeded`, `RoleOutputInvalid`, `IncompleteResponse` and `LLMError` raised from `llm.py`/`roles.py`, all imported from the frozen `src.agent.errors` (C's T2); the typed raise at `roles.py:52, 63` is C's own step. *Fallback:* the message check is permanent, so this never blocks you.
- **C** — an optional `feedback: str | None = None` keyword on `ResearchRoles.research()`, `critic_preflight()` and `build()` that appends `PREVIOUS ATTEMPT REJECTED: <error>` **after** the volatile state, so prompt caching is unaffected (C's T3). *Fallback:* `_role_call` re-samples the same role with no feedback — still bounded and non-fatal, just less likely to succeed on attempt 2.
- **B** — the six `failure_class` values in `candidate_runner.py` (I-3), `test_scores_path` on `ExperimentOutcome` (I-2), and the I11 two-sided baseline gate (`official.py::within_baseline_tolerance`) as a ≤20-line PR into `research_controller.py:55, 62`. *Fallback:* `failure_class is None` keeps today's behaviour; merge B's gate PR after T2 so B rebases once.
- **D** — `render_data_card` (I-4), `render_reports` (I-5), and the `.gitignore` lines for personal run dirs plus the final-run exception (D's T3 step 1). *Fallback:* the stubs return `""`/`None` and every T10 criterion is written against the stubs.
- **E** — `Family.grid` / `Family.defaults` on the registry entries, `families.family_names()` and `families.coverage_families()` (E's T3), which T10 step 4 reads. *Fallback:* `getattr(entry, "grid", {})` keeps today's hard-coded bounds until E's fields land, and `coverage_families()` falls back to the literal pair.

**Notes from the Step 0 review assigned to you** — both are task steps above: `policy.py`'s literal family set → `families.family_names()` (T10 step 4; E trims the `families.py` docstring themselves in their T3 step 4, so send no PR), and `max_iterations: 50` overloading three semantics (T3).

## 7. Rules

1. **One owner per file.** Need a change in someone else's file? Ask the owner, or send them a ≤20-line PR they merge.
2. **New tests in new files.** Never edit another owner's test file.
3. **Rebase on `main` twice a day; PRs ≤ 300 lines; `pytest` green before merge.** Small, frequent merges beat one big one.
4. **Config: add files, don't edit shared ones** (`configs/offline_smoke.json`, `configs/features_run.json`).
5. **Run directories are personal** (`runs/<initials>_…`, gitignored) until the final run, which A commits.
6. **Shared surfaces only move in the freeze PR** — `types.py`, `contracts.py`, `configs/ranking_losses.json`. A second freeze PR (by A) is the way to change them again, not five drive-by edits.

Plus: never run `git add -A`; never commit `runs/` except the final run (A only); never commit `.env`; PR ≤ 300 lines; rebase on `main` twice a day. And note the resume caveat: the harness refuses to resume a run whose `src/**/*.py` manifest changed (`controller.py:26-36`, checked at `research_controller.py:114-116`), so every merged PR invalidates in-flight `--resume` until the tree settles.

## 8. Daily checkpoints

**Day 1 end.** T1–T3 merged (unkillable loop, baseline selection, split caps) plus T10's gate and data-card wiring against the stubs. The offline e2e run is green on `main` — C's scripted provider + B's gate + your loop — with `stop_reason` never `controller_error`.

**Day 2 end.** T4–T9 merged: `summary.json` carries both convergence verdicts and a real intervention count, and `intervene` works end to end. C's `feedback` keyword and B's `failure_class` consumed if they landed. The first **live run** started overnight on `configs/run_<initials>.json`, with the head commit and run id posted in the team thread.

**Day 3.** Fix what the live run exposed, same day, in small PRs. Start the final run early enough to converge inside 6 h. Commit it with full logs and no absolute paths, and hand D the run id for the journal and Devpost.
