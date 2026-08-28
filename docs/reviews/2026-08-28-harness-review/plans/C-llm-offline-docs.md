# Owner C — LLM layer, offline mode, docs
Branch: `feat/llm-offline-docs`   ·   Base: `main` after the Step 0 merge (`cbf8330` step 0 + `553095d` step 0b, the branch head)   ·   Estimated effort: ~5 h

## 1. Mission

Make the LLM layer survive a 6-hour unattended run, and make the whole loop runnable offline, for free, in under a minute. Typed SDK retries and typed harness exceptions are what A's C4 recovery classifier catches (Technical/robustness, ~10%); a scripted end-to-end run through the *real* training subprocess is the only test that covers `run_candidate.py` at all, and it unblocks everyone's Day-1 integration. Then write the deliverables the organizers grade directly: README (setup, reproduce, limitations, contributions, diagram) and `docs/devpost.md` — Presentation is 10%, and Feasibility (15%) is scored from the numbers you report.

## 2. Files you own (exclusive) / files you must not touch

**Own:** `src/agent/llm.py`, `src/agent/roles.py`, `tests/test_openai_runtime.py`, `tests/test_research_runtime.py`, new `tests/test_llm_retry.py`, new `configs/offline_smoke.json`, new `tests/fixtures/offline_smoke_script.json`, `README.md`, `AGENTS.md`, `PLAN.md`, new `docs/devpost.md`.

**Must not touch:** A's `research_controller.py`, `policy.py`, `convergence.py`, `controller.py`, `configs/baseline.json`, `tests/test_research_loop.py`, `tests/test_agent.py` · B's `official.py`, `gate.py`, `run_candidate.py`, `run_baseline.py`, `candidate_runner.py`, `runs/` · D's `datacard.py`, `report.py`, `audit.py`, `logger.py`, `.gitignore` · E's `safety.py`, `families.py`, `src/models/*`, `research/methods/*.md` · the frozen shared surfaces `src/agent/types.py`, `src/experiments/contracts.py`, `configs/ranking_losses.json`, `src/agent/errors.py` (import the exception classes from it; never add a class to it, and never define one in `llm.py` either) and the shared `tests/test_interfaces.py` · the unowned `src/agent/catalog.py` (work around it from `roles.py`) and `requirements.txt`. Need a change in someone else's file? Ask the owner, or send a ≤20-line PR they merge — you have exactly one such ask (T2 step 8), listed in §6.

## 3. Setup (15 minutes)

```bash
cd ~/Desktop/personal/techjam-2026-SeniorPromptEngineers
git checkout main && git pull                    # must contain cbf8330 + 553095d
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt        # numpy, openai>=1.68, python-dotenv
python -m pip install pytest                     # test runner only; do NOT add it to requirements.txt
unset OPENAI_API_KEY
python -m pytest -q -W error                     # expect: 47 passed
python -m pytest -q -W error -m "not slow"       # the quick loop: 46 passed, 1 deselected
git checkout -b feat/llm-offline-docs
```

Verified on this tree: `pytest -q -W error` → 47 passed (28 existing + 19 added by Step 0/0b). Outside the venv, `tests/test_openai_runtime.py` errors on the missing `openai`/`dotenv` imports and `tests/test_errors.py` on the missing `pytest` import — those are absent packages, not defects. You never need `OPENAI_API_KEY`; every task below runs offline. You do need the dataset extracted at `data/KuaiRand-Pure/data/` for T1's end-to-end test.

## 4. Tasks, in order

### T1 · I14 / I-6 · Scripted provider, end to end through the real trainer   (effort L, ~1.5 h)

- **Why:** `ScriptedProvider` exists (`llm.py:301-322`) but nothing reaches it from a config, and the real generate→train→evaluate path (`run_candidate.py:100-138`) has **zero** coverage — `test_research_loop.py:92-104` fakes the executor. The Day-1 milestone ("offline e2e green on `main`") is this task.
- **Where:** new `configs/offline_smoke.json`, new `tests/fixtures/offline_smoke_script.json`; `llm.py:325-336` (`build_provider`), `llm.py:313` (`payload.pop`); consumers `candidate_runner.py:64-81, 83-155` and `run_candidate.py:100-138`; shapes to copy from `tests/test_research_loop.py:16-89, 127-157`.
- **Do:**
  1. `configs/offline_smoke.json` — copy `configs/ranking_losses.json`, then set `"name": "kuairand-pure-offline-smoke"`; `"llm": {"provider": "scripted", "script_path": "tests/fixtures/offline_smoke_script.json", "max_total_tokens": 100000}`; `"budgets": {"max_iterations": 1, "max_training_attempts": 1, "max_wall_clock_seconds": 900, "experiment_timeout_seconds": 300, "test_timeout_seconds": 60, "max_debug_repairs": 2}`. `max_iterations: 1` is load-bearing: the loop stops at `research_controller.py:359` after one scored iteration, so it never pops a fifth scripted response.
  2. Write `candidate.py` by hand from `research/methods/bpr.md:15-33`, `src/models/fm_core.py:33-89` and `src/models/sampling.py:24-44`. This exact body is measured on the real data — 3 epochs, primary **0.6023** (GAUC 0.6686 / nDCG@5 0.5361), **14 s** including 6 s of load+encode:

     ```python
     import numpy as np
     from src.experiments.contracts import CandidateOutput
     from src.models.fm_core import FMRanker, sigmoid
     from src.models.sampling import sample_bpr_pairs

     def run(context, parameters):
         seed = int(parameters.get("seed", 0))
         batch_size = int(parameters.get("batch_size", 4096))
         per_positive = int(parameters.get("negatives_per_positive", 1))
         model = FMRanker(context.field_dimension, embedding_dim=int(parameters.get("k", 16)),
                          learning_rate=float(parameters.get("learning_rate", 0.001)), seed=seed)
         rng = np.random.default_rng(seed)
         trace, best_primary, best_state = [], -1.0, model.state_dict()
         for epoch in range(1, int(parameters.get("epochs", 3)) + 1):
             positive_rows, negative_rows = sample_bpr_pairs(context.train_users, context.train_y, rng, per_positive)
             order = rng.permutation(len(positive_rows))
             for start in range(0, len(order), batch_size):
                 batch = order[start : start + batch_size]
                 positive_x = context.train_x[positive_rows[batch]]
                 negative_x = context.train_x[negative_rows[batch]]
                 difference = model.logits(positive_x)[0] - model.logits(negative_x)[0]
                 gradient = ((sigmoid(difference) - 1.0) / len(batch)).astype(np.float32)
                 grad_v_p, grad_w_p, _ = model.gradients(positive_x, gradient)
                 grad_v_n, grad_w_n, _ = model.gradients(negative_x, -gradient)
                 model.apply_gradients(grad_v_p + grad_v_n, grad_w_p + grad_w_n, 0.0)
             metrics = context.evaluate_validation(model.predict(context.valid_x))
             trace.append({"epoch": epoch, "primary": float(metrics["primary"])})
             if float(metrics["primary"]) > best_primary:
                 best_primary, best_state = float(metrics["primary"]), model.state_dict()
         model.load_state_dict(best_state)
         return CandidateOutput(
             validation_scores=model.predict(context.valid_x), checkpoint_state=best_state,
             training_trace=trace, diagnostics={"pairs": int(len(positive_rows)), "best_primary": best_primary},
             test_scores=None if context.test_x is None else model.predict(context.test_x))
     ```

     `dL/dd = sigmoid(d) − 1` is the method card's own derivative (`bpr.md:24`); the bias gradient is 0.0 because it cancels in a score difference (`bpr.md:39`). The guarded `test_scores` line makes the same fixture correct before *and* after B lands `test_x` — both fields are already frozen (`contracts.py:18, 27`).
  3. Write its `test_candidate.py`: one test that `sample_bpr_pairs` over a hand-built `("u1","u1","u2","u2")` / `[1,0,1,0]` array returns same-user, opposite-label pairs, and one that `candidate.run` is callable. Imports limited to `unittest`, `numpy`, `candidate`, `src.models.sampling` (`safety.py:8-19`), and no dataset access — `roles.py:164-165` requires exactly that.
  4. Build the fixture JSON once from a shell heredoc (let `json.dump` escape the sources; never hand-escape): four payloads in call order — `research_decision`, `critic_decision`, `candidate_manifest`, `critic_decision` — plus a fifth `debug_decision`, each carrying `"_usage": {"input_tokens": …, "output_tokens": …, "total_tokens": …}`. Parameters everywhere: `seed 0, k 16, learning_rate 0.001, epochs 3, batch_size 4096, patience 2, negatives_per_positive 1, negatives_per_group null, temperature null` — all inside `policy.sanitize_parameters` (`policy.py:36-52`). Set `"needs_web_search": false` with one non-empty evidence entry, or `roles.research` makes a second call (`roles.py:113-123`) and the script desynchronises. The fifth (Debugger) payload is **not** consumed on the happy path — `ScriptedProvider` is positional, not role-keyed — so note that in the file and cover it with a unit test instead.
  5. Fix Minor **M9** while you are here: `llm.py:313`'s `payload.pop("_usage", …)` mutates the stored dict, so a re-read of the same entry loses its usage. Use `payload.get("_usage", …)` plus a comprehension dropping `_usage`.
- **Interface:** *I-6 Provider factory* — `src/agent/llm.py::build_provider(config) -> LLMProvider` (frozen). **C** owns it and adds `configs/offline_smoke.json` (`"llm": {"provider": "scripted", "script_path": "tests/fixtures/<...>.json"}`). The scripted fixture must carry a complete, valid `candidate.py` so the real training subprocess runs.
- **Tests** (new `tests/test_llm_retry.py`, class `OfflineSmokeTests`):
  - `test_fixture_candidate_passes_the_safety_validators` — `validate_source(code)`, `validate_source(tests, test_file=True)`, `validate_family_contract(code, "bpr")`. Fast, no dataset.
  - `test_fixture_debug_payload_parses_as_a_debug_decision` — `DebugDecision.from_dict(script[4])` plus `validate_source` on its `replacement_code`.
  - `test_offline_smoke_config_points_at_the_committed_fixture` — the config's `script_path` resolves and `build_provider(config)` returns a `ScriptedProvider` holding 5 responses.
  - `test_scripted_loop_scores_one_real_bpr_iteration` — **the e2e**, marked `@pytest.mark.slow`. The marker is registered in the committed `pytest.ini` (Step 0b: *"slow: end-to-end tests that run the real training subprocess"*), so it is not an unregistered-mark error under `-W error`; `tests/test_errors.py:32` already uses it. Everyone's quick loop is `pytest -q -W error -m "not slow"`, and the full `pytest -q -W error` runs this one too. Add the same dataset guard B and D use — `skipUnless(<data_dir>/log_standard_4_08_to_4_21_pure.csv is a file)` — because D untracks `data/` on Day 2 and a fresh clone runs `scripts/download_data.sh` first. Build a config in a `TemporaryDirectory` with its own `run_root`/`generated_root`, then `ResearchLoop(config, config_path, provider=ScriptedProvider(script), baseline_summary={… "primary": 0.6016 …})` — passing `baseline_summary` skips `_ensure_baseline` (`research_controller.py:87`). Assert: `summary["stop_reason"] != "controller_error"` (print `error.json` in the message the way `test_research_loop.py:160-165` does); `summary["training_attempts"] == 1`; the node's `status == "success"`; `0.47 <= metrics["primary"] <= 0.80`; `outcome["failure_class"] is None`; `len(provider.calls) == 4` with roles `["researcher", "critic_preflight", "builder", "critic_postflight"]`. If `test_scores.npy` exists in the node directory (B's I-2), assert its length is 170,588.
- **Acceptance criteria:**
  - [ ] `python -m pytest -q -W error -m "not slow"` → 46 + your new fast tests, all green, `OPENAI_API_KEY` unset.
  - [ ] `python -m pytest -q -W error -m slow tests/test_llm_retry.py` passes in < 60 s.
  - [ ] `python -m src.agent.controller --config configs/offline_smoke.json` ends with `best.metrics.primary` ≈ 0.602 and `"stop_reason"` either `"iteration_budget_reached"` (today) or `"training_attempt_budget_reached"` (after A's I5 splits the knobs — `max_training_attempts: 1` then owns `research_controller.py:359`). Confirm which with A when their T3 merges, then pin the one value. (The first invocation also reproduces the baseline ladder, ~21 s, unless a passing baseline run already exists.)
  - [ ] `git diff --stat` touches only `configs/offline_smoke.json`, `tests/fixtures/…`, `tests/test_llm_retry.py`, `src/agent/llm.py`.
- **Depends on / blocks:** blocks nobody, unblocks everybody — announce the moment it merges. If `failure_class == "missing_test_scores"` appears after B merges, `run_candidate.py` is not passing `test_x` into the context: ping B rather than editing the fixture.

### T2 · I4 / C4 · Typed retries and typed exceptions   (effort M, ~1.25 h)

- **Why:** `llm.py:211` disables SDK retries, `:269-272` allows 2 attempts at 0.5 s → 1 s and ignores `Retry-After`, and `:214-221` matches exceptions by **class-name string**. A 429 burst or a 30-second 503 window ends the run through C4. And A's classifier cannot key on `if "token budget" in str(exc).lower()` (`research_controller.py:402`) forever — `roles.py:52, 63` only happen to contain that phrase.
- **Where:** `llm.py:193-298`, specifically `:209, :211, :213-221, :262-276`; `roles.py:51-52, 62-63` (budget) and `roles.py:111, 175, 207` (consistency failures).
- **Do:**
  1. Module constants in `llm.py`: `RETRY_ATTEMPTS = 5`, `BACKOFF_INITIAL_SECONDS = 2.0`, `BACKOFF_MAX_SECONDS = 60.0`, `RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})`.
  2. Typed exceptions: **do not define any.** They are frozen in `src/agent/errors.py` (Step 0b, nobody owns it) — `LLMError(RuntimeError)` and its subclasses `TokenBudgetExceeded`, `RoleOutputInvalid`, `IncompleteResponse`. Add `from .errors import IncompleteResponse, LLMError, RoleOutputInvalid, TokenBudgetExceeded` to `llm.py` and `roles.py` and raise them from there: `RoleOutputInvalid` for schema / JSON / cross-field consistency, `IncompleteResponse` for `status == "incomplete"` or an empty `output_text`, `TokenBudgetExceeded` for the budget stop. A imports the same four names from `src.agent.errors` and only catches them.
  3. Resolve the SDK types lazily in `__init__`, keeping the guarded import at `:199-202` so `llm.py` still imports without the SDK: store `self._retryable = (APIConnectionError, APITimeoutError, RateLimitError)` and `self._status_error = APIStatusError`. Delete the class-name set at `:216-221`.
  4. Replace `_retryable` with `_retry_delay(self, error, attempt) -> float | None`, where `None` means fatal. Retry when `isinstance(error, self._retryable)`, or `isinstance(error, self._status_error)` and `error.status_code in RETRYABLE_STATUS_CODES or error.status_code >= 500`. Delay = `min(BACKOFF_MAX_SECONDS, BACKOFF_INITIAL_SECONDS * 2 ** attempt)` → 2, 4, 8, 16 s. Honour `Retry-After` from `error.response.headers.get("retry-after")` when it parses as a float, clamped to `BACKOFF_MAX_SECONDS`; ignore an unparseable (HTTP-date) value.
  5. `self.max_retries = int(config.get("max_retries", RETRY_ATTEMPTS))` at `:209`; the loop at `:264-272` runs at most `self.max_retries` attempts and re-raises the last error unchanged once they are gone.
  6. After the call read `status = str(getattr(response, "status", "") or "")` and `text = getattr(response, "output_text", "") or ""`. If `status == "incomplete"` or `not text`, count it as a retryable attempt (include `getattr(response, "incomplete_details", None)` in the message); when the attempts are exhausted raise `IncompleteResponse` in place of the bare `ValueError` at `:276`. Wrap `json.loads(text)` so a `JSONDecodeError` becomes `RoleOutputInvalid`.
  7. In `roles.py` (same import from `.errors`): `TokenBudgetExceeded` at `:52` and `:63`; `RoleOutputInvalid` at `:111`, `:175`, `:207`. Keep the message strings unchanged so A's current text check still works during the overlap. This is the conversion A's T8 is waiting on — it is your step, not their PR.
  8. Ask A for one line in `configs/ranking_losses.json:21` → `"max_retries": 5` (A carries it in their T3 config freeze PR).
- **Interface:** provided to A's C4 from the frozen module: `src.agent.errors.TokenBudgetExceeded`, `src.agent.errors.RoleOutputInvalid`, `src.agent.errors.LLMError`, `src.agent.errors.IncompleteResponse` — C raises them from `llm.py`/`roles.py`, A catches them, neither side declares them anywhere else. `TokenBudgetExceeded` maps to `stop_reason="llm_token_budget_reached"`; `RoleOutputInvalid` and `IncompleteResponse` are re-promptable (A: ≤2 attempts, then `continue`).
- **Tests** (`tests/test_llm_retry.py`, class `RetryPolicyTests`; a fake `responses.create`, `patch("src.agent.llm.time.sleep")` recording delays, and real SDK errors built from `httpx.Response(429, headers={"retry-after": "7"}, request=httpx.Request("POST", "https://api.openai.com/v1/responses"))` — if a constructor signature differs in the installed SDK, build the instance however that SDK's own tests do; the assertion is about your retry policy):
  - `test_rate_limit_is_retried_with_exponential_backoff` — two failures then success; `result.retries == 2`, sleeps `[2.0, 4.0]`. · `test_retry_after_header_wins_over_the_backoff` — the recorded sleep is `7.0`.
  - `test_five_attempts_then_the_error_propagates` — 5 `create` calls, original exception type re-raised. · `test_client_error_is_not_retried` — a 400 `APIStatusError` → exactly 1 call.
  - `test_incomplete_response_raises_a_typed_error` — `status="incomplete"`, empty `output_text` → `IncompleteResponse`, and `isinstance(exc, LLMError)`. · `test_unparseable_output_text_raises_role_output_invalid`.
  - `test_token_budget_uses_a_typed_exception` — `ResearchRoles(..., max_total_tokens=0)` + `ScriptedProvider` → `TokenBudgetExceeded`. · `test_family_mismatch_raises_role_output_invalid` — `roles.research(state, 1, "bpr")` against a scripted `group_softmax` decision.
- **Acceptance criteria:**
  - [ ] `grep -n "__class__.__name__" src/agent/llm.py` → no hits.
  - [ ] `python -m pytest -q -W error` green with `OPENAI_API_KEY` unset.
  - [ ] `python -c "from src.agent.errors import TokenBudgetExceeded, RoleOutputInvalid, LLMError, IncompleteResponse"` exits 0, and `grep -n "class .*Error\|class .*Exceeded\|class .*Invalid\|class .*Response" src/agent/llm.py src/agent/roles.py` → no hits.
- **Depends on / blocks:** **blocks A's C4.** Merge before A starts classifying exceptions; if A is ahead of you, tell them to catch `RuntimeError` temporarily and narrow it after your merge.

### T3 · Prompt structure — cache prefix, I-4 data card, I-2 `test_scores`, I-7 registry   (effort M, ~0.75 h)

- **Why:** `prompt_cache_key` is set per role (`llm.py:251`) but the volatile state summary is inlined *before* the method cards (`roles.py:102-106`), so the cached prefix changes every call and never hits (spec-compliance §9, "Prompt caching | partial"). Nothing tells the Researcher about the data (`roles.py:66-87`, I15). The Builder is never told to return `test_scores` (I-2). The sampler names are hard-coded (`roles.py:166-167`), which forces E to edit your file.
- **Where:** `roles.py:66-87` (`_state_summary`), `:97-107`, `:141-148`, `:155-171`, `:188-196`, `:218-226`; `llm.py:52` and `:103` (`"enum": ["bpr", "group_softmax"]` in two schemas).
- **Do:**
  1. Add `_stable_prefix(self, state, family) -> str` returning, in a fixed order: the immutable task text, the candidate-contract text (lift the invariant sentences out of the Builder prompt into a module constant), `self.catalog.prompt_text(card_key)`, and the data card. Every role prompt becomes `_stable_prefix(...) + "\n\n" + <volatile block>`; the volatile block keeps the state summary, the proposal/metrics/error JSON and the `family_rule` line (`roles.py:92-96`). Render JSON with `sort_keys=True` so a re-ordered dict cannot invalidate the prefix. The prefix must clear ~1k tokens for caching to engage — two method cards plus the contract and data card do.
  2. `_data_card_text(self, state)`: read `state.data_card_path` (frozen at `types.py:241`), return `""` on `None` or `OSError`, and **memoise on the instance** so five role passes per iteration emit byte-identical text.
  3. Builder instructions (`roles.py:155-171`): add "Return `test_scores` — one finite score per row of `context.test_x`, same row order, from the same trained model. Return `test_scores=None` only when `context.test_x` is None." (I-2.)
  4. Import from `.families` in `roles.py`; replace the hard-coded sampler sentences at `:166-167` with `families.builder_brief(decision.family)` — E's renderer (their T3 step 3) returns the mandatory trusted calls plus the family's grid, so a new family never reaches your file. Until E's T3 merges, fall back to `FAMILIES[decision.family].trusted_sampler`. Select the method card by `Path(FAMILIES[name].method_card).stem` as the catalog key (`catalog.py:29-31` keys cards by file stem) instead of assuming the key equals the family name, and pass the selected family to `catalog.prompt_text(...)` rather than `None` — `prompt_text(None)` concatenates every card, which E's two new cards would add ~1.2k tokens to on every Researcher call. (I-7.)
  5. In `llm.py`, build both family enums from `sorted(family_names())` instead of the literals at `:52` and `:103`, so E adding `history_features` needs no edit in your files.
  6. Add an optional `feedback: str | None = None` **keyword** to `ResearchRoles.research()`, `critic_preflight()` and `build()` — A's C4 re-prompt path passes it (their T1 step 3). When it is set, append `PREVIOUS ATTEMPT REJECTED: <error>` as the **last** block of the prompt, after the volatile state, so the stable prefix and the cache hit are unaffected; when it is `None` the prompt is byte-identical to today's. Default `None`, so every existing call site keeps working.
- **Interface:** *I-4 Data card* — `src/evaluation/datacard.py::render_data_card(data_dir: Path) -> str` (Markdown). Provided by **D**. **A** wires it at run start: write the string to `<run_dir>/DATA_CARD.md` and set `RunState.data_card_path` to that path (skip silently if the string is empty). **C** reads `state.data_card_path` in `roles.py` and places the text in the *stable* prompt prefix (before volatile state). *I-2 Test predictions* (C's half) — **C** updates the Builder prompt/schema instructions in `roles.py` so generated `candidate.py` returns `test_scores` (until C's change lands, B's worker treats a missing `test_scores` as `failure_class="missing_test_scores"`, not a crash).
- **Tests** (append to `tests/test_research_runtime.py`, which you own):
  - `test_method_cards_precede_the_volatile_state_in_every_prompt` — for each captured prompt, `prompt.index("METHOD CARD") < prompt.index("RESEARCH STATE")`. · `test_stable_prefix_is_identical_across_two_iterations` — two research calls with different node lists share the prefix up to the volatile marker.
  - `test_data_card_text_is_inserted_in_the_prefix` — a temp Markdown file set as `state.data_card_path` appears before `RESEARCH STATE`; with `data_card_path=None` the prompt still builds. · `test_builder_prompt_requires_test_scores` — `"test_scores"` appears in the Builder prompt.
  - `test_builder_prompt_names_the_sampler_from_the_registry` — for every family in `FAMILIES` the prompt contains `family.trusted_sampler`. · `test_schema_family_enum_follows_the_registry` — `SCHEMAS["research_decision"]…["enum"] == sorted(family_names())`.
  - `test_feedback_keyword_appends_after_the_volatile_state` — `research(state, 1, "bpr", feedback="…")` puts `PREVIOUS ATTEMPT REJECTED` last and leaves the prefix byte-identical to the `feedback=None` call.
- **Acceptance criteria:**
  - [ ] `grep -n "sample_bpr_pairs\|sample_softmax_groups\|\"bpr\"" src/agent/roles.py src/agent/llm.py` → no hard-coded family strings remain.
  - [ ] `python -m pytest -q -W error -m "not slow"` green; `python -m pytest -q -W error -m slow` still green (prompts changed, fixture responses did not — `ScriptedProvider` ignores prompt text).
- **Depends on / blocks:** the data-card read is inert until D's I15 and A's wiring land — test it with a temp file so your criterion never waits on them. Step 4 prefers E's `families.builder_brief(name)` (their T3, Day 1) and falls back to `trusted_sampler` if it has not merged, so it never blocks you either. Tell E when steps 4–5 merge, and A when step 6 does; that is what keeps `roles.py` off their branches.

### T4 · Step 0 hand-off · `build_provider` coverage   (effort S, ~0.25 h)

- **Why:** the Step 0 review note assigned to C (team-split.md:50) — `build_provider`'s scripted branch is tested only with an empty `[]` payload (`tests/test_interfaces.py:106-114`) and the repo-relative `script_path` branch (`llm.py:332-335`) has no test at all.
- **Where:** `llm.py:331-335`. Add the cases to `tests/test_llm_retry.py`; do **not** edit `tests/test_interfaces.py`, it is shared (rule 2).
- **Tests:** `test_scripted_provider_round_trips_a_non_empty_script` — two payloads with `_usage`; the first `complete(role="critic", …)` returns payload 1's data with matching `usage.total_tokens`, the second returns payload 2. `test_relative_script_path_resolves_against_the_repo_root` — `script_path="tests/fixtures/offline_smoke_script.json"` with no leading slash loads 5 responses.
- **Acceptance criteria:** [ ] both pass from any working directory (`cd /tmp && python -m pytest -q -W error <repo>/tests/test_llm_retry.py`).
- **Depends on:** T1's fixture file must exist first.

### T5 · I17 · README, `docs/devpost.md`, AGENTS.md, PLAN.md   (effort M, ~1.25 h)

- **Why:** `problem_statement.md:262-270` names README contents the repo lacks; `:252-260` names a Devpost description that does not exist. `README.md:29, 46, 73, 83, 104` are PowerShell-only on a POSIX-graded repo, and `README.md:110-111` / `AGENTS.md:39-41` build a guardrail around `data/judge/**`, **a directory that does not exist** (correctness §M1) — the real hidden-test surface is the 20220429–20220508 date range inside `log_standard_4_22_to_5_08_pure.csv`.
- **Where:** all of `README.md`; `AGENTS.md:30-43`; `PLAN.md:5, 34, 47, 73-79, 147-155, 170`; new `docs/devpost.md`.
- **Do:**
  1. **README**, in this order: overview · setup (venv, `pip install -r requirements.txt`, dataset download — reference D's download script by name once it lands) · run / resume / intervene, POSIX fence first and PowerShell second, for `configs/baseline.json`, `configs/ranking_losses.json`, `configs/offline_smoke.json`, `--resume runs/<id>` and A's `intervene --reason` (I-8) · how to reproduce the final numbers (which run directory, `python -m src.agent.report <run_dir>`, B's `submission.csv` path) · results table with the validation delta vs 0.6016 · resource accounting (tokens, wall-clock, iterations, GPU-hours = 0, manual interventions) · **iteration and convergence definitions** · limitations and what we would do with more time · team contributions (A–E, one line each) · architecture diagram. Keep `python -m pytest -q` beside the existing `python -m unittest discover -s tests -v` (`README.md:73`); both work. Every number traces to a committed `summary.json` / `resources.json`; `README.md:155-166`'s table and its "~212 seconds" come from the old Windows run (C5), so replace them with B's regenerated run and its real wall-clock (21 s on the review machine).
  2. The definitions are A's (I5 splits the three `max_iterations` semantics; I6 separates the official ε=0.002 / N=3 verdict from the harness stop rule). Write them as clearly marked placeholders — `<!-- A/I5: fill in -->`, `<!-- A/I6: fill in -->` — with today's behaviour written underneath so the section is never blank, and send A those two exact questions.
  3. Architecture diagram: adapt the spec's §2 ASCII block (`docs/superpowers/specs/2026-08-28-autonomous-mle-agent-design.md:51-64`) to the harness's real module names — Steward → `official.py::load_train_valid` + `datacard.py`; Scientist → `roles.research`; *(no spec equivalent)* → `roles.critic_preflight` / `critic_postflight`; Engineer → `roles.build`; Medic → `roles.debug`; Sandbox → `safety.validate_source` + `candidate_runner.CandidateExecutor` + `run_candidate.py`; Scorekeeper → `official.official_evaluate`; Ledger → `audit.ResearchAudit` + `report.render_reports`; Conductor → `research_controller.ResearchLoop`; Gate → `gate.run_gate`.
  4. `docs/devpost.md`, one heading per bullet of `problem_statement.md:256-260`: how the solution addresses the problem statement · development tools · **APIs** — OpenAI Responses API (`gpt-5.5`, structured outputs with `strict: true`, the `web_search` tool, `prompt_cache_key`) · **libraries** — numpy, `openai`, `python-dotenv`, stdlib `csv`/`ast`/`subprocess`/`unittest`, and state plainly that there is no pandas/torch/LightGBM · **datasets** — KuaiRand-Pure plus the organizers' starter kit, no external data. Close with the results table and resource usage.
  5. `AGENTS.md:30-43`: name the real hidden-test surface (the date range, not the phantom directory) and keep the prohibition. When B's gate merges, "A label-free `data/judge/test.csv` may be read only when the user explicitly asks…" must become: the Gate scores the validation-best checkpoint once, automatically, at the end of the run, reading test **metadata** only — it is not a manual intervention. Mirror it at `README.md:110-111`.
  6. `PLAN.md`: `:5` ("The whole agent is **not built yet**") is stale — replace with what exists plus the five-owner split; `:34` "Retry transient API failures twice" → 5 attempts, 2 s → 60 s, honour `Retry-After`; `:47` and `:155` "eight development candidates" → 50 iterations; `:73-79` add `test_scores` to the `CandidateOutput` block; `:147-153` POSIX fences; `:170` note the resume test is A's I3.
- **Tests:** none (documentation). Verify by executing every command block you write, in a clean shell, from the repo root.
- **Acceptance criteria:**
  - [ ] Every fenced command in `README.md` has been run and its output matches what the README claims.
  - [ ] Each PowerShell block has a POSIX sibling above it in `README.md` and `PLAN.md`.
  - [ ] `grep -rn "data/judge" README.md AGENTS.md` → only where the text explains the directory does not exist here.
  - [ ] `docs/devpost.md` has a heading for each of the five bullets at `problem_statement.md:256-260`.
  - [ ] No number in either document is unsourced: each traces to a committed `summary.json` or `resources.json`.
- **Depends on / blocks:** the results table needs B's regenerated run (C5) and A's final run (day 3); the definitions need A's I5/I6; the AGENTS.md sentence needs B's gate. Write everything else on day 2; land the numbers last.

## 5. Definition of done (whole plan)

- [ ] `python -m pytest -q -W error` green with `OPENAI_API_KEY` unset (and `-m "not slow"` for the quick loop); no new warnings.
- [ ] `python -m pytest -q -W error -m slow tests/test_llm_retry.py` green in < 60 s.
- [ ] `python -m src.agent.controller --config configs/offline_smoke.json` produces a scored iteration offline, with no API key.
- [ ] Every task's acceptance checklist ticked; each PR ≤ 300 lines; all merged to `main`.
- [ ] Hand-offs delivered and announced: the typed exceptions raised from `src.agent.errors` and the `feedback=` keyword (A), Builder `test_scores` prompt (B), registry-driven prompts and schema enums (E), offline config + fixture (everyone).
- [ ] README, `docs/devpost.md`, `AGENTS.md` and `PLAN.md` describe the merged tree, every number sourced from a committed run directory.

## 6. Hand-offs

**You provide:**
- The four frozen exceptions actually raised: `src.agent.errors.TokenBudgetExceeded` (`roles.py:52, 63`), `RoleOutputInvalid` (`roles.py:111, 175, 207` and the JSON decode in `llm.py`), `IncompleteResponse` and `LLMError` from `llm.py` — **A's C4 and I13 depend on these.** Tell A the moment T2 merges; it is the highest-priority hand-off in this plan.
- An optional `feedback: str | None = None` keyword on `ResearchRoles.research()`, `critic_preflight()` and `build()` (T3 step 6) — **A's C4 re-prompt path uses it**; announce it with T3.
- `configs/offline_smoke.json` + `tests/fixtures/offline_smoke_script.json` — the free end-to-end path. Announce at the end of Day 1; the sequencing table expects "offline e2e run green on `main`" that evening.
- The Builder prompt requiring `test_scores` (I-2, C's half) — tell **B**, so their worker can move `missing_test_scores` from tolerated to enforced.
- Registry-driven prompts and schema family enums (I-7) — tell **E**; after this they add a family without touching `roles.py` or `llm.py`.

**You consume:**
- **D → I-4** `render_data_card(data_dir) -> str`, written by **A** to `<run_dir>/DATA_CARD.md` with `RunState.data_card_path` set. *Fallback:* the field stays `None`, the prefix omits the block, your test uses a temp file.
- **B → I-2** `CandidateContext.test_x` populated in `run_candidate.py` and `test_scores.npy` persisted per node. *Fallback:* the guarded `test_scores` line is correct either way and the e2e assertion is conditional.
- **A** — one line in `configs/ranking_losses.json:21` → `"max_retries": 5` (their file, rule 1; A carries it in their T3 config freeze PR).
- **E → I-7** `families.builder_brief(name) -> str`, the rendered mandatory calls and grid that replace the hard-coded sentences at `roles.py:166-167` (T3 step 4). *Fallback:* `FAMILIES[name].trusted_sampler`, which is already frozen.
- **A → I5/I6** — the iteration and convergence definitions for the README, and confirmation that `configs/offline_smoke.json` still stops after one iteration once the knobs are split.
- **B → C1** — the regenerated baseline run and the gate's submission path, for the README results table.

**Note from the Step 0 review assigned to you** (team-split.md:50): `build_provider`'s scripted branch is tested with an empty `[]` payload and the repo-relative `script_path` branch has no test — add a non-empty round-trip and a relative-path case when wiring I14. Covered by **T4**, in `tests/test_llm_retry.py`, not in `tests/test_interfaces.py`.

## 7. Rules

1. **One owner per file.** Need a change in someone else's file? Ask the owner, or send them a ≤20-line PR they merge.
2. **New tests in new files.** Never edit another owner's test file.
3. **Rebase on `main` twice a day; PRs ≤ 300 lines; `pytest` green before merge.** Small, frequent merges beat one big one.
4. **Config: add files, don't edit shared ones** (`configs/offline_smoke.json`, `configs/features_run.json`).
5. **Run directories are personal** (`runs/<initials>_…`, gitignored) until the final run, which A commits.
6. **Shared surfaces only move in the freeze PR** — `types.py`, `contracts.py`, `configs/ranking_losses.json`. A second freeze PR (by A) is the way to change them again, not five drive-by edits.

Plus: never run `git add -A`; never commit `runs/` except the final run (A only); never commit `.env`; PR ≤ 300 lines; rebase on `main` twice a day.

## 8. Daily checkpoints

**Day 1** — T1 and T2 merged. `configs/offline_smoke.json` runs a scored BPR iteration offline on `main`, and the typed exceptions are importable and announced to A. The evening's "offline e2e green on `main`" milestone is yours to call.

**Day 2** — T3 and T4 merged (prompt prefix, data-card read, `test_scores` instruction, registry-driven families, provider coverage). T5 drafted: README structure, diagram, `docs/devpost.md`, AGENTS.md and PLAN.md corrections — everything except the numbers, which A's first live run is still producing.

**Day 3** — T5 finished against the final run: results table, deltas vs 0.6016, resource accounting, iteration/convergence definitions filled in by A, limitations, contributions. Re-run every README command in a clean shell before the last merge.
