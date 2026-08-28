# Owner E — Search surface, safety, method cards

Branch: `feat/search-safety` · Base: `main` after the Step 0 merge (`cbf8330` step 0 + `553095d` step 0b, the branch head) · Estimated effort: ≈ 6.5 h core + 1.25 h stretch

## 1. Mission

Close the four verified bypasses of the AST validator (C2) so the isolation story survives a judge reading `safety.py`, then widen the agent's
search space by one honest axis: a `history_features` family whose leakage-sensitive work stays in trusted code (I8 / I-7). C2 defends Technical
Execution (35%, which includes robustness and the credibility of "no hidden-test access"). I8 is where the dataset's headroom is (0.5946 baseline vs
0.8645 oracle) and is the largest lever on Innovation & Problem Insight (20%), because today the agent varies one loss over a ~12-point grid.

## 2. Files you own (exclusive) / files you must not touch

**Own:** `src/agent/safety.py` · `src/agent/families.py` (post-freeze) · `src/models/features.py` · `src/models/sampling.py` ·
`src/models/fm_core.py` · `src/models/baselines.py` · `research/methods/*.md` · `tests/test_safety.py` · `tests/test_sampling.py` · new
`tests/test_features.py`.

**Must not touch:** everything in A's, B's, C's and D's lists in `team-split.md`; the frozen shared surfaces `types.py`, `contracts.py`,
`configs/ranking_losses.json`, `src/agent/errors.py`; and the two files nobody owns, `src/agent/catalog.py` (treat it as A's) and
`tests/test_interfaces.py` — that one pins the Step 0 stubs, and the single sanctioned edit is the owner who fills a stub updating that stub's
assertions in the same PR, ≤ 5 lines. For you that is `test_history_features_are_not_implemented_yet` when T4 lands and the
`frozenset({"bpr", "group_softmax"})` literal in `test_family_names_expose_the_registered_families` when T5 registers a third family; nothing else
in that file. To change any other owner's file: ask the owner or send a ≤20-line PR they merge; every such change here is in §6. Two of your own files need **no** edit: `fm_core.py` (verified
field-count agnostic, T4) and `baselines.py`, whose output must stay bit-identical to the committed baseline (`empirical.md:285-300`).
`tests/test_sampling.py` is ownership only.

## 3. Setup (15 minutes)

```sh
cd /Users/Ke_Jun_YEO_from.TP/Desktop/personal/techjam-2026-SeniorPromptEngineers
git checkout main && git pull            # must contain cbf8330 + 553095d
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt pytest
pytest -q -W error                       # expect: 47 passed
git checkout -b feat/search-safety
```

No `OPENAI_API_KEY` is needed anywhere here; keep it unset so you notice if a test reaches for one. Outside the venv three tests error on missing
`openai`/`python-dotenv` (M3) — that is the venv, not you. T4's real-data check needs `data/KuaiRand-Pure/data/`.

## 4. Tasks, in order

### T1 · C2 (a)(b)(c) · Close the AST and text bypasses (M, ~1.5 h)

- **Why:** `validate_source` is bypassable four ways, all verified by execution (`correctness-safety.md:105-131`); the AST check is the only thing
  between LLM-written code and the raw logs.
- **Where:** `safety.py:85-111` (`validate_source`), `:60-72` (`FORBIDDEN_TEXT`), `:35-59` (`FORBIDDEN_ATTRIBUTES`), `:8-19` (`ALLOWED_IMPORTS`,
  `TEST_ONLY_IMPORTS`).
- **Do:**
  1. **(a) Dunder names.** In the `ast.walk` loop reject an `ast.Name` whose `id` starts with `__` unless it is in `ALLOWED_DUNDER_NAMES =
     {"__name__"}` — without that exception every generated test with `if __name__ == "__main__":` is rejected. Add a second explicit branch for an
     `ast.Subscript` whose `.value` is a dunder `ast.Name`: redundant today, but it keeps the hole shut if the allowlist grows. Kills
     `__builtins__['open']` and `__builtins__['__import__']('os')`.
  2. **(b) Attribute access, not just calls.** Replace the `ast.Call`-only attribute check (`:108-109`) with a check on every `ast.Attribute`:
     `node.attr in FORBIDDEN_ATTRIBUTES` → raise. Keep the `ast.Call`/`ast.Name` check at `:106-107`. Kills `f = np.load ; f('x.npz')`.
  3. **(c) Text blocklist.** Add `log_standard`, `log_random`, `kuairand`, `.csv`, `/data/`. Remove the stale `data/judge` and `data\judge` (that
     directory does not exist — M1, `correctness-safety.md:293-299`) and `kuairand-starter-kit` (subsumed by `kuairand`). Matching is already
     case-insensitive (`:86-88`); `tests/test_safety.py:21-23` still passes, via `test_truth`.
  4. **False positives** (`spec-compliance.md:88` — they burn Debugger repairs). Drop `replace`, `rename` and `call` from `FORBIDDEN_ATTRIBUTES`.
     The rule, in the module docstring: *an attribute is banned only if a capability with that name is reachable from an allowed module.* With
     `os`/`pathlib`/`subprocess` unimportable, the only receivers for `.replace`/`.rename` are `str`, `bytes` and numpy arrays, where those methods
     are pure. Everything left names a real numpy or OS capability (`load`, `save`, `savez*`, `loadtxt`, `genfromtxt`, `fromfile`, `tofile`,
     `memmap`, `load_library`, `system`, `popen`, `check_output`, `remove`, `unlink`, `rmtree`, `read_*`, `write_*`) and stays banned on *access* —
     `FMRanker.load_state_dict` (`fm_core.py:86`) is not `load`, so it passes.
  5. Add `"__future__"` to `ALLOWED_IMPORTS` (today `from __future__ import annotations`, which the Builder writes by habit, is rejected) and factor
     the membership test into `is_allowed_import(name, allowed)` = `name in allowed or any(name.startswith(p + ".") for p in allowed)`, so `import
     numpy.random` stops being a false positive. Use it for `ast.Import` and `ast.ImportFrom`; reject `ImportFrom` with `level > 0`.
- **Interface:** `validate_source(source, *, test_file=False) -> None` raising `SafetyViolation` is unchanged; called from
  `candidate_runner.py:33-35, 65-66` and `run_candidate.py:44`.
- **Tests** (`tests/test_safety.py`): `test_builtins_open_subscript_is_rejected` · `test_builtins_import_subscript_is_rejected` ·
  `test_aliased_forbidden_attribute_is_rejected` (`f = np.load` on its own line) · `test_literal_dataset_path_is_rejected` (subTests over the
  `log_standard_…csv` path, `log_random`, `KuaiRand`, `/data/`) · `test_main_guard_and_string_methods_are_accepted` (a source with `if __name__ ==
  "__main__":`, `"a_b".replace("_","")` and `from __future__ import annotations` passes).
- **Acceptance:** `pytest -q -W error tests/test_safety.py` green with ≥ 11 tests · the four snippets at `correctness-safety.md:112-118` each raise
  `SafetyViolation` (paste that script and its output in the PR) · `pytest -q -W error` still ≥ 47 passed.
- **Depends on / blocks:** nothing. Do it first.

### T2 · C2 (d) · Restricted `__builtins__` for executed candidates (S–M, ~1.0 h)

- **Why:** defence in depth — today `__builtins__` inside a loaded candidate is the real builtins dict and `open` is reachable at runtime
  (`correctness-safety.md:119-125`). You provide the mapping, B spends one line.
- **Where:** new code in `safety.py`; the call site is `run_candidate.py:42-52` (B's file).
- **Do:**
  1. `SAFE_BUILTIN_NAMES`: the pure builders, iterators, numerics and common exception types — everything in `builtins` a training loop needs,
     including `print`, `hasattr`, `object`, `super` and `type`, and explicitly excluding every name in `FORBIDDEN_CALLS` (`safety.py:20-34`).
  2. `restricted_builtins(*, test_file: bool = False) -> dict[str, object]` = those names pulled from `builtins`, plus `"__build_class__"` (without
     it any `class` statement fails) and `"__import__": _guarded_import(test_file=test_file)`.
  3. `_guarded_import` returns a closure with the real signature `(name, globals=None, locals=None, fromlist=(), level=0)` raising `SafetyViolation`
     when `level != 0` or `not is_allowed_import(name, allowed)`, with `allowed = ALLOWED_IMPORTS | {"__future__"} | (TEST_ONLY_IMPORTS if test_file
     else set())` — the same predicate T1 gave `validate_source`, so the static and dynamic rules cannot drift.
  4. Docstring the limit: the mapping covers the **training** run (`run_candidate.py` execs the module), not the unit-test subprocess, which imports
     `candidate.py` through the normal import system (`candidate_runner.py:64-70`) — T1's AST rules stay the primary defence.
- **Interface — the exact hand-off to B.** E provides `safety.py::restricted_builtins(*, test_file: bool = False) -> dict[str, object]`. B adds one
  line to `_load_candidate` (`run_candidate.py:42-52`), between `module = importlib.util.module_from_spec(spec)` and
  `spec.loader.exec_module(module)`:
  ```python
  module.__dict__["__builtins__"] = restricted_builtins()
  ```
  and extends `run_candidate.py:14` to `from src.agent.safety import restricted_builtins, validate_source`. `exec` injects the real builtins only
  when the key is absent, so pre-setting it wins.
- **Tests:** `test_restricted_builtins_block_open_and_import` — with `ns = {"__builtins__": restricted_builtins()}`, `exec("open('x')", ns)` raises
  `NameError` and `exec("import os", ns)` raises `SafetyViolation` · `test_guarded_import_allows_numpy_and_project_modules` (`import numpy as np`,
  `import numpy.random`, `from src.models.sampling import sample_bpr_pairs` all succeed in `ns`) ·
  `test_restricted_builtins_support_class_definitions`.
- **Acceptance:** `pytest -q -W error tests/test_safety.py` green · `python -c "from src.agent.safety import restricted_builtins as r; b=r();
  print('open' in b, '__import__' in b, '__build_class__' in b)"` → `False True True` · the one-line diff sent to B (§6).
- **Depends on / blocks:** independent of B — the mapping is inert until called. Ping once on Day 2 if the line has not landed.

### T3 · I-7 · Point `validate_family_contract` at the registry (S, ~0.5 h)

- **Why:** `safety.py:114-133` hard-codes `{"bpr": …, "group_softmax": …}`; a third family would need a second literal list. It is also what makes
  A's I-7 change possible.
- **Where:** `safety.py:114-133`; `families.py:8-24`.
- **Do:**
  1. Extend `Family`: `grid: dict = field(default_factory=dict, compare=False)`, `defaults: dict = field(default_factory=dict, compare=False)`,
     `required_calls: tuple = ()`. `compare=False` keeps the frozen dataclass hashable despite the dicts. `required_calls` is a tuple of *one-of
     groups* — the candidate must call at least one name from each; empty means `((trusted_sampler,),)`.
  2. Fill `grid`/`defaults` for the two existing families with **exactly** today's values from `policy.py:27-64`, so A's switch-over changes no
     behaviour: shared `{"seed": range(0, 1000), "k": (16,), "learning_rate": (0.0003, 0.0005, 0.001), "epochs": range(1, 41), "patience": range(1,
     7)}`; bpr adds `{"batch_size": (2048, 4096), "negatives_per_positive": (1, 2)}`; group_softmax adds `{"batch_size": (512, 1024, 2048),
     "negatives_per_group": (4, 8), "temperature": (0.5, 1.0, 2.0)}`. Defaults are the `raw.get` fallbacks at `policy.py:28-35, 48, 54-55`. Grid
     values are `tuple` or `range`; membership is tested with `in`, exact and O(1) for both.
  3. Add `coverage_families() -> frozenset` returning `{"bpr", "group_softmax"}` — the *minimum* coverage set, so new families do not make
     `policy.py:23-24, 98-99` unsatisfiable — and `builder_brief(name) -> str` rendering the mandatory calls plus the grid (C consumes it).
  4. Rewrite `validate_family_contract` to look up `FAMILIES[family]` (raise `SafetyViolation` on `KeyError`, keeping today's message) and require
     one call per group in `required_calls`, reusing the existing `ast.Call` walk (`safety.py:122-129`). Trim the `families.py` docstring to match.
- **Interface:** `FAMILIES` keeps the frozen shape `Family(name, method_card, trusted_sampler)` plus the added fields; `family_names()` unchanged.
- **Tests:** `tests/test_safety.py::test_family_contract_reads_the_registry` — a bpr source that never calls `sample_bpr_pairs` raises, one that
  calls it passes, an unregistered family raises. `tests/test_interfaces.py:39-51` must stay green untouched here — T3 adds fields, not families, so
  `family_names()` is still `{"bpr", "group_softmax"}`; the ≤5-line update to that literal belongs to T5, in T5's PR.
- **Acceptance:** `pytest -q -W error` green · `python -c "from src.agent.families import FAMILIES as F; print(F['bpr'].grid['learning_rate'], 16 in
  F['bpr'].grid['k'])"` → `(0.0003, 0.0005, 0.001) True`.
- **Depends on / blocks:** blocks T5; unblocks A's I-7 change — tell A when it merges.

### T4 · I8 · `build_features` — trusted, train-only, leakage-safe (L, ~2.0 h)

- **Why:** features are frozen at the kit's 5 fields (`run_candidate.py:110-114`); user behaviour sequences are the kit's own #2 untested direction
  and "a completely blank direction" (`README.en.md:150-160`).
- **Where:** `src/models/features.py:8-9` (the Step 0 stub); reads through `official.py:33-69`; consumed by generated candidates and by
  `fm_core.py:33-52`.
- **Do:**
  1. Implement `build_features(rows, spec: dict) -> np.ndarray`. `rows` is the caller's encoded id matrix for one split (`context.train_x` /
     `valid_x` / `test_x`), used for its length and checked against the trusted row count; `spec["split"]` ∈ `{"train","valid","test"}`. Raw rows
     come from trusted code, never from the candidate: an `@lru_cache(maxsize=1)` wrapper around `official.load_train_valid(data_dir)`, which
     `continue`s on test dates *before* reading the label (`official.py:50-57`), so one parse (~10 s) serves all three calls in a worker process.
     `data_dir` is `spec.get("data_dir")` → `os.environ.get("KUAIRAND_DATA_DIR")` → `REPO_ROOT/data/KuaiRand-Pure/data` (what every config uses,
     `configs/ranking_losses.json:4`). Row order is the kit's, so index *i* of `rows` is index *i* of the loaded split — assert equal lengths, else
     raise `ValueError`.
  2. Six feature groups, each one extra int32 column:

     | group | value, from train rows only | slots |
     |---|---|---|
     | `user_rate` | smoothed long_view rate of the user | 9 |
     | `user_author` | smoothed rate of (user, author) — the DIN affinity signal | 9 |
     | `user_tab` | smoothed rate of (user, tab) | 9 |
     | `recency` | days since the user's last long_view before this row, capped at 14 | 9 |
     | `video_age` | `date − upload_dt` in days (`video_features_basic_pure.csv`, `YYYY-MM-DD`) | 9 |
     | `tab_cross` | smoothed rate of the (tab, duration-bucket) cell | 9 |

     Smoothing: `(positives + m·prior) / (count + m)`, `prior` = the global train long_view rate, `m = spec["smoothing"]`. Each value is bucketed
     with 8 quantile edges computed from the **train** values only (the kit's trick, `data.py:32-33`) via `np.searchsorted`; slot 8 is reserved for
     "no history / unknown" — an unseen key or a missing `upload_dt` — never the prior bucket, so the model can learn "unknown" separately.
  3. **Time-respecting rule inside train: strictly-earlier days (expanding window), `spec["scheme"] = "prior_days"` by default.** For a train row on
     day *d* the tables count only that key's rows on days < *d* (the finest timestamp the trusted loader keeps is `date`, `official.py:58-68`);
     valid and test rows use **all** of train. Why, and not leave-one-out: every valid/test row is scored from statistics built out of strictly
     earlier data, so building train rows the same way keeps the train-time and inference-time feature distributions aligned, whereas LOO lets a
     train row see the same user's *later* days — the model over-trusts the feature and the gain does not transfer. Cost: day 1 (≈7% of train rows)
     has no history and lands in the unknown slot. Offer `"leave_one_out"` as the other grid value (same tables, total minus the row's own
     contribution) so the agent can measure the difference.
  4. **Never read valid/test labels.** Element 6 of the row tuple (`long_view`) is read **only** when `spec["split"] == "train"`; the tables are
     built solely from train rows. Docstring it, pin it with a test.
  5. `feature_dimension(spec) -> int` = `9 × (enabled groups)` — a pure function of `spec`, no data, so the candidate can size the FM first.
     Returned indices are already offset by `spec["field_offset"]` (the candidate passes `context.field_dimension`).
  6. Deterministic: pure counting plus fixed quantiles, no RNG; sort keys before any tie-sensitive step.
- **Interface (what a candidate writes; how the FM consumes it):**
  ```python
  from src.models.features import build_features, feature_dimension
  spec = {"split": "train", "field_offset": context.field_dimension, **parameters}
  extra = build_features(context.train_x, spec)                 # (n, g) int32
  train_features = np.concatenate([context.train_x, extra], axis=1)
  model = FMRanker(context.field_dimension + feature_dimension(spec), embedding_dim=16, ...)
  ```
  The field dimension grows by `9 × g` (≤ 54 on top of 40 260), the row width from 5 to 5 + g (≤ 11). **`fm_core.py` needs no change:** `logits`
  (`fm_core.py:33-40`) gathers `V[features]` and uses the sum-of-squares identity over `axis=1`, and both `np.add.at` calls in `gradients`
  (`:46-51`) broadcast over the field axis — verified field-count agnostic. The new columns become ordinary FM fields that interact with
  `user_id`/`video_id`/`author_id`/`tab`/`dur_bucket`. Keep `k == 16`: capacity is a measured dead end (`README.en.md:133-139`; k=8/16/32 →
  0.5895/0.5902/0.5887).
- **Tests** (new `tests/test_features.py`; synthetic rows injected through `spec["history_rows"]`, the documented test override for the cached
  loader — no dataset needed): `test_train_features_use_only_strictly_earlier_days` (a user whose only long_view is on the last day gets the unknown
  slot on day 1 and a populated bucket later) · `test_valid_rows_use_all_of_train` · `test_labels_of_non_train_rows_are_never_read` (element 6 is an
  object that raises on any use, with `split="valid"`; the call must return normally) · `test_output_is_deterministic_and_within_range` (two calls
  `array_equal`; every value in `[field_offset, field_offset + feature_dimension(spec))`; dtype int32) ·
  `test_feature_dimension_matches_enabled_groups` · `test_unknown_keys_fall_in_the_reserved_slot`.
- **Acceptance:** `pytest -q -W error tests/test_features.py` green · real-data smoke (≤ 60 s): load the splits, `encode` them, call
  `build_features` for train and valid, print shape, dtype, `min() >= dim`, `max() < dim + feature_dimension(spec)` → `(1141112, 6) (124909, 6)
  int32 True True`, script kept in the PR · `grep -n "long_view\|\[6\]" src/models/features.py` shows every label read inside the train branch ·
  `tests/test_interfaces.py:146-148` (`test_history_features_are_not_implemented_yet`) turned into a shape assertion **in this same PR** — ≤ 5 lines,
  the sanctioned stub-pinning edit for the owner who fills a stub (team-split.md's second ruling); no PR to A.
- **Depends on / blocks:** blocks T5. `split="test"` needs B's `load_test_meta()` (I-2 / C1); until it exists raise a typed error there — the family
  still trains and scores validation, and B's worker records `failure_class="missing_test_scores"` instead of crashing. Work on train/valid
  meanwhile.

### T5 · I8 / I-7 · The `history_features` family and its method card (M, ~1.25 h)

- **Why:** the registry is how the family reaches the prompts (C), the parameter sanitiser (A) and the AST contract check. Without it
  `build_features` is dead code.
- **Where:** `families.py:15-20`; `safety.py:8-18`; new `research/methods/history_features.md`.
- **Do:**
  1. `ALLOWED_IMPORTS += {"src.models.features"}` so generated code may import it. Nothing else moves.
  2. Register:
     ```python
     "history_features": Family(
         name="history_features", method_card="research/methods/history_features.md",
         trusted_sampler="sample_bpr_pairs",
         grid={**SHARED_GRID, "epochs": range(1, 21), "batch_size": (2048, 4096),
               "negatives_per_positive": (1, 2), "smoothing": (5.0, 20.0, 100.0),
               "scheme": ("prior_days", "leave_one_out"),
               **{f"use_{g}": (True, False) for g in GROUPS}},   # GROUPS: the six T4 names, literal in families.py
         defaults={**SHARED_DEFAULTS, "epochs": 20, "batch_size": 2048, "negatives_per_positive": 1,
                   "smoothing": 20.0, "scheme": "prior_days", **{f"use_{g}": True for g in GROUPS}},
         required_calls=(("sample_bpr_pairs", "sample_softmax_groups"), ("build_features",)),
     )
     ```
     `trusted_sampler` stays a BPR/group-softmax sampler — the loss is unchanged, the *features* are the axis under test; `required_calls` enforces
     "one of the two samplers **and** `build_features`". Keep `GROUPS` a literal tuple in `families.py` rather than importing it from `features.py`:
     `types.py` imports `families`, so that import must stay light. `epochs` caps at 20 because six extra fields roughly double the gather/scatter
     cost (one FM epoch ≈ 12 s, `empirical.md:300-303`) against `experiment_timeout_seconds: 900` (`configs/ranking_losses.json:27`).
  3. Write `research/methods/history_features.md` in the **exact** section order of `bpr.md` (`# …`, `## Primary source`, `## Hypothesis`, `##
     Objective`, `## Safe initial search space`, `## Known failure modes`). Cite Zhou et al., "Deep Interest Network for Click-Through Rate
     Prediction", KDD 2018 (https://arxiv.org/abs/1706.06978) for the motivation — per-user history and (user, author) affinity — plus the kit's
     ranking of the direction (`README.en.md:150-160`). Objective: the loss is unchanged, the field set is what changes — give the concatenation,
     the `9 × g` extra slots and `k == 16`. Failure modes: (i) user-side-only features are constant within a user and contribute exactly 0 through
     first-order terms, acting only through crosses (`README.en.md:141-148`); (ii) leave-one-out inside train sees the user's future days; (iii) low
     counts without smoothing memorise the label; (iv) unknown keys belong in the reserved slot, not the prior bucket; (v) six extra fields ≈ 2×
     epoch time, so `epochs ≤ 20`; (vi) more fields is not more capacity — `k` stays 16.
  4. No catalog change: `MethodCatalog.load` globs `*.md` and keys by stem (`catalog.py:19-27`), so the filename stem must equal the family name.
  5. In the same PR, update the one `tests/test_interfaces.py` assertion a third family invalidates: the
     `frozenset({"bpr", "group_softmax"})` literal at `:41` (`test_family_names_expose_the_registered_families`) → assert against `frozenset(FAMILIES)`
     and the three names. ≤ 5 lines, the sanctioned stub-pinning edit; the card/sampler and decision checks at `:44-58` then cover three families
     unchanged.
- **Interface:** I-7, verbatim — *`src/agent/families.py::FAMILIES` (frozen shape: `Family(name, method_card, trusted_sampler)`). **E** may add
  fields to `Family` (e.g. `grid: dict[str, list]` describing the allowed parameter values) and new entries (`"history_features"`, `"multi_task"`).
  **A** makes `policy.py::sanitize_parameters` read the family's `grid` from the registry instead of its hard-coded checks, and **A** replaces
  `policy.py`'s literal family set with `families.family_names()`. Until A lands that, a new family's parameters pass through A's C4 non-fatal path
  (rejection, not a crash).*
- **Tests:** `tests/test_features.py::test_registry_entry_matches_the_builder_contract` — `required_calls` names `build_features`, the `method_card`
  file exists and carries all five headings of `bpr.md`, every `grid` value is a `tuple` or a `range`.
- **Acceptance:** `python -c "from src.agent.families import FAMILIES; print(sorted(FAMILIES))"` → `['bpr', 'group_softmax', 'history_features']` ·
  `pytest -q -W error` green, `tests/test_interfaces.py` included (its card/sampler check now covers three families, and step 5's ≤5-line update to
  the family-names literal is in this PR) · `diff <(grep '^## ' research/methods/bpr.md) <(grep '^## ' research/methods/history_features.md)` prints
  nothing.
- **Depends on / blocks:** **register only after A's C4 has merged** — until then an unregistered-family proposal raises at `policy.py:62-63` and
  ends the run (`research_controller.py:395-407`). T4 and the card can merge earlier; keep the registry line as the last commit and confirm with A
  first.

### T6 · I8 (stretch) · `multi_task` family (M, ~1.25 h — drop if Day 3 is tight)

- **Why:** the kit's #3 untested direction (`README.en.md:161-165`) — `is_click` is 44% of rows and strongly linked to `long_view`; spec Appendix A
  idea 9 (`…design.md:503-505`).
- **Where:** `src/models/features.py` (new function), `families.py`, new `research/methods/multi_task.md`.
- **Do:** add `build_aux_labels(rows, spec) -> np.ndarray` returning `(n_train, t)` float32 auxiliary targets read from train dates only —
  `is_click`, `is_like`, `log1p(play_time_ms)` min-max scaled on train. A loss touches train rows only, so no valid/test path exists and none may be
  added: raise if `spec["split"] != "train"`. Register `multi_task` with `required_calls=(("sample_bpr_pairs","sample_softmax_groups"),
  ("build_aux_labels",))` and a grid over `aux_weight ∈ (0.1, 0.3, 1.0)` plus which heads are on. The card cites Ma et al., ESMM (SIGIR 2018,
  https://arxiv.org/abs/1804.07931) and Ma et al., MMoE (KDD 2018); failure modes: auxiliary gradients swamping the ranking head, `play_time_ms`
  censored at video length (Zhao et al., KDD 2024), and the shared-embedding assumption.
- **Tests:** `test_aux_labels_are_train_only_and_finite` — shape, dtype, finiteness, and `split="valid"` raises.
- **Acceptance:** `pytest -q -W error` green · `sorted(FAMILIES)` has four entries · the card passes the same heading diff as T5.
- **Depends on / blocks:** T4, T5, and a healthy live run. If the Day-2 live run is still failing, skip it and spend the time on §8's Day-3 duty.

## 5. Definition of done (whole plan)

- [ ] `pytest -q -W error` green with `OPENAI_API_KEY` unset — 47 existing plus ≥ 16 new.
- [ ] Every acceptance box in T1–T5 ticked (T6 optional).
- [ ] The four bypasses from `correctness-safety.md:112-118` all raise, and the restricted-builtins namespace exposes neither `open` nor an
  unguarded `__import__`.
- [ ] `build_features` reads no label outside train, and no row outside train/valid unless B's `load_test_meta()` supplied it.
- [ ] `research/methods/history_features.md` (plus `multi_task.md` if T6 lands) exist in the card format and are reachable through `MethodCatalog`.
- [ ] PRs merged, ≤ 300 lines each: one for T1–T3, one for T4–T5, one for T6. Hand-offs in §6 delivered, each naming the file, the line and the
  change. No `runs/`, no `data/`, no `.env` in any commit.

## 6. Hand-offs

**You provide:**

- **To B — the restricted-builtins call (C2 d).** `safety.py::restricted_builtins(*, test_file: bool = False) -> dict[str, object]`; B adds the
  single line `module.__dict__["__builtins__"] = restricted_builtins()` to `_load_candidate` (`run_candidate.py:42-52`), between `module_from_spec`
  and `exec_module`, and imports it at `run_candidate.py:14` (their T4 step 2). Tell B when T2 merges. Ask B for two things in return: (i)
  `KUAIRAND_DATA_DIR=<data_dir>` in the minimal env B builds for C3 (`candidate_runner.py:58-62`) — one entry, and the default path works without
  it; (ii) `official.py::load_test_meta(data_dir, *, expected_rows=None) -> TestSplit` (C1), whose `.meta` is `(row_id, user_id, video_id)` per row
  and whose `.rows` are the kit-shaped 7-tuples `(date, user_id, video_id, author_id, tab, duration_ms, LABEL_PLACEHOLDER)` — **no label column** —
  which is what `build_features(spec={"split": "test"})` needs.
- **To A — the registry (I-7).** `FAMILIES[name].grid: dict[str, tuple | range]` and `.defaults: dict[str, Any]`; `sanitize_parameters` becomes
  "fill from `defaults`, then reject any key whose value is `not in` the grid entry" — `in` works for a tuple of allowed values and for a `range`
  integer bound. T3 seeds bpr/group_softmax with today's exact values from `policy.py:27-64`, so the switch-over changes no behaviour. Also:
  `families.family_names()` replaces the literal at `policy.py:8`, and `families.coverage_families()` is what `coverage_complete`/`required_family`
  (`policy.py:11-24`, used at `:98-99`) must use — otherwise every family E adds makes the harness stop rule unsatisfiable. Last: when T4 lands,
  `tests/test_interfaces.py:146-148` (`test_history_features_are_not_implemented_yet`) is false — you fill that stub, so you turn it into a shape
  assertion yourself in the T4 PR (≤ 5 lines, the sanctioned edit), and likewise the family-names literal in the T5 PR. No PR to A for either.
- **To C — the prompt text.** `families.builder_brief(name) -> str` renders the mandatory trusted calls and the family's grid; it replaces the
  hard-coded sentences at `roles.py:166-167`, so C never tracks a new family. Cards need no C change (`catalog.py:19-27` globs and keys by stem).
  One warning: `catalog.prompt_text(None)` (`catalog.py:29-31`, called at `roles.py:106`) concatenates **all** cards, so two new cards add roughly
  1.2k tokens to every Researcher call against the 150 000-token budget (`configs/ranking_losses.json:19`) — pass the selected family.

**You consume:** D's note that `tests/test_datacard.py` imports `safety.FORBIDDEN_TEXT` dynamically, so your T1 step 3 additions (`log_standard`, `log_random`, `kuairand`, `.csv`, `/data/`) can turn D's card test red — intentional, and D fixes the card, not you; just tell D when T1 merges · A's C4 (non-fatal proposal errors) before registering `history_features`, else the first proposal for it ends the run — fallback:
merge T4 and the card, hold the one-line registry commit · A's registry-driven `sanitize_parameters` (I-7) before the new grid keys can pass —
fallback: parameters are rejected non-fatally on A's C4 path and validation-only experiments still run · B's `load_test_meta()` for `split="test"` —
fallback: a typed error, the family works on train/valid.

**Notes from the Step 0 review assigned to you:** none — the hand-off table names A, B and C only. The stale `data/judge` guard (M1) and the
`.replace`/`.load` false positives (`spec-compliance.md:88`) fall in your files and are steps 3 and 4 of T1.

## 7. Rules

1. **One owner per file.** Need a change in someone else's file? Ask the owner, or send them a ≤20-line PR they merge.
2. **New tests in new files.** Never edit another owner's test file.
3. **Rebase on `main` twice a day; PRs ≤ 300 lines; `pytest` green before merge.** Small, frequent merges beat one big one.
4. **Config: add files, don't edit shared ones** (`configs/offline_smoke.json`, `configs/features_run.json`).
5. **Run directories are personal** (`runs/<initials>_…`, gitignored) until the final run, which A commits.
6. **Shared surfaces only move in the freeze PR** — `types.py`, `contracts.py`, `configs/ranking_losses.json`. A second freeze PR (by A) is the way
   to change them again, not five drive-by edits.

Plus: never run `git add -A`; never commit `runs/` (only A, and only the final run); never commit `.env`; PR ≤ 300 lines; rebase on `main` twice a
day.

## 8. Daily checkpoints

**Day 1 (end).** T1–T3 merged: the four bypasses raise, the false-positive guards pass, `pytest -q -W error` is green, B has the one-line
`restricted_builtins()` diff. `build_features` is designed on paper — the group table and the strictly-earlier rule drafted into
`history_features.md`.

**Day 2 (end).** T4 and T5 merged. `build_features` runs on the real dataset inside 60 s and passes the range/determinism check; `history_features`
is registered (after A confirms C4 and the registry-driven sanitiser are on `main`); the card is in the catalog; A and C have their hand-off
messages. One candidate using the family has trained end-to-end on validation — never register a family you have not run.

**Day 3.** On call for A's live run: `grep -c SafetyViolation <run_dir>/iterations.jsonl`; any false positive that burns a Debugger repair is yours
to fix within 30 minutes, with a named regression test in `tests/test_safety.py`. If the run is healthy by mid-morning do T6; otherwise help A read
failures. Do not widen the grid or the allowlist during the live run.
