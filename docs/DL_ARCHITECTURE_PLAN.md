# Deep-Learning Model Architecture & Integration Plan

> Target: push KuaiRand-Pure validation primary from **0.6037** toward **≥0.7** (oracle ceiling 0.8645).
> Decision: **PyTorch**, via a **trusted training primitive** that keeps the autonomous agent loop in charge.
> Status: designed and adversarially verified (3 designs × 3 critics; 2/3 survived all lenses).

---

## 1. Framework Decision: PyTorch

**JAX/FLAX NNX is rejected for this task/platform/agent-loop, but kept as a documented fallback** if the model later outgrows the 900s budget. Only `din_trainer.py` changes in a fallback; the parameter/grid wiring is framework-agnostic.

| Reason | Grounding (verified against repo) |
|---|---|
| Contract fit (imperative → eager) | `run_candidate.py:193` calls `module.run(context, parameters) -> CandidateOutput` of numpy arrays; trusted worker computes GAUC/nDCG itself (`:88`). PyTorch eager is imperative — a 1:1 match with the existing imperative `FMRanker`. JAX/FLAX NNX is functional (pytrees, PRNG threading); more friction. `torch state_dict -> numpy` is one line; NNX state pytree needs a traversal. |
| Platform | Apple Silicon, CPU-only. JAX has no MPS; PyTorch has MPS (future knob). For the small model in scope (vocab ~7,538, embed 16-32), CPU-eager torch beats MPS anyway. Neither is installed today. |
| 900s budget on 1.14M rows | ~278 batches/epoch at batch 4096. DIN fwd/bwd ~8-22s/epoch eager; 20-40 epochs → ~160-650s + eval, inside 900s. JAX JIT is faster per-batch *after compile*, but first compile is seconds-to-tens-of-seconds, any shape change recompiles, and 900s is a hard cap with no partial credit. |
| Debuggability in captured stdout | `candidate_runner.py:177-178` captures stdout/stderr to files; the agent repairs failures from those logs. Torch eager errors are normal Python stack traces. JAX JIT errors are opaque traced-back transforms — harder for the LLM Builder to repair in a sandbox. Decisive for an autonomous loop. |
| Reference impls | DIN/DIEN/DeepFM/DCN/xDeepFM + the cited CWM paper (xDeepFM in torch). DeepCTR/FuxiCTR/CWM are all PyTorch. The LLM Builder was trained on far more torch CTR code. |
| Sandbox fit | `safety.py` only AST-validates *candidate* source; trusted `src/models` modules are NOT AST-gated, so `din_trainer.py` can `import torch` freely while the candidate still cannot (torch **not** added to `ALLOWED_IMPORTS`). Exactly the `build_features`/`build_aux_labels` pattern. |
| NNX maturity | FLAX NNX (2024) is still settling; on Apple Silicon CPU far less battle-tested than torch CPU. The kit's reliability premise favors the most boring combo. |

---

## 2. Integration Architecture: Path A (trusted training primitive)

A new **trusted** module `src/models/din_trainer.py` (human-authored, allowlisted, **not** LLM-generated) owns all PyTorch math and exposes:

```python
run_din_trainer(context: CandidateContext, parameters: dict)
    -> (validation_scores: np.float32 (n_valid,),
        test_scores: np.float64 (n_test,),
        checkpoint_state: dict[str, np.ndarray],
        training_trace: list[dict],
        diagnostics: dict)
```

The LLM-generated candidate is a **~15-line thin wrapper** that imports it and returns a `CandidateOutput`. The agent's propose/critique/build loop keeps full control via a new `din` family grid. **Path B (score-only scripts/train_deep.py → test_scores.npy → gate.py) is rejected** — it bypasses the propose/critique/build loop for the model itself.

### End-to-end data flow

1. `run_candidate.py:182` builds `CandidateContext{train_x (N,5) int32, train_y, train_users, valid_x, valid_users, field_dimension, evaluate_validation, test_x}` and calls `module.run(context, parameters)`.
2. The **thin wrapper candidate** imports `run_din_trainer` + `build_user_sequences`, calls `run_din_trainer(context, parameters)`, wraps the arrays into `CandidateOutput`. It never touches raw logs or builds its own sequences.
3. **The trusted trainer** `run_din_trainer`:
   - Re-loads raw kit rows **itself** via `official.load_train_valid` + `load_test_meta` + `build_user_sequences` using `KUAIRAND_DATA_DIR` (CandidateContext carries only encoded int32 matrices). **Must not accept candidate-supplied rows** — candidate is untrusted.
   - Builds multi-task aux labels from `train_y` only: `is_click` (within-user r=0.72) + censored `play_time` (r=0.58). **Drops** is_like/is_follow/is_comment/is_forward (all <0.10, noise).
   - Constructs the DIN+FM model in torch (CPU eager, `torch.device('cpu')` explicit, `torch.set_num_threads()` bound to the scratch env's OMP cap).
   - Trains with within-user **listwise** loss (group-softmax, reusing `sample_softmax_groups` semantics) + multi-task aux heads weighted by `aux_weight`, epochs from grid, early-stop on valid GAUC via `context.evaluate_validation` (**early-stopping only, never an optimization objective**).
   - Predicts `validation_scores` (float32, len valid) + `test_scores` (float64, len test) from the **same** trained model — no retraining on valid, no peeking at test labels.
   - Converts **every** torch tensor via `.detach().cpu().numpy()` before building the return tuple. Internal **~820s wall-clock guard** (defense-in-depth so a misconfigured grid can't trip the 900s subprocess timeout).
4. `validate_and_persist_output` (`run_candidate.py:80-149`) computes GAUC/nDCG via `official_evaluate` — **trusted, candidate metrics ignored**, validates finiteness/shape (checkpoint ≤50M elements, keys alnum, test_scores float64 1-D len=170588 finite), persists `model.npz` + `test_scores.npy`, writes `result.json`.
5. `candidate_runner` applies the sanity band `[0.47, 0.80]` and `test_scores_status`, promotes/rejects.
6. `gate.py` reads the best node's `test_scores.npy`, builds `submission.csv`, runs `submit.py --check`. **No change to gate.py.**

### How the loop stays in charge

The Researcher proposes DIN knobs from the `din` grid; the Critic checks leakage (no valid/test label read, prior-days sequences, train-only vocab); the Builder writes the thin wrapper with exact import spellings.

> ⚠️ Requires fixing the **parameter channel** — see Phase 1, Step 3. `PARAMETER_SCHEMA` is `additionalProperties:False` with a fixed property set; `_normalize_schema_output` silently strips family-specific keys. Without this fix, "Researcher picks architecture knobs" is false.

---

## 3. Model Architecture

DIN target-conditional attention + listwise ranking + multi-task (is_click + censored play_time) + FM backbone, all in torch inside the trusted primitive. **Strict superset of the 0.6016 FM** — the FM block is ported verbatim to torch, so a sanity anchor reproduces 0.6016 before the attention block is added.

### Trusted primitive signatures

- `run_din_trainer(context, parameters) -> (validation_scores, test_scores, checkpoint_state, training_trace, diagnostics)`
- `build_user_sequences(split, ...) -> (hist_items (N,L), hist_mask (N,L), cand_item (N,))`  ← already built in `src/models/sequence.py`
- `_DIN.forward(id_fields (B,5), hist_items (B,L), hist_mask (B,L), cand_item (B,)) -> (long_view_logit, click_logit, play_pred)`
- `state_to_npz(model) -> dict[str, np.ndarray]`

### The math

**FM block** (exact `FMRanker` math, ported to torch):
```
logits_fm = b + sum(W[features]) + 0.5*((sum_emb)^2 - sum(emb^2))   # reproduces 0.6016 inside torch
```

**DIN target-conditional attention** (the key piece — the only mechanism that survives within-user=0; first-order user-side terms contribute exactly 0 to within-user ranking, so user signal only helps via cross/attention terms):
```
query = E[cand_item]                                   (B,d)
keys  = E[hist_items]                                   (B,L,d)
interactions = [query*keys, query-keys, query·keys]     (B,L,3d) -> MLP -> (B,L,1)
attention_logits = masked_fill(..., hist_mask==0, -1e9)
alpha = softmax(attention_logits, dim=1)                 (B,L)
pooled_interest = sum(alpha * keys, dim=1)               (B,d)
# empty-history fallback: all-zero-mask rows use the UNK embedding as pooled_interest
```
The query **is the candidate item** → two candidates for the same user get different attention weights → intra-user ordering changes. This is the structural break from FM.

**Tower:** `MLP([fm_interactions, fm_linear, pooled_interest, E[cand_item]]) -> long_view_logit`

**Aux heads:** `is_click` BCE (r=0.72) + censored `play_time` head (r=0.58, cap below the long_view threshold or a survival/quantile target). `aux_loss = aux_weight*(BCE_click + huber_play)`. **Drop** is_like/is_follow/is_comment/is_forward (all <0.10).

**Ranking loss:** within-user **listwise** group-softmax (not BPR pairwise — pairwise aligns with GAUC only weakly and with nDCG@5 not at all). v1 uniform group-softmax; flag a positives-weighted/LambdaRank top-5 variant if nDCG@5 lags GAUC.

**Optimizer:** AdamW, lr from params, weight_decay 1e-6; ~20-40 epochs, batch 4096 (~278 batches/epoch on 1.14M rows), early-stop patience 1-6 on valid GAUC.

### Thin wrapper candidate (~15 lines, LLM-generated, AST-validated)

```python
import numpy as np
from src.models.din_trainer import run_din_trainer
from src.experiments.contracts import CandidateOutput
def run(context, parameters):
    val, test, ckpt, trace, diag = run_din_trainer(context, parameters)
    return CandidateOutput(validation_scores=val, checkpoint_state=ckpt,
                           training_trace=trace, diagnostics=diag, test_scores=test)
```

---

## 4. Leakage Strategy (all in trusted code, never candidate)

| Invariant | Enforcement |
|---|---|
| Item vocabulary | train-only with UNK slot — `sequence.py:build_user_sequences` (`_build_video_vocab`). Trainer calls it; never builds its own vocab. |
| Time-respecting sequences | prior-days for train, all-train for valid/test — `sequence.py`. Trainer owns sequence construction by calling `build_user_sequences(split=...)` internally; **must not accept candidate-supplied sequences**. |
| Valid labels | never read; `evaluate_validation` is early-stopping only, never an optimization objective. `valid_y` is hidden in `CandidateContext`. |
| Test labels | never touched; test rows arrive as `test_x` with no labels (`run_candidate.py:167` loads features only). |
| Test scores / checkpoint | from the **same** trained model; checkpoint is `dict[str,np.ndarray]` (tensors → `.detach().cpu().numpy()`), shape/finiteness-checked before persist. No fs access from candidate (`safety.py` FORBIDDEN_ATTRIBUTES blocks write/save/savez; FORBIDDEN_TEXT blocks /data/, .csv, kuairand). |

The trusted `din_trainer` must **assert** (not trust) these invariants at entry, and a unit test pins that the valid/test sequence path never touches the label column.

---

## 5. Files to Touch

| File | Change |
|---|---|
| `src/models/din_trainer.py` | **NEW**, trusted: torch DIN+FM+aux trainer, `run_din_trainer`, leakage enforcement, wall-clock guard, numpy-conversion contract. Import torch as `import torch as _torch` (don't bind `torch` so candidate can't reexport it). |
| `src/agent/safety.py` | `ALLOWED_IMPORTS += 'src.models.din_trainer', 'src.models.sequence'`; **do NOT add `'torch'`**. |
| `src/agent/families.py` | new `din` Family: grid (embedding_dim/k {16,24,32}, seq_len {20,50}, attention_dim, epochs capped for 900s, batch_size {2048,4096}, aux_weight {0.1,0.3,1.0}, use_is_click, use_play_time), defaults, `required_calls=(('run_din_trainer',),)`, `TRUSTED_CALL_MODULES += run_din_trainer/build_user_sequences`. |
| `src/agent/llm.py` | extend `PARAMETER_PROPERTIES` with DIN knobs **or** relax the parameters schema to permit family-specific keys; extend `schema_fields_note` with `run_din_trainer` + `build_user_sequences` signatures. |
| `src/agent/policy.py` | wire DL knobs into `_SHARED/_FAMILY_KEYS` coercion + bounds. |
| `src/agent/roles.py` | add `src.models.din_trainer`, `src.models.sequence` to the allowed-import list in the Builder prompt. |
| `requirements.txt` | `+= torch>=2.2` (CPU-only pin / index URL). |
| `research/methods/din.md` | **NEW** method card (stem must equal `'din'`; `MethodCatalog` globs `*.md`, keys by stem). |
| `src/models/sequence.py` | **no change** — already the leakage-safe input producer. |

---

## 6. Implementation Phases & Steps

### Phase 1 — Prerequisites & plumbing (no torch math yet)

**Step 1.1 — Install torch into the repo interpreter**
```bash
# MUST go into pyenv site-packages, NOT --user: candidate_runner._environment()
# overrides HOME and Python drops ~/.local user-site from sys.path in the scratch subprocess.
pip install torch  # CPU arm64 wheel
# Verify it imports in a scratch-like env:
env -i HOME=/tmp/probe PYTHONPATH=. python -c "import numpy,torch; torch.set_num_threads(4); print(torch.__version__)"
```
Add `torch>=2.2` (CPU-only pin) to `requirements.txt`.

**Step 1.2 — Trusted trainer skeleton (plumbing before math)**

Create `src/models/din_trainer.py` with `run_din_trainer(context, parameters)` returning **zero-filled arrays of the right shapes**:
- `validation_scores`: float32 1-D, len = `len(valid_y)`
- `test_scores`: float64 1-D, len = expected test rows
- `checkpoint_state`: `dict[str, np.ndarray]`, all-finite, ≤50M elements
- `training_trace`, `diagnostics`

Conventions: `import torch as _torch` (don't bind `torch`); `torch.device('cpu')` explicit; `torch.set_num_threads()`; the 820s wall-clock guard.

**Step 1.3 — Fix the parameter channel (the active bug, blocks all 4 families)**

This is a pure-numpy change, runnable now without torch. `PARAMETER_SCHEMA` (`llm.py:35-51`) is `additionalProperties:False` with a fixed property set; `_normalize_schema_output` silently strips keys not in it before `sanitize_parameters`. So family-specific knobs (DL knobs, and the existing `smoothing`/`scheme`/`use_*`/`aux_weight`) are stripped.

Fix: extend `PARAMETER_PROPERTIES` with the DIN knobs (`seq_len`, `attention_dim`, `dropout`, `aux_weight`, `use_is_click`, `use_play_time`) **or** relax the parameters schema to permit family-specific extra keys. Wire DL knobs into `policy.py` `_SHARED/_FAMILY_KEYS` coercion and bounds. Update `schema_fields_note` with `run_din_trainer` + `build_user_sequences` signatures and exact import spellings.

**Step 1.4 — Wire the allowlist + family registry**

- `safety.py`: `ALLOWED_IMPORTS += 'src.models.din_trainer', 'src.models.sequence'` (NOT `'torch'`).
- `families.py`: register the `din` Family with grid/defaults/`required_calls=(('run_din_trainer',),)`; `TRUSTED_CALL_MODULES += run_din_trainer→'src.models.din_trainer', build_user_sequences→'src.models.sequence`.
- `roles.py`: add `src.models.din_trainer`, `src.models.sequence` to the Builder prompt's allowed-import list.

**Step 1.5 — End-to-end plumbing test (before any torch math)**

Write the thin-wrapper candidate and run it through `validate_and_persist_output` to confirm the plumbing (allowlist, family contract, schema, parameter channel) all work with zero-filled scores.

### Phase 2 — The model (torch math)

**Step 2.1 — FM-in-torch sanity anchor**

Port `FMRanker.logits` (`fm_core.py`) to torch verbatim: `logits = b + sum(W[features]) + 0.5*((sum_emb)^2 - sum(emb^2))`. Train with within-user listwise loss. **Assert it reproduces the 0.6016 FM baseline on valid** — the known sanity anchor before adding the attention block.

**Step 2.2 — DIN attention + multi-task**

Add the DIN target-conditional attention block (query=cand_item_emb, keys=hist_items, masked softmax, pooled_interest) + the tower MLP + the is_click BCE and censored play_time aux heads with learned loss weighting (`aux_weight` start 0.1). Compose on the group-softmax listwise loss. Keep small (d=16-32).

**Step 2.3 — Leakage + reproducibility tests**

`tests/test_din_trainer.py` on synthetic small data:
- (a) `run_din_trainer` returns finite `validation_scores` of correct length,
- (b) `test_scores` length matches `expected_test_rows`,
- (c) the valid/test sequence path never touches the label column (ExplosiveLabel pattern from `test_sequence.py`),
- (d) checkpoint is `dict[str,np.ndarray]` all-finite ≤50M elements,
- (e) reproducible across seeds.

**Step 2.4 — 900s benchmark in the actual scratch env**

Run a timed DIN training in the **actual** scratch env (`HOME`/`TMPDIR` rewritten, only `PASSTHROUGH_KEYS`, thread caps set, `PYTHONPATH=repo`, no provider keys) to verify it finishes < 900s. **Treat this as the primary feasibility gate.** Cap `seq_len` at 64; cache sequences per split (build once).

### Phase 3 — Agent loop integration

**Step 3.1 — Method card + agent loop integration**

Write `research/methods/din.md` (MethodCatalog keys by stem = `'din'`). Document the data path (trusted row reload via `KUAIRAND_DATA_DIR`), sequence build one-time cost, the listwise loss, the aux heads, and the knobs the Researcher may vary.

**Step 3.2 — Loss alignment + scaling knob**

If nDCG@5 lags GAUC, implement a positives-weighted or LambdaRank-style top-5 listwise variant (GAUC is positives-weighted by per-user npos; nDCG@5 cares about top-5 position). Expose as a Researcher knob.

---

## 7. Roadmap Summary

| Phase | Step | Effort | Risk | Depends on |
|---|---|---|---|---|
| 1 | 1.1 Framework install + smoke | 0.5d | low | — |
| 1 | 1.2 Trusted trainer skeleton + plumbing | 1d | medium | 1.1 |
| 1 | 1.3 Parameter channel fix | 1d | medium | 1.2 |
| 1 | 1.4 Allowlist + family registry | 0.5d | low | 1.2 |
| 1 | 1.5 End-to-end plumbing test | 0.5d | low | 1.4, 1.3 |
| 2 | 2.1 FM-in-torch sanity anchor | 1d | low | 1.5 |
| 2 | 2.2 DIN attention + multi-task | 2d | medium | 2.1, 1.3 |
| 2 | 2.3 Leakage + reproducibility tests | 1d | low | 2.2 |
| 2 | 2.4 900s benchmark in scratch env | 1d | high | 2.2 |
| 3 | 3.1 Method card + agent loop integration | 1d | low | 2.4, 1.3 |
| 3 | 3.2 Loss alignment + scaling knob | 1-2d | medium | 3.1 |

---

## 8. Open Questions & Risks

- **HARD BLOCKER — torch not installed.** The trusted subprocess inherits the interpreter; the scratch sandbox (no network, no pip) can't install it. Must pre-install a CPU wheel into pyenv site-packages (not `--user`). Without this, the design doesn't run at all.
- **ACTIVE BUG — parameter channel.** `PARAMETER_SCHEMA` is `additionalProperties:False`; `_normalize_schema_output` silently strips family-specific keys. DL knobs (and the existing history_features/multi_task `smoothing`/`scheme`/`use_*`/`aux_weight`) are stripped — the Builder literally cannot pass architecture choices. **This is why 2 of the 4 families are registered but never run.** Fix is a real wiring task across `llm.py` + `policy.py` + `families.py`.
- **COUPLED EDIT SITES — registering the DIN family is not a single `families.py` edit.** Six coordinated sites must all land together (families, `llm.PARAMETER_PROPERTIES`, `llm.schema_fields_note`, `policy._SHARED/_FAMILY_KEYS`, `safety.ALLOWED_IMPORTS`, `methods/din.md`) or the loop breaks.
- **IMPORT-REEXPORT HOLE — adding `src.models.din_trainer` to `ALLOWED_IMPORTS` lets candidate code attempt `from src.models.din_trainer import torch`.** Mitigation: `import torch as _torch` inside the trainer and don't bind `torch`. `torch.hub` network fetchers aren't covered by the current allowlist — confirm no reexports.
- **PLATFORM SPEED — Apple Silicon CPU torch is slower than Linux CUDA.** If the one-time numpy sequence build (1.14M rows) exceeds ~60s the epoch budget shrinks. Cap `seq_len` at 64, cache sequences per split. The 900s claim must be verified by a timed run in the actual scratch env.
- **PLAY_TIME CENSORING — play_time is near-collinear with long_view** (long_view derives from it). The censored target must be defined so it's **not reducible to the long_view label** — else the aux head is a leakage channel the Critic must audit.
- **LOSS-LABEL ALIGNMENT — GAUC is positives-weighted by per-user npos; nDCG@5 cares about top-5 position.** A uniform full-list group-softmax slightly mismatches both. v1 uses uniform group-softmax (acceptable approximation); flag a positives-weighted/LambdaRank variant for the Researcher if nDCG@5 lags GAUC.
- **NNX MATURITY (runner-up) — FLAX NNX (2024) is still settling**; on Apple Silicon CPU far less battle-tested than torch CPU. JAX JIT only clearly wins if the model outgrows the 900s budget — keep it as the documented fallback (only `din_trainer.py` changes).

---

## 9. Honest Expectations

- **Conservative floor:** ~0.62 (the deep backbone + listwise + multi-task modestly beats FM).
- **Stretch:** ~0.68-0.69 (DIN target-attention pays off at the top of its believable range + ensemble compounds).
- **Reaching 0.7:** ~15-20% probability — requires the top of every stretch + ensemble compounding. The oracle is 0.8645; nDCG is hard-capped (27% of test users are all-negative → nDCG permanently 0). 0.7 means capturing ~39% of remaining headroom.
- **The one bet it all rests on:** target-conditional sequence attention (DIN) in PyTorch, trained outside the numpy sandbox via the trusted primitive, beating the `user_id × video_id` cross. The minimal DIN experiment (Phase 2, Step 2.2) clearing ~0.606 across 3 seeds is the decisive test — run it first, alone, before the full stack.

## Kill conditions (when to stop, honestly)

- Minimal DIN+listwise doesn't beat **0.606** (0.6037 + 2.5σ) across 3 seeds → sequences aren't paying off on KuaiRand-Pure; don't build DIEN/SIM.
- DIN beats FM on biased validation but **not on the unbiased `log_random` validation** → overfitting biased traffic → kill the deep variants.
- Target-attention weights collapse to near-uniform (entropy within 5% of max) across candidates → the within-user=0 mechanism isn't firing → kill.
- After the full stack, validation stalls below **GAUC ~0.74** → 0.7 not reachable by this feature class → report the honest floor and stop. **Don't keep stacking components hoping for a jump.**
- If validation ever exceeds **0.80** → halt and audit for test-label leakage; do not submit.
