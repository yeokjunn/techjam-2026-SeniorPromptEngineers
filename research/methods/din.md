# DIN — Target-Conditional Sequence Attention

## Method

Deep Interest Network (DIN) models a user's interest *in the context of the
candidate item* by attending over the user's time-ordered interaction history,
where the **candidate item is the attention query**. Unlike mean-pooling (which
produces a user-constant vector that contributes exactly 0 to within-user
ranking), target-conditional attention gives two candidates for the same user
**different** attention weights, so the intra-user ordering changes. This is the
one mechanism the task contract blesses for user-side signal to survive
within-user ranking, and it is structurally unreachable by the FM's per-user
embedding (which is constant within a user).

Reference: Zhou et al., *Deep Interest Network for Click-Through Rate
Prediction* (KDD 2018). Composed on a within-user listwise (group-softmax) loss
aligned with GAUC/nDCG@5, plus multi-task aux heads sharing the backbone.

## Applicability

- The committee's #2 unexplored direction: each user has hundreds-to-thousands
  of train interactions; sequence interest modelling is "completely blank."
- Target: push primary well past the FM baseline (0.6016 validation). The
  within-user residual correlation diagnostic (this session) confirmed
  `is_click` (r=0.72) and `play_time` (r=0.58) transfer to within-user ranking;
  `is_like`/`is_follow`/`is_comment`/`is_forward` (all <0.10) do not and are
  dropped.

## Trusted primitive

All torch math lives in `src/models/din_trainer.py:run_din_trainer`. The
candidate is a thin wrapper:

```python
from src.models.din_trainer import run_din_trainer
from src.experiments.contracts import CandidateOutput
def run(context, parameters):
    val, test, ckpt, trace, diag = run_din_trainer(context, parameters)
    return CandidateOutput(val, ckpt, trace, diag, test)
```

The trusted trainer: reloads kit rows via `KUAIRAND_DATA_DIR`, builds
leakage-safe item-id history via `build_user_sequences` (prior-days for train,
all-train for valid/test, train-only vocab), ports the FM block to torch
(sanity anchor reproducing 0.6016), adds the DIN attention tower, trains with a
within-user listwise loss + multi-task aux heads, early-stops on valid GAUC via
`context.evaluate_validation` (early-stopping only, never an optimization
objective), and converts every tensor via `.detach().cpu().numpy()` before
return. The candidate never imports torch, loads data, or constructs sequences.

## Architecture

- **FM block** (exact `FMRanker` math, ported to torch): a sanity anchor.
- **DIN attention:** query = `E[candidate_item]`; keys = `E[history_items]`;
  interactions = `[query*keys, query-keys, query·keys]` → MLP → masked softmax
  over the history → `pooled_interest = Σ α_i · E[history_i]`. Empty-history rows
  fall back to the UNK embedding.
- **Tower:** `MLP([fm_interactions, fm_linear, pooled_interest, E[candidate]])`
  → `long_view_logit`.
- **Loss:** within-user **listwise** group-softmax (not BPR pairwise — pairwise
  aligns with GAUC only weakly and with nDCG@5 not at all). A positives-weighted
  or LambdaRank-style top-5 variant is a follow-up if nDCG@5 lags GAUC.
- **Multi-task aux:** `is_click` BCE + censored `play_time` (Tobit one-sided
  loss, censored at duration), weighted by `aux_weight`. **Drop**
  is_like/is_follow/is_comment/is_forward (within-user r < 0.10).
- **Optimizer:** AdamW, lr from `learning_rate`, weight_decay 1e-6; epochs from
  grid, batch 4096 (~278 batches/epoch on 1.14M rows); early-stop patience on
  valid GAUC. CPU eager; an 820s wall-clock guard defends the 900s subprocess
  timeout.

## Tunable parameters (grid)

| parameter | grid | default | note |
|---|---|---|---|
| `embedding_dim` | 16, 32 | 32 | item-history embedding width (distinct from `k`, the FM-block width, pinned at 16) |
| `seq_len` | 20, 50 | 50 | max history length per row (capped for the 900s CPU budget) |
| `attention_dim` | 16, 32 | 32 | attention MLP hidden width |
| `dropout` | 0.0, 0.2, 0.5 | 0.2 | attention/tower dropout |
| `aux_weight` | 0.1, 0.3, 1.0 | 0.1 | multi-task aux loss weight (start small — a large aux head swamps the ranking head) |
| `use_is_click` | True, False | True | is_click aux head (within-user r=0.72) |
| `use_play_time` | True, False | False | censored play_time aux head (r=0.58) |
| `learning_rate` | 0.0003, 0.0005, 0.001 | 0.001 | |
| `epochs` | 1-20 | 20 | capped for the 900s CPU timeout |
| `batch_size` | 2048, 4096 | 4096 | |
| `patience` | 1-6 | 4 | early-stop patience on valid GAUC |
| `seed` | 0-999 | 0 | |

## Known failure modes

- **Target attention collapses to uniform** (entropy within 5% of max across
  candidates) → the within-user=0 mechanism is not firing → the family is not
  paying off; do not escalate to DIEN/SIM.
- **Beats FM on biased validation but not on the unbiased `log_random` exposure
  set** → overfitting biased traffic, not learning preference → kill deep
  variants.
- **play_time censoring** is near-collinear with `long_view` (long_view derives
  from it); the censored target must NOT be reducible to the long_view label, or
  the aux head is a leakage channel the Critic must reject.
- **900s CPU timeout** is the hard feasibility gate; if a full DIN run exceeds
  it, cap `seq_len`/`epochs` before abandoning the family.
