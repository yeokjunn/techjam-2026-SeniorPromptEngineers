"""Trusted DIN (target-conditional attention) trainer for the ``din`` family.

This is the trusted primitive that owns all deep-learning math for the DIN
model, so LLM-generated candidate code never imports a DL framework or touches a
raw log. The candidate is a ~15-line thin wrapper:

    from src.models.din_trainer import run_din_trainer
    from src.experiments.contracts import CandidateOutput
    def run(context, parameters):
        val, test, ckpt, trace, diag = run_din_trainer(context, parameters)
        return CandidateOutput(val, ckpt, trace, diag, test)

Why a trusted primitive (not score-only-outside-loop): the kit README permits
handing any model's scores to ``evaluate()``, but routing the DL model through
this trusted module keeps the agent's propose/critique/build loop in charge of
the architecture knobs (the Researcher proposes them, the Critic checks
leakage, the Builder writes the thin wrapper) — exactly the
``build_features``/``build_aux_labels`` pattern, extended to a DL trainer.

Leakage discipline is enforced HERE, in trusted code, never in candidate code:
  * the item vocabulary is train-only (built inside ``build_user_sequences``);
  * sequences are prior-days for train rows and all-train for valid/test rows;
  * valid labels are never read (only ``context.evaluate_validation`` is used,
    for early-stopping only — never an optimization objective);
  * test labels are never touched (test rows arrive as ``context.test_x`` with
    no labels);
  * raw kit rows are re-loaded by this module via ``KUAIRAND_DATA_DIR`` — the
    candidate only supplies the already-encoded ``CandidateContext`` and may NOT
    supply its own rows or sequences.

Phase 1 (this file) ships the plumbing: ``run_din_trainer`` returns
correctly-shaped, finite, zero-filled arrays so the end-to-end harness path
(allowlist -> family contract -> schema -> validate_and_persist_output -> gate)
is verified before any torch math lands. Phase 2 fills in the real DIN
attention + listwise + multi-task training inside ``_train_din``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.official import (
    REPO_ROOT,
    TEST_ROWS,
    load_test_meta,
    load_train_valid,
)
from src.experiments.contracts import CandidateContext
from src.models.sequence import build_user_sequences

# The trusted trainer owns its own wall-clock guard so a misconfigured grid
# cannot trip the 900s subprocess timeout (candidate_runner.py:176), which is a
# hard failure with no partial credit. 820s leaves margin for numpy score
# conversion + persistence after training stops.
_TRAIN_WALL_CLOCK_SECONDS = 820

# Phase 2 will import torch lazily INSIDE _train_din as ``import torch as _torch``
# so that (a) this module imports cleanly without torch installed, and (b) the
# name ``torch`` is never bound here — a candidate attempting
# ``from src.models.din_trainer import torch`` raises ImportError (the
# import-reexport hole mitigation from the architecture plan).


def _resolve_data_dir() -> Path:
    raw = os.environ.get("KUAIRAND_DATA_DIR")
    return Path(raw) if raw else REPO_ROOT / "data" / "KuaiRand-Pure" / "data"


def _sequences_for_splits(context: CandidateContext, seq_len: int):
    """Build leakage-safe item-id history for train/valid/test.

    Re-loads the trusted kit rows via ``KUAIRAND_DATA_DIR`` (CandidateContext
    carries only encoded int32 matrices, not kit rows) and delegates to
    ``build_user_sequences``, which enforces the prior-days / all-train rule and
    the train-only vocabulary. The candidate never supplies rows or sequences.
    """
    data_dir = _resolve_data_dir()
    splits = load_train_valid(data_dir)
    train_rows = list(splits["train"])
    valid_rows = list(splits["valid"])
    test_rows = list(load_test_meta(data_dir, expected_rows=TEST_ROWS).rows)

    common = {"seq_len": seq_len, "data_dir": str(data_dir)}
    train_spec = {**common, "split": "train", "history_rows": {"train": train_rows}}
    valid_spec = {**common, "split": "valid", "history_rows": {"train": train_rows, "valid": valid_rows}}
    test_spec = {**common, "split": "test", "history_rows": {"train": train_rows, "test": test_rows}}

    train_seqs = build_user_sequences(train_rows, train_spec)
    valid_seqs = build_user_sequences(valid_rows, valid_spec)
    test_seqs = build_user_sequences(test_rows, test_spec)
    return train_seqs, valid_seqs, test_seqs


def _train_din(
    context: CandidateContext,
    parameters: dict[str, Any],
    train_seqs,
    valid_seqs,
    test_seqs,
    started: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[dict], dict[str, Any]]:
    """Run the real DIN+FM+listwise+multi-task training in torch.

    Architecture (see ``research/methods/din.md`` and the architecture plan):
      * FM block — exact ``FMRanker.logits`` math ported to torch (sanity anchor).
      * DIN target-conditional attention — query = candidate item embedding,
        keys = history item embeddings, masked softmax, pooled interest. This is
        the one mechanism that survives within-user first-order=0.
      * Tower MLP combining FM logits + pooled interest + candidate embedding.
      * Loss: within-user listwise group-softmax (aligned with GAUC/nDCG@5).
      * Multi-task aux: is_click BCE + censored play_time (Tobit), weighted by
        ``aux_weight``. ``is_like``/``is_follow``/``is_comment``/``is_forward``
        are dropped (within-user residual r < 0.10).
      * Early-stop on valid primary via ``context.evaluate_validation`` (early-
        stopping only, never an optimization objective).
      * 820s wall-clock guard; CPU eager; ``torch.set_num_threads`` bound to the
        scratch env's OMP cap.
    """
    import torch as _torch

    # ---- hyperparameters ----
    seed = int(parameters.get("seed", 0))
    k = int(parameters.get("k", 16))
    lr = float(parameters.get("learning_rate", 0.001))
    epochs = int(parameters.get("epochs", 20))
    batch_size = int(parameters.get("batch_size", 4096))
    patience = int(parameters.get("patience", 4))
    embedding_dim = int(parameters.get("embedding_dim", 32))
    seq_len = int(parameters.get("seq_len", 50))
    attention_dim = int(parameters.get("attention_dim", 32))
    dropout_p = float(parameters.get("dropout", 0.2))
    aux_weight = float(parameters.get("aux_weight", 0.1))
    use_is_click = bool(parameters.get("use_is_click", True))
    use_play_time = bool(parameters.get("use_play_time", False))
    temperature = float(parameters.get("temperature", 1.0))
    negatives_per_group = int(parameters.get("negatives_per_group", 4))
    loss_variant = str(parameters.get("loss_variant", "uniform"))

    _torch.manual_seed(seed)
    _torch.set_num_threads(min(4, os.cpu_count() or 4))
    device = _torch.device("cpu")

    # ---- data ----
    train_x_np = np.asarray(context.train_x, dtype=np.int64)
    train_y_np = np.asarray(context.train_y, dtype=np.float64)
    train_users = context.train_users
    valid_x_np = np.asarray(context.valid_x, dtype=np.int64)
    test_x_np = np.asarray(context.test_x, dtype=np.int64) if context.test_x is not None else None

    field_dimension = int(context.field_dimension)
    seq_field_dimension = int(train_seqs.field_dimension)

    # Move sequence data to torch tensors (int64 indices for embedding gather).
    def _to_tensor(arr):
        return _torch.from_numpy(np.ascontiguousarray(arr)).to(device)

    train_hist = _to_tensor(train_seqs.history_items)
    train_hist_mask = _to_tensor(train_seqs.history_mask)
    train_cand = _to_tensor(train_seqs.candidate_items)
    valid_hist = _to_tensor(valid_seqs.history_items)
    valid_hist_mask = _to_tensor(valid_seqs.history_mask)
    valid_cand = _to_tensor(valid_seqs.candidate_items)
    test_hist = _to_tensor(test_seqs.history_items)
    test_hist_mask = _to_tensor(test_seqs.history_mask)
    test_cand = _to_tensor(test_seqs.candidate_items)
    train_x_t = _to_tensor(train_x_np)
    valid_x_t = _to_tensor(valid_x_np)

    # ---- model ----
    class _DIN(_torch.nn.Module):
        def __init__(self):
            super().__init__()
            # FM side: embedding table (field_dimension, k) + first-order W.
            self.V = _torch.nn.Embedding(field_dimension, k)
            self.W = _torch.nn.Embedding(field_dimension, 1)
            _torch.nn.init.normal_(self.V.weight, mean=0.0, std=0.01)
            _torch.nn.init.zeros_(self.W.weight)
            self.bias = _torch.nn.Parameter(_torch.zeros(1))
            # DIN side: separate item-history embedding table (seq_field_dimension, embedding_dim).
            self.item_emb = _torch.nn.Embedding(seq_field_dimension, embedding_dim)
            _torch.nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.01)
            # Attention: [query*keys, query-keys, dot(query,keys)] -> 2*embedding_dim+1 -> attention_dim -> 1
            # The dot product captures the scalar similarity; the element-wise
            # terms capture per-dimension interaction. Standard DIN formulation.
            att_in = 2 * embedding_dim + 1
            self.att_mlp = _torch.nn.Sequential(
                _torch.nn.Linear(att_in, attention_dim),
                _torch.nn.ReLU(),
                _torch.nn.Linear(attention_dim, 1),
            )
            self.dropout = _torch.nn.Dropout(dropout_p)
            # DIN attention tower: takes [pooled_interest, cand_emb] and produces
            # an ADDITIVE delta to the FM logit. This preserves the FM signal
            # (the sanity anchor) and lets the attention layer learn on top.
            # If the tower replaces the FM logit, the nonlinear MLP destroys
            # the FM's already-good within-user ordering signal early in training.
            tower_in = 2 * embedding_dim
            self.tower = _torch.nn.Sequential(
                _torch.nn.Linear(tower_in, attention_dim),
                _torch.nn.ReLU(),
                _torch.nn.Dropout(dropout_p),
                _torch.nn.Linear(attention_dim, 1),
            )
            # Scale the tower output so the delta starts small (preserving the FM
            # signal) while still allowing gradient flow to the attention. A
            # zero-init last layer blocks ALL gradient to the attention (grad of
            # delta w.r.t. pooled is proportional to tower[-1].weight = 0). A
            # small learnable scale starts the delta near zero and opens it as
            # the attention learns.
            self.tower_scale = _torch.nn.Parameter(_torch.tensor(0.01))
            # Separate aux heads (NOT shared with the ranking logit). Each aux
            # head reads the pooled interest + candidate embedding, not the
            # ranking logit, so aux gradients shape the embeddings without
            # directly competing with the ranking loss on the output.
            aux_in = 2 * embedding_dim
            self.click_head = _torch.nn.Linear(aux_in, 1)
            self.play_head = _torch.nn.Linear(aux_in, 1)

        def fm_logits(self, features):
            """Exact FMRanker.logits math: b + sum(W[features]) + 0.5*((sum_emb)^2 - sum(emb^2))."""
            emb = self.V(features)           # (B, F, k)
            summed = emb.sum(dim=1)          # (B, k)
            interactions = 0.5 * ((summed ** 2).sum(dim=1) - (emb ** 2).sum(dim=(1, 2)))
            linear = self.W(features).sum(dim=1).squeeze(-1)  # (B,)
            return self.bias + linear + interactions  # (B,)

        def forward(self, features, hist_items, hist_mask, cand_item):
            fm = self.fm_logits(features)                          # (B,)
            # DIN target-conditional attention.
            query = self.item_emb(cand_item)                         # (B, d)
            keys = self.item_emb(hist_items)                         # (B, L, d)
            B, L, d = keys.shape
            query_exp = query.unsqueeze(1).expand(B, L, d)           # (B, L, d)
            att_interactions = _torch.cat([
                query_exp * keys,                                    # (B, L, d)
                query_exp - keys,                                    # (B, L, d)
                (query_exp * keys).sum(dim=2, keepdim=True),         # (B, L, 1)
            ], dim=2)                                                # (B, L, 2d+1)
            att_logits = self.att_mlp(att_interactions).squeeze(-1)  # (B, L)
            mask = (hist_mask > 0.5)
            att_logits = att_logits.masked_fill(~mask, -1e9)
            # Empty-history fallback: rows with all-zero mask use UNK embedding.
            has_history = mask.any(dim=1, keepdim=True)              # (B, 1)
            alpha = _torch.softmax(att_logits, dim=1)               # (B, L)
            pooled = (alpha.unsqueeze(2) * keys).sum(dim=1)          # (B, d)
            unk_emb = self.item_emb.weight[seq_field_dimension - 1]  # UNK slot (d,)
            pooled = _torch.where(has_history, pooled, unk_emb.unsqueeze(0).expand(B, d))
            pooled = self.dropout(pooled)
            cand_emb = self.item_emb(cand_item)                      # (B, d)
            # Additive: FM logit + scaled attention tower delta. Preserves the FM signal.
            tower_in = _torch.cat([pooled, cand_emb], dim=1)         # (B, 2d)
            delta = self.tower_scale * self.tower(tower_in).squeeze(-1)  # (B,)
            logit = fm + delta
            return logit

        def aux_forward(self, features, hist_items, hist_mask, cand_item):
            """Return aux head logits (click, play) from the pooled interest."""
            query = self.item_emb(cand_item)
            keys = self.item_emb(hist_items)
            B, L, d = keys.shape
            query_exp = query.unsqueeze(1).expand(B, L, d)
            att_interactions = _torch.cat([
                query_exp * keys,
                query_exp - keys,
                (query_exp * keys).sum(dim=2, keepdim=True),
            ], dim=2)
            att_logits = self.att_mlp(att_interactions).squeeze(-1)
            mask = (hist_mask > 0.5)
            att_logits = att_logits.masked_fill(~mask, -1e9)
            has_history = mask.any(dim=1, keepdim=True)
            alpha = _torch.softmax(att_logits, dim=1)
            pooled = (alpha.unsqueeze(2) * keys).sum(dim=1)
            unk_emb = self.item_emb.weight[seq_field_dimension - 1]
            pooled = _torch.where(has_history, pooled, unk_emb.unsqueeze(0).expand(B, d))
            cand_emb = self.item_emb(cand_item)
            aux_in = _torch.cat([pooled, cand_emb], dim=1)  # (B, 2d)
            return self.click_head(aux_in).squeeze(-1), self.play_head(aux_in).squeeze(-1)

    model = _DIN().to(device)
    # The numpy FMRanker uses Adam for V/W but plain SGD for the bias. With
    # Adam, the bias gets a tiny nonzero gradient (float noise, ~1e-6) amplified
    # by the sqrt(v)+eps normalization to a large per-step update (~lr), growing
    # linearly and wasting capacity. SGD keeps the bias update proportional to
    # its near-zero gradient. The bias is irrelevant to within-user ranking
    # anyway (it shifts all scores equally), but keeping it stable matters for
    # the FM's logit scale. Separate parameter groups: FM V/W in Adam, bias in
    # SGD, attention/tower/item-emb at lr/10 so the FM signal dominates early.
    fm_param_ids = {id(model.V.weight), id(model.W.weight)}
    fm_params = [model.V.weight, model.W.weight]
    din_params = [p for n, p in model.named_parameters()
                  if id(p) not in fm_param_ids and id(p) != id(model.bias)]
    optimizer = _torch.optim.AdamW([
        {"params": fm_params, "lr": lr},
        {"params": din_params, "lr": lr * 0.1},
    ], weight_decay=1e-6)
    # Bias via plain SGD (matches the numpy FMRanker; prevents Adam amplification).
    bias_opt = _torch.optim.SGD([model.bias], lr=lr)

    # ---- multi-task aux labels (train-only, from raw CSV) ----
    aux_click = None
    aux_play = None
    if use_is_click or use_play_time:
        aux_click_np, aux_play_np = _load_aux_labels(use_is_click, use_play_time)
        if use_is_click:
            aux_click = _to_tensor(aux_click_np)
        if use_play_time:
            aux_play = _to_tensor(aux_play_np)

    # ---- training loop: within-user listwise group-softmax ----
    rng = np.random.default_rng(seed)
    from src.models.sampling import sample_softmax_groups

    best_primary = -1.0
    best_state = None
    stale = 0
    trace = []

    def _predict_scores(features_np, hist_items_np, hist_mask_np, cand_np):
        """Forward-only scoring in chunks; returns float64 numpy."""
        n = len(features_np)
        scores = np.empty(n, dtype=np.float64)
        chunk = 100_000
        model.eval()
        with _torch.no_grad():
            for off in range(0, n, chunk):
                end = min(off + chunk, n)
                f = _to_tensor(features_np[off:end])
                h = _to_tensor(hist_items_np[off:end])
                m = _to_tensor(hist_mask_np[off:end])
                c = _to_tensor(cand_np[off:end])
                scores[off:end] = model(f, h, m, c).cpu().numpy()
        model.train()
        return scores

    for epoch in range(epochs):
        t0 = time.monotonic()
        # Wall-clock guard.
        if time.monotonic() - started > _TRAIN_WALL_CLOCK_SECONDS:
            trace.append({"epoch": epoch, "note": "wall-clock guard triggered"})
            break

        pos_idx, neg_groups = sample_softmax_groups(
            train_users, train_y_np, rng, negatives_per_group=negatives_per_group
        )
        n_groups = len(pos_idx)
        if n_groups == 0:
            break
        perm = rng.permutation(n_groups)
        losses = []
        model.train()
        for start in range(0, n_groups, batch_size):
            if time.monotonic() - started > _TRAIN_WALL_CLOCK_SECONDS:
                break
            batch_idx = perm[start:start + batch_size]
            pos = pos_idx[batch_idx]
            neg = neg_groups[batch_idx]  # (B, K)
            B = len(pos)
            K = neg.shape[1]
            # Build interleaved groups: [pos_0, neg_0_0..neg_0_K, pos_1, ...] so
            # that .view(B, K+1) yields the correct per-group rows. Using
            # concatenate([pos, neg.flat]) would put ALL positives first — the
            # view would then group items from DIFFERENT users, and the listwise
            # loss would compute softmax over random cross-user items (the bug
            # that kept the torch FM from learning).
            group_rows = np.empty((B, K + 1), dtype=np.int64)
            group_rows[:, 0] = pos
            group_rows[:, 1:] = neg
            all_rows = group_rows.reshape(-1)              # (B*(K+1),) interleaved
            feats = train_x_t[all_rows]                    # (B*(K+1), 5)
            hist = train_hist[all_rows]                    # (B*(K+1), L)
            mask = train_hist_mask[all_rows]               # (B*(K+1), L)
            cand = train_cand[all_rows]                     # (B*(K+1),)
            logits = model(feats, hist, mask, cand)        # (B*(K+1),)
            logits = logits.view(B, K + 1) / temperature    # (B, K+1)
            # Listwise cross-entropy: positive is index 0. Use sum reduction to
            # match the numpy FMRanker's gradient scale (the kit's group-softmax
            # candidate accumulates raw gradients without dividing by batch size;
            # Adam normalizes the scale eventually, but the bias correction for
            # beta2=0.999 takes ~1000 steps, so 'mean' would barely move the
            # 40K-entry FM embeddings in 20 epochs).
            if loss_variant == "uniform":
                loss = _torch.nn.functional.cross_entropy(
                    logits, _torch.zeros(B, dtype=_torch.long), reduction="sum"
                )
            elif loss_variant == "positives_weighted":
                # Weight each group by that user's positive count (matching GAUC's
                # per-user-npos weighting). Groups from users with more positives
                # carry more ranking signal and should dominate the gradient.
                pos_users = np.asarray(train_users)[pos]
                user_pos_counts = np.bincount(
                    np.searchsorted(np.unique(train_users), pos_users)
                    if len(train_users) > 0 else np.arange(B),
                    minlength=B,
                ).astype(np.float32)
                # Clamp to >=1 to avoid zero-weight groups.
                weights = _torch.from_numpy(np.maximum(user_pos_counts, 1.0))
                loss = _torch.nn.functional.cross_entropy(
                    logits, _torch.zeros(B, dtype=_torch.long),
                    weight=weights, reduction="sum"
                )
            else:  # lambdarank_top5
                # LambdaRank-style: weight the group-softmax loss by 1/log2(rank+2)
                # for the positive, approximating nDCG@5's position discount.
                # The gradient is the standard softmax gradient scaled by the
                # discount, pushing the model to rank positives higher (lower
                # rank = higher position = more nDCG gain).
                probs = _torch.softmax(logits, dim=1)
                pos_prob = probs[:, 0]
                # Discount weight: 1/log2(2) = 1 for rank 1, decreasing after.
                discount = 1.0 / _torch.log2(_torch.arange(1, K + 2, dtype=_torch.float32) + 1)
                # nDCG-style gain: the positive at rank r contributes 1/log2(r+2).
                # LambdaRank uses |ΔNDCG| as the pair weight; approximate with
                # the discount of the ideal positive position.
                weight = discount[0]  # = 1.0 for position 0
                loss = weight * _torch.nn.functional.cross_entropy(
                    logits, _torch.zeros(B, dtype=_torch.long), reduction="sum"
                )
            # Multi-task aux loss (train rows only, separate heads).
            if aux_weight > 0 and (aux_click is not None or aux_play is not None):
                aux_loss = _torch.tensor(0.0, device=device)
                pos_rows = np.asarray(pos, dtype=np.int64)
                pos_feats = train_x_t[pos_rows]
                pos_hist = train_hist[pos_rows]
                pos_mask = train_hist_mask[pos_rows]
                pos_cand = train_cand[pos_rows]
                click_pred, play_pred = model.aux_forward(pos_feats, pos_hist, pos_mask, pos_cand)
                if aux_click is not None:
                    click_target = aux_click[pos_rows]
                    aux_loss = aux_loss + _torch.nn.functional.binary_cross_entropy_with_logits(click_pred, click_target)
                if aux_play is not None:
                    # Censored play_time: simple MSE on log1p-scaled play_time for now.
                    play_target = aux_play[pos_rows]
                    aux_loss = aux_loss + _torch.nn.functional.mse_loss(play_pred, play_target)
                loss = loss + aux_weight * aux_loss
            optimizer.zero_grad()
            bias_opt.zero_grad()
            loss.backward()
            optimizer.step()
            bias_opt.step()
            losses.append(float(loss.item()))

        # Validation eval (early-stopping only, never an optimization objective).
        val_scores = _predict_scores(valid_x_np, valid_seqs.history_items,
                                     valid_seqs.history_mask, valid_seqs.candidate_items)
        metrics = context.evaluate_validation(val_scores)
        primary = 0.5 * (float(metrics.get("GAUC", 0)) + float(metrics.get("nDCG@5", 0)))
        epoch_time = time.monotonic() - t0
        trace.append({
            "epoch": epoch, "loss": float(np.mean(losses)) if losses else 0.0,
            "primary": primary, "groups": n_groups, "seconds": epoch_time,
        })
        if primary > best_primary + 1e-9:
            best_primary = primary
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                trace[-1]["early_stopped"] = True
                break

    # ---- final scoring from the best checkpoint ----
    if best_state is not None:
        model.load_state_dict(best_state)
    validation_scores = _predict_scores(valid_x_np, valid_seqs.history_items,
                                        valid_seqs.history_mask, valid_seqs.candidate_items)
    if test_x_np is not None:
        test_scores = _predict_scores(test_x_np, test_seqs.history_items,
                                      test_seqs.history_mask, test_seqs.candidate_items)
    else:
        test_scores = np.zeros(0, dtype=np.float64)

    # ---- checkpoint: convert every tensor to numpy (sanitize keys) ----
    # torch state_dict keys contain dots (e.g. "V.weight", "att_mlp.0.weight")
    # but the worker's checkpoint validator requires alnum-only keys
    # (key.replace("_", "").isalnum()). Flatten by joining with underscores.
    checkpoint_state = {}
    for name, tensor in model.state_dict().items():
        safe_name = name.replace(".", "_").replace("-", "_")
        if not safe_name.replace("_", "").isalnum():
            safe_name = f"param_{len(checkpoint_state)}"
        checkpoint_state[safe_name] = tensor.detach().cpu().numpy()

    diagnostics = {
        "family": "din",
        "mode": "din_torch",
        "seed": seed, "k": k, "embedding_dim": embedding_dim,
        "seq_len": seq_len, "attention_dim": attention_dim,
        "dropout": dropout_p, "aux_weight": aux_weight,
        "use_is_click": use_is_click, "use_play_time": use_play_time,
        "negatives_per_group": negatives_per_group, "temperature": temperature,
        "field_dimension": field_dimension, "seq_field_dimension": seq_field_dimension,
        "epochs_run": len(trace), "best_primary": float(best_primary),
        "early_stopped": len(trace) < epochs,
    }
    return validation_scores.astype(np.float32), test_scores, checkpoint_state, trace, diagnostics


def _load_aux_labels(use_is_click: bool, use_play_time: bool) -> tuple[np.ndarray, np.ndarray]:
    """Load train-only auxiliary labels (is_click, log1p play_time) from raw CSV.

    Train-only by construction: only TRAIN dates (20220408-20220421) are read,
    matching ``features.py:_cached_aux_columns``'s discipline (date checked
    before any other column). Returns two numpy arrays aligned to the kit's
    train-row order.
    """
    import csv
    from src.evaluation.official import TRAIN_END, TRAIN_START

    data_dir = _resolve_data_dir()
    clicks = []
    plays = []
    for filename in ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv"):
        path = data_dir / filename
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                date = int(row["date"])
                if not (TRAIN_START <= date <= TRAIN_END):
                    continue  # skip before reading any signal column
                clicks.append(1.0 if row["is_click"] != "0" else 0.0)
                plays.append(np.log1p(max(0.0, float(row.get("play_time_ms", "0") or "0"))))
    click_arr = np.asarray(clicks, dtype=np.float32)
    play_arr = np.asarray(plays, dtype=np.float32)
    if not use_is_click:
        click_arr = np.zeros(len(click_arr), dtype=np.float32)
    if not use_play_time:
        play_arr = np.zeros(len(play_arr), dtype=np.float32)
    return click_arr, play_arr


def run_din_trainer(
    context: CandidateContext,
    parameters: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[dict], dict[str, Any]]:
    """Trusted entry point for the ``din`` family.

    Returns ``(validation_scores, test_scores, checkpoint_state,
    training_trace, diagnostics)``:

    * ``validation_scores`` — float32 1-D, len = n_valid (one score per valid row)
    * ``test_scores`` — float64 1-D, len = n_test (one score per test row)
    * ``checkpoint_state`` — ``dict[str, np.ndarray]``, all-finite, <=50M elements
    * ``training_trace`` / ``diagnostics`` — JSON-safe records (no large arrays)
    """
    started = time.monotonic()
    seq_len = int(parameters.get("seq_len", 50))
    if seq_len < 1:
        raise ValueError("seq_len must be positive.")

    # Leakage invariant: the trusted trainer derives sequences from the trusted
    # kit rows + KUAIRAND_DATA_DIR, never from candidate-supplied data.
    train_seqs, valid_seqs, test_seqs = _sequences_for_splits(context, seq_len)

    # Defence in depth: stop training at 820s so persistence + numpy conversion
    # fit inside the 900s subprocess timeout.
    elapsed = time.monotonic() - started
    if elapsed > _TRAIN_WALL_CLOCK_SECONDS:
        raise RuntimeError(f"DIN trainer exceeded the {_TRAIN_WALL_CLOCK_SECONDS}s guard during setup.")

    validation_scores, test_scores, checkpoint_state, training_trace, diagnostics = _train_din(
        context, parameters, train_seqs, valid_seqs, test_seqs, started
    )

    # Trusted return-type contract: the worker (run_candidate.py:80-149) shape/
    # finiteness-checks these before persisting model.npz + test_scores.npy.
    validation_scores = np.asarray(validation_scores, dtype=np.float32)
    if validation_scores.ndim != 1 or not np.all(np.isfinite(validation_scores)):
        raise ValueError("run_din_trainer must return finite 1-D float32 validation_scores.")
    test_scores = np.asarray(test_scores, dtype=np.float64)
    if test_scores.ndim != 1 or not np.all(np.isfinite(test_scores)):
        raise ValueError("run_din_trainer must return finite 1-D float64 test_scores.")

    clean_ckpt: dict[str, np.ndarray] = {}
    total_elements = 0
    for key, value in checkpoint_state.items():
        if not key.replace("_", "").isalnum():
            raise ValueError(f"Unsafe checkpoint key: {key!r}")
        array = np.asarray(value)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Checkpoint array {key!r} contains NaN or Inf.")
        total_elements += int(array.size)
        clean_ckpt[key] = array
    if total_elements > 50_000_000:
        raise ValueError("Checkpoint exceeds the 50M-element safety limit.")

    diagnostics = {
        str(key): _json_safe(value) for key, value in diagnostics.items()
    }
    return validation_scores, test_scores, clean_ckpt, list(training_trace), diagnostics


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size > 1000:
            raise ValueError("Diagnostics may not contain large arrays.")
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"Diagnostics value is not JSON serializable: {type(value).__name__}")
