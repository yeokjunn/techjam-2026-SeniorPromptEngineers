# Multi-Task Auxiliary Targets

## Primary source

- Xiao Ma, Liqin Zhao, Guan Huang, Zhi Wang, Zelin Hu, Xiaoqiang Zhu, Kun Gai,
  "Entire Space Multi-Task Model: An Effective Approach for Estimating
  Post-Click Conversion Rate," SIGIR 2018. https://arxiv.org/abs/1804.07931
- Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, Ed H. Chi, "Modeling
  Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts,"
  KDD 2018.
- The starter kit ranks multi-task auxiliary signals third among untested
  directions (`kuairand-starter-kit/README.en.md:161-165`).

## Hypothesis

`long_view` is one of several behaviours the log records for the same
impression. `is_click` fires on 46.3% of train rows and correlates with
`long_view` at 0.76; scaled `play_time` correlates at 0.60. ESMM's argument is
that these related signals share structure, so training the shared embeddings
against them as well as against the ranking objective gives the embeddings more
supervision per row than the ranking loss alone provides — particularly for the
long tail of users and videos where `long_view` positives are scarce.

## Objective

The ranking loss is unchanged and the field set is unchanged. What is added is a
weighted auxiliary term over the *same* shared FM embeddings:

```text
loss = ranking_loss + aux_weight * mean_t( binary_cross_entropy(head_t, target_t) )
```

`src.models.features.build_aux_labels(rows, spec)` returns `(n, t)` float32
targets in `[0, 1]`, one column per enabled head:

| head | target |
|---|---|
| `is_click` | binary, 46.3% of train rows |
| `is_like` | binary, 1.9% of train rows |
| `play_time` | `log1p(play_time_ms)`, min-max scaled on train |

The auxiliary targets are **train-only by construction**. A loss touches train
rows only, so no validation or test path exists and none may be added:
`build_aux_labels` raises for any other split rather than returning something a
scorer could misuse.

Keep `k == 16` and the standard field set; this family varies the *supervision*,
not the capacity or the features.

## Safe initial search space

- Either trusted same-user sampler; `build_aux_labels` is mandatory
- FM embedding dimension fixed at 16 for attribution
- Learning rate: 0.0003, 0.0005, or 0.001
- Batch size: 2048 or 4096; one or two negatives per positive
- `aux_weight`: 0.1, 0.3, or 1.0
- Any subset of the three heads via `use_<head>`
- Epochs up to 40 — auxiliary targets add a loss term, not FM fields, so the
  per-epoch cost is close to plain BPR

## Known failure modes

- Auxiliary gradients swamping the ranking head. At `aux_weight = 1.0` the
  auxiliary term can dominate, and the model optimises click prediction while
  GAUC on `long_view` degrades. Start at 0.1–0.3 and check that validation
  primary still moves.
- `play_time_ms` is censored at video length: a fully-watched short video and a
  half-watched long one can produce similar values, so the scaled target is not a
  clean measure of interest (Zhao et al., KDD 2024, on watch-time bias). It is
  the weakest of the three heads and the first to disable.
- `is_like` fires on only 1.9% of train rows. As a lone auxiliary head it
  contributes very little signal and mostly adds variance.
- The shared-embedding assumption is the whole bet. If the auxiliary behaviour is
  driven by different factors than `long_view` — clicks driven by thumbnails,
  say — the shared parameters are pulled toward a competing objective and the
  ranking head gets worse, not better. MMoE exists precisely because that
  assumption often fails; with a single shared embedding table there is no gating
  to fall back on.
- Because the auxiliary heads read row-level behaviour columns, they are train-
  only. Any attempt to extend them to validation or test is a leak, not a feature.
