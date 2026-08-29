# AGENTS.md

These instructions apply to the entire repository. They are guardrails for all
human and coding-agent work on the TechJam Track 2 submission.

## Project objective

Build an autonomous ML research agent for the required KuaiRand-Pure benchmark.
The agent must propose experiments, change or generate pipeline code, train,
evaluate on validation data, reflect on results, recover from failures, and stop
at convergence. A strong recommender model alone is not the complete project.

The authoritative task contract is:

- Relevance label: `long_view`
- Task: within-user ranking over logged impressions
- Metrics: GAUC and nDCG@5
- Primary score: `(GAUC + nDCG@5) / 2`
- Required benchmark: KuaiRand-Pure
- Development data: train and validation only
- Convergence: no improvement greater than `0.002` for 3 consecutive iterations
- Hard limits: 50 iterations and 6 hours per benchmark run
- Official validation baseline: primary `0.6016`
- Official hidden-test baseline: primary `0.5946`

The contradictory `click`, Recall@50, or NDCG@10 wording in one limits-table row
is not the working task definition. Follow the repeated conventions above and
`docs/problem_statement.md`.

## Never access hidden judge truth

- `data/judge/**` is not part of this repository; if such a path appears, treat it
  as judge-owned and opaque.
- Do not open, print, parse, search, summarize, copy, rename, edit, or delete any
  judge truth, answer key, hidden label, or similarly named file.
- In particular, never access files matching `*truth*`, `*label*`, `*answer*`, or
  `*ground_truth*` under `data/judge/`.
- Do not use hidden-test results for feature selection, hyperparameter tuning,
  early stopping, checkpoint selection, ensembling, or debugging.
- The real test surface is the 2022-04-29 through 2022-05-08 segment of
  `log_standard_4_22_to_5_08_pure.csv`. The final Gate may automatically read
  only its metadata and score the validation-best checkpoint once. This is not a
  manual intervention and must never influence model selection.
- If hidden labels become visible accidentally, stop and tell the user. Do not
  incorporate any information learned from them.

## Read-only reference files

Do not modify these files unless the user explicitly asks to update the official
reference material:

- `docs/problem_statement.md`
- `kuairand-starter-kit/README.md`
- `kuairand-starter-kit/README.en.md`
- `kuairand-starter-kit/evaluate.py`
- `kuairand-starter-kit/baseline.py`
- `kuairand-starter-kit/data.py`
- `kuairand-starter-kit/submit.py`
- `kuairand-starter-kit/ablation_features.py`
- `kuairand-starter-kit/baseline_scores.json`

Do not modify raw KuaiRand dataset files. Treat extracted CSV files and downloaded
archives as immutable inputs. Extend or wrap the starter kit from new files so the
official baseline remains reproducible.

## Where new work belongs

Prefer this layout for project-owned code:

- `src/agent/` — controller, proposer, runner, reflector, recovery logic
- `src/models/` — ranking models and losses
- `src/features/` — leakage-safe feature and sequence construction
- `src/evaluation/` — wrappers around the official evaluator; do not reimplement
  metric conventions unless required for testing
- `configs/` — reproducible experiment and benchmark configuration
- `tests/` — unit and integration tests
- `runs/` — lightweight run logs and summaries
- `artifacts/` — generated predictions and checkpoints; normally gitignored

If these directories do not exist, create them only as needed. Do not reorganize
the repository merely for aesthetics.

## Data and leakage rules

- Use no external training data.
- Fit vocabularies, scalers, buckets, aggregates, and learned preprocessing using
  training data only.
- Validation may be used only for evaluation, early stopping, and experiment
  selection.
- Test data may be scored only by a train-selected, validation-selected pipeline.
- For temporal and behavioural features, use only events that occurred before the
  row being predicted. Never aggregate future interactions into a feature.
- Do not join public labels back into judge rows, infer hidden labels from file
  ordering, or exploit duplicate rows as a label side channel.
- Preserve official date splits and row order.

## Evaluation rules

- Use the official `kuairand-starter-kit/evaluate.py` as the metric authority.
- Rank impressions only against other impressions belonging to the same user.
- Keep zero-positive users in nDCG with a score of zero.
- Compute GAUC only for users with both positive and negative impressions, using
  the official weighting.
- Select the final checkpoint by validation score, never by hidden-test score.
- Before trusting experiments, reproduce the validation ladder approximately:
  random `0.4834`, popularity `0.5807`, FM `0.6016` primary.
- Do not claim an improvement smaller than normal seed variance without repeated
  runs. Record seeds and report mean/variance when feasible.

## Autonomous-run requirements

Every iteration must record:

- iteration number and parent experiment
- hypothesis and rationale
- exact configuration and command
- code diff or immutable code revision
- validation GAUC, nDCG@5, and primary score
- elapsed time and LLM token usage when available
- checkpoint and artifact locations
- errors, retries, recovery actions, and final status
- whether a human intervened

The controller must:

- preserve the validation-best checkpoint even if later runs regress
- enforce per-experiment timeouts and the global iteration/wall-clock budgets
- reject NaN/Inf metrics and prediction scores
- handle failed experiments without corrupting the previous best state
- stop according to the official convergence rule
- make runs reproducible from configuration plus seed

## Experiment priorities

Prioritize fast, evidence-driven experiments in this order:

1. Reproduce and lock the official baselines.
2. Pairwise BPR or listwise objectives aligned with ranking metrics.
3. Pointwise/pairwise score blending.
4. Leakage-safe temporal and user-behaviour sequence features.
5. Multi-task learning using in-dataset auxiliary feedback.
6. More complex architectures only after the autonomous loop is reliable.

Do not spend iterations repeating the starter kit's known dead ends: simply
adding all static features or only increasing FM embedding dimension. Do not work
on KuaiRand-1k or KuaiRand-27k until KuaiRand-Pure is working end to end and beats
the validation baseline consistently.

## Submission safety

Final prediction CSVs must have exactly:

```text
row_id,user_id,video_id,score
```

Requirements:

- one row per evaluation row
- `row_id` starts at zero and increases without gaps
- user and video IDs preserve official row alignment
- scores are finite real numbers
- duplicate `(user_id, video_id)` pairs remain separate rows
- run the official submission checker before delivery

## Repository safety

- Preserve user-authored and unrelated working-tree changes.
- Do not delete or overwrite files unless the user explicitly requests it.
- Do not run destructive Git commands such as `git reset --hard` or discard
  changes to make the tree clean.
- Do not commit datasets, downloaded archives, secrets, large checkpoints, or
  generated prediction files unless explicitly requested.
- Never put API keys or credentials in source, configs, logs, or prompts.
- Prefer small, reviewable changes and verify behavior after each material edit.
- Do not change task definitions or metric conventions to make results look
  better.

## Documentation and reporting

Keep the root README, reproduction commands, dependency setup, results table,
limitations, team contributions, and architecture description current as the
implementation evolves. Final reporting must include validation metrics and delta
over baseline, iteration count, wall-clock time, LLM tokens, GPU-hours if any, and
the number of manual interventions.
