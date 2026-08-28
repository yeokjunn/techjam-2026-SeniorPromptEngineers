# Senior Prompt Engineers — TechJam 2026

Autonomous ML research agent for the KuaiRand-Pure within-user ranking task.
The working target is `long_view`, evaluated with GAUC and nDCG@5.

## Current vertical slice

The first implementation provides:

- a deterministic experiment proposer with an interface that can later be backed
  by an LLM
- an isolated subprocess runner with timeouts and failure capture
- train/validation-only data loading that skips test dates before reading labels
- the untouched starter-kit evaluator and FM implementation
- random, item-popularity, and official-FM validation baselines
- JSONL iteration logs, best-checkpoint tracking, reflection, convergence, and
  global budget enforcement

## Setup

Python 3.9+ and NumPy are required:

```powershell
python -m pip install -r requirements.txt
```

Place the extracted KuaiRand-Pure files under:

```text
data/KuaiRand-Pure/data/
```

The runner never reads `data/judge/` and never loads rows after the validation
cutoff while developing models.

## Run the baseline agent

From the repository root:

```powershell
python -m src.agent.controller --config configs/baseline.json
```

The baseline ladder should be approximately:

| Experiment | Validation primary |
|---|---:|
| Random | 0.4834 |
| Item popularity | 0.5807 |
| Official FM | 0.6016 |

Each run creates `runs/<run-id>/` containing:

- `run_config.json` — frozen configuration
- `source_manifest.json` — SHA-256 revision of every project-owned source file
- `iterations.jsonl` — hypotheses, configurations, metrics, failures, recoveries,
  intervention counts, and reflections
- `best.json` — validation-best experiment and checkpoint
- `summary.json` — stop reason, resource use, and final result
- `stdout/` and `artifacts/` — generated logs and checkpoints

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Next experiment

After reproducing the official FM baseline, the first improvement should be a
pairwise BPR objective using positive/negative interactions from the same user.
This aligns optimization with the ranking metrics and avoids repeating the
starter kit's known unproductive capacity/static-feature ablations.

## Latest verified baseline

Run `20260828T141646Z_baseline` completed all three experiments with zero manual
interventions:

| Experiment | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Random, seed 0 | 0.4990 | 0.4663 | 0.4827 |
| Item popularity | 0.6387 | 0.5227 | 0.5807 |
| Official FM, seed 0 | 0.6671 | 0.5358 | **0.6015** |

The FM result reproduces the published validation baseline of `0.6016` within
rounding/noise. The run used 3 iterations, 0 LLM tokens, and approximately 212
seconds wall-clock on the current Windows/OneDrive workspace.

See `AGENTS.md` for data-leakage, judge-data, repository-safety, and experiment
logging rules.
