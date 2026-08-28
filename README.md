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
- a single role-based autonomous research loop using Researcher, Critic, Builder,
  and Debugger passes over shared persisted memory
- OpenAI Responses API structured outputs, optional primary-source web-search
  fallback, retry handling, and exact token accounting
- restricted agent-generated candidates with trusted metric computation
- BPR and same-user group-softmax research cards and audited sampling utilities

## Setup

Python 3.9+, NumPy, and the OpenAI Python SDK are required:

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

This command remains fully deterministic and does not require an API key.

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

The test suite uses a scripted provider and does not make paid API calls.

## Run the autonomous ranking-loss researcher

Create your local environment file from the committed template, then add your
API key to `.env`:

```powershell
Copy-Item .env.example .env
# Edit .env and set OPENAI_API_KEY, then run:
python -m src.agent.controller --config configs/ranking_losses.json
```

`.env` is loaded automatically and ignored by Git. An `OPENAI_API_KEY` already
set in PowerShell or CI takes precedence over the file.

The research run uses `gpt-5.5` with medium reasoning and low verbosity by
default. Edit `configs/ranking_losses.json` to use a model available to your
OpenAI project. It has an explicit 150,000-token research budget, eight training
attempts, two debugger repairs per candidate, and the official six-hour ceiling.

The run first reuses a passing official-FM run; if none exists it automatically
reproduces the baseline gate. It must execute at least one BPR and one
group-softmax candidate before convergence is allowed. An improvement greater
than `0.002` queues exact seed-1 and seed-2 replications.

Resume an interrupted research run with:

```powershell
python -m src.agent.controller `
  --config configs/ranking_losses.json `
  --resume runs/<research-run-id>
```

The agent never reads `data/judge/`. Final judge prediction generation remains a
separate, explicit user-authorized step.

### Generated candidate contract

The Builder writes only under `generated_experiments/<run-id>/<iteration>/`.
Candidate code defines:

```python
def run(context, parameters) -> CandidateOutput:
    ...
```

Candidates receive encoded train features/labels, validation features without
labels, and a trusted validation callback for early stopping. They return scores,
checkpoint arrays, a training trace, and diagnostics. The trusted worker checks
shape and finiteness and computes final GAUC/nDCG@5 itself; candidate-reported
metric values are ignored.

### Research audit files

Research runs add:

- `state.json` — atomic resumable state
- `experiment_tree.json` — branch, family, parameters, outcome, and parent
- `research_memory.jsonl` — role-call records and recovery events
- `resources.json` — time, tokens, iterations, interventions, and GPU-hours
- `passes/` — structured prompts and outputs for each role
- `interventions.json` — explicit human-intervention ledger

Generated source is statically checked before writing/execution. Judge paths,
official evaluator imports, file/network/process access, dynamic execution, path
traversal, and non-approved imports are rejected. Candidate unit tests run before
training, and the Debugger gets at most two hypothesis-preserving repairs.

## First research agenda

The initial approved method catalog contains pairwise BPR and same-user
group-softmax. The Researcher chooses the order and parameters, the Critic checks
evidence and leakage, and the Builder generates each implementation. After both
families have coverage, the deterministic policy permits exploitation,
replication, or convergence.

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
