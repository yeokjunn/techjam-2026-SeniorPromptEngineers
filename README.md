# Senior Prompt Engineers — TechJam 2026

Autonomous ML research agent for the KuaiRand-Pure within-user ranking task.
The working target is `long_view`, evaluated with GAUC and nDCG@5.

## What is implemented

The repository provides an end-to-end autonomous research harness:

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
- OpenAI Responses API and OpenAI-compatible Chat Completions adapters, including
  GLM, with structured outputs, retry handling, and token accounting
- restricted agent-generated candidates with trusted metric computation
- BPR and same-user group-softmax research cards and audited sampling utilities
- coarse live-stage observability, structured agent decision notes, and immutable
  per-iteration candidate patches for the local dashboard

## Setup

Python 3.9+, NumPy, and the OpenAI Python SDK are required. POSIX:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/download_data.sh
```

PowerShell (download the dataset with Git Bash or WSL first):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place the extracted KuaiRand-Pure files under:

```text
data/KuaiRand-Pure/data/
```

Development uses only the official train/validation dates. The final gate loads
label-free test metadata, scores the validation-selected checkpoint once, and
never exposes test labels to model selection.

## Run the baseline agent

From the repository root:

```bash
python -m src.agent.controller --config configs/baseline.json
```

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

```bash
python -m pytest -q -W error
python -m unittest discover -s tests -v
```

```powershell
python -m pytest -q -W error
python -m unittest discover -s tests -v
```

The test suite uses a scripted provider and does not make paid API calls.

## Run the autonomous ranking-loss researcher

Create your local environment file from the committed template, then add your
API key to `.env`:

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY, then run:
python -m src.agent.controller --config configs/ranking_losses.json
```

```powershell
Copy-Item .env.example .env
# Edit .env and set OPENAI_API_KEY, then run:
python -m src.agent.controller --config configs/ranking_losses.json
```

`.env` is loaded automatically and ignored by Git. An `OPENAI_API_KEY` already
set in PowerShell or CI takes precedence over the file.

The research run uses `gpt-5.5` with medium reasoning and low verbosity by
default. Edit `configs/ranking_losses.json` to use a model available to your
OpenAI project. It has an explicit 150,000-token research budget, 50 training
attempts, two debugger repairs per candidate, and the official six-hour ceiling.

### Use GLM or another OpenAI-compatible endpoint

GLM uses an OpenAI-compatible **Chat Completions** endpoint rather than the
Responses API used by the default GPT configuration. A ready-to-edit GLM config
is provided at `configs/ranking_losses_glm.json`:

```bash
# Put the provider key in .env; never put it in the JSON config.
ZAI_API_KEY=your-provider-key
python -m src.agent.controller --config configs/ranking_losses_glm.json
```

The adapter also accepts `provider: "openai_compatible"` for other compatible
services. Configure `model` plus either `base_url`/`endpoint` in JSON or
`OPENAI_MODEL` plus `OPENAI_BASE_URL` in the environment. The API-key variable is
selected by `api_key_env` and defaults to `OPENAI_API_KEY`, so a generic setup can
look like:

```json
{
  "llm": {
    "provider": "openai_compatible",
    "model": "your-model-name",
    "api_key_env": "OPENAI_API_KEY",
    "endpoint": "https://provider.example/v1/chat/completions"
  }
}
```

The key must be issued by the service behind that endpoint; an OpenAI-issued key
does not authenticate to GLM. Provider-specific reasoning and search are opt-in
through `thinking` and `web_search_tool`. If the endpoint does not implement
those extensions, omit them. JSON responses are validated locally against the
same role schemas used by the GPT path. The example uses Z.AI's global API URL;
accounts on the mainland China platform should replace it with the base URL shown
for their account.

The run first reuses a passing official-FM run; if none exists it automatically
reproduces the baseline gate. It must execute at least one BPR and one
group-softmax candidate before convergence is allowed. An improvement greater
than `0.002` queues exact seed-1 and seed-2 replications.

Resume an interrupted research run with:

```bash
python -m src.agent.controller \
  --config configs/ranking_losses.json \
  --resume runs/<research-run-id>
```

```powershell
python -m src.agent.controller `
  --config configs/ranking_losses.json `
  --resume runs/<research-run-id>
```

There is not yet an `intervene` CLI. Operator actions must be recorded manually;
adding an append-only intervention command and derived counter is listed below as
a limitation.

## Run the read-only research dashboard

For the complete two-terminal UI and live-agent walkthrough, see
[`docs/UI_QUICKSTART.md`](docs/UI_QUICKSTART.md).

Install the optional UI dependencies:

```bash
python -m pip install -r requirements-ui.txt
```

Optionally generate the aggregate-only train/validation EDA profile:

```bash
python -m src.ui.profile_data --config configs/ui.json
```

Then launch the dashboard from the repository root:

```bash
python -m streamlit run streamlit_app.py
```

Launching the nested file directly is also supported:

```bash
python -m streamlit run src/ui/app.py
```

The Pipeline tab refreshes every five seconds only for a running research run.
Its translucent execution overlay shows the active coarse stage, elapsed time,
structured Agent Notes, finalized candidate changes, errors/repairs, and the
recent transition timeline. It does not stream raw reasoning or full prompts.

The dashboard is read-only: it cannot launch, resume, cancel, reconfigure, or
authorize a run. It discovers artifacts under `runs/`, reads the explicit
aggregate EDA profile under `artifacts/ui/`, and rejects judge-owned paths. Its
CSV uploader performs local schema/finiteness checks only; judge row alignment
is truthfully reported as unchecked.
### Run the offline smoke test

For a complete end-to-end test of the real training subprocess without an API
key or network access, run:

```bash
python -m src.agent.controller --config configs/offline_smoke.json
```

```powershell
python -m src.agent.controller --config configs/offline_smoke.json
```

This test uses a scripted LLM provider and fixed mock decisions. It runs exactly
one iteration with the BPR candidate, trains on real data, evaluates on the
validation set, and exits. Expected validation primary score is approximately
`0.602`. This configuration confirms that the full generate→train→evaluate path
works end-to-end before depending on live LLM calls.

### Generated candidate contract

The Builder writes only under `generated_experiments/<run-id>/<iteration>/`.
Candidate code defines:

```python
def run(context, parameters) -> CandidateOutput:
    ...
```

Candidates receive encoded train features/labels, label-free validation/test
features, and a trusted validation callback. They return validation and test scores,
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
- `activity.json` — atomic snapshot of the latest coarse stage transition
- `activity.jsonl` — append-only live-stage transition timeline
- `changes/` — per-iteration file/line summaries and reproducible candidate patches

Generated source is statically checked before writing/execution. Judge paths,
official evaluator imports, file/network/process access, dynamic execution, path
traversal, and non-approved imports are rejected. Candidate unit tests run before
training, and the Debugger gets at most two hypothesis-preserving repairs.

## Iteration, Training, and Convergence Definitions

The research loop tracks three distinct counts:

**Iteration** (`max_iterations`, cap 50)
: One scored candidate experiment. The loop increments the iteration count once a candidate has run to completion and been evaluated on validation data. Stop reason `candidate_budget_reached` fires when iteration count reaches `max_iterations`.

**Training attempt** (`max_training_attempts`)
: One executor subprocess call to train a candidate model. Repairs may create
multiple attempts for one candidate. `training_attempt_budget_reached` fires at
`max_training_attempts`.

**Proposal** (`max_proposals`, default `max_iterations × 2`)
: One research→build cycle, including rejected candidates. The Critic may reject a candidate's preflight check; rejected proposals increment the proposal counter but do not run training. Stop reason `proposal_budget_reached` fires when proposal count reaches `max_proposals`.

**Convergence** uses ε = **0.002** and patience **3**: after the meaningful best
is established, three successful scored candidates without an improvement greater
than ε stop the harness with `stop_reason: "converged"`. Both required families
must be covered first. The current summary does not yet publish a separate
`converged_official` verdict; this is a reporting limitation.

## First research agenda

The initial approved method catalog contains pairwise BPR and same-user
group-softmax. The Researcher chooses the order and parameters, the Critic checks
evidence and leakage, and the Builder generates each implementation. After both
families have coverage, the deterministic policy permits exploitation,
replication, or convergence.

## Latest verified baseline

Run `20260829T041834051989Z_baseline` completed all three experiments with zero manual
interventions:

| Experiment | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Random, seed 0 | 0.4990 | 0.4663 | 0.4827 |
| Item popularity | 0.6387 | 0.5227 | 0.5807 |
| Official FM, seed 0 | 0.6671 | 0.5358 | **0.6015** |

The FM result reproduces the published validation baseline of `0.6016` within
rounding/noise. The run used 3 iterations, 0 LLM tokens, 0 manual interventions,
and 32.02 seconds wall-clock. Source:
[`summary.json`](runs/20260829T041834051989Z_baseline/summary.json).

## Latest verified autonomous integration run

Committed run `20260829T060130480764Z_research` executed the real BPR trainer and
the label-free submission gate:

| GAUC | nDCG@5 | Primary | Delta vs official 0.6016 |
|---:|---:|---:|---:|
| 0.6679 | 0.5360 | **0.6019** | **+0.0003** |

It used 1 scored iteration/training attempt, 2,440 scripted tokens, 141.31 seconds,
0 GPU-hours, and 0 manual interventions. This is an offline integration result,
not a converged live-LLM claim. Sources: [`summary.json`](runs/20260829T060130480764Z_research/summary.json),
[`resources.json`](runs/20260829T060130480764Z_research/resources.json), and
[`results.json`](runs/20260829T060130480764Z_research/results.json).

Reproduce its rendered report with:

```bash
python -m src.agent.report runs/20260829T060130480764Z_research
```

## Architecture

```text
Conductor: research_controller.ResearchLoop
  ├─ Steward: official.load_train_valid + datacard.render_data_card
  ├─ Scientist/Critics: roles.research + critic_preflight/postflight
  ├─ Engineer/Medic: roles.build + roles.debug
  ├─ Sandbox: safety.validate_source → CandidateExecutor → run_candidate
  ├─ Scorekeeper/Gate: official_evaluate → gate.run_gate
  └─ Ledger/UI: ResearchAudit → report.render_reports → read-only Streamlit UI
```

## Limitations and next work

- No committed converged live-LLM run exists yet; the reported autonomous run is
  the deterministic one-iteration smoke test.
- The harness lacks the append-only `intervene` command and separate persisted
  official-convergence verdict.
- Only BPR and group-softmax are currently registered; sequence features,
  multi-task feedback, and repeated-seed variance estimates remain next steps.

## Team contributions

- A: research controller, budgets, recovery, convergence, and integration wiring.
- B: official loaders/evaluator, trusted candidate runner, and submission gate.
- C: LLM provider/retries, scripted offline path, prompt structure, and docs.
- D: data card, audit/report rendering, repository hygiene, and run journal.
- E: safety validation, family registry, sampling primitives, and method cards.

See `AGENTS.md` for data-leakage, judge-data, repository-safety, and experiment
logging rules.
