# Senior Prompt Engineers — TechJam 2026

Autonomous ML research agent for the KuaiRand-Pure within-user ranking task.
The working target is `long_view`, evaluated with GAUC and nDCG@5.

## What is implemented

The repository provides an end-to-end autonomous research harness:

- a live LLM-driven proposer (Researcher/Critic/Builder/Debugger, plus an EDA pass),
  with a deterministic scripted provider for offline testing
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
- a user-level validation holdout so early stopping and reporting use disjoint halves
- four registered research families -- BPR, same-user group-softmax, history features
  and multi-task auxiliary targets -- each with a method card and audited trusted helpers
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
OpenAI project. Full benchmark configs use the official 50-iteration and
six-hour caps. Per-iteration execution is capped at 432 seconds so 50 iterations
sum to six hours. LLM tokens, training attempts, GPU-hours, and proposal counts
are recorded for feasibility and debugging; they are not official benchmark stop
conditions unless a config explicitly sets an engineering guard.

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
reproduces the baseline gate. A promising family is now given a short controlled
follow-up window before broad exploration, and a failed or clearly regressed new
family falls back to the current validation-best family. Family coverage is
reported for auditability, but it is not a hard lock that overrides attribution
of the best lead. An improvement greater than `0.002` queues exact seed-1 and
seed-2 replications.

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

**Training attempt** (`max_training_attempts`, optional engineering guard)
: One executor subprocess call to train a candidate model. Repairs may create
multiple attempts for one candidate. By default this is telemetry only; if a
config explicitly sets `max_training_attempts`, `training_attempt_budget_reached`
fires at that engineering guard.

**Proposal** (`max_proposals`, default `max_iterations × 2`)
: One research→build cycle, including rejected candidates. The Critic may reject a candidate's preflight check; rejected proposals increment the proposal counter but do not run training. Stop reason `proposal_budget_reached` fires when proposal count reaches `max_proposals`.

**Convergence** uses ε = **0.002** and patience **3** over successful validation
scores only. Failed experiments do not count as convergence evidence because
they have no validation score. The run summary reports the official convergence
verdict separately from the harness `stop_reason`; the harness may spend an
extra pass on queued replications or best-family follow-ups before stopping.

## First research agenda

The approved method catalog currently contains pairwise BPR, same-user
group-softmax, leakage-safe history features, and multi-task auxiliary feedback.
The Researcher chooses the order and parameters, the Critic checks evidence and
leakage, and the Builder generates each implementation. The deterministic policy
prioritizes exploiting and attributing the current validation-best lead before
moving into broader family exploration.

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

Latest local live run `20260830T070229778050Z_research` completed four iterations
and preserved the best validation checkpoint from the first history-features
candidate:

| Iteration | Candidate | Family | Status | Primary | Notes |
|---:|---|---|---|---:|---|
| 1 | `cand_hf_tabcross_prior_days_v1` | `history_features` | success | **0.6031** | Best result; all six history feature groups enabled. |
| 2 | `cand_bpr_sameuser_v1` | `bpr` | failed by controller timeout | — | `result.json` existed with primary `0.6023`, but the controller finalized it as failed after the timeout; previous best was preserved. |
| 3 | `cand_bpr_sameuser_v2` | `bpr` | failed | — | Candidate bug: treated the validation metrics dict as a float. |
| 4 | `cand_bpr_sameuser_v3` | `bpr` | success | 0.5929 | Pure BPR regressed; next policy should fall back to history-feature ablations. |

Best validation metrics from that run:

| GAUC | nDCG@5 | Primary | Delta vs official 0.6016 |
|---:|---:|---:|---:|
| 0.6695 | 0.5366 | **0.6031** | **+0.0015** |

It used 4 iterations, 4 training attempts, 111,450 reported LLM tokens,
1,879.65 seconds wall-clock, 0 GPU-hours, and 0 manual interventions. It stopped
on the global wall-clock budget, not convergence. The run artifacts remain
generated outputs under `runs/` and are intentionally not committed in this PR.

Previously committed integration run `20260829T060130480764Z_research` executed
the real BPR trainer and the label-free submission gate:

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

## Results

The committed converged run is `runs/final/` (promote another with
`python scripts/promote_final_run.py runs/<id>_research`). Validation, KuaiRand-Pure:

| Metric | Official baseline | Agent best | Δ |
|---|---|---|---|
| GAUC | 0.6674 | 0.6705 | +0.0031 |
| nDCG@5 | 0.5357 | 0.5375 | +0.0018 |
| **primary** | **0.6016** | **0.6040** | **+0.0024** |

`score_dataset` (the judging formula, mean of the two metric deltas) = **+0.0025** on
validation. The ranked score applies the same formula to the hidden test, which is scored
once and is not computable here.

**Read that delta with the caveat below.** Progress should be judged against the 0.8645
attainable ceiling rather than 1.0 — 27.1% of test users have no positive label and score
nDCG 0 for any model — so +0.0024 is roughly 0.9% of the baseline-to-ceiling span.

Resource usage to reach the converged result (Feasibility reporting):

| | |
|---|---|
| LLM tokens (input + output) | 354,290 |
| Agent wall-clock | 49.4 min |
| GPU-hours | 0.0 |
| Iterations used | 9 of 50 (stopped on the convergence rule, not the cap) |
| **Manual interventions** | **0** |

### How much of that delta is real

Every candidate early-stops on validation and reports that same validation number, and the
run reports the best of ~14 candidates — roughly 112 selections against the same 124,909
rows. Measured across 21 candidates, a reported score sits **+0.0025 above the median of the
epoch curve it came from**, which is the size of the delta itself.

`src/evaluation/holdout.py` therefore splits validation by user into a selection half (early
stopping, candidate choice) and a reporting half (never consulted during training), and the
baseline is scored on both. The halves are not equally hard — the reporting half is +0.0053
easier — so the comparison must be made within a half:

| comparison | delta |
|---|---|
| candidate report-half vs baseline **full** validation | +0.0049 ← wrong, mismatched populations |
| candidate report-half vs baseline **report** half | **+0.0023** ← apples to apples |

The honest figure agrees with the full-validation +0.0024, so the improvement appears real
and consistent rather than a selection artefact. Runs before `runs/final/` predate the split
and carry no `report_primary`.

## Limitations and next work

- **The improvement is small and near the noise floor.** The baseline's own 5-seed std is
  0.0008 and candidates are scored on a single seed, so a +0.002 delta is roughly 3σ of
  seed noise. Scoring each candidate on 2–3 seeds and comparing means is the first thing we
  would add; the budget allows it (9 of 50 iterations used).
- **Web search never fires.** 0 search calls across 207 role calls: the provider in use
  accepts the tool parameter and no-ops it, so the Researcher's evidence is limited to the
  four method cards in `research/methods/` that we wrote. Two cited URLs in earlier run logs
  did not resolve. Cited-source validation, and a provider that genuinely searches, are
  needed before the agent can be said to draw on the literature autonomously.
- **The model class is fixed.** Candidates must use the trusted `FMRanker`; the competition
  permits PyTorch, LightGBM and others. We probed the obvious extension directly rather than
  assuming: a numpy DeepFM (h=32/64) peaked *lower* than the FM (0.6025 vs 0.6030) and
  overfit from epoch 1. Combined with the kit's measured k-sweep, capacity and nonlinearity
  both look like dead ends on 1.14M rows.
- **Two registered families are barely tested.** `history_features` and `multi_task` were
  unreachable until a prompt fix; `history_features` has since produced six successful
  candidates in a partial run, `multi_task` still has none.
- **What we measured and ruled out**, so others need not repeat it: L2 regularisation moves
  the peak by 0.00005 across three orders of magnitude (Adam normalises gradient-L2 away);
  cold start is not a factor (1.6% of validation rows contain an unseen id); the binding
  constraint is that validation primary peaks at epoch 3–5 and then decays below baseline.

## Team contributions

- A: research controller, budgets, recovery, convergence, and integration wiring.
- B: official loaders/evaluator, trusted candidate runner, and submission gate.
- C: LLM provider/retries, scripted offline path, prompt structure, and docs.
- D: data card, audit/report rendering, repository hygiene, and run journal.
- E: safety validation, family registry, sampling primitives, and method cards.

See `AGENTS.md` for data-leakage, judge-data, repository-safety, and experiment
logging rules.
