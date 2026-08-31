# UI and live-agent quickstart

This guide starts the read-only Streamlit dashboard and a bounded autonomous
research run in two terminals. Run every command from the repository root.

## What is ready

The current development loop can:

- reuse or reproduce the official FM validation baseline;
- call the Researcher, Critic, Builder, and Debugger roles;
- safety-check generated source and run candidate tests;
- train BPR or group-softmax candidates and compute trusted validation metrics;
- persist live stage transitions, structured Agent Notes, candidate patches,
  metrics, errors, repairs, and completed iteration records; and
- expose those artifacts in the dashboard with a five-second active-run refresh.

This is ready for observing development runs. Final submission gating and report
generation are implemented — see the committed [`runs/final/`](../runs/final/)
artifacts (`results.md`, `journal.md`, `submission.csv`) — but the dashboard
itself remains read-only observation tooling and cannot drive a run.

## 1. Open the repository

In PowerShell, from your local clone:

```powershell
cd <repository-root>
```

## 2. Install dependencies

Use the same Python interpreter for both the UI and agent:

```powershell
python -m pip install -r requirements-ui.txt
python -m streamlit version
```

Using `python -m streamlit` avoids accidentally invoking Streamlit from a
different Python environment.

## 3. Configure the API key

The autonomous research configuration uses the OpenAI provider. If `.env` does
not exist, create it from the template:

```powershell
Copy-Item .env.example .env
```

Open `.env` locally and set:

```text
OPENAI_API_KEY=your-key-here
```

Never commit `.env`, paste the key into terminal logs, or place it in a JSON
configuration. The runner loads `.env` without overriding an existing shell or
CI environment variable.

The default model in the smoke configuration is `gpt-5.4-nano`. If that model is
not available to your API project, change only the `llm.model` value in
`configs/ranking_losses_smoke.json` to a model your project can use.

## 4. Start the UI in terminal A

```powershell
python -m streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit, normally:

```text
http://localhost:8501
```

The dashboard initially displays existing completed runs. It never launches,
resumes, cancels, or modifies an experiment.

Optional: generate aggregate-only EDA data once before starting the UI:

```powershell
python -m src.ui.profile_data --config configs/ui.json
```

The profiler uses only the trusted train and validation date ranges and writes
aggregate output under `artifacts/ui/`.

## 5. Start a bounded agent run in terminal B

The recommended first run is bounded to 20 iterations, 8,640 seconds (2.4 h) of
wall-clock, and 100,000 total LLM tokens:

```powershell
python -m src.agent.controller --config configs/ranking_losses_smoke.json
```

This is a real autonomous run and can incur API usage. The limits are ceilings,
not expected consumption. The existing passing FM baseline should be reused; if
no valid baseline exists, the controller will reproduce it before research.

## 6. Follow the run in the UI

Once terminal B creates a directory ending in `_research` under `runs/`:

1. Refresh the browser page once so the new run appears in the sidebar.
2. Select the newest `_research` run.
3. Open the **Pipeline** tab.

The live overlay should move through coarse stages such as:

```text
Research → Preflight → Build → Safety + tests → Train + validate → Reflect → Persist
```

Use the collapsible panels to inspect:

- **Agent Notes** — hypothesis, rationale, evidence, decision, concerns, and next focus;
- **Changes** — changed files, added/deleted lines, and the candidate patch;
- **Errors & repairs** — safety, test, training, timeout, and Debugger information; and
- **Recent timeline** — active/completed/failed stage transitions.

The UI does not display raw hidden reasoning or full prompts. The finalized
iteration record remains the permanent source of truth.

## 7. Inspect the artifacts directly

For a run at `runs/<run-id>/`, the main observability files are:

```text
activity.json             latest atomic stage snapshot
activity.jsonl            append-only stage transition timeline
iterations.jsonl          finalized iteration records
state.json                resumable research state
experiment_tree.json      experiment nodes and parents
resources.json            time, token, and attempt usage
changes/*.json            per-candidate change summaries
changes/*.patch           reproducible candidate diffs
passes/*.json             structured role-call audit records
```

Do not inspect or authorize anything under `data/judge/` during development.

## 8. Run the full research budget later

After the two-iteration smoke run behaves correctly:

```powershell
python -m src.agent.controller --config configs/ranking_losses.json
```

The full configuration allows up to 50 iterations, 100 proposals, six hours of
wall-clock, and a 2,500,000-token engineering guard. Use it only when you
intentionally want a benchmark run.

## Resume an interrupted research run

If at least one state save completed and project-owned source has not changed:

```powershell
python -m src.agent.controller `
  --config configs/ranking_losses_smoke.json `
  --resume runs/<research-run-id>
```

Use the same configuration that created the run. Resume is intentionally rejected
when the frozen configuration or project source revision differs.

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Use the canonical root launcher from the repository root:

```powershell
python -m streamlit run streamlit_app.py
```

Both entry points are tested, but the root launcher is the documented default.

### `OPENAI_API_KEY is required`

Confirm `.env` exists at the repository root and contains a non-empty
`OPENAI_API_KEY`. Restart terminal B after editing it.

### The new run is not in the sidebar

Wait until `runs/<run-id>_research/` exists, then refresh the browser page once.
After selecting an active run, the Pipeline fragment refreshes every five seconds.

### The overlay says `Possibly stale`

The controller may still be inside a long LLM or training call, or it may have
been interrupted. Check terminal B and the run's latest `activity.json`. The UI
does not invent progress when no transition has been persisted.

### Port 8501 is already in use

Choose another port:

```powershell
python -m streamlit run streamlit_app.py --server.port=8502
```
