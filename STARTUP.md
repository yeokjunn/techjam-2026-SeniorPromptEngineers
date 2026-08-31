# Quick Startup Guide

## Prerequisites

```powershell
# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies (first time only)
pip install -r requirements.txt
pip install -r requirements-ui.txt   # only if running the dashboard
```

## Download Dataset (first time only)

```powershell
python scripts/download_data.py
```

## Available Configs

| Config | Purpose |
|---|---|
| `configs/baseline.json` | Deterministic baseline ladder (random → popularity → FM). No API key needed. |
| `configs/offline_smoke.json` | Scripted-LLM end-to-end smoke test. No API key needed. |
| `configs/ranking_losses_smoke.json` | LLM-driven research loop (OpenAI), 20 iterations, smoke-test budgets |
| `configs/ranking_losses.json` | Full research loop (OpenAI, 50 iterations / 6 h caps) |
| `configs/ranking_losses_glm.json` | Full research loop (GLM) |

## Run the Agent

```powershell
# Baseline ladder (deterministic, no LLM)
python -m src.agent.controller --config configs/baseline.json

# Smoke test — 20 iterations, short budgets (quick sanity check)
python -m src.agent.controller --config configs/ranking_losses_smoke.json

# Full run — 50 iterations, 6h wall clock, 2 replication seeds (GLM)
python -m src.agent.controller --config configs/ranking_losses_glm.json
```

## Run the Dashboard (separate terminal)

```powershell
streamlit run streamlit_app.py
```

The dashboard auto-refreshes every 5 seconds while a run is active. It reads from `runs/` and is read-only.

## Run Tests

```powershell
python -m pytest -q
```

