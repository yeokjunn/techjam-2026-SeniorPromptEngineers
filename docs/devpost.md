# Senior Prompt Engineers — Devpost description

## How the solution addresses the problem statement

This is an autonomous ML research loop for KuaiRand-Pure `long_view` within-user ranking. A Researcher proposes an experiment, Critics check it before and after execution, a Builder generates code/tests, and a Debugger performs bounded, hypothesis-preserving repairs. Trusted code owns splits, subprocess isolation, GAUC/nDCG@5 evaluation, checkpoint promotion, budgets, logging, and the final label-free submission gate.

## Development tools used

Development used Git/GitHub, VS Code-compatible Python tooling, pytest/unittest, PowerShell on Windows, and POSIX shell scripts. The optional Streamlit dashboard is read-only and renders persisted run artifacts.

## APIs used

The harness supports the OpenAI Responses API (JSON-schema Structured Outputs with `strict: true`, optional `web_search`, `prompt_cache_key`; see the [Responses create API](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)), any OpenAI-compatible Chat Completions endpoint, and GLM. The committed final run used `deepseek-v4-flash` through an OpenAI-compatible endpoint with locally validated structured outputs (frozen in [`runs/final/run_config.json`](../runs/final/run_config.json)); offline tests replace the live provider with `ScriptedProvider`.

## Libraries and frameworks used

Runtime dependencies are NumPy, the `openai` SDK, and `python-dotenv`; the standard library supplies `csv`, `ast`, `subprocess`, JSON, hashing, and `unittest`. Streamlit is optional. The trusted training path does not use pandas, PyTorch, LightGBM, scikit-learn, or external model services.

## Datasets and assets used

Training uses only the organizer-provided KuaiRand-Pure dataset and starter kit. No external training data or manual labels are used. Selection uses train and validation only; the final gate reads label-free test metadata in official order.

## Verified results and resources

All numbers below come from the committed final run [`runs/final/`](../runs/final/results.md)
(a copy of live run `20260831T115602777469Z_research` with local absolute paths
rewritten to repo-relative form). Validation metrics, scored against the official
baseline (GAUC 0.6674 / nDCG@5 0.5357 / primary 0.6016):

| Checkpoint | GAUC | nDCG@5 | Primary | Δ primary vs 0.6016 |
|---|---:|---:|---:|---:|
| Best single checkpoint (`gs_hard_neg_temp2_run1_seed2`) | 0.6710 | 0.5374 | 0.6042 | +0.0026 |
| 4-candidate blended ensemble — designated submission | 0.6717 | 0.5379 | 0.6048 | +0.0032 |

The run converged at **iteration 7 of 50** under the official ε = 0.002 /
patience-3 rule, before the 50-iteration and six-hour limits. Resource use:
**579,031 LLM tokens** (427,391 input / 151,640 output), **1,906 seconds
(0.53 h) wall-clock**, 11 training attempts, 11 proposal attempts, **0
GPU-hours**, and **0 manual interventions**. Three of seven iterations failed
and were handled autonomously by bounded, hypothesis-preserving debugger
repairs; the run then recovered and converged. The label-free submission gate
passed on all 170,588 test rows: the designated submission is
[`runs/final/submission.csv`](../runs/final/submission.csv) (SHA-256
`dcdfc43d…85218c2`), which encodes the blended-ensemble test scores. Hidden-test
scores are unknown by construction and did not guide model selection; the
official hidden-test baseline is primary 0.5946.

## Limitations and next steps

The final committed run converged and passed the label-free submission gate, but hidden-test score remains unknown and must not guide model selection. Validation lift is still modest (+0.0032 primary over the official baseline, close to seed variance), so next priorities are more diverse leakage-safe history contrast, more robust generated multi-task implementations, and score blending that reduces baseline over-weighting without changing the official GAUC/nDCG@5 criteria.
