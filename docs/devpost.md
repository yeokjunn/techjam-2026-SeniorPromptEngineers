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
(a copy of live run `20260831T141845874517Z_research` with local absolute paths
rewritten to repo-relative form). Validation metrics, scored against the official
baseline (GAUC 0.6674 / nDCG@5 0.5357 / primary 0.6016):

| Checkpoint | GAUC | nDCG@5 | Primary | Δ primary vs 0.6016 |
|---|---:|---:|---:|---:|
| Best single checkpoint (`hist_prior_days_var_gs2_3b9a_seed1`) | 0.6707 | 0.5382 | 0.6045 | +0.0029 |
| 4-candidate blend — designated submission | 0.6711 | 0.5383 | 0.6047 | +0.0031 |

The run converged at **iteration 6 of 50** under the official ε = 0.002 /
patience-3 rule, before the 50-iteration and six-hour limits. Resource use:
**446,210 LLM tokens** (311,696 input / 134,514 output), **1,772 seconds
(0.49 h) wall-clock**, 8 training attempts, 7 proposal attempts, **0
GPU-hours**, and **0 manual interventions**, with 0 failed iterations. The loop
first tried two below-baseline objectives (top-weighted BPR, group-softmax with
history crossing), then found the prior-days history-feature family, replicated
it across two seeds, and converged on a multi-task probe. The label-free
submission gate passed on all 170,588 test rows: the designated submission is
[`runs/final/submission.csv`](../runs/final/submission.csv) (SHA-256
`ffda31d4…ca6e`), which encodes the blended test scores. Hidden-test scores are
unknown by construction and did not guide model selection; the official
hidden-test baseline is primary 0.5946.

## Limitations and next steps

The final committed run converged and passed the label-free submission gate, but hidden-test score remains unknown and must not guide model selection. Validation lift is still modest (+0.0031 primary over the official baseline, close to seed variance), so next priorities are more diverse leakage-safe history contrast, more robust generated multi-task implementations, and score blending that reduces baseline over-weighting without changing the official GAUC/nDCG@5 criteria.
