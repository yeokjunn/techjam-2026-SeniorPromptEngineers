# Senior Prompt Engineers — Devpost description

## How the solution addresses the problem statement

This is an autonomous ML research loop for KuaiRand-Pure `long_view` within-user ranking. A Researcher proposes an experiment, Critics check it before and after execution, a Builder generates code/tests, and a Debugger performs bounded, hypothesis-preserving repairs. Trusted code owns splits, subprocess isolation, GAUC/nDCG@5 evaluation, checkpoint promotion, budgets, logging, and the final label-free submission gate.

## Development tools used

Development used Git/GitHub, VS Code-compatible Python tooling, pytest/unittest, PowerShell on Windows, and POSIX shell scripts. The optional Streamlit dashboard is read-only and renders persisted run artifacts.

## APIs used

Live research uses the OpenAI Responses API with `gpt-5.5`, JSON-schema Structured Outputs (`strict: true`), optional `web_search`, and `prompt_cache_key`; offline tests replace it with `ScriptedProvider`. Official OpenAI documentation confirms these capabilities: [GPT-5.5 model](https://developers.openai.com/api/docs/models/gpt-5.5) and [Responses create API](https://developers.openai.com/api/reference/cli/resources/responses/methods/create).

## Libraries and frameworks used

Runtime dependencies are NumPy, the `openai` SDK, and `python-dotenv`; the standard library supplies `csv`, `ast`, `subprocess`, JSON, hashing, and `unittest`. Streamlit is optional. The trusted training path does not use pandas, PyTorch, LightGBM, scikit-learn, or external model services.

## Datasets and assets used

Training uses only the organizer-provided KuaiRand-Pure dataset and starter kit. No external training data or manual labels are used. Selection uses train and validation only; the final gate reads label-free test metadata in official order.

## Verified results and resources

| Committed run | GAUC | nDCG@5 | Primary | Delta vs 0.6016 |
|---|---:|---:|---:|---:|
| `20260829T041834051989Z_baseline` | 0.6671 | 0.5358 | 0.6015 | -0.0001 |
| `20260829T060130480764Z_research` | 0.6679 | 0.5360 | 0.6019 | +0.0003 |
| `kj_20260831T020238331867Z_research` | 0.6701 | 0.5373 | 0.6037 | +0.0021 |

The `20260829T060130480764Z_research` row is the scripted offline end-to-end run: 1 scored iteration and training attempt, 2,440 scripted tokens, 141.31 seconds wall-clock, 0 GPU-hours, and 0 manual interventions. It validates autonomy plumbing. The `kj_20260831T020238331867Z_research` row is the latest live autonomous run: it converged at iteration 12 under the official epsilon `0.002`, patience-3 rule, used 13 training attempts, 44 proposal attempts, 1,855,199 reported LLM tokens, 3,664.28 seconds wall-clock, 0 GPU-hours, and 0 manual interventions. Its final diverse ensemble reached validation primary `0.6048` with GAUC `0.6718` and nDCG@5 `0.5378`, and the label-free submission gate passed on 170,588 rows.

## Limitations and next steps

The latest live-LLM run converged and passed the label-free submission gate, but hidden-test score remains unknown and must not guide model selection. Validation lift is still modest, so next priorities are more diverse leakage-safe history contrast, more robust generated multi-task implementations, and score blending that reduces baseline over-weighting without changing the official GAUC/nDCG@5 criteria.
