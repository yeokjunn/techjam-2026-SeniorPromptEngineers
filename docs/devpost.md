# Senior Prompt Engineers: autonomous ML research agent for KuaiRand-Pure

We built an agent that tries to improve within-user `long_view` ranking on KuaiRand-Pure by actually doing research: proposing experiments, running them in a sandbox, checking the results are real, and writing up what worked and what didn't. The thing we care most about, and what we think sets this apart, is that the agent doesn't get to grade its own homework. Every number it reports can be checked against files it cannot fake.

## How it addresses the problem statement

Five LLM agents handle the thinking. An **EDA agent** goes first: it profiles the real dataset (label rates, history coverage, where the signal actually lives) and its report is fed to every proposal that follows. A **Researcher agent** proposes an experiment and has to predict, in advance, how much it will help. A **Critic agent** checks every proposal before and after execution and rejects ones that are confounded or unsafe (it caught a real one mid-run: a proposal that changed two variables at once). A **Builder agent** writes the candidate code, and a **Debugger agent** fixes failures. It's told what *kind* of failure happened, so it doesn't waste repairs on unfixable things like leaks.

Everything that touches correctness lives in trusted code the LLM can't modify: data splits, subprocess sandboxing, the official GAUC/nDCG@5 evaluation, token/time budgets, audit logs, and the label-free submission gate. Generated code can't read the dataset directly, can't see test labels, and can't report its own score.

## The part we're proud of

Early on we noticed our results clustered suspiciously close to the baseline, so we investigated instead of shipping the best number we'd seen. It turned out identical experiments with identical seeds were scoring ±0.001 apart (a Python hash-randomization bug in our sampler), and our loop was keeping the max of dozens of noisy tries. The expected max of pure noise was almost exactly the "improvement" we'd been reporting. That was uncomfortable, and we rebuilt around it:

- The harness now measures its own noise floor and only promotes a result that clears it. Promising candidates get replicated across seeds, and the summary reports the honest claim, the raw max, and the spread separately.
- The **Researcher agent**'s predictions get scored. On our evidence run it landed within epsilon 75% of the time, with a slight optimism bias, which we report rather than hide.
- Each run writes a `falsified.md` listing what was tried and measured flat. Each run also leaves a short digest that the next run reads; we watched the following run skip every dead end and go straight for the one family nobody had explored.
- The final submission is a blend of the near-best candidates, since that's the one place a free half-point of variance reduction matters.

## Results

Everything below comes from the committed final run in [`runs/final/`](../runs/final/results.md), a copy of live run `20260831T141845874517Z_research` with local paths rewritten to repo-relative form. Validation metrics, against the official baseline:

| Checkpoint | GAUC | nDCG@5 | Primary | vs official 0.6016 |
|---|---:|---:|---:|---:|
| Official FM baseline | 0.6674 | 0.5357 | 0.6016 | 0 |
| Best single checkpoint (`hist_prior_days_var_gs2_3b9a_seed1`) | 0.6707 | 0.5382 | 0.6045 | +0.0029 |
| 4-candidate blend, the designated submission | 0.6711 | 0.5383 | 0.6047 | +0.0031 |

The run tried two objectives that landed below baseline (top-weighted BPR, then group-softmax with history crossing), wrote them off, found the prior-days history-feature family, and replicated it across three seeds at +0.0020, +0.0022, and +0.0029, clearing our measured noise margin every time. It converged at iteration 6 of 50 under the official epsilon-0.002/patience-3 rule: 446,210 LLM tokens, about 30 minutes wall-clock, no GPU, and zero manual interventions (there's an intervention log; it's empty). The designated submission is [`runs/final/submission.csv`](../runs/final/submission.csv), which passed the label-free gate on all 170,588 test rows. Hidden-test scores are unknown by construction and never guided selection; the official hidden-test baseline is primary 0.5946.

Honest context: these gains look small until you know that published models on KuaiRand (including half-billion-parameter ones) land in the same GAUC band, that our measured ceiling on these features is about 0.606, and that roughly 42% of the metric can't be moved by any model (users whose impressions are all one label). The task is hard. We'd rather report a small, replicated, calibrated gain than a big number we can't defend.

## Tools, APIs, libraries, data

- **Tools:** Git/GitHub with per-owner branches and reviewed PRs, VS Code, pytest/unittest run with `-W error` (400+ tests), Playwright for dashboard QA, and a read-only Streamlit dashboard that shows the run's story, every LLM call, and the audit trail.
- **APIs:** DeepSeek (`deepseek-v4-flash`) over its OpenAI-compatible chat API; the exact config is frozen in [`runs/final/run_config.json`](../runs/final/run_config.json). The endpoint ignores server-side structured outputs, so we validate JSON schemas client-side and re-prompt on bad replies. The OpenAI Responses API is supported too. Offline, a scripted provider stands in for the model, so the whole test suite runs without a key.
- **Libraries:** NumPy, the `openai` SDK, `jsonschema`, `python-dotenv`, and the standard library (`ast` for the code-safety validator, `subprocess` for the sandbox). No pandas, PyTorch, or scikit-learn in the training path; pandas and Streamlit only in the dashboard.
- **Data:** the organizer-provided KuaiRand-Pure dataset and starter kit, nothing external, no manual labels. All learned features come from the train window only: we audited the platform's pre-aggregated video statistics file, found its window spans the test dates, and dropped it even though it scored better.

## What we'd do next

Better train-only item features, more robust generated multi-task implementations, a proper paired multi-seed test in the stopping rule, and overlapping the LLM calls with training so iterations cost less wall-clock.
