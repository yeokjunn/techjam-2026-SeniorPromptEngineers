# Role Skill: Hyperparameter Optimization and Experiment Tracking

Use hyperparameter optimization only when it is cheaper and more informative than a targeted
manual experiment. The registered tools are optional research-environment tools:

- Optuna: preferred default for define-by-run studies, pruning, and reproducible seeded samplers.
- scikit-optimize (`skopt`): useful for small bounded Bayesian searches with explicit dimensions.
- Hyperopt: useful for legacy TPE searches; keep the search space and random seed explicit.
- MLflow: use for tracking parameters, metrics, artifacts, and the selected validation checkpoint.

These tools are controller-side or research-side capabilities. They must not be imported into
generated `candidate.py` unless the runtime contract explicitly allows them; the candidate sandbox
remains restricted to its declared imports. Do not use an optimizer to access hidden-test scores,
future rows, labels, or judge-owned files. Optimize only on train-derived features and official
validation GAUC/nDCG@5, and select the final checkpoint by validation primary.

Every study must declare:

- the objective and exact search dimensions, including bounds and distributions;
- a fixed seed, trial budget, timeout, and parallelism limit within the run budget;
- the immutable parent configuration and the reason tuning is expected to help;
- pruning or early-stopping rules that do not inspect hidden-test results;
- the best validation metrics, trial number, full parameters, and artifact/checkpoint path.

Prefer a small, family-specific search over broad joint tuning. Keep embedding dimension fixed
when attribution requires it, avoid retuning known dead ends, and do not claim an improvement
without recording seed variance when practical. MLflow runs must not contain credentials or raw
datasets; log lightweight metadata and artifact paths instead.

Suggested workflow: define the approved search space, run a bounded seeded study, report all
trials and the validation-selected configuration in the iteration ledger, then hand the selected
configuration to the normal Builder/trusted worker path for final evaluation.
