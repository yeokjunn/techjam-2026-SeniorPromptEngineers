# Review Instructions

Instructions for the automated code reviewer. Human reviewers may also
use this as a checklist.

## Skip paths

Do not review changes confined to these paths — mention them only if
they break something outside them:

- `**/package-lock.json`, `**/pnpm-lock.yaml`, `**/yarn.lock`
- `**/dist/**`, `**/build/**`, `**/coverage/**`
- `**/*.min.js`, `**/*.min.css`, `**/*.map`
- `**/__pycache__/**`, `**/*.pyc`, `.venv/**`
- Datasets and artifacts: `data/**`, `*.npy`, `*.npz`, `*.csv` over 1 MB

## Always flag as Important

Security and contract violations, regardless of size:

- `subprocess` / `os.system` / `exec` / `eval` with `shell=True` or
  unsanitized input — command injection
- SQL built by string concatenation; filesystem paths joined from
  user-controlled input without normalization
- Hardcoded secrets, tokens, or credentials
- **Test-label leakage** (project-specific): any code path that loads
  test-set labels, or fits encoders/bucket edges/vocabularies on test
  data instead of train-only — this violates the isolation contract
  from the harness design spec

## Acceptance criteria

Cross-check the PR description's stated acceptance criteria against the
actual diff. Flag any claim the code does not deliver, and cite the PR
description when doing so.

## Severity calibration

- Important: correctness bugs, contract violations, security issues
- Nit: style, naming, minor preferences — cap at 5 per review, and
  don't spend review effort on them before the Important items
