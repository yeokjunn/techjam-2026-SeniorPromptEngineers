# Additions plan — innovation + workflow (2026-08-31)

Inputs: investigation/7-innovation.md (field survey), investigation/6-new-methods.md, the user's
two workflow picks. Lens per item: INNOVATION (verified rare?) vs WORKFLOW (helps results/ops).
"Pre-run safe" = cannot destabilize the imminent final submission run.

## Build tonight (pre-run safe)

### A. Pre-registration + calibration  [INNOVATION: no prior art found — lead Devpost claim]
The Researcher already emits hypothesis + expected_metric; extend the decision contract with
`predicted_delta` (float, vs current best) via prompt text (no schema risk: derived-optional key
in the registry-driven schema). Report side: per-iteration (predicted vs realized) pairs ->
`summary.calibration` {n, mean_abs_error, overconfidence_ratio} + a results.md table.
Files: roles.py prompt text (+1 grid-free optional key), research_controller summary block,
report hand-off to D. Effort S. Risk: near-zero (text + reporting).

### B. Negative-result artifact  [INNOVATION: rare — no surveyed agent ships one]
Auto-generate a "What we falsified" section: flat families/axes (from nodes), the noise floor,
the E[max] argument, published-effect-size context table (numbers from investigation reports,
checked in as a static reference file). Report-side only; fault-contained like render_reports.
Files: research_controller (small summary fields) + report.py hand-off to D (or an A-side
appendix file the journal links). Effort S. Risk: near-zero.

### C. Cross-run memory  [WORKFLOW: user pick; innovation-neutral (Agent K/DS-Agent precedent)]
Append-per-run digest (5 lines: families tried, score band, verdict, falsified axes) to
`research/campaign_log.md`; Researcher prompt includes the last ~3 digests in the STABLE prefix.
Note: #38's DiscoveryStore half-does this in-run — build the cross-run digest ON it, don't duplicate.
Files: research_controller (write at run end), roles.py (prompt block), one test file. Effort S-M.
Risk: low (additive; prompt grows ~300 tokens).

## After the final run (flagged or deferred)

### D. Pipelined loop / multi-hypothesis proposals  [WORKFLOW: user pick; wall-clock x~2]
Two parts: (1) Researcher emits 2-3 ranked hypotheses per call; Critic filters; best builds.
(2) Overlap i+1 proposal with i training (thread + queue around the executor).
HONESTY NOTE: last run used 9% of wall clock — time is not binding today; value rises once
memory/wider search give the agent more to try. Concurrency in the core loop = the riskiest
change on this list -> config-flagged (`pipeline: false` default), built AFTER the submission run.
Effort M-L. Risk: HIGH pre-run, moderate post-run.

### E. Red-team critic  [SKIP per scout: MLE-STAR precedent + trusted layer already blocks the
probed vectors]. Revisit only if judges ask about robustness testing.

### F. Self-tuning exploration (live sigma -> margin)  [SKIP pre-run: touches promotion logic.
The static loop-closure (measured sigma -> margin constant) is already shipped and claimable.]

## Recommended order
Tonight: A -> B -> C (one SDD wave, parallel-safe: A+C share roles.py -> sequence A then C, B parallel).
Post-submission: D behind a flag; E/F only on demand.
