# Campaign log

One five-line digest per research run, appended at run end and read back into the
Researcher's stable prompt prefix (last 3). Written by
`src/agent/discoveries.py::append_campaign_digest`; the per-hypothesis detail
lives in the discovery store beside it. Safe to prune from the top — the reader
only ever takes the most recent entries.

## run kj_20260830T185458454134Z_research
- families: bpr 0.602220-0.602220 (n=1, best d=+0.000750); group_softmax 0.602337-0.603748 (n=6, best d=+0.002278); history_features 0.602356-0.602356 (n=1, best d=+0.000887)
- verdict: best group_softmax_001 primary 0.603748 (d=+0.002278 vs baseline 0.601470); margin 0.0010 CLEARED
- falsified: bpr, history_features
- note: 8 scored / 9 iterations; stop_reason=converged; calibration_n=4
