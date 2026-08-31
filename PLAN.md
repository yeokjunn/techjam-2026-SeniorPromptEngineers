# Single-Agent Autonomous Research Loop

## Summary

The end-to-end agent is implemented: role-based proposing/critique/build/repair,
trusted subprocess training and evaluation, persistence/resume, a final gate, and
a read-only dashboard. Remaining limitations are tracked in the README.

Build one role-based research agent first:

```text
Observe → Research → Critique → Generate code → Validate → Execute
                                      ↑                 ↓
                                  Debug/repair ← Failure
                                                        ↓
                                              Reflect + remember
```

Use one shared memory and one OpenAI client. Researcher, critic, builder, and debugger are separate structured passes, not independent agents. This minimizes tokens, handoff failures, and manual interventions while preserving a future multi-agent upgrade path.

## Key Changes

### 1. OpenAI research runtime

- Add a provider interface with an OpenAI Responses API implementation.
- Default to a configurable model (`gpt-5.4-nano` in `configs/ranking_losses.json`), low reasoning, low verbosity, `store=False`.
- Read `OPENAI_API_KEY` only from the environment; never log it.
- Use JSON-schema Structured Outputs for:
  - `ResearchDecision`
  - `CriticDecision`
  - `CandidateManifest`
  - `DebugDecision`
- Record response ID, model, input/output/total tokens, cached tokens, latency, tool calls, and retry count.
- Use stateless calls built from the persisted research memory instead of relying on conversational state.
- Retry transient API failures for 5 total attempts with 2–60 second bounded
  exponential backoff, honoring numeric `Retry-After` values.
- Use the Responses API web-search tool only when the curated catalog has an evidence gap. This follows current [official OpenAI Responses API guidance](https://developers.openai.com/api/reference/cli/resources/beta/subresources/responses).

### 2. Curated research agenda and policy

Create approved method cards for BPR and group-softmax containing primary citations, objective formulas, applicability, tunable parameters, and known failure modes.

The first autonomous run must:

1. Pass or reuse the official FM baseline gate.
2. Attempt at least one BPR candidate.
3. Attempt at least one group-softmax candidate.
4. After both families are covered, explore variants from the better family.
5. Replicate any improvement greater than `0.002` using seeds 1 and 2.
6. Stop after three non-meaningful successful iterations, 50 scored candidates,
   50 training attempts, 100 proposals, or six hours.

The researcher chooses:

- Which family to attempt first
- Same-user sampling details
- Negative count
- Temperature
- Batch size and learning rate
- Whether to exploit, branch, or replicate

It may not change the dataset split, label, evaluator, budgets, or hidden-test policy.

### 3. Generated candidate contract and safety

Replace the baseline-only worker with a generic candidate runner.

Each generated candidate lives under:

```text
generated_experiments/<run-id>/<iteration>/
```

Candidate code receives trusted train/validation structures and parameters. It returns:

```python
CandidateOutput(
    validation_scores: np.ndarray,
    test_scores: np.ndarray | None,
    checkpoint_state: dict[str, np.ndarray],
    training_trace: list[dict],
    diagnostics: dict,
)
```

The trusted runner—not candidate code—must:

- Validate score count, ordering, dtype, and finiteness
- Calculate GAUC and nDCG@5 with the official evaluator
- Persist the checkpoint and metrics
- Prevent candidates from supplying or overriding their own metrics

Add a deterministic safety validator before execution:

- Permit imports from NumPy, standard mathematical utilities, and approved project model primitives.
- Reject filesystem, network, process, dynamic-import, `eval`, `exec`, and path-traversal operations.
- Reject references to `data/judge`, test truth, the official evaluator implementation, or files outside the generated candidate directory.
- Reject attempts to edit trusted reference files.
- Execute candidates in subprocesses with time and output limits.

### 4. Role passes, memory, and recovery

Implement these sequential passes:

- **Researcher:** observes the experiment tree and proposes an evidence-backed hypothesis.
- **Critic preflight:** checks novelty, attribution, leakage, compute feasibility, and whether the experiment isolates a meaningful variable.
- **Builder:** generates candidate code, tests, and configuration.
- **Deterministic validator:** enforces paths, AST policy, schemas, and task invariants.
- **Debugger:** receives validation/runtime errors and may repair the same candidate twice. It cannot silently change the scientific hypothesis.
- **Critic post-run:** interprets GAUC, nDCG@5, primary delta, training behavior, and whether the hypothesis was supported.
- **Search policy:** deterministically promotes the validation-best candidate and decides explore/exploit/replicate/stop from structured critic output.

Persist atomically:

- `state.json` for resume
- `experiment_tree.json`
- `research_memory.jsonl`
- `iterations.jsonl`
- `interventions.jsonl`
- `resources.json`
- per-pass prompts, structured outputs, evidence URLs, patches, tests, stdout/stderr, and checkpoints

A syntax/test repair does not consume a scientific iteration, but every repair consumes wall-clock and tokens and is recorded. Every training attempt counts toward the iteration budget.

## Public Interfaces

Extend the current agent types with:

```python
ResearchDecision
CandidateManifest
CriticDecision
DebugDecision
CandidateOutput
ExperimentNode
RunState
TokenUsage
```

Replace `ConfigProposer` in research runs with:

```python
class ResearchProposer:
    def propose(self, state: RunState) -> ResearchDecision | None:
        ...
```

Keep `ConfigProposer` for deterministic baseline and offline tests.

Provide two commands:

```bash
# Deterministic existing baseline
python -m src.agent.controller --config configs/baseline.json

# Autonomous ranking-loss research
python -m src.agent.controller --config configs/ranking_losses.json
```

The research configuration defaults to 50 candidate iterations, 50 training
attempts, 100 proposals, two debugger repairs per candidate, a 2,500,000-token
engineering guard, and the official six-hour ceiling.

## Test Plan

- Structured-output parsing rejects missing or malformed fields.
- Token accounting aggregates all researcher, critic, builder, debugger, and web-search calls.
- Curated evidence is used before web fallback.
- Web-derived evidence retains title and URL attribution.
- Path traversal, judge-data access, forbidden imports, filesystem calls, and evaluator modification are rejected.
- Candidate-supplied fake metrics cannot bypass trusted evaluation.
- Score length mismatch and NaN/Inf predictions fail safely.
- BPR sampling uses positive/negative rows from the same user.
- Group-softmax groups contain one positive and the configured same-user negatives.
- Users lacking both classes are skipped safely.
- Debugger repairs are capped at two and preserve the hypothesis.
- Resume continues without duplicating completed iterations.
- Both BPR and group-softmax receive initial coverage before convergence is allowed.
- Replication is triggered for improvements greater than `0.002`.
- Mocked OpenAI responses exercise a complete offline loop.
- One real-data acceptance run reproduces FM, generates both ranking-loss candidates, records zero human interventions, and produces a trusted comparison table.

## Assumptions

- Start with a single role-based agent; multi-agent branching is deferred until measured evidence shows it improves results enough to justify extra tokens and complexity.
- BPR and group-softmax are seeded as an approved research agenda, but implementations and variants are agent-generated.
- External papers and web research are allowed; external training data remains prohibited.
- Judge data remains completely outside research runs.
- The existing official starter-kit files remain unchanged.
