# Introduction to AI Agents

**Mode:** Pro
**Theme:** [varies per test — run once per theme]
**Length:** 13 slides
**Images:** eval/assets/ (screenshot.svg, logo.svg, hero.svg)

---

## Slide 1 — Title
**Title:** AI Agents
**Subtitle:** From chatbots to autonomous systems
**Tag:** Introducing

## Slide 2 — Statement
**Statement:** An agent is an AI system that can reason, plan, and take action.
**Subtitle:** Unlike traditional chatbots, agents maintain state, use tools, and work toward goals autonomously.

## Slide 3 — Flip Cards (4 cards)
**Tag:** Core Capabilities
**Heading:** Four Building Blocks

Card 1 — Front: "Reasoning" / Back: "Analyze the situation and choose the best next step. Chain-of-thought, tree search, or learned heuristics."
Card 2 — Front: "Memory" / Back: "Carry context across steps. Short-term working memory plus long-term knowledge retrieval."
Card 3 — Front: "Tools" / Back: "Take actions in the real world. File operations, API calls, code execution, database queries."
Card 4 — Front: "Evaluation" / Back: "Detect errors and confirm completion. Self-check outputs, run tests, validate against requirements."

## Slide 4 — Architecture Flow (3 boxes)
**Tag:** How It Works
**Heading:** The Agent Loop

Box 1 (blue): "Observe" — Read environment, user input, tool results
Arrow: "context"
Box 2 (yellow): "Think" — Reason about next action, plan steps
Arrow: "decision"
Box 3 (green): "Act" — Execute tool, write code, send message

## Slide 5 — Code Block
**Tag:** Under the Hood
**Heading:** Agent Loop in Python

```python
async def agent_loop(task):
    memory = Memory()
    while not task.is_complete():
        observation = await observe(task, memory)
        thought = await think(observation)
        result = await act(thought)
        memory.add(result)
        if evaluate(result, task):
            return result
```

## Slide 6 — Stats Cards (3 stats)
**Tag:** Impact
**Heading:** Measured Results

Stat 1 (blue): "3.2x" — Faster code review with AI assistance
Stat 2 (green): "47%" — Fewer production incidents
Stat 3 (orange): "89%" — Developer satisfaction rating

## Slide 7 — VS Compare
**Tag:** The Shift
**Heading:** Traditional AI vs Agentic AI

Left card: "Traditional" — Stateless, single-turn, prompt-in/response-out, no tool use, human orchestrates
Right card: "Agentic" — Stateful, multi-step, plans and executes, tool-equipped, self-directed

## Slide 8 — Status Timeline (5 items)
**Tag:** Evolution
**Heading:** The Path to Agents

Item 1 (green): "2022 — ChatGPT launches, sparking the LLM era"
Item 2 (green): "2023 — Function calling enables tool use"
Item 3 (green): "2024 — Agent frameworks emerge (LangChain, CrewAI)"
Item 4 (yellow): "2025 — Production agents ship (Claude Code, Codex)"
Item 5 (orange): "2026 — Autonomous multi-agent systems"

## Slide 9 — Table
**Tag:** Ecosystem
**Heading:** Agent Frameworks Compared

| Framework | Language | Stars | Key Feature |
|-----------|----------|-------|-------------|
| LangChain | Python | 82K | Chain composition |
| CrewAI | Python | 15K | Multi-agent crews |
| AutoGen | Python | 28K | Conversation patterns |
| Claude Agent SDK | Python/TS | — | First-party Anthropic |

## Slide 10 — Image Slide (Default)
**Tag:** Demo
**Heading:** Agent Dashboard

Image: `eval/assets/screenshot.svg` (default variant — rounded corners, hover scale)
Caption: "The agent monitoring dashboard showing active sessions"

## Slide 11 — Image Slide (Logo)
**Tag:** Brand
**Heading:** HTML Slides

Image: `eval/assets/logo.svg` (logo variant — smaller, centered)
Caption: "Built for AI agents"

## Slide 12 — Image Slide (Full-Bleed)
**Tag:** Vision
**Heading:** Present Anywhere

Image: `eval/assets/hero.svg` (full-bleed variant — background image with text overlay)
Subtitle: "One file. Every screen. No dependencies."

## Slide 13 — CTA
**Heading:** Start Building Agents
**Resources:**
- Anthropic docs: docs.anthropic.com
- Claude Agent SDK: github.com/anthropics/claude-agent-sdk
- Community: discord.gg/anthropic
