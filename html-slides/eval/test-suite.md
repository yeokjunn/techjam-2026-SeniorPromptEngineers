# HTML Slides Visual Test Suite

Reusable test suite for validating Pro and Vibe presentations using agent-browser screenshots.

## How It Works

```
eval/content/*.md          → Agent generates HTML using local SKILL.md
                           → agent-browser opens each HTML
                           → Navigate slides with ArrowRight (2s delay for animations)
                           → Screenshot each slide
                           → Visual inspection (manual or automated)
```

## Test Cases

### Pro Mode Tests

| Test ID | Content File | Theme | Slides | What it validates |
|---------|-------------|-------|--------|-------------------|
| pro-obsidian | pro-core.md | Obsidian | 10 | Core components, particles, glow blobs |
| pro-excalidraw-light | pro-core.md | Excalidraw Light | 10 | Hachure fills, rounded corners, offset shadows |
| pro-excalidraw-dark | pro-core.md | Excalidraw Dark | 10 | Same as above on dark background |
| pro-editorial | pro-core.md | Editorial Light | 10 | No-line rule, ambient shadows, gradient accents |
| pro-binary | pro-core.md | Binary Architect | 10 | 0px corners, grid overlay, neon glow, monospace |
| pro-charts | pro-charts.md | Obsidian | 6 | Chart.js bar, line, pie, doughnut |
| pro-mermaid | pro-mermaid.md | Obsidian | 6 | Mermaid flowchart TD, sequence, mind map |

### Vibe Mode Tests

All use the same content (Aura pitch deck) to compare visual variety across presets.

| Test ID | Content File | Preset | Slides | What it validates |
|---------|-------------|--------|--------|-------------------|
| vibe-bold-signal | vibe-aura.md | Bold Signal | 6 | Colored cards, section numbers, dark gradient |
| vibe-dark-botanical | vibe-aura.md | Dark Botanical | 6 | Serif typography, warm accents, gradient circles |
| vibe-swiss-modern | vibe-aura.md | Swiss Modern | 6 | Bauhaus grid, red accent, asymmetric layout |
| vibe-creative-voltage | vibe-aura.md | Creative Voltage | 6 | Electric blue + neon yellow, halftone, sharp corners |
| vibe-notebook-tabs | vibe-aura.md | Notebook Tabs | 6 | Paper card, colorful tabs, binder holes |
| vibe-vintage-editorial | vibe-aura.md | Vintage Editorial | 6 | Fraunces serif, geometric shapes, cream bg |
| vibe-excalidraw | vibe-devops.md | Excalidraw Light | 8 | CSS-only hand-drawn aesthetic, CSS hachure, Caveat font |

---

## Content Files

### pro-core.md — Core Components (10 slides)

Used by all 5 Pro theme tests. Covers the most common component types with varied content.

```
Topic: "Introduction to AI Agents"
Theme: [varies per test]

Slide 1 — Title
  "AI Agents" / "From chatbots to autonomous systems"

Slide 2 — Statement
  "An agent is an AI system that can reason, plan, and take action."

Slide 3 — Flip Cards (4 cards)
  Reasoning — choose the next step
  Memory — carry context across steps
  Tools — take actions in the real world
  Evaluation — detect errors, confirm completion

Slide 4 — Architecture Flow (3 boxes)
  User Input → Agent Core → Tool Execution

Slide 5 — Code Block
  Python function: agent loop (observe, think, act, evaluate)

Slide 6 — Stats Cards (3 stats)
  3.2x faster code review
  47% fewer production incidents
  89% developer satisfaction

Slide 7 — VS Compare
  Traditional AI (stateless, single-turn) vs Agentic AI (stateful, multi-step)

Slide 8 — Status Timeline (5 items)
  2022: ChatGPT launch
  2023: Function calling
  2024: Agent frameworks
  2025: Production agents
  2026: Autonomous systems

Slide 9 — Table
  Framework | Language | Stars | Key Feature
  LangChain | Python | 82K | Chain composition
  CrewAI | Python | 15K | Multi-agent
  AutoGen | Python | 28K | Conversation patterns
  Claude Agent SDK | Python/TS | — | First-party Anthropic

Slide 10 — CTA
  Resources: docs, GitHub, community
```

### pro-charts.md — Chart.js Tests (6 slides)

```
Topic: "Engineering Metrics Dashboard"
Theme: Obsidian

Slide 1 — Title
Slide 2 — Bar Chart: Deploy frequency by team (6 teams)
Slide 3 — Line Chart: Incident count over 6 months (2 datasets)
Slide 4 — Pie Chart: Error category distribution (5 categories)
Slide 5 — Doughnut Chart: Time allocation (4 categories)
Slide 6 — CTA
```

### pro-mermaid.md — Mermaid.js Tests (6 slides)

```
Topic: "MCP Architecture Deep Dive"
Theme: Obsidian

Slide 1 — Title
Slide 2 — Flowchart TD: MCP client-server connection flow (max 8 nodes)
  Must use: .mermaid-wrap container with bg, border, border-radius
  Must use: themeVariables matching Obsidian palette
  Must use: fontSize 18px+
Slide 3 — Sequence Diagram: Tool call handshake (Host, Client, Server)
Slide 4 — Mind Map: MCP ecosystem (Modes, Themes, Libraries, Agents)
Slide 5 — Content slide explaining primitives (no Mermaid)
Slide 6 — CTA
```

### vibe-aura.md — Aura Pitch Deck (6 slides)

Used by all 6 Vibe preset tests (same content, different styles).

```
Slide 1 — Title: "Aura" / "Lighting that learns your life"
Slide 2 — Problem: "Smart lighting isn't smart"
Slide 3 — Solution: "Aura adapts to you" (Circadian Sync, Presence Awareness, Mood Sensing)
Slide 4 — How it works: "The Aura Engine" (Sense → Decide → Act)
Slide 5 — Traction: 12K homes, 4.8 rating, 73% reduction, 91% satisfaction
Slide 6 — Closing: "Stop configuring. Start living."
```

### vibe-devops.md — DevOps Pipeline (8 slides, Excalidraw CSS-only)

```
Slide 1 — Title: "DevOps Pipeline" / "From commit to production"
Slide 2 — Pipeline overview with CSS-drawn arrows between cards (Build → Test → Deploy)
Slide 3 — Metrics with CSS decorative circles around numbers (8min, 99.2%, 14/day, <2min)
Slide 4 — Architecture with CSS connectors (GitHub → CI → Registry → K8s)
Slide 5 — Key takeaways with CSS decorative annotations (underline, circle, arrow)
Slide 6 — Tool stack (6 cards in 2x3 grid)
Slide 7 — Timeline (4 milestones)
Slide 8 — Closing
```

---

## Screenshot Script

### Usage

```bash
# Run a single test
bash eval/screenshot.sh <test-id>

# Run all Pro tests
bash eval/screenshot.sh pro-all

# Run all Vibe tests
bash eval/screenshot.sh vibe-all

# Run everything
bash eval/screenshot.sh all
```

### Screenshot Rules

1. **Wait 2 seconds** after each ArrowRight before screenshotting (scroll-snap animation)
2. **Wait 1 second** after opening a file before first screenshot (initial load)
3. Save screenshots to `eval/screenshots/<test-id>/slide-{N}.png`
4. Generate an index HTML showing all screenshots side-by-side for quick review

### Visual Assertions (per screenshot)

| Check | How to verify |
|-------|--------------|
| Content visible | Screenshot is not blank/black |
| No overlap | Previous slide content not bleeding through |
| Centered layout | Content block centered in viewport |
| Theme colors | Background and text match the expected palette |
| Font loaded | Not falling back to system fonts |
| Cards rendered | Card borders, backgrounds visible |
| Charts rendered | Canvas has drawn content (not empty box) |
| Mermaid rendered | SVG diagram visible inside container |
| Mermaid container | Has background, border, rounded corners |
| Navigation chrome | Progress bar, slide counter, nav dots visible |
| Viewport fit | No scrollbars, content within 100vh |

---

## Running the Tests

### Step 1: Generate presentations

For each test case, spawn an agent with the local SKILL.md. Use this exact prompt template to ensure the LOCAL skill is used, not the installed plugin:

```
Agent prompt template:

You are testing a LOCAL version of the html-slides skill.
Do NOT use the installed html-slides plugin. Do NOT invoke any Skill tool.
Instead, read the SKILL.md file below and follow it as your instructions.

Base directory: /Users/danzhu/playground/html-slides
Skill file: /Users/danzhu/playground/html-slides/SKILL.md

ALL file references in SKILL.md are relative to the base directory above.
For example:
- "assets/components.css" means /Users/danzhu/playground/html-slides/assets/components.css
- "references/libraries.md" means /Users/danzhu/playground/html-slides/references/libraries.md

First, read the SKILL.md file. Then follow it to generate the presentation.

Task: Generate using content from eval/content/{content-file}.md
Theme/Preset: {theme}
Mode: {pro|vibe}
Save to: eval/outputs/{test-id}/presentation.html

Simulate AskUserQuestion answers: [list answers]
Do NOT open in browser.
```

### Step 2: Screenshot with agent-browser

```bash
# For each test output:
agent-browser open "file:///path/to/presentation.html"
sleep 1
agent-browser screenshot eval/screenshots/{test-id}/slide-1.png
for each remaining slide:
  agent-browser press ArrowRight
  sleep 2
  agent-browser screenshot eval/screenshots/{test-id}/slide-{N}.png
```

### Step 3: Review

Open the screenshot index to compare all slides across themes visually.

---

## Adding New Tests

1. Create a content file in `eval/content/`
2. Add a row to the test table above
3. Run the test
4. Screenshots are saved for comparison

Keep content files small (6-10 slides) — easier to review and faster to generate.
