# HTML Slides Eval System — Design Spec

## Goal

A self-evaluation system where the AI agent generates a presentation using the html-slides skill, then evaluates its own output for spec compliance, layout integrity, and visual quality. Results are written as JSON (machine-readable, diffable across versions) and Markdown (human-readable summary).

## Architecture

The eval is **agent-driven** — no orchestrator script. The agent reads `eval/EVAL.md`, follows the steps, and writes the report. You run it by telling your agent: "Run the html-slides eval for Obsidian theme."

The eval lives inside the html-slides repo as `eval/`, alongside the existing `testing/` directory. Generated artifacts (HTML, screenshots, reports) go into `eval/output/` which is gitignored.

## Directory Structure

```
eval/
├── EVAL.md                       # Master instructions the agent reads
├── content/
│   ├── all-components.md         # Test content covering all 14 components
│   └── assets/                   # Test images (screenshot.jpg, logo.jpg, hero.jpg)
├── metrics/
│   └── visual-criteria.md        # Evaluation criteria for screenshot review
├── output/                       # Gitignored
│   └── {date}-{mode}-{theme}/
│       ├── presentation.html     # Generated HTML
│       ├── presentation.notes.json
│       ├── slides/               # Per-slide screenshots
│       │   ├── slide-0.png
│       │   ├── slide-0-snapshot.md
│       │   └── ...
│       ├── results.json          # Machine-readable scores
│       └── report.md             # Human-readable summary
└── .gitignore                    # Ignores output/
```

## Test Content

A single comprehensive content document (`content/all-components.md`) that exercises:

- All 14 Pro component types (title, statement, flip cards, VS/comparison, architecture flow, code block, auth flip compare, stats cards, expandable cards, status timeline, table, chart, CTA box, image slide)
- All 8 chart subtypes (bar, line, pie, doughnut, radar, polar, scatter, bubble)
- All 3 image slide variants (default, screenshot, logo) using real images from `content/assets/`
- Edge cases: max-density slides (6 cards, 10 lines of code, 6-row table) to stress viewport fitting

The same content is used for every theme and mode, enabling fair comparison.

Topic: "Introducing HTML Slides" — a meta-presentation about the skill itself, which naturally covers technical and visual concepts.

`eval/content/all-components.md` is a new file based on `testing/test-all-components.md` with these additions:
- Image Slide (default variant) with `screenshot.jpg`
- Image Slide (logo variant) with `logo.jpg`
- Image Slide (full-bleed variant) with `hero.jpg`
- All 8 chart subtypes (existing doc only covers bar)

Image assets (`eval/content/assets/`) are committed to the repo. Sourced from the existing `testing/assets/` directory (screenshot.jpg 77KB, logo.jpg 41KB, hero.jpg 179KB).

The agent must save the generated presentation as `presentation.html` in the output directory — EVAL.md specifies this filename explicitly.

## Evaluation Pipeline

The agent follows these steps as described in `EVAL.md`:

### Step 1 — Generate

- Read `eval/content/all-components.md` for test content
- Invoke `/html-slides` with the specified mode and theme
- All interactive answers are baked into the prompt (mode, theme, editing preference, images) so the skill skips questions
- Save output to `eval/output/{date}-{mode}-{theme}/presentation.html`
- Save the `.notes.json` sidecar file alongside it

### Step 2 — Spec Compliance

- Run `node testing/bin/validate-slides.js` on the generated HTML
- Record pass/fail for all 8 rules:
  1. Deck container (`<div class="deck" id="deck">`)
  2. At least one `<div class="slide">`
  3. First slide has `active` class (note: validator does not check uniqueness — only that the first slide has it)
  4. Sequential `data-slide` attributes (0 to N-1)
  5. Global `goTo()` function at brace-depth 0 (note: `next()` and `prev()` are required by SKILL.md but not checked by the validator — known gap)
  6. Inline CSS only (except font imports)
  7. Inline JS only (except Chart.js CDN)
  8. Generator meta tag present
- If any rule fails, flag it but continue (don't abort)

### Step 3+4 — Screenshot & Visual Evaluation (merged)

Uses `agent-browser` (Vercel's headless browser CLI) via shell commands. Any agent with shell access can run this — no MCP plugin dependency.

```bash
# Install validator dependencies (first time only)
cd testing && npm install && cd ..

# Serve the HTML — capture PID for reliable cleanup
python3 -m http.server 8765 -d eval/output/{run}/ &
SERVER_PID=$!

# Open browser
agent-browser open http://localhost:8765/presentation.html

# For each slide (use absolute paths for screenshots):
agent-browser screenshot /absolute/path/to/eval/output/{run}/slides/slide-{N}.png
agent-browser snapshot > /absolute/path/to/eval/output/{run}/slides/slide-{N}-snapshot.md
agent-browser get box ".slide.active"    # verify viewport fit
agent-browser wait 500                    # wait for CSS animations to settle
agent-browser press ArrowRight

agent-browser close
kill $SERVER_PID  # reliable cleanup via PID
```

After all screenshots are captured, the agent reads each screenshot (multimodal) and the accessibility snapshot, then evaluates against `metrics/visual-criteria.md`.

### Step 5 — Score

Aggregate results into `results.json`:

- `metadata`: date, skill version, mode, theme, agent, slide count
- `spec_compliance`: 8 rule results (pass/fail)
- `slides[]`: per-slide record with tier 2 checks, tier 3 scores, and issues
- `summary`: pass rates, averages, grade, worst slides, common issues

### Step 6 — Report

Write `report.md` with:
- Summary table (spec compliance, layout issue count, average visual score, letter grade)
- Issues table (slide number, component, specific issue)
- Per-slide score table with screenshot references

## Evaluation Metrics

### Tier 1 — Spec Compliance (binary pass/fail)

The 8 validation rules, already implemented in `testing/lib/validate.js`. Plus viewport dimension check via `agent-browser get box ".slide.active"`.

### Tier 2 — Layout Integrity (binary pass/fail per check)

Per slide, check:
- Content overflows viewport
- Elements overlapping each other
- Empty/blank slide (component didn't render)
- Images not loading (broken img)
- Chart canvas empty
- Text truncated or unreadable
- Interactive elements visible (flip cards have front content, expandable cards show icons)

### Tier 3 — Aesthetic Quality (1-5 score per category)

Per slide, score:

| Category | 1 (Bad) | 3 (Acceptable) | 5 (Excellent) |
|----------|---------|-----------------|---------------|
| Spacing | Cramped or too sparse | Adequate | Balanced, breathing room |
| Typography | Wrong sizes, unreadable | Correct hierarchy | Beautiful, consistent |
| Color | Clashing, low contrast | Theme-consistent | Vibrant, harmonious |
| Alignment | Off-center, jagged | Mostly centered | Pixel-perfect |
| Overall | Looks broken | Looks fine | Looks polished |

### Grading Scale

- **A** (4.5+) — Ship it
- **B** (3.5–4.4) — Minor issues, acceptable
- **C** (2.5–3.4) — Needs improvement
- **D** (1.5–2.4) — Significant problems
- **F** (<1.5) — Broken

No plus/minus modifiers — whole letter grades only.

## Report Formats

### results.json (machine-readable)

```json
{
  "metadata": {
    "date": "2026-03-25",
    "skill_version": "0.6.6",
    "mode": "pro",
    "theme": "obsidian",
    "agent": "claude-code",
    "slide_count": 21
  },
  "spec_compliance": {
    "passed": 8,
    "total": 8,
    "rules": [{ "rule": 1, "passed": true, "message": "..." }]
  },
  "slides": [
    {
      "index": 0,
      "component": "title-slide",
      "screenshot": "slides/slide-0.png",
      "tier2": {
        "overflow": false,
        "overlapping": false,
        "blank": false,
        "image_broken": false,
        "chart_empty": false,
        "text_truncated": false,
        "interactive_visible": true
      },
      "tier3": {
        "spacing": 4,
        "typography": 5,
        "color": 4,
        "alignment": 5,
        "overall": 4
      },
      "issues": []
    }
  ],
  "summary": {
    "spec_pass_rate": "8/8",
    "tier2_pass_rate": "19/21",
    "tier3_avg": 3.8,
    "grade": "B",
    "worst_slides": [3, 15],
    "common_issues": ["Flip card text overflow on dense content"]
  }
}
```

### report.md (human-readable)

Contains: summary table, issues found, per-slide scores with screenshot references.

## Usage

Run one eval at a time. Tell your agent:

> "Read eval/EVAL.md and run the html-slides eval. Mode: pro, theme: obsidian."

The agent does everything: generates, validates, screenshots, evaluates, reports.

Works with any agent that has shell access (for `agent-browser`) and multimodal capability (for screenshot evaluation): Claude Code, Gemini CLI, etc.

## Dependencies

- `agent-browser` (`npm install -g agent-browser && agent-browser install`) — headless browser CLI for screenshots
- `node` — for running the spec validator
- `node-html-parser` — validator dependency, install via `cd testing && npm install`
- `python3` — for serving HTML locally
- Multimodal AI agent — for visual evaluation of screenshots
- **Network access** — required for Chart.js CDN (`cdn.jsdelivr.net`). Chart slides will show blank canvases if offline. EVAL.md should verify CDN reachability before eval or note this as a known limitation.

## What Gets Committed

- `eval/EVAL.md` — master instructions
- `eval/content/` — test content and images
- `eval/metrics/` — evaluation criteria
- `eval/.gitignore` — ignores `output/`

What does NOT get committed:
- `eval/output/` — all generated HTML, screenshots, and reports (gitignored)
