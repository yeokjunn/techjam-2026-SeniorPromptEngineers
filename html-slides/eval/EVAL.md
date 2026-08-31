# HTML Slides Eval

Self-evaluation workflow: generate a presentation, validate it, screenshot every slide, evaluate visual quality, and produce a report.

## Prerequisites

Before running, ensure these are installed:

1. `agent-browser` — `npm install -g agent-browser && agent-browser install`
2. `node` — for the spec validator
3. Validator dependencies — `cd testing && npm install && cd ..` (first time only)
4. `python3` — for serving HTML locally

## How to Run

**All commands assume the working directory is the repo root (`html-slides/`).**

You (the agent) will be told which mode and theme to evaluate. Example:

> "Read eval/EVAL.md and run the html-slides eval. Mode: pro, theme: obsidian."

Follow Steps 1-6 below in order.

**Shorthand:** `{RUN}` = `{DATE}-{MODE}-{THEME}` (e.g., `2026-03-25-pro-obsidian`)

## Step 1 — Generate

**Input:** `eval/content/all-components.md`
**Output:** `eval/output/{DATE}-{MODE}-{THEME}/presentation.html`

1. Read `eval/content/all-components.md` — this is your slide content.
2. Create the output directory: `eval/output/{DATE}-{MODE}-{THEME}/` (e.g., `eval/output/2026-03-25-pro-obsidian/`)
3. Create `eval/output/{DATE}-{MODE}-{THEME}/slides/` for screenshots.
4. If the content references images in `assets/`, copy `eval/content/assets/` to the output directory: `cp -r eval/content/assets eval/output/{DATE}-{MODE}-{THEME}/assets`
5. Invoke `/html-slides` to generate the presentation with these parameters (provide all upfront so the skill skips interactive questions):
   - **Mode:** as specified (pro or vibe)
   - **Theme:** as specified (e.g., obsidian, excalidraw-light, bold-signal, etc.)
   - **Editing:** No
   - **Images:** Yes, I have an image folder — path: `eval/output/{DATE}-{MODE}-{THEME}/assets/`
   - **Content:** All content ready — provide the full content from `all-components.md`
6. Save the generated HTML as `eval/output/{DATE}-{MODE}-{THEME}/presentation.html`

**Important:** The filename MUST be `presentation.html`. If the skill saves with a different name, rename it.

## Step 2 — Spec Compliance

**Input:** `presentation.html`
**Output:** Validation results (stored in memory for Step 5)

Run the validator:
```bash
node testing/bin/validate-slides.js eval/output/{RUN}/presentation.html
```

Record:
- Total rules passed (out of 8)
- Per-rule pass/fail with messages
- If any rule fails, note it but continue — do NOT abort the eval.

## Step 3 — Screenshot, Evaluate & Structural Check

**Input:** `presentation.html`
**Output:** `slides/slide-{N}.png`, `slides/slide-{N}-snapshot.md`, and per-slide evaluation scores

Screenshot each slide and evaluate it immediately (avoids loading all screenshots into memory at once).

1. Read `eval/metrics/visual-criteria.md` — this is your scoring rubric for the visual evaluation.

2. Start an HTTP server:
```bash
python3 -m http.server 8765 -d eval/output/{RUN}/ &
SERVER_PID=$!
```

3. Verify Chart.js CDN is reachable (chart slides need it):
```bash
curl -s -o /dev/null -w "%{http_code}" https://cdn.jsdelivr.net/npm/chart.js@4.4.8/dist/chart.umd.min.js
```
If not 200, note that chart slides may render blank — this is a network issue, not a skill issue.

4. Open the presentation and wait for initial render:
```bash
agent-browser open http://localhost:8765/presentation.html
agent-browser wait 1000
```

5. Get total slide count:
```bash
agent-browser get count ".slide"
```

6. For each slide (0 to N-1), using absolute paths:
```bash
# Screenshot and structural data
agent-browser screenshot {ABSOLUTE_PATH}/eval/output/{RUN}/slides/slide-{N}.png
agent-browser snapshot > {ABSOLUTE_PATH}/eval/output/{RUN}/slides/slide-{N}-snapshot.md
agent-browser get box ".slide.active"

# Evaluate THIS slide immediately:
# a. Read the screenshot image (you are multimodal — view it directly)
# b. Read the snapshot .md for structural context
# c. Complete Tier 2 checks (pass/fail per check)
# d. Score Tier 3 categories (1-5 per category)
# e. Note specific issues (be concrete — e.g., "Card 3 text overflows")
# Store results in memory for Step 5.

# Navigate to next slide (wait for animations to settle BEFORE next screenshot)
agent-browser press ArrowRight
agent-browser wait 500
```

Record the bounding box for each slide. If width or height doesn't match the viewport dimensions, flag as overflow.

7. Cleanup:
```bash
agent-browser close
kill $SERVER_PID
```

## Step 4 — Score & Write results.json

**Input:** Results from Steps 2-3
**Output:** `eval/output/{RUN}/results.json`

Aggregate all results into `results.json` with this exact schema:

```json
{
  "metadata": {
    "date": "{DATE}",
    "skill_version": "{VERSION from SKILL.md frontmatter}",
    "mode": "{pro|vibe}",
    "theme": "{theme name}",
    "agent": "{your agent name, e.g. claude-code}",
    "slide_count": {N}
  },
  "spec_compliance": {
    "passed": {N},
    "total": 8,
    "rules": [
      { "rule": "deck-container", "passed": true, "message": "Deck container exists" },
      { "rule": "slide-elements", "passed": true, "message": "..." }
    ]
  },
  "slides": [
    {
      "index": 0,
      "component": "{component type, e.g. title-slide, flip-cards, chart-bar}",
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
    "spec_pass_rate": "{passed}/{total}",
    "tier2_pass_rate": "{slides without tier2 issues}/{total slides}",
    "tier3_avg": {average of tier3.overall scores across all slides},
    "grade": "{A|B|C|D|F}",
    "worst_slides": [list of slide indices with lowest overall scores],
    "common_issues": ["list of issues that appear on 2+ slides"]
  }
}
```

**Grading scale** (based on `tier3_avg`):
- **A** (4.5+) — Ship it
- **B** (3.5–4.4) — Minor issues
- **C** (2.5–3.4) — Needs improvement
- **D** (1.5–2.4) — Significant problems
- **F** (<1.5) — Broken

## Step 5 — Write report.md

**Input:** `results.json`
**Output:** `eval/output/{RUN}/report.md`

Write a human-readable report:

```markdown
# HTML Slides Eval Report

**Theme:** {theme} | **Mode:** {mode} | **Date:** {date} | **Skill:** v{version}

## Summary

| Metric | Result |
|--------|--------|
| Spec Compliance | {passed}/{total} |
| Layout Issues | {count} slides with Tier 2 problems |
| Avg Visual Score | {tier3_avg}/5 |
| **Grade** | **{grade}** |

## Issues Found

| Slide | Component | Issue |
|-------|-----------|-------|
{for each slide with issues, one row per issue}

## Per-Slide Scores

| # | Component | Layout | Visual | Screenshot |
|---|-----------|--------|--------|------------|
| 0 | title-slide | ✓ | 4.5 | [view](slides/slide-0.png) |
{one row per slide, Layout = ✓ if no tier2 issues else ✗, Visual = tier3 overall score}

## Spec Compliance Details

{list each of the 8 rules with ✓/✗ and message}
```

**Done!** The eval is complete. Summarize the grade and any critical issues to the user.
