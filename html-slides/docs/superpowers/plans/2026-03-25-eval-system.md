# HTML Slides Eval System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an agent-driven self-evaluation system that generates slides, validates spec compliance, screenshots every slide, evaluates visual quality, and produces JSON + Markdown reports.

**Architecture:** No orchestrator script. The eval is a set of instruction files (`EVAL.md`, `visual-criteria.md`) that the agent reads and follows. Test content drives generation, `agent-browser` handles screenshots, the existing validator checks spec compliance, and the agent's multimodal capability judges visual quality.

**Tech Stack:** agent-browser (headless browser CLI), node (spec validator), python3 (HTTP server), markdown (instructions + reports)

**Spec:** `docs/superpowers/specs/2026-03-25-eval-system-design.md`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `eval/.gitignore` | Ignore `output/` directory |
| `eval/content/all-components.md` | Comprehensive test content — all 14 components, 8 chart types, 3 image variants |
| `eval/content/assets/screenshot.jpg` | Test image for default image slide (copied from `testing/assets/`) |
| `eval/content/assets/logo.jpg` | Test image for logo variant (copied from `testing/assets/`) |
| `eval/content/assets/hero.jpg` | Test image for full-bleed variant (copied from `testing/assets/`) |
| `eval/metrics/visual-criteria.md` | Tier 2 + Tier 3 evaluation criteria with scoring rubric |
| `eval/EVAL.md` | Master instruction file — the 6-step pipeline the agent follows |

---

### Task 1: Scaffold eval directory and gitignore

**Files:**
- Create: `eval/.gitignore`
- Create: `eval/content/assets/` (copy images)

- [ ] **Step 1: Create eval directory structure**

```bash
mkdir -p eval/content/assets eval/metrics eval/output
```

- [ ] **Step 2: Create .gitignore**

Create `eval/.gitignore`:
```
output/
```

- [ ] **Step 3: Copy test images from testing/assets/**

```bash
cp testing/assets/screenshot.jpg eval/content/assets/
cp testing/assets/logo.jpg eval/content/assets/
cp testing/assets/hero.jpg eval/content/assets/
```

- [ ] **Step 4: Verify files exist**

```bash
ls -la eval/content/assets/
```
Expected: 3 files (screenshot.jpg ~77KB, logo.jpg ~41KB, hero.jpg ~179KB)

- [ ] **Step 5: Commit**

```bash
git add eval/.gitignore eval/content/assets/
git commit -m "scaffold eval directory with test images"
```

---

### Task 2: Write test content (`all-components.md`)

**Files:**
- Create: `eval/content/all-components.md`
- Reference: `testing/test-all-components.md` (base content, 21 slides for 13 components)

The new file extends the existing test content with:
- 3 image slide variants (slides 22-24)
- Already has all 8 chart subtypes (slides 12-19)
- Max-density edge cases: Slide 3 already has 4 flip cards — update to 6 cards (max allowed). Slide 6 code block should have 10 lines (max). Slide 11 table should have 6 rows (max).
- Total: ~24 slides covering all 14 components

- [ ] **Step 1: Create all-components.md**

Read `testing/test-all-components.md` as the base. Create `eval/content/all-components.md` with:
- All 21 slides from the base (slides 1-21)
- Modify Slide 3 (Flip Cards) to use 6 cards instead of 4 (max density per spec)
- Modify Slide 6 (Code Block) to use 10 lines of code (max density per spec)
- Modify Slide 11 (Table) to use 6 rows (max density per spec)
- Add Slide 22 — Image Slide (default variant) using `assets/screenshot.jpg`
- Add Slide 23 — Image Slide (logo variant) using `assets/logo.jpg`
- Add Slide 24 — Image Slide (full-bleed variant) using `assets/hero.jpg`
- Update the header to say "24 slides" and "14 component templates"

The format for each new slide follows the existing pattern:

```markdown
---

## Slide 22 — Image Slide: Default (Template: Image Slide)

**Tag:** Visual Content
**Heading:** Product Screenshot

Image: `assets/screenshot.jpg` (default variant — rounded corners, hover scale)
Caption: "The HTML Slides presenter app showing a live deck"

---

## Slide 23 — Image Slide: Logo (Template: Image Slide, variant: logo)

**Tag:** Branding
**Heading:** HTML Slides

Image: `assets/logo.jpg` (logo variant — smaller, circular crop)
Caption: "Designed for AI agents"

---

## Slide 24 — Image Slide: Full-Bleed (Template: Image Slide, variant: full-bleed)

**Tag:** Immersive
**Heading:** Present Anywhere

Image: `assets/hero.jpg` (full-bleed variant — background image with text overlay)
Subtitle: "One file. Every screen. No dependencies."
```

- [ ] **Step 2: Verify slide count and component coverage**

Manually check the file covers:
- 14 distinct component types
- 8 chart subtypes
- 3 image variants
- Total ~24 slides

- [ ] **Step 3: Commit**

```bash
git add eval/content/all-components.md
git commit -m "add comprehensive test content for eval (24 slides, 14 components)"
```

---

### Task 3: Write visual evaluation criteria (`visual-criteria.md`)

**Files:**
- Create: `eval/metrics/visual-criteria.md`

This file tells the agent exactly what to look for when evaluating each screenshot. It must be precise enough that different agents produce comparable scores.

- [ ] **Step 1: Create visual-criteria.md**

```markdown
# Visual Evaluation Criteria

Use this guide when evaluating each slide screenshot. For every slide, complete Tier 2 checks first, then score Tier 3.

## Tier 2 — Layout Integrity (pass/fail)

Check each item. Record `true` for issues found, `false` for no issues.

| Check | What to look for |
|-------|-----------------|
| `overflow` | Any content extends beyond the visible viewport. Text, images, or components are clipped at edges. Scrollbars visible. |
| `overlapping` | Elements overlap each other unintentionally. Text on top of other text. Cards stacking instead of side-by-side. |
| `blank` | The slide is empty or mostly empty when it should have content. Component failed to render. |
| `image_broken` | Image placeholder icon visible. Alt text showing instead of image. Image area is empty/gray. |
| `chart_empty` | Chart canvas is blank (white or transparent rectangle where a chart should be). Usually means Chart.js failed to load. |
| `text_truncated` | Text is cut off mid-word or mid-sentence. Ellipsis where there shouldn't be. Important content hidden. |
| `interactive_visible` | For flip cards, expandable cards: the front face content is visible (icons, titles, subtitles). Set to `false` if the component appears empty or unstyled. |

**Note:** `interactive_visible` is `true` when content IS visible (good). All other checks are `true` when an issue IS found (bad).

## Tier 3 — Aesthetic Quality (1-5 per category)

Score each category independently. Use the rubric below.

### Spacing (1-5)
- **1:** Content jammed against edges or other elements. No breathing room. Padding missing entirely.
- **2:** Noticeably cramped. Some padding exists but feels tight.
- **3:** Adequate spacing. Nothing feels broken but not generous.
- **4:** Good spacing. Content has comfortable room. Visual hierarchy aided by whitespace.
- **5:** Excellent. Balanced and deliberate use of space. Presentation-quality.

### Typography (1-5)
- **1:** Text unreadable. Wrong font rendered. Sizes way off (headings smaller than body).
- **2:** Readable but hierarchy unclear. Heading sizes too similar to body.
- **3:** Correct hierarchy. Headings > subheadings > body. Readable at normal distance.
- **4:** Good typography. Font weights and sizes create clear visual flow.
- **5:** Beautiful. Type choices enhance the theme. Perfect weight contrast.

### Color (1-5)
- **1:** Colors clash. Text invisible against background. Accent colors fight each other.
- **2:** Low contrast issues. Some text hard to read. Colors feel random.
- **3:** Theme-consistent. All colors come from the chosen palette. Readable.
- **4:** Good use of accent colors. Highlights draw attention correctly.
- **5:** Vibrant and harmonious. Colors enhance content meaning. Professional.

### Alignment (1-5)
- **1:** Elements visibly off-center. Jagged left edges. Cards different sizes.
- **2:** Mostly aligned but noticeable inconsistencies.
- **3:** Centered and aligned. No obvious misalignment.
- **4:** Clean alignment. Grid structure visible. Consistent gaps.
- **5:** Pixel-perfect. Every element precisely placed. Professional quality.

### Overall Impression (1-5)
- **1:** Looks broken. Would not present this.
- **2:** Looks amateurish. Functional but unpolished.
- **3:** Looks acceptable. Could present if needed.
- **4:** Looks good. Would confidently present this.
- **5:** Looks polished. Indistinguishable from a professional designer's work.

## Scoring Notes

- Score what you SEE, not what you know the code does.
- Compare against other slides in the same deck for consistency.
- A chart slide with blank canvas gets `chart_empty: true` AND `overall: 1`.
- Title slides and statement slides should score high on spacing (they're intentionally minimal).
- Flip cards and expandable cards show front face only in screenshots — that's correct.
```

- [ ] **Step 2: Verify criteria completeness**

Check that:
- All 7 Tier 2 checks from the spec are documented
- All 5 Tier 3 categories have full 1-5 rubrics
- The `interactive_visible` polarity is clearly explained (true = good, unlike others)

- [ ] **Step 3: Commit**

```bash
git add eval/metrics/visual-criteria.md
git commit -m "add visual evaluation criteria with scoring rubrics"
```

---

### Task 4: Write master instruction file (`EVAL.md`)

**Files:**
- Create: `eval/EVAL.md`
- Reference: `docs/superpowers/specs/2026-03-25-eval-system-design.md` (the spec)

This is the core file. The agent reads this and follows it step by step. It must be unambiguous — the agent should never need to guess.

- [ ] **Step 1: Create EVAL.md**

```markdown
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
7. Save the `.notes.json` file alongside it as `presentation.notes.json`

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

Steps 3 and 4 from the spec are merged: screenshot each slide and evaluate it immediately (avoids loading all screenshots into memory at once).

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

## Step 5 — Score & Write results.json

**Input:** Results from Steps 2-4
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

## Step 6 — Write report.md

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
```

- [ ] **Step 2: Review EVAL.md for completeness**

Verify:
- All 6 steps are present with exact commands
- File paths are explicit (no ambiguity)
- `results.json` schema matches the spec exactly
- Grading scale matches the spec
- Prerequisites section covers all dependencies
- Cleanup (kill server, close browser) is in Step 3

- [ ] **Step 3: Commit**

```bash
git add eval/EVAL.md
git commit -m "add EVAL.md — master instruction file for agent-driven eval"
```

---

### Task 5: End-to-end verification

**Files:**
- Read: all files in `eval/`

This task verifies the complete eval system is coherent before considering it done.

- [ ] **Step 1: Verify directory structure matches spec**

```bash
find eval/ -type f | sort
```

Expected:
```
eval/.gitignore
eval/EVAL.md
eval/content/all-components.md
eval/content/assets/hero.jpg
eval/content/assets/logo.jpg
eval/content/assets/screenshot.jpg
eval/metrics/visual-criteria.md
```

- [ ] **Step 2: Cross-reference EVAL.md against spec**

Read `eval/EVAL.md` and `docs/superpowers/specs/2026-03-25-eval-system-design.md`. Verify:
- All 6 pipeline steps are implemented (Steps 3+4 merged into one)
- `results.json` schema matches exactly (7 tier2 fields, 5 tier3 fields, string rule IDs like `"deck-container"`)
- `{RUN}` alias is defined as `{DATE}-{MODE}-{THEME}`
- Grading scale: A/B/C/D/F with correct thresholds
- `tier3_avg` is defined as average of `tier3.overall` across all slides
- Dependencies listed in prerequisites
- Cleanup uses PID-based kill (not `kill %1`)
- Wait comes after `press ArrowRight` (before next screenshot), plus initial wait after `open`

- [ ] **Step 3: Cross-reference visual-criteria.md against spec**

Read `eval/metrics/visual-criteria.md`. Verify:
- 7 Tier 2 checks documented
- 5 Tier 3 categories with full 1-5 rubrics
- `interactive_visible` polarity clearly explained

- [ ] **Step 4: Verify test content covers all components**

Read `eval/content/all-components.md`. Count:
- 14 distinct component types
- 8 chart subtypes (bar, line, pie, doughnut, radar, polar, scatter, bubble)
- 3 image slide variants
- ~24 total slides

- [ ] **Step 5: Final commit (if any fixes needed)**

```bash
git add -A eval/
git commit -m "fix: address eval system verification issues"
```

Only if changes were needed. If everything passes, skip this step.
