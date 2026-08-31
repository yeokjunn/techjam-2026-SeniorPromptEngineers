# Changelog

## v0.8.0

### Problem
Pro mode generated every presentation with the same slide sequence — Title → Statement → Flip Cards → VS Compare → Architecture → Code → Stats → Timeline → Table → CTA. The agent did a 1:1 lookup from the decision table in component-templates.md, ignoring what the content actually needed. Every deck looked identical regardless of topic.

### Fix: Content-Driven Design
- **Removed the decision table and narrative structure table** from component-templates.md. These caused deterministic component cycling. Component templates are now a style reference — the agent uses them when they fit the content, not as a menu to walk through.
- **Added content analysis step** to Phase 3. Before generating HTML, the agent analyzes each slide's purpose, content, and best visual treatment. If two slides have similar content, it varies the approach — different layouts, different component types.
- **Added "Content-Driven Design" as core principle #4** in SKILL.md.

### New: CDN Library Support
- **Mermaid.js v11.13.0** — flowcharts, sequence diagrams, ER diagrams, state machines, mind maps. Includes: container styling with background/border, dark/light theming, sizing guidance (TD default, 18px font), zoomable lightbox with +/- buttons.
- **anime.js v3.2.2** — orchestrated staggered entrances, SVG path drawing, count-up number animations. Pinned to v3 (v4 is 6x larger with breaking changes).
- **New file: `references/libraries.md`** — integration guide for all CDN libraries (Chart.js, Mermaid, anime.js) with theming, sizing for slides, gotchas, and "when to use what" decision table.

### New: Improved Vibe Style Discovery
- **Mood → numbered text list** replaces the old 3-preview-only flow. After selecting a mood (Impressed/Excited/Calm/Inspired), the user sees all matching presets (4-5 options) as a numbered list and picks by number. Shows more options without AskUserQuestion's 4-option limit.

### New: Shared Presentation Spec
- **New file: `references/presentation-layer.md`** — centralized spec for slide structure, speaker notes format, 8 validation rules, content density limits, and navigation requirements. Previously scattered across SKILL.md.

### Fix: Vibe Mode Slide Conflict
- **Added instruction** not to redefine `.slide` positioning after viewport-base.css in Vibe mode. The upstream viewport-base.css uses `position: relative` + scroll-snap, which conflicts if the agent also generates `position: absolute; opacity: 0`. This caused blank or overlapping slides in some Vibe presets.
- **Fixed html-template.md** — removed conflicting `.slide` definition that contradicted viewport-base.css. Added `scrollIntoView()` to `goTo()` function for scroll-snap navigation.

### Fix: Excalidraw Theme Borders
- **Removed `.rough-border` CSS class** from excalidraw.css and excalidraw-dark.css. The old design set `border: none` on cards expecting Rough.js JavaScript to draw hand-drawn borders. This broke flip card transforms and caused position shifts. CSS hachure fills + offset shadows provide the hand-drawn aesthetic without JavaScript.

### Removed: Rough.js
- Initially added for hand-drawn graphics in Excalidraw presets. Removed after testing showed it breaks flip card 3D transforms, produces ugly thick borders, and the CSS-only approach from excalidraw.css works better. The Excalidraw hand-drawn aesthetic is achieved entirely with CSS (repeating-linear-gradient hachure fills, offset box-shadows, 14px border-radius).

### Test Suite
- **12 eval cases** covering 5 Pro themes, 6 Vibe presets, and Excalidraw CSS-only
- **Content files** for reusable testing: pro-core.md (13 slides), pro-charts.md (6 slides), pro-mermaid.md (6 slides), vibe-aura.md (6 slides), vibe-devops.md (8 slides)
- **Screenshot script** using agent-browser with 2s animation delay
- **Test images** (SVG) for image slide variants

---

## v0.7.1

Initial version based on frontend-slides.
