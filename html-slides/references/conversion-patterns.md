# Conversion Patterns

Reference for detecting HTML slide frameworks and extracting content during conversion.

## Framework Detection

Check the HTML for these patterns to identify the source format.

### reveal.js

**Detection:**
- `<div class="reveal">` container
- `<section>` elements as slides (may be nested for vertical slides)
- `Reveal.initialize()` or `new Reveal()` in script
- CDN link to `reveal.js` or `revealjs`

**Extraction:**
- Each top-level `<section>` = one slide
- Nested `<section>` elements = vertical slides (flatten into sequential slides)
- Speaker notes in `<aside class="notes">` — preserve as inline `<script class="slide-notes">` blocks
- Fragments (`class="fragment"`) — include all content (remove fragment animation)
- Code blocks in `<pre><code>` — preserve language class for syntax highlighting
- Backgrounds via `data-background` attributes — convert to CSS background on slide div

**Strip:**
- All reveal.js JS (`Reveal.initialize()`, plugin configs)
- reveal.js CSS (`reveal.css`, theme CSS)
- Plugin scripts (markdown, highlight, notes, math)

---

### Marp

**Detection:**
- `<!-- marp: true -->` HTML comment
- `class="marpit"` on container
- `<style>` with Marp-specific CSS (e.g., `section` styling)
- `data-marpit-` attributes

**Extraction:**
- Each `<section>` = one slide
- Headings, paragraphs, lists map directly
- Images with Marp sizing syntax (`w:` / `h:`) — convert to standard CSS
- Speaker notes after `<!-- _notes: -->` comments
- Background images via `![bg]()` syntax — extract URL

**Strip:**
- Marp engine JS
- Marp theme CSS (replace with HTMLSlides styling)
- `data-marpit-*` attributes

---

### impress.js

**Detection:**
- `<div id="impress">` container
- Elements with `class="step"`
- `impress().init()` in script
- `data-x`, `data-y`, `data-z` positioning attributes

**Extraction:**
- Each `.step` element = one slide
- Ignore 3D positioning (flatten to sequential slides)
- Extract text content, images, code blocks from each step
- `data-rotate`, `data-scale` — ignore (not applicable to HTMLSlides)

**Strip:**
- impress.js core and plugins
- 3D transform CSS
- Positioning attributes (`data-x/y/z/rotate/scale`)

---

### Slidev

**Detection:**
- `class="slidev-layout"` or `class="slidev-page"`
- `<div id="app">` with Slidev structure
- Vue component artifacts

**Extraction:**
- Each `.slidev-page` or `<section>` = one slide
- Markdown-rendered content (headings, lists, code blocks)
- Component blocks — extract text content, ignore Vue interactivity
- Speaker notes from `<!-- notes -->` blocks

**Strip:**
- Vue/Slidev runtime JS
- Slidev layout CSS
- Component interactivity (keep static content)

---

### Google Slides Export

**Detection:**
- Deeply nested `<div>` structure with auto-generated class names
- `<svg>` elements for shapes and text
- Google-specific class patterns (e.g., `c1`, `c2`, sequential naming)
- `<style>` with `.c0 { }` type selectors
- Sometimes `punch-viewer-content` class

**Extraction:**
- Each top-level container div typically = one slide
- Text is often in `<span>` elements with inline styles — extract plain text
- Images may be embedded as base64 or referenced externally
- Layout is absolute-positioned — ignore positioning, reflow content
- Speaker notes in a separate panel structure if present

**Strip:**
- All Google-generated CSS (meaningless class names)
- SVG shape decorations (keep SVG images if they contain content)
- Inline positioning styles
- Google Fonts may already be present — keep those

---

### Partially Compliant HTMLSlides

**Detection:**
- Has `<div class="deck">` but may be missing `id="deck"`
- Has `<div class="slide">` elements
- May be missing `data-slide` attributes, `active` class, or navigation functions

**Extraction:**
- Keep the entire structure intact
- Only fix the specific failing rules:
  - Missing `id="deck"` → add it
  - Missing `data-slide` → add sequential numbering
  - Missing `class="active"` on first slide → add it
  - Missing `goTo/next/prev` → inject standard navigation JS
  - External CSS → inline it
  - External JS → inline it (except Chart.js)

**Do NOT restructure** — minimal fixes only.

---

### Article / Blog HTML

**Detection:**
- `<article>` element, or `<main>` with heading-structured content
- Sequential headings (h1 → h2 → h3) with paragraphs between them
- No slide-framework-specific patterns

**Extraction rules:**
- `<h1>` → Title slide (first one becomes the presentation title)
- `<h2>` → New slide with that heading
- `<h3>` within an h2 section → Sub-section, stays on same slide if fits density limits, otherwise new slide
- `<p>` paragraphs → Bullet points or content text (summarize if too long for a slide)
- `<ul>` / `<ol>` → Bullet lists (split if > 6 items)
- `<pre><code>` → Code slide
- `<table>` → Table slide
- `<blockquote>` → Quote slide
- `<img>` → Image slide or included in content slide if space permits
- `<figure>` → Image slide with caption
- Navigation, footer, sidebar, header elements → Strip (not content)

**Content chunking:**
- Maximum 4-6 bullet points per slide
- Maximum 2 short paragraphs per slide
- Code blocks limited to 8-10 lines (truncate with `// ...` if longer)
- Tables limited to 5 columns x 6 rows

---

### Generic HTML

**Detection:**
- None of the above patterns match

**Fallback extraction:**
- Look for `<main>`, `<article>`, or `<body>` as root content
- Strip `<nav>`, `<header>`, `<footer>`, `<aside>`, `<sidebar>` elements
- Apply article extraction rules to remaining content
- If no heading structure exists, split every ~200 words into a new slide

---

## CSS Inlining Strategy

When the source has external stylesheets:

1. **Local CSS files** — Read the file and include its contents in `<style>`
2. **CDN CSS** — Do NOT fetch. Instead:
   - For framework CSS (reveal.css, marp themes): discard entirely (replaced by HTMLSlides CSS)
   - For utility CSS (normalize.css, reset.css): discard (HTMLSlides has its own reset)
   - For font CSS (Google Fonts, Fontshare): keep the `<link>` element (allowed by spec)
3. **Inline styles** — Preserve on elements, but ensure they use `clamp()` for font sizes
4. **Important:** After inlining, verify no `<link rel="stylesheet">` remains except allowed font imports

## JS Replacement Strategy

Framework navigation JS is **completely replaced**, not merged:

1. **Strip all of:**
   - Framework initialization (`Reveal.initialize()`, `impress().init()`, etc.)
   - Framework event handlers and plugins
   - Slide transition JS
   - Framework-specific keyboard handlers

2. **Keep:**
   - Chart.js chart configurations (convert to HTMLSlides chart component format)
   - Custom animation JS that doesn't conflict with HTMLSlides
   - Data-processing scripts that feed into charts or dynamic content

3. **Replace with:**
   - Standard HTMLSlides navigation from [html-template.md](html-template.md) (for Vibe presets)
   - Or [slides-runtime.js](../assets/slides-runtime.js) verbatim (for Pro mode)

## Image Handling

- **Relative URLs** — Keep as-is (they work when opened from the same directory)
- **Absolute URLs** — Keep but warn user about offline accessibility
- **Base64-encoded** — Keep if < 500KB per image. If larger, warn user to save as separate file
- **SVG inline** — Keep inline SVGs as-is
- **All images** must have: `max-height: min(50vh, 400px); width: auto; object-fit: contain;`
