# Libraries Reference

Optional CDN libraries for slides that need more than pure HTML/CSS. Only include what the presentation actually needs — most slides need zero external JS.

**Rule: include a library only when the slide content genuinely calls for it.** A timeline doesn't need Mermaid — CSS works fine. A simple animation doesn't need anime.js — CSS transitions are enough. But a 10-node architecture diagram? That's Mermaid territory.

---

## Chart.js — Data Visualizations

Use for bar charts, line charts, pie/doughnut charts, radar charts, and other data-driven visualizations. Don't use for simple numbers or stats — styled HTML is better for those.

**CDN (v4.4.8):**
```html
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.8/dist/chart.umd.min.js"></script>
```

### Basic Setup

```html
<div style="position: relative; max-height: 50vh; margin: 0 auto;">
  <canvas id="myChart"></canvas>
</div>

<script>
  new Chart(document.getElementById('myChart'), {
    type: 'bar',
    data: {
      labels: ['Q1', 'Q2', 'Q3', 'Q4'],
      datasets: [{
        label: 'Revenue',
        data: [45, 62, 78, 91],
        backgroundColor: '#58a6ff33',
        borderColor: '#58a6ff',
        borderWidth: 2,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 800, easing: 'easeOutQuart' },
      plugins: {
        legend: {
          labels: { color: '#8b949e', font: { family: "'Space Grotesk', sans-serif", size: 12 } }
        }
      },
      scales: {
        x: {
          ticks: { color: '#6e7681', font: { family: "'Space Grotesk', sans-serif", size: 11 } },
          grid: { color: 'rgba(255,255,255,0.06)' }
        },
        y: {
          ticks: { color: '#6e7681', font: { family: "'Space Grotesk', sans-serif", size: 11 } },
          grid: { color: 'rgba(255,255,255,0.06)' }
        }
      }
    }
  });
</script>
```

### Theming for Slides

Since you're generating all CSS from scratch, you already know the presentation's color palette. Use those colors directly in Chart.js config — no need to read CSS variables at runtime.

- **Font family**: match the presentation's body font
- **Text colors**: use the palette's muted/dim text colors for labels and ticks
- **Grid color**: use the palette's border color at low opacity
- **Dataset colors**: use the palette's accent colors with `33` (20%) alpha for backgrounds, full color for borders
- **For pie/doughnut**: assign each segment a different accent color

### Sizing for Slides

Charts must fit within 100vh slides without scrolling:

```css
.chart-container {
  position: relative;
  max-height: min(50vh, 400px);
  margin: 0 auto;
  padding: clamp(0.5rem, 2vw, 1.5rem);
}
```

Set `maintainAspectRatio: false` and `responsive: true` so Chart.js fills the container.

### Slide Transition Lifecycle

If a presentation has multiple chart slides, create charts when the slide becomes active and destroy when leaving. This prevents memory leaks:

```js
function goTo(index) {
  // ... slide transition logic ...

  // Destroy charts on previous slide
  if (typeof Chart !== 'undefined') {
    const prevCanvases = slides[prevIndex].querySelectorAll('canvas');
    prevCanvases.forEach(c => {
      const instance = Chart.getChart(c);
      if (instance) instance.destroy();
    });
  }

  // Create charts on new slide (after transition)
  setTimeout(() => {
    slides[index].querySelectorAll('canvas[data-chart]').forEach(c => {
      initChart(c); // your chart init function
    });
  }, 400);
}
```

### Chart Types Quick Reference

| Type | Best for | Key options |
|------|----------|-------------|
| `bar` | Comparisons, rankings | `borderRadius`, `barPercentage` |
| `line` | Trends over time | `tension: 0.4` for curves, `fill: true` for area |
| `pie` / `doughnut` | Proportions | `cutout: '60%'` for doughnut |
| `radar` | Multi-dimensional comparison | `pointBackgroundColor` |
| `scatter` | Correlations | `pointRadius`, `pointHoverRadius` |
| `polarArea` | Magnitude comparison | Similar to pie but area-proportional |

---

## Mermaid.js — Diagrams & Flowcharts

Use for flowcharts, sequence diagrams, ER diagrams, state machines, mind maps, and class diagrams — any diagram where automatic node positioning and edge routing saves effort. Mermaid handles layout; you handle theming.

Don't use for simple layouts that CSS Grid can handle (e.g., 4 feature cards). Reserve Mermaid for actual relational or flow-based diagrams.

**CDN (v11.13.0) — ESM import required:**
```html
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11.13.0/dist/mermaid.esm.min.mjs';
  mermaid.initialize({ startOnLoad: true, theme: 'base', themeVariables: { /* ... */ } });
</script>
```

**With ELK layout** (for complex graphs that need better auto-layout):
```html
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11.13.0/dist/mermaid.esm.min.mjs';
  import elkLayouts from 'https://cdn.jsdelivr.net/npm/@mermaid-js/layout-elk@0.2.1/dist/mermaid-layout-elk.esm.min.mjs';
  mermaid.registerLayoutLoaders(elkLayouts);
  mermaid.initialize({ startOnLoad: true, layout: 'elk', theme: 'base', themeVariables: { /* ... */ } });
</script>
```

Without the ELK import, `layout: 'elk'` silently falls back to dagre. Only import ELK when needed — it adds significant weight.

### Theming

Always use `theme: 'base'` — it's the only theme where all `themeVariables` work. The built-in themes (`default`, `dark`, `forest`, `neutral`) ignore most overrides.

**CRITICAL: Match themeVariables to your presentation's color palette.** If your slide has a dark background, Mermaid nodes must also be dark with light text. Default Mermaid colors (navy/light blue) will look broken on dark themes.

```js
// Example for a DARK theme presentation (e.g., Obsidian)
mermaid.initialize({
  startOnLoad: true,
  theme: 'base',
  look: 'classic',
  themeVariables: {
    // Match your slide's --bg-card and --text colors
    primaryColor: '#131720',        // node bg — use your --bg-card
    primaryBorderColor: '#58a6ff',  // node border — use your --accent-blue
    primaryTextColor: '#e6edf3',    // node text — use your --text
    secondaryColor: '#1a1f2e',      // secondary node bg
    secondaryBorderColor: '#3fb950',
    secondaryTextColor: '#e6edf3',
    tertiaryColor: '#1a1f2e',
    tertiaryBorderColor: '#f0883e',
    tertiaryTextColor: '#e6edf3',
    lineColor: '#8b949e',           // edges — use your --text-muted
    fontSize: '18px',
    fontFamily: 'var(--font-body)',
    background: '#0b0e14',          // diagram bg — use your --bg
    noteBkgColor: '#131720',
    noteTextColor: '#e6edf3',
    noteBorderColor: '#d29922',
    // ER diagram specific
    attributeBackgroundColorEven: '#131720',
    attributeBackgroundColorOdd: '#1a1f2e',
  }
});

// Example for a LIGHT theme presentation (e.g., Swiss Modern)
mermaid.initialize({
  startOnLoad: true,
  theme: 'base',
  look: 'classic',
  themeVariables: {
    primaryColor: '#f5f5f5',
    primaryBorderColor: '#1a1a1a',
    primaryTextColor: '#1a1a1a',
    secondaryColor: '#ffffff',
    secondaryBorderColor: '#ff3300',
    secondaryTextColor: '#1a1a1a',
    lineColor: '#666666',
    fontSize: '18px',
    background: '#ffffff',
  }
});
```

### Sizing for Slides

Mermaid diagrams in slides need aggressive sizing to be readable:

- **Use `fontSize: '18px'` minimum** in `mermaid.initialize()` — default 16px is too small on projected slides
- **Max 8 nodes per diagram** — more than that and the diagram shrinks to unreadable sizes within 100vh
- **Container must fill the slide** — use `width: 100%; max-width: 900px; min-height: 40vh;`
- **Default to `flowchart TD`** (top-down) — vertical diagrams fill 100vh slides much better than horizontal ones. `LR` diagrams become extremely short (100-200px tall) when squeezed into a slide-width container.
- **Only use `flowchart LR`** for very simple 3-4 node linear flows where each node label is short
- **If the diagram needs 10+ nodes**, split into 2 slides or use the hybrid pattern (simple Mermaid overview + CSS Grid detail cards)

```css
/* Container — card-like wrapper with background, same style as .chart-container */
.mermaid-wrap {
  width: 100%;
  max-width: 920px;
  min-height: 40vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--bg-card, #131720);
  border: 1px solid var(--border, rgba(255,255,255,0.07));
  border-radius: 14px;
  padding: 24px;
  margin-top: 24px;
  cursor: zoom-in;
  transition: all 0.35s ease;
}
.mermaid-wrap:hover {
  border-color: var(--border-hover, rgba(255,255,255,0.18));
  box-shadow: 0 12px 40px rgba(0,0,0,0.3);
}

/* Center and scale Mermaid SVG */
.mermaid-wrap .mermaid {
  width: 100%;
  display: flex;
  justify-content: center;
}
.mermaid-wrap .mermaid svg {
  max-width: 100%;
  max-height: 50vh;
  height: auto;
}
```

### Zoomable Diagram Lightbox

Diagrams on slides are often too small to read. Add a click-to-zoom lightbox so the audience can inspect details:

```css
/* Lightbox overlay */
.diagram-lightbox {
  position: fixed; inset: 0; z-index: 9999;
  background: rgba(0, 0, 0, 0.92);
  display: none; align-items: center; justify-content: center;
  cursor: default;
}
.diagram-lightbox.active { display: flex; }
.diagram-lightbox-inner {
  transform-origin: center center;
  transition: transform 0.2s ease;
  max-width: 95vw; max-height: 95vh;
  overflow: visible;
}
.diagram-lightbox-controls {
  position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
  display: flex; gap: 12px; align-items: center;
}
.diagram-lightbox-controls button {
  width: 36px; height: 36px; border: 1px solid rgba(255,255,255,0.2);
  background: rgba(255,255,255,0.1); color: white; font-size: 18px;
  cursor: pointer; display: flex; align-items: center; justify-content: center;
  border-radius: 4px;
}
.diagram-lightbox-controls button:hover { background: rgba(255,255,255,0.2); }
.diagram-lightbox-hint {
  font-size: 12px; color: rgba(255,255,255,0.4);
  font-family: var(--font-mono, monospace);
}

/* Clickable diagram container */
.mermaid-wrap { cursor: zoom-in; }
```

```js
/* Diagram Lightbox — click to open, +/- to zoom, Escape to close */
(function() {
  const lb = document.createElement('div');
  lb.className = 'diagram-lightbox';
  lb.innerHTML = '<div class="diagram-lightbox-inner"></div>'
    + '<div class="diagram-lightbox-controls">'
    + '<button id="lb-zoom-out">−</button>'
    + '<span class="diagram-lightbox-hint">click backdrop or ESC to close</span>'
    + '<button id="lb-zoom-in">+</button>'
    + '</div>';
  document.body.appendChild(lb);
  const inner = lb.querySelector('.diagram-lightbox-inner');
  let scale = 1;

  document.querySelectorAll('.mermaid-wrap').forEach(wrap => {
    wrap.addEventListener('click', (e) => {
      e.stopPropagation();
      const svg = wrap.querySelector('svg');
      if (!svg) return;
      inner.innerHTML = '';
      const clone = svg.cloneNode(true);
      inner.appendChild(clone);
      // Auto-fit: calculate scale so SVG fills 90% of viewport
      const vw = window.innerWidth * 0.9;
      const vh = window.innerHeight * 0.9;
      const svgW = svg.viewBox?.baseVal?.width || svg.getBoundingClientRect().width;
      const svgH = svg.viewBox?.baseVal?.height || svg.getBoundingClientRect().height;
      scale = Math.min(vw / svgW, vh / svgH, 3);
      inner.style.transform = 'scale(' + scale + ')';
      lb.classList.add('active');
    });
  });

  lb.addEventListener('click', (e) => {
    if (e.target === lb) lb.classList.remove('active');
  });
  document.getElementById('lb-zoom-in').addEventListener('click', (e) => {
    e.stopPropagation();
    scale = Math.min(5, scale + 0.3);
    inner.style.transform = 'scale(' + scale + ')';
  });
  document.getElementById('lb-zoom-out').addEventListener('click', (e) => {
    e.stopPropagation();
    scale = Math.max(0.3, scale - 0.3);
    inner.style.transform = 'scale(' + scale + ')';
  });
  document.addEventListener('keydown', (e) => {
    if (!lb.classList.contains('active')) return;
    if (e.key === 'Escape') lb.classList.remove('active');
    if (e.key === '+' || e.key === '=') { scale = Math.min(5, scale + 0.3); inner.style.transform = 'scale(' + scale + ')'; }
    if (e.key === '-') { scale = Math.max(0.3, scale - 0.3); inner.style.transform = 'scale(' + scale + ')'; }
  });
})();
```

Add both the CSS and JS to any presentation that uses Mermaid or complex diagrams. Wrap each diagram in a `<div class="mermaid-wrap">` to make it clickable.

### CSS Overrides on Mermaid SVG

Mermaid renders SVG. Override its classes for precise control:

```css
/* Container */
.mermaid-wrap {
  position: relative;
  border-radius: 12px;
  padding: clamp(1rem, 2vw, 1.5rem);
  overflow: auto;
  max-height: min(55vh, 500px);
}

/* Force text to follow your color scheme */
.mermaid .nodeLabel { color: var(--text) !important; }
.mermaid .edgeLabel { color: var(--text-muted) !important; background-color: transparent !important; }
.mermaid .edgeLabel rect { fill: transparent !important; }

/* Node shapes */
.mermaid .node rect,
.mermaid .node circle,
.mermaid .node polygon { stroke-width: 1.5px; }

/* Edge labels — smaller for visual hierarchy */
.mermaid .edgeLabel { font-size: 13px !important; }

/* Node labels */
.mermaid .nodeLabel { font-size: 16px !important; }
```

### Diagram Syntax

**Flowchart:**
```html
<pre class="mermaid">
graph TD
  A[Request] --> B{Authenticated?}
  B -->|Yes| C[Load Dashboard]
  B -->|No| D[Login Page]
  D --> E[Submit Credentials]
  E --> B
</pre>
```

**Sequence diagram:**
```html
<pre class="mermaid">
sequenceDiagram
  participant C as Client
  participant G as Gateway
  participant S as Service
  C->>G: POST /api/data
  G->>G: Validate JWT
  G->>S: Forward request
  S-->>G: Response
  G-->>C: 200 OK
</pre>
```

**State diagram:**
```html
<pre class="mermaid">
stateDiagram-v2
  [*] --> Draft
  Draft --> Review : submit
  Review --> Approved : approve
  Review --> Draft : request_changes
  Approved --> Published : publish
</pre>
```

**Mind map:**
```html
<pre class="mermaid">
mindmap
  root((Project))
    Frontend
      React
      Next.js
    Backend
      Node.js
      PostgreSQL
    Infra
      AWS
      Docker
</pre>
```

**Architecture (use flowchart, not native C4):**
```html
<pre class="mermaid">
graph TD
  user("User") -->|HTTPS| app["Web App"]
  subgraph Backend
    app --> db[("Database")]
    app --> cache["Redis"]
  end
  app -->|API| ext["External Service"]:::ext
  classDef ext fill:none,stroke-dasharray:5 5
</pre>
```

Don't use native `C4Context` / `C4Container` syntax — it ignores theme settings.

### Which Diagram Type?

| You want to show... | Use | Syntax |
|---|---|---|
| Process flow, decisions, pipelines | Flowchart | `graph TD` / `graph LR` |
| Request/response, API calls | Sequence | `sequenceDiagram` |
| Database tables and relationships | ER diagram | `erDiagram` |
| Classes, domain models | Class diagram | `classDiagram` |
| System architecture | C4-style | `graph TD` + `subgraph` |
| State transitions, lifecycles | State diagram | `stateDiagram-v2` |
| Hierarchical breakdowns | Mind map | `mindmap` |

### Gotchas

1. **Max 10-12 nodes per diagram.** Beyond that, readability collapses. For complex architectures, use a simple 5-8 node Mermaid overview + CSS Grid detail cards.

2. **Use `TD` (top-down) over `LR` (left-to-right)** for anything with branches or 5+ nodes. `LR` scales everything down to fit width, making text unreadable.

3. **Quote labels with special characters.** Parentheses, colons, brackets break the parser:
   ```
   A["handleRequest(ctx)"] --> B["DB: query"]
   ```

4. **Keep IDs simple.** Node IDs should be alphanumeric. Put readable names in labels:
   ```
   userSvc["User Service"] --> authSvc["Auth Service"]
   ```

5. **Sequence diagram messages must be plain text.** No braces `{}`, brackets `[]`, angle brackets `<>`, or `&`:
   ```
   %% WRONG: A->>B: call({ query: [...] })
   %% RIGHT: A->>B: Call search with queries
   ```

6. **Multi-line labels use `<br/>` not `\n`** (flowcharts only):
   ```
   A["Web Server<br/>Port 3000"]
   ```

7. **`classDef` — never set `color:`** (hardcodes text color that breaks in opposite color scheme). Let CSS overrides handle text color. Use semi-transparent fills:
   ```
   classDef highlight fill:#b5761433,stroke:#b57614,stroke-width:2px
   ```

8. **Arrow styles for meaning:**
   - `-->` solid = primary flow
   - `-.->` dotted = optional/async
   - `==>` thick = critical path
   - `--x` cross = blocked/rejected

9. **Mermaid initializes once** — it can't switch themes reactively. Set dark/light at load time.

10. **State diagram labels are limited.** No `<br/>`, no parentheses in labels, no multiple colons. Use flowcharts instead if you need these.

---

## anime.js — Orchestrated Animations

Use when a slide has 10+ elements that need choreographed entrance sequences — staggered reveals, path drawing, count-up numbers. For simpler animations, CSS `animation-delay` staggering is sufficient and lighter.

**CDN (v3.2.2):**
```html
<script src="https://cdn.jsdelivr.net/npm/animejs@3.2.2/lib/anime.min.js"></script>
```

Pinned to v3 — v4 is 6x larger (106 KB vs 17 KB) with breaking API changes. v3 covers all presentation animation needs.

### Common Patterns

**Staggered card entrance:**
```js
anime({
  targets: '.card',
  opacity: [0, 1],
  translateY: [20, 0],
  delay: anime.stagger(80, { start: 200 }),
  easing: 'easeOutCubic',
  duration: 500
});
```

**SVG path drawing:**
```js
anime({
  targets: '.connector path',
  strokeDashoffset: [anime.setDashoffset, 0],
  easing: 'easeInOutCubic',
  duration: 800,
  delay: anime.stagger(150, { start: 600 })
});
```

**Count-up numbers:**
```js
document.querySelectorAll('[data-count]').forEach(el => {
  anime({
    targets: { val: 0 },
    val: parseInt(el.dataset.count),
    round: 1,
    duration: 1200,
    delay: 400,
    easing: 'easeOutExpo',
    update: (anim) => { el.textContent = anim.animations[0].currentValue; }
  });
});
```

### Reduced Motion Support

Always respect `prefers-reduced-motion`. Set initial state in CSS and skip animations:

```css
/* Initial state — hidden until animated */
.card { opacity: 0; }

/* Respect reduced motion */
@media (prefers-reduced-motion: reduce) {
  .card { opacity: 1 !important; transform: none !important; }
}
```

```js
const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
if (!prefersReduced) {
  anime({ /* ... */ });
}
```

### Slide-Aware Animation

Only animate when the slide becomes active:

```js
function goTo(index) {
  // ... slide transition ...

  // Trigger animations on new slide
  setTimeout(() => {
    animateSlide(index);
  }, 300);
}

function animateSlide(index) {
  const slide = slides[index];
  const cards = slide.querySelectorAll('.card');
  if (cards.length === 0) return;

  anime({
    targets: cards,
    opacity: [0, 1],
    translateY: [30, 0],
    delay: anime.stagger(100),
    easing: 'easeOutCubic',
    duration: 600
  });
}
```

### Key Easing Values

| Easing | Feel | Best for |
|--------|------|----------|
| `easeOutCubic` | Smooth deceleration | Card entrances |
| `easeOutExpo` | Fast start, gentle stop | Count-ups, number reveals |
| `easeInOutCubic` | Smooth both ends | Path drawing |
| `spring(1, 80, 10, 0)` | Bouncy | Playful entrances |
| `easeOutElastic(1, .5)` | Overshoot + settle | Attention-grabbing |

---

## When to Use What

| Content | Library | Why |
|---------|---------|-----|
| Bar/line/pie chart | Chart.js | Data visualization with animations |
| Architecture diagram | Mermaid.js | Auto node positioning + edge routing |
| Sequence of API calls | Mermaid.js | `sequenceDiagram` syntax |
| Database schema | Mermaid.js | `erDiagram` syntax |
| 10+ elements staggered entrance | anime.js | Orchestrated timing |
| Count-up statistics | anime.js | Animated number reveal |
| Hand-drawn aesthetic | **CSS only** | CSS hachure fills, offset shadows, rounded corners |
| Simple card entrance | **CSS only** | `animation-delay` staggering |
| Feature grid | **CSS only** | CSS Grid layout |
| Timeline | **CSS only** | Styled list or flexbox |
| Stats/numbers (no animation) | **CSS only** | Styled HTML |
| Code snippet | **CSS only** | `<pre><code>` with styling |
| Quote | **CSS only** | Styled blockquote |

**Default to CSS.** Reach for a library only when CSS can't do it well.
