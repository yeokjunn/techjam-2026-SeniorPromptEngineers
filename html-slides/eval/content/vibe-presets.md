# Vibe Preset Test Cases

**Purpose:** Test creative visual variety across Vibe presets. Same content, different styles — each should look distinctively different.
**Mode:** Vibe
**Content:** A short 6-slide pitch deck for "Aura" — an AI-powered home lighting system

---

## Shared Content (used for all preset tests)

### Slide 1 — Title
**Title:** Aura
**Subtitle:** Lighting that learns your life

### Slide 2 — Problem
**Heading:** Smart lighting isn't smart
**Body:** Current systems require manual scenes, complex apps, and don't adapt. You end up with the same brightness all day.

### Slide 3 — Solution
**Heading:** Aura adapts to you
**Body:** Three capabilities:
- Circadian sync — matches natural light throughout the day
- Presence awareness — adjusts when you enter, dims when you leave
- Mood sensing — reads ambient signals (music, time, calendar) to set the tone

### Slide 4 — How It Works
**Heading:** The Aura Engine
**Body:** Three-step pipeline:
- Sense: ambient light sensors + calendar + music API
- Decide: on-device ML model selects scene
- Act: Zigbee/Matter commands to bulbs in <200ms

### Slide 5 — Traction
**Heading:** Early results
**Stats:**
- 12K homes in beta
- 4.8 App Store rating
- 73% reduction in manual adjustments
- 91% say "it just works"

### Slide 6 — Closing
**Heading:** Aura
**Subtitle:** Stop configuring. Start living.
**Contact:** hello@aura.lighting

---

## Test Cases

### Test A — Bold Signal (Dark, Impressed/Confident)

**Preset:** Bold Signal
**Expected:** Confident, bold, modern. Archivo Black display font. Colored card focal points on dark gradient. Large section numbers. High-impact layout.

### Test B — Dark Botanical (Dark, Inspired/Moved)

**Preset:** Dark Botanical
**Expected:** Elegant serif typography (Cormorant). Warm color accents (pink, gold, terracotta). Abstract soft gradient circles. Thin vertical accent lines. Premium, artistic feel.

### Test C — Swiss Modern (Light, Calm/Focused)

**Preset:** Swiss Modern
**Expected:** Bauhaus-inspired. Archivo + Nunito. Pure white/black with red accent. Visible grid, asymmetric layouts, geometric shapes. Clean and precise.

### Test D — Creative Voltage (Dark, Excited/Energized)

**Preset:** Creative Voltage
**Expected:** Retro-modern. Syne + Space Mono. Electric blue + neon yellow. Halftone textures, neon badges. Script typography accents. Energetic, bold.

### Test E — Notebook Tabs (Light, Calm/Focused)

**Preset:** Notebook Tabs
**Expected:** Editorial, tactile. Bodoni Moda + DM Sans. Cream paper card on dark background. Colorful section tabs on right edge. Binder hole decorations. Organized, elegant.

### Test F — Vintage Editorial (Light, Inspired/Moved)

**Preset:** Vintage Editorial
**Expected:** Witty, personality-driven. Fraunces + Work Sans. Cream background. Abstract geometric shapes (circle outline + line + dot). Bold bordered CTA boxes. Distinctive serif character.

---

## Test G — Excalidraw Light (CSS-only)

**Preset:** Excalidraw Light
**Content:** Different topic — "DevOps Pipeline Overview" (8 slides, not the Aura deck)

### Slide 1 — Title
**Title:** DevOps Pipeline
**Subtitle:** From commit to production

### Slide 2 — Overview
**Heading:** Pipeline stages
**Body:** Three stages connected by CSS-drawn arrows:
- Build — Compile, lint, unit tests
- Test — Integration, E2E, security scan
- Deploy — Staging → approval → production

Cards use Excalidraw CSS theme (hachure fills via CSS gradient, 14px border-radius, offset shadows). Arrows drawn with CSS borders/pseudo-elements.

### Slide 3 — Metrics
**Heading:** Pipeline health
**Stats:** 4 metrics with CSS decorative circles around each number:
- 8min avg build time
- 99.2% pass rate
- 14 deploys/day
- <2min rollback

Use CSS border-radius circles as decorative annotation around each stat number.

### Slide 4 — Architecture
**Heading:** Infrastructure
**Body:** Simple architecture with CSS connectors:
- GitHub → CI Runner → Registry → Kubernetes
Cards are CSS-styled. Connectors drawn with CSS borders/pseudo-elements.

### Slide 5 — Key Takeaways
**Heading:** What we learned
**Body:** Three points with CSS decorative annotations:
- CSS underline beneath "Automate everything" (wavy border or pseudo-element)
- CSS circle around "Observability" (border-radius + offset shadow)
- CSS arrow pointing to "Shift left on security" (pseudo-element)

### Slide 6 — Tools
**Heading:** Our stack
**Body:** 6 tool cards in a 2x3 grid: GitHub Actions, Docker, ArgoCD, Datadog, Terraform, Vault

### Slide 7 — Timeline
**Heading:** Migration journey
**Body:** 4 milestones: Q1 manual deploys → Q2 CI pipeline → Q3 CD automation → Q4 full GitOps

### Slide 8 — Closing
**Heading:** DevOps Pipeline
**Subtitle:** Automate. Observe. Iterate.

**Expected:** Caveat hand-drawn font. Excalidraw Light colors (white bg, #1971c2 blue accent). CSS hachure fills on cards (repeating-linear-gradient). CSS-drawn connectors, arrows, circles, and underlines — all pure CSS, no JavaScript. 14px border-radius on all cards. Offset drop shadows.

**Extra assertions:**
- No Rough.js CDN included — hand-drawn aesthetic is CSS-only
- Card borders are CSS (border-radius: 14px)
- Connectors and decorations use CSS borders/pseudo-elements
- Caveat font loaded from Google Fonts
- No external JS libraries for visual effects

---

## Assertions

For each test case, verify:

1. **Distinctive fonts** — Uses the preset's specified font pairing, NOT Inter/Roboto/Arial
2. **Correct color palette** — Colors match the preset's CSS variables
3. **Signature elements present** — At least 2 of the preset's signature elements are visible
4. **viewport-base.css included** — Has scroll-snap, clamp() sizing, height breakpoints
5. **No component CSS** — Does NOT include components.css class names (.flip-front, .arch-box, etc.)
6. **Speaker notes** — Every slide has a slide-notes JSON block
7. **Visual variety** — No two test cases should look similar
8. **Content preserved** — All 6 slides' content is present and accurate
9. **Spec rules pass** — All 8 validation rules satisfied
10. **Animations** — Slide entrance animations match the preset's feeling (e.g., bouncy for playful, slow fade for calm)
