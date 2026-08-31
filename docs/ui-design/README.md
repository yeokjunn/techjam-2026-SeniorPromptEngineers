# Run Ledger — UI screenshots

Captured headless (Playwright/Chromium, 1440px, 2x) against the finished run
`runs/kj_20260830T160447546759Z_research`. Both themes are Streamlit's own
`[theme.light]` / `[theme.dark]` palettes from `.streamlit/config.toml`, and
every capture flips the theme through the app's OWN menu (⋮ → Light/Dark) in
a live session — never by reloading with an emulated color scheme — because
the custom CSS derives all its tones from `currentColor` via `color-mix()`
and must follow the toggle instantly.

| File | View |
| --- | --- |
| `00-toggle-before.png` / `00-toggle-after.png` | The no-reload proof: one browser session, theme flipped Light → Dark through the app menu; background `#F3F5F4 → #0F1517` and ink `#1B2421 → #E4ECE9` changed with zero navigations |
| `01-story-light.png` / `01-story-dark.png` | Story (landing): 60-second band (health, margin-gated claim, budget), wayfinding, score trajectory; full verdict ledger one click away |
| `02-activity-light.png` / `02-activity-dark.png` | Activity · Loop stages: last/live activity, stage strip, role passes, notes, timeline (Iterations and Model calls are sibling sections behind the switcher) |
| `03-evidence-light.png` / `03-evidence-dark.png` | Evidence · Trust & audit: gate, baseline provenance, interventions ledger, telemetry, journal, submission checker (Dataset and Features behind the switcher) |
| `04-judge-light.png` / `04-judge-dark.png` | Judge sheet: the print-ready one-pager — verdict ledger, trajectory, lineage, iteration ledger, spend, provenance |
