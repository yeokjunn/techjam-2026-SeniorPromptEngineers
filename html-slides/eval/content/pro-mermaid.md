# MCP Architecture Deep Dive

**Mode:** Pro
**Theme:** Obsidian
**Length:** 6 slides

**Library requirements:** Mermaid.js v11.13.0 (ESM import)
**Read references/libraries.md before generating — follow Mermaid section exactly.**

---

## Slide 1 — Title
**Title:** Model Context Protocol
**Subtitle:** Architecture deep dive
**Tag:** MCP

## Slide 2 — Mermaid Flowchart (TD)
**Tag:** Architecture
**Heading:** Client-Server Connection Flow

```mermaid
graph TD
  H[MCP Host] --> C1[Client 1]
  H --> C2[Client 2]
  C1 --> S1[Local Server]
  C2 --> S2[Remote Server]
  S1 --> D1[Filesystem]
  S2 --> D2[Database]
```

Requirements:
- Wrap in `<div class="mermaid-wrap">` with background, border, border-radius
- Use flowchart TD (not LR)
- Max 8 nodes
- themeVariables matching Obsidian palette
- fontSize: '18px'
- Include diagram lightbox with +/- zoom buttons

## Slide 3 — Mermaid Sequence Diagram
**Tag:** Protocol
**Heading:** Tool Call Lifecycle

```mermaid
sequenceDiagram
  participant H as Host
  participant C as Client
  participant S as Server
  C->>S: initialize
  S-->>C: capabilities
  C->>S: tools/list
  S-->>C: available tools
  C->>S: tools/call
  S-->>C: result
```

Same container requirements as Slide 2.

## Slide 4 — Mermaid Mind Map
**Tag:** Ecosystem
**Heading:** MCP Components

```mermaid
mindmap
  root((MCP))
    Primitives
      Tools
      Resources
      Prompts
    Transport
      STDIO
      HTTP
    Clients
      Claude
      VS Code
      Cursor
```

Same container requirements.

## Slide 5 — Content Slide (no Mermaid)
**Tag:** Primitives
**Heading:** Three Building Blocks

Explain Tools, Resources, Prompts as three cards or sections (use component templates).

## Slide 6 — CTA
**Heading:** Get Started with MCP
**Resources:**
- Spec: modelcontextprotocol.io
- SDKs: TypeScript, Python, Java, C#
- Servers: github.com/modelcontextprotocol/servers
