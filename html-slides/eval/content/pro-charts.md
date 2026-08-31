# Engineering Metrics Dashboard

**Mode:** Pro
**Theme:** Obsidian
**Length:** 6 slides

---

## Slide 1 — Title
**Title:** Engineering Metrics
**Subtitle:** Q1 2026 Performance Dashboard
**Tag:** Dashboard

## Slide 2 — Bar Chart
**Tag:** Velocity
**Heading:** Deploy Frequency by Team

Type: bar
Labels: ["Platform", "Frontend", "Backend", "Data", "Mobile", "Infra"]
Dataset: "Deploys per week" — [18, 31, 24, 12, 15, 8]

## Slide 3 — Line Chart
**Tag:** Reliability
**Heading:** Incident Count Over Time

Type: line
Labels: ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
Dataset 1: "P1 Incidents" — [8, 6, 9, 4, 3, 2]
Dataset 2: "P2 Incidents" — [22, 18, 24, 15, 12, 9]

## Slide 4 — Pie Chart
**Tag:** Analysis
**Heading:** Error Category Distribution

Type: pie
Labels: ["Timeout", "Auth Failure", "Rate Limit", "Bad Input", "Server Error"]
Dataset: [32, 24, 18, 15, 11]

## Slide 5 — Doughnut Chart
**Tag:** Allocation
**Heading:** Engineering Time Distribution

Type: doughnut
Labels: ["Feature Work", "Bug Fixes", "Tech Debt", "Ops/On-call"]
Dataset: [45, 20, 22, 13]

## Slide 6 — CTA
**Heading:** Next Steps
**Resources:**
- Grafana dashboard: grafana.internal/eng-metrics
- Weekly sync: Thursdays 2pm
- Playbook: confluence/incident-response
