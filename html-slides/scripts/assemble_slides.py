import os
import json

def main():
    # Base paths
    workspace_dir = r"c:\Users\Admin\OneDrive - Nanyang Technological University\2026_projects\tiktok_techjam_clean\techjam-2026-SeniorPromptEngineers"
    html_slides_dir = os.path.join(workspace_dir, "html-slides")
    artifact_dir = r"C:\Users\Admin\.gemini\antigravity\brain\95648f60-48a0-4b65-a1c2-3d4417a44fc4"

    # Read dark-interactive.css theme
    theme_path = os.path.join(html_slides_dir, "assets", "dark-interactive.css")
    with open(theme_path, "r", encoding="utf-8") as f:
        theme_css = f.read()

    # Read slides-runtime.js runtime
    runtime_path = os.path.join(html_slides_dir, "assets", "slides-runtime.js")
    with open(runtime_path, "r", encoding="utf-8") as f:
        runtime_js = f.read()

    # Custom styles to append (reverted to contain for diagram preservation, restored padding)
    custom_css = """
    .hidden { display: none !important; }
    .vs-container { gap: 16px; margin-top: 16px; }
    .vs-card { padding: 20px 16px; }
    .vs-name { font-size: 28px; margin-bottom: 8px; }
    .vs-desc { font-size: 12px; margin-bottom: 8px; line-height: 1.4; }
    
    .slide-layout-cols {
      display: grid;
      grid-template-columns: 2fr 3fr;
      gap: 28px;
      align-items: center;
      width: 100%;
      max-width: 980px;
      margin-top: 16px;
    }
    
    .slide-layout-gallery {
      display: grid;
      grid-template-columns: 1.5fr 3.5fr;
      gap: 28px;
      align-items: center;
      width: 100%;
      max-width: 1020px;
      margin-top: 16px;
    }
    
    .gallery-grid {
      display: grid;
      grid-template-rows: 1fr 1fr;
      gap: 12px;
    }
    
    .slide-img-framed {
      border: 1.5px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      background: var(--bg-card);
      box-shadow: 0 8px 24px rgba(0,0,0,0.3);
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.3s ease;
      padding: 4px !important;
    }
    .slide-img-framed:hover {
      border-color: var(--border-hover);
      box-shadow: 0 12px 36px rgba(0,0,0,0.4);
    }
    .slide-img-framed img {
      width: 100%;
      height: auto;
      max-height: min(42vh, 310px);
      object-fit: contain;
      transition: transform 0.5s cubic-bezier(0.4, 0, 0.2, 1);
      transform-origin: center;
    }
    .gallery-grid .slide-img-framed img {
      max-height: min(23vh, 175px);
    }
    """

    # Build slides content (with speaker notes)
    # Slide 0: Cover
    slide_0 = """
    <div class="slide active" data-slide="0">
      <div class="glow-blob glow-blue" style="top:-100px;left:-100px;"></div>
      <div class="glow-blob glow-purple" style="bottom:-120px;right:-80px;"></div>
      <p class="slide-tag anim-1">Track 2 Recommendation Challenge</p>
      <h1 class="anim-2">Autonomous ML Research Agent<br><span class="rainbow-text">for Recommender Systems</span></h1>
      <p class="subtitle anim-3">An autonomous research loop that proposes hypotheses, generates pipeline code, trains, and reflects under the KuaiRand-Pure benchmark.</p>
      <div class="rainbow-line anim-4" style="margin: 24px auto;"></div>
      <p class="subtitle anim-5" style="font-size:14px;">by <strong>Senior Prompt Engineers</strong></p>
      <script type="application/json" class="slide-notes">
      {
        "title": "Introduction",
        "script": "Welcome to our pitch. Today, we present our Autonomous ML Research Agent designed to optimize recommendation models under the KuaiRand-Pure benchmark.",
        "notes": [
          "Focus: Track 2 Recommendation Challenge",
          "Objective: Build an autonomous researcher to optimize recommendation pipelines",
          "Team: Senior Prompt Engineers"
        ]
      }
      </script>
    </div>
    """

    # Slide 1: Problem Statement
    slide_1 = """
    <div class="slide" data-slide="1">
      <p class="slide-tag anim-1">1. Problem &amp; Constraints</p>
      <h2 class="anim-2">KuaiRand-Pure Engagement Ranking</h2>
      
      <div class="vs-container anim-3">
        <div class="vs-card card-left slide-l">
          <div class="vs-label-top highlight-purple">Target Task</div>
          <div class="vs-name highlight-purple">long_view</div>
          <div class="vs-desc">Rank video impressions against other impressions belonging to the same user. Evaluated strictly within-user.</div>
          <div class="vs-badge">Metric: (GAUC + nDCG@5) / 2</div>
        </div>
        
        <div class="vs-plus anim-3">+</div>
        
        <div class="vs-card card-right slide-r">
          <div class="vs-label-top highlight-blue">Validation Baseline</div>
          <div class="vs-name gradient-text">0.60160</div>
          <div class="vs-desc">Max 50 iterations or 6 hours. Convergence triggered when validation score improvement <= 0.002 for 3 rounds.</div>
          <div class="vs-badge">Hidden Test: 0.5946</div>
          
          <!-- Interactive Trigger -->
          <div style="margin-top: 16px; display: flex; flex-direction: column; align-hidden: center; justify-content: center; align-items: center;">
            <button id="beat-button" onclick="revealBeat(event)" style="padding: 6px 14px; background: var(--accent-green); border: none; border-radius: 6px; color: var(--bg); font-weight: 700; font-size: 11px; cursor: pointer; transition: transform 0.2s ease;">
              Did we beat the baseline?
            </button>
            <div id="beat-result" class="hidden" style="margin-top: 8px; text-align: center; animation: fadeInUp 0.4s ease both;">
              <div style="color: var(--accent-green); font-weight: 800; font-size: 13px; display: flex; align-items: center; gap: 4px; justify-content: center; animation: bounceInSoft 0.5s ease both;">
                <span>🎉</span> YES! WE BEAT IT!
              </div>
              <div style="font-size: 10px; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; margin-top: 2px;">
                Diverse Ensemble: <strong>0.60480</strong> vs Floor: 0.60160 (<span style="color: var(--accent-green);">+0.00320</span>)
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <script type="application/json" class="slide-notes">
      {
        "title": "Problem & Constraints",
        "script": "The task requires ranking impressions based on the long_view label. The strict convergence threshold requires us to stop if improvements stagnate. Let's see if we beat it.",
        "notes": [
          "Label: long_view ranking within-user",
          "Primary score: average of GAUC and nDCG@5",
          "Strict budgets: 50 iterations, 6 hours max"
        ]
      }
      </script>
    </div>
    """

    # Slide 2: High-Level Agent Architecture
    slide_2_workspace = """
    <div class="slide" data-slide="2">
      <p class="slide-tag anim-1">2. High-Level Architecture</p>
      <h2 class="anim-2">Autonomous Single-Controller Loop</h2>
      <p class="subtitle anim-2" style="font-size: 13px; margin-bottom: 8px;">Click cards below to zoom; click diagram to zoom back out</p>
      
      <div class="slide-layout-cols anim-3" style="grid-template-columns: 3fr 4fr;">
        <div id="arch-cards" style="text-align: left; display: grid; grid-template-columns: 1fr 1fr; gap: 10px;" class="slide-l">
          <div class="card-v2 glow-blue" style="padding: 12px; height: 110px;" onclick="zoomDiagram(this, 'loop')">
            <div class="card-title" style="font-size: 13px; color: var(--accent-blue); margin-bottom: 2px;">🔄 Decoupled Passes</div>
            <p class="card-desc" style="font-size: 10px; line-height: 1.3;">Specialized Observe, Research, Critic, Builder, Validator steps.</p>
          </div>
          <div class="card-v2 glow-green" style="padding: 12px; height: 110px;" onclick="zoomDiagram(this, 'debugger')">
            <div class="card-title" style="font-size: 13px; color: var(--accent-green); margin-bottom: 2px;">⚙️ Self-Correction</div>
            <p class="card-desc" style="font-size: 10px; line-height: 1.3;">Debugger catches compilation or leakage errors to apply hotfixes.</p>
          </div>
          <div class="card-v2 glow-orange" style="padding: 12px; height: 110px;" onclick="zoomDiagram(this, 'families')">
            <div class="card-title" style="font-size: 13px; color: var(--accent-orange); margin-bottom: 2px;">🌟 Model Families</div>
            <p class="card-desc" style="font-size: 10px; line-height: 1.3;">Optimized listwise Group Softmax, BPR with decay, and MTL.</p>
          </div>
          <div class="card-v2 glow-purple" style="padding: 12px; height: 110px;" onclick="zoomDiagram(this, 'state')">
            <div class="card-title" style="font-size: 13px; color: var(--accent-purple); margin-bottom: 2px;">💾 Persisted State</div>
            <p class="card-desc" style="font-size: 10px; line-height: 1.3;">Synchronized JSON states track telemetry and memory logs.</p>
          </div>
        </div>
        
        <div class="slide-img-framed slide-r" style="cursor: pointer;" onclick="resetZoom()">
          <img id="arch-img" src="./assets/architecture_diagram.png" alt="High Level Agent Architecture Diagram" />
        </div>
      </div>
      
      <script type="application/json" class="slide-notes">
      {
        "title": "Agent Architecture",
        "script": "The agent coordinates specialized passes under a single controller. If the validator or compiler throws an error, the Debugger automatically intercepts and repairs candidate files.",
        "notes": [
          "Single controller coordinating role-based sub-agents",
          "Automated debugger repairing candidate compilation errors",
          "Shared persisted run state logging telemetry"
        ]
      }
      </script>
    </div>
    """

    slide_2_artifact = slide_2_workspace.replace("./assets/architecture_diagram.png", "./architecture_diagram.png")

    # Slide 3: Exploratory Search & Feature Lab (Stacked Up & Down)
    slide_3_workspace = """
    <div class="slide" data-slide="3">
      <p class="slide-tag anim-1">3. Search &amp; Feature Engineering</p>
      <h2 class="anim-2">Exploratory Search &amp; Feature Lab</h2>
      
      <!-- Top Row: Cards side-by-side -->
      <div class="anim-3" style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; width: 100%; max-width: 980px; margin-top: 12px;">
        <div class="card-v2 glow-yellow" style="padding: 10px; height: 105px;">
          <div class="card-title" style="font-size: 13px; color: var(--accent-yellow); margin-bottom: 2px;">🧠 Search Policy &amp; Exploration</div>
          <p class="card-desc" style="font-size: 9.5px; line-height: 1.35;">Thompson Sampling selects pipeline branches; Fisher Information bounds guide parameters.</p>
        </div>
        <div class="card-v2 glow-red" style="padding: 10px; height: 105px;">
          <div class="card-title" style="font-size: 13px; color: var(--accent-red); margin-bottom: 2px;">📊 Permutation Importance</div>
          <p class="card-desc" style="font-size: 9.5px; line-height: 1.35;">Dynamically measures model sensitivity to feature noise, screening candidates without leakage.</p>
        </div>
        <div class="card-v2 glow-blue" style="padding: 10px; height: 105px;">
          <div class="card-title" style="font-size: 13px; color: var(--accent-blue); margin-bottom: 2px;">🛠️ Counterfactual Watch Modeling</div>
          <p class="card-desc" style="font-size: 9.5px; line-height: 1.35;">Novel CWM bias-scaling handles video duration discrepancies across long-view impressions.</p>
        </div>
      </div>
      
      <!-- Bottom Row: Images side-by-side -->
      <div class="anim-4" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; width: 100%; max-width: 980px; margin-top: 16px;">
        <div class="slide-img-framed" style="height: min(24vh, 180px);">
          <img src="./assets/lineage_dag.png" alt="Visual Experiment Lineage DAG" style="height: 100%;" />
        </div>
        <div class="slide-img-framed" style="height: min(24vh, 180px);">
          <img src="./assets/eda_tab.png" alt="EDA Duration Distribution" style="height: 100%;" />
        </div>
      </div>
      
      <script type="application/json" class="slide-notes">
      {
        "title": "Exploratory Search & CWM",
        "script": "To pitch Counterfactual Watch Modeling, explain that recommender models are inherently biased towards recommending longer videos because they accumulate more watch time. CWM resolves this length bias by modeling the ratio of a user's watch duration against the video catalog's median duration baseline, ensuring short-form and long-form videos compete fairly. We discover pipelines using Tree Search and guide exploration parameters via Fisher Information, screening feature candidates safely with Permutation Importance.",
        "notes": [
          "Explain duration bias: Long videos naturally collect more watch time",
          "CWM solution: Models relative watch ratio against video median duration",
          "Benefit: Level playing field for ranking content fairly, boosting nDCG@5 score"
        ]
      }
      </script>
    </div>
    """

    slide_3_artifact = slide_3_workspace.replace("./assets/lineage_dag.png", "./lineage_dag.png").replace("./assets/eda_tab.png", "./eda_tab.png")

    # Slide 4: Observatory UI
    slide_4_workspace = """
    <div class="slide" data-slide="4">
      <p class="slide-tag anim-1">4. Live Observatory</p>
      <h2 class="anim-2">Observatory UI &amp; Trace Telemetry</h2>
      
      <div class="slide-layout-gallery anim-3">
        <div style="text-align: left; display: flex; flex-direction: column; gap: 14px;" class="slide-l">
          <div class="status-timeline" style="margin-top: 0; width: 100%;">
            <div class="status-item">
              <div class="status-dot green"></div>
              <div class="status-text" style="font-size: 12px;"><strong>Live Agent Tracker</strong>: Dynamic highlighting of active execution stages.</div>
            </div>
            <div class="status-item">
              <div class="status-dot yellow"></div>
              <div class="status-text" style="font-size: 12px;"><strong>Execution Trace Stream</strong>: Bookkeeping of models, token consumption, and repairs.</div>
            </div>
            <div class="status-item">
              <div class="status-dot orange"></div>
              <div class="status-text" style="font-size: 12px;"><strong>Leakage-Safe EDA</strong>: Fully dynamic visual charts calculated on train partitions.</div>
            </div>
          </div>
        </div>
        
        <div class="gallery-grid slide-r" style="grid-template-columns: 1fr 1fr; grid-template-rows: none;">
          <div class="slide-img-framed" style="height: min(24vh, 180px);">
            <img src="./assets/pipeline_tab.png" alt="Pipeline Tab UI" style="height: 100%;" />
          </div>
          <div class="slide-img-framed" style="height: min(24vh, 180px);">
            <img src="./assets/iterations_tab.png" alt="Iterations Tab UI" style="height: 100%;" />
          </div>
        </div>
      </div>
      
      <script type="application/json" class="slide-notes">
      {
        "title": "Observatory Dashboard",
        "script": "The Observatory UI keeps developers in control. It offers real-time pipeline status overlay, detailed agent traces, and fully dynamic leakage-safe profiling.",
        "notes": [
          "Active stage updates and live overlay rendering",
          "Chronological multi-role sub-agent trace stream",
          "Dynamic EDA charts directly mapped from train split summaries"
        ]
      }
      </script>
    </div>
    """

    slide_4_artifact = slide_4_workspace.replace("./assets/pipeline_tab.png", "./pipeline_tab.png").replace("./assets/iterations_tab.png", "./iterations_tab.png")

    # Slide 5: Experimental Results (Interactive Zoom on Best primary score)
    slide_5_workspace = """
    <div class="slide" data-slide="5">
      <p class="slide-tag anim-1">5. Experimental Results</p>
      <h2 class="anim-2">Outperforming the Target Baseline</h2>
      <p class="subtitle anim-2" style="font-size: 13px; margin-bottom: 8px;">Click green scores below to zoom in on Best (Iter 10) bar graph</p>
      
      <div class="slide-layout-cols anim-3">
        <div style="text-align: left; display: flex; flex-direction: column; gap: 14px;" class="slide-l">
          <div class="table-wrap" style="margin-top: 0;">
            <table>
              <thead>
                <tr>
                  <th>Experiment Run</th>
                  <th>Primary Score</th>
                  <th>GAUC</th>
                  <th>nDCG@5</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td class="cell-highlight">Official Baseline</td>
                  <td class="cell-muted font-mono">0.60160</td>
                  <td class="cell-muted font-mono">0.60160</td>
                  <td class="cell-muted font-mono">-</td>
                </tr>
                <tr>
                  <td class="cell-highlight">Run 1 (kj_0202)</td>
                  <td class="font-mono zoom-trigger" style="color: var(--accent-green); font-weight:700; cursor:pointer;" onclick="zoomResults(this)">0.60480</td>
                  <td class="font-mono">0.67184</td>
                  <td class="font-mono">0.53777</td>
                </tr>
                <tr>
                  <td class="cell-highlight">Run 2 (kj_0601)</td>
                  <td class="font-mono">0.60443</td>
                  <td class="font-mono">0.67110</td>
                  <td class="font-mono">0.53777</td>
                </tr>
                <tr>
                  <td class="cell-highlight">Run 3 (kj_0519)</td>
                  <td class="font-mono">0.60287</td>
                  <td class="font-mono">0.66980</td>
                  <td class="font-mono">0.53594</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div class="cta-box" style="margin-top: 0; padding: 14px 20px; font-size:12px;">
            🎉 All 3 best runs beat validation baseline! Max: <strong class="zoom-trigger" style="color: var(--accent-green); cursor:pointer;" onclick="zoomResults(this)">0.60480 (+0.00320)</strong>.
          </div>
        </div>
        
        <div class="slide-img-framed slide-r" style="height: min(42vh, 310px); cursor: pointer;" onclick="resetResultsZoom()">
          <img id="results-img" src="./assets/results_tab.png" alt="Results Tab UI" />
        </div>
      </div>
      
      <script type="application/json" class="slide-notes">
      {
        "title": "Experimental Results",
        "script": "Our runs consistently beat the validation baseline of 0.6016. Run 1 achieved 0.60480, demonstrating the effectiveness of the autonomous loop optimization.",
        "notes": [
          "Run 1 (kj_0202) leads with 0.60480 primary score",
          "Both listwise Softmax and pairwise BPR options beat baseline",
          "Submission files verified through the official validator checker"
        ]
      }
      </script>
    </div>
    """

    slide_5_artifact = slide_5_workspace.replace("./assets/results_tab.png", "./results_tab.png")

    # Combine CSS
    full_css = f"{theme_css}\n{custom_css}"

    # Build Workspace HTML
    html_workspace = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="generator" content="html-slides v0.9.4">
  <title>Autonomous ML Research Agent Presentation</title>
  <style>
    {full_css}
  </style>
</head>
<body>
  <!-- Particles background -->
  <div id="particles" class="particles"></div>
  
  <!-- Chrome elements -->
  <div class="branding">
    <div class="branding-icon">🤖</div>
    <span>TechJam Track 2 Presentation</span>
  </div>
  <div class="progress-bar" id="progress"></div>
  <div class="slide-counter" id="counter">1 / 6</div>
  <div class="nav-hints">
    Use <kbd>←</kbd> <kbd>→</kbd> or <kbd>Space</kbd> to navigate
  </div>
  <div class="slide-nav" id="slideNav"></div>

  <!-- Slide container -->
  <div class="deck" id="deck">
    {slide_0}
    {slide_1}
    {slide_2_workspace}
    {slide_3_workspace}
    {slide_4_workspace}
    {slide_5_workspace}
  </div>

  <script>
    {runtime_js}

    function revealBeat(event) {{
      event.stopPropagation(); // Prevent slide transition on click
      document.getElementById('beat-button').classList.add('hidden');
      document.getElementById('beat-result').classList.remove('hidden');
    }}

    function zoomDiagram(card, target) {{
      // Select the active slide's architecture image
      const activeSlide = document.querySelector('.slide.active');
      if (!activeSlide) return;
      const img = activeSlide.querySelector('#arch-img');
      if (!img) return;

      // Check if clicked card is already active
      const isCurrentlyActive = card.classList.contains('expanded');

      // Reset all cards' visual states
      const cardsContainer = document.getElementById('arch-cards');
      if (cardsContainer) {{
        cardsContainer.querySelectorAll('.card-v2').forEach(c => {{
          c.classList.remove('expanded');
          c.style.borderColor = 'var(--border)';
          c.style.boxShadow = 'none';
        }});
      }}

      if (!isCurrentlyActive) {{
        // Activate current card
        card.classList.add('expanded');
        card.style.borderColor = 'var(--accent-blue)';
        card.style.boxShadow = '0 0 16px rgba(88, 166, 255, 0.25)';

        // Apply specific zoom transforms based on target section of architecture diagram
        if (target === 'loop') {{
          img.style.transform = 'scale(1.7) translate(-5%, 8%)';
        }} else if (target === 'debugger') {{
          img.style.transform = 'scale(2.3) translate(-2%, -6%)';
        }} else if (target === 'families') {{
          img.style.transform = 'scale(2.1) translate(-2%, 5%)';
        }} else if (target === 'state') {{
          img.style.transform = 'scale(1.75) translate(0%, -24%)';
        }}
      }} else {{
        resetZoom();
      }}
    }}

    function resetZoom() {{
      const activeSlide = document.querySelector('.slide.active');
      if (!activeSlide) return;
      const img = activeSlide.querySelector('#arch-img');
      if (img) img.style.transform = 'scale(1) translate(0, 0)';
      
      const cardsContainer = activeSlide.querySelector('#arch-cards');
      if (cardsContainer) {{
        cardsContainer.querySelectorAll('.card-v2').forEach(c => {{
          c.classList.remove('expanded');
          c.style.borderColor = 'var(--border)';
          c.style.boxShadow = 'none';
        }});
      }}
    }}

    function zoomResults(trigger) {{
      const activeSlide = document.querySelector('.slide.active');
      if (!activeSlide) return;
      const img = activeSlide.querySelector('#results-img');
      if (!img) return;

      const isCurrentlyActive = trigger.classList.contains('highlighted-trigger');

      // Clear previous triggers highlight
      activeSlide.querySelectorAll('.zoom-trigger').forEach(t => {{
        t.classList.remove('highlighted-trigger');
        t.style.textDecoration = 'none';
      }});

      if (!isCurrentlyActive) {{
        trigger.classList.add('highlighted-trigger');
        trigger.style.textDecoration = 'underline';
        // Zoom in on the Best (Iter 10) bar graph on the right-top of the results image
        img.style.transform = 'scale(1.95) translate(-21%, 19%)';
      }} else {{
        resetResultsZoom();
      }}
    }}

    function resetResultsZoom() {{
      const activeSlide = document.querySelector('.slide.active');
      if (!activeSlide) return;
      const img = activeSlide.querySelector('#results-img');
      if (img) img.style.transform = 'scale(1) translate(0, 0)';
      
      activeSlide.querySelectorAll('.zoom-trigger').forEach(t => {{
        t.classList.remove('highlighted-trigger');
        t.style.textDecoration = 'none';
      }});
    }}
  </script>
</body>
</html>
"""

    # Build Artifact HTML
    html_artifact = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="generator" content="html-slides v0.9.4">
  <title>Autonomous ML Research Agent Presentation</title>
  <style>
    {full_css}
  </style>
</head>
<body>
  <!-- Particles background -->
  <div id="particles" class="particles"></div>
  
  <!-- Chrome elements -->
  <div class="branding">
    <div class="branding-icon">🤖</div>
    <span>TechJam Track 2 Presentation</span>
  </div>
  <div class="progress-bar" id="progress"></div>
  <div class="slide-counter" id="counter">1 / 6</div>
  <div class="nav-hints">
    Use <kbd>←</kbd> <kbd>→</kbd> or <kbd>Space</kbd> to navigate
  </div>
  <div class="slide-nav" id="slideNav"></div>

  <!-- Slide container -->
  <div class="deck" id="deck">
    {slide_0}
    {slide_1}
    {slide_2_artifact}
    {slide_3_artifact}
    {slide_4_workspace}
    {slide_5_artifact}
  </div>

  <script>
    {runtime_js}

    function revealBeat(event) {{
      event.stopPropagation(); // Prevent slide transition on click
      document.getElementById('beat-button').classList.add('hidden');
      document.getElementById('beat-result').classList.remove('hidden');
    }}

    function zoomDiagram(card, target) {{
      // Select the active slide's architecture image
      const activeSlide = document.querySelector('.slide.active');
      if (!activeSlide) return;
      const img = activeSlide.querySelector('#arch-img');
      if (!img) return;

      // Check if clicked card is already active
      const isCurrentlyActive = card.classList.contains('expanded');

      // Reset all cards' visual states
      const cardsContainer = document.getElementById('arch-cards');
      if (cardsContainer) {{
        cardsContainer.querySelectorAll('.card-v2').forEach(c => {{
          c.classList.remove('expanded');
          c.style.borderColor = 'var(--border)';
          c.style.boxShadow = 'none';
        }});
      }}

      if (!isCurrentlyActive) {{
        // Activate current card
        card.classList.add('expanded');
        card.style.borderColor = 'var(--accent-blue)';
        card.style.boxShadow = '0 0 16px rgba(88, 166, 255, 0.25)';

        // Apply specific zoom transforms based on target section of architecture diagram
        if (target === 'loop') {{
          img.style.transform = 'scale(1.7) translate(-5%, 8%)';
        }} else if (target === 'debugger') {{
          img.style.transform = 'scale(2.3) translate(-2%, -6%)';
        }} else if (target === 'families') {{
          img.style.transform = 'scale(2.1) translate(-2%, 5%)';
        }} else if (target === 'state') {{
          img.style.transform = 'scale(1.75) translate(0%, -24%)';
        }}
      }} else {{
        resetZoom();
      }}
    }}

    function resetZoom() {{
      const activeSlide = document.querySelector('.slide.active');
      if (!activeSlide) return;
      const img = activeSlide.querySelector('#arch-img');
      if (img) img.style.transform = 'scale(1) translate(0, 0)';
      
      const cardsContainer = activeSlide.querySelector('#arch-cards');
      if (cardsContainer) {{
        cardsContainer.querySelectorAll('.card-v2').forEach(c => {{
          c.classList.remove('expanded');
          c.style.borderColor = 'var(--border)';
          c.style.boxShadow = 'none';
        }});
      }}
    }}

    function zoomResults(trigger) {{
      const activeSlide = document.querySelector('.slide.active');
      if (!activeSlide) return;
      const img = activeSlide.querySelector('#results-img');
      if (!img) return;

      const isCurrentlyActive = trigger.classList.contains('highlighted-trigger');

      // Clear previous triggers highlight
      activeSlide.querySelectorAll('.zoom-trigger').forEach(t => {{
        t.classList.remove('highlighted-trigger');
        t.style.textDecoration = 'none';
      }});

      if (!isCurrentlyActive) {{
        trigger.classList.add('highlighted-trigger');
        trigger.style.textDecoration = 'underline';
        // Zoom in on the Best (Iter 10) bar graph on the right-top of the results image
        img.style.transform = 'scale(1.95) translate(-21%, 19%)';
      }} else {{
        resetResultsZoom();
      }}
    }}

    function resetResultsZoom() {{
      const activeSlide = document.querySelector('.slide.active');
      if (!activeSlide) return;
      const img = activeSlide.querySelector('#results-img');
      if (img) img.style.transform = 'scale(1) translate(0, 0)';
      
      activeSlide.querySelectorAll('.zoom-trigger').forEach(t => {{
        t.classList.remove('highlighted-trigger');
        t.style.textDecoration = 'none';
      }});
    }}
  </script>
</body>
</html>
"""

    # Write Workspace Slide Deck
    workspace_deck_path = os.path.join(html_slides_dir, "techjam-2min-deck.html")
    with open(workspace_deck_path, "w", encoding="utf-8") as f:
        f.write(html_workspace)
    print(f"Successfully generated workspace slide deck at: {workspace_deck_path}")

    # Write Artifact Slide Deck
    artifact_deck_path = os.path.join(artifact_dir, "techjam-2min-deck.html")
    with open(artifact_deck_path, "w", encoding="utf-8") as f:
        f.write(html_artifact)
    print(f"Successfully generated artifact slide deck at: {artifact_deck_path}")

if __name__ == "__main__":
    main()
