#!/bin/bash
# Visual test suite for html-slides
# Usage: bash eval/screenshot.sh <test-id|pro-all|vibe-all|all>

set -e

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$EVAL_DIR")"
OUTPUTS_DIR="$EVAL_DIR/outputs"
SCREENSHOTS_DIR="$EVAL_DIR/screenshots"
ANIMATION_DELAY=2  # seconds to wait after navigation for animations

screenshot_presentation() {
  local test_id="$1"
  local html_file="$2"
  local expected_slides="$3"

  local out_dir="$SCREENSHOTS_DIR/$test_id"
  mkdir -p "$out_dir"

  echo "📸 Screenshotting: $test_id ($expected_slides slides)"

  # Open the file
  agent-browser open "file://$html_file" 2>&1 | grep -v "^$" || true
  sleep 1

  # Screenshot first slide
  agent-browser screenshot "$out_dir/slide-1.png" 2>&1 | grep -v "^$" || true

  # Navigate and screenshot remaining slides
  for i in $(seq 2 "$expected_slides"); do
    agent-browser press ArrowRight 2>&1 > /dev/null || true
    sleep "$ANIMATION_DELAY"
    agent-browser screenshot "$out_dir/slide-${i}.png" 2>&1 | grep -v "^$" || true
  done

  echo "✅ $test_id: $expected_slides screenshots saved to $out_dir/"
}

generate_index() {
  local index_file="$SCREENSHOTS_DIR/index.html"

  cat > "$index_file" << 'HEADER'
<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<title>HTML Slides Visual Test Results</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, sans-serif; background: #0d1117; color: #e6edf3; padding: 24px; }
  h1 { font-size: 24px; margin-bottom: 24px; }
  h2 { font-size: 18px; margin: 32px 0 12px; color: #58a6ff; }
  .test-group { margin-bottom: 40px; }
  .slides { display: flex; gap: 8px; overflow-x: auto; padding: 8px 0; }
  .slides img { height: 200px; border: 1px solid rgba(255,255,255,0.1); border-radius: 4px; cursor: pointer; transition: transform 0.2s; }
  .slides img:hover { transform: scale(1.05); border-color: #58a6ff; }
  .slides img.selected { border-color: #3fb950; border-width: 2px; }
  .preview { margin-top: 16px; text-align: center; }
  .preview img { max-width: 100%; max-height: 70vh; border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; }
  .label { font-size: 12px; color: #8b949e; margin-top: 4px; }
</style>
</head><body>
<h1>HTML Slides Visual Test Results</h1>
<div id="preview" class="preview"></div>
HEADER

  # Find all test directories and generate sections
  for dir in "$SCREENSHOTS_DIR"/*/; do
    local test_id=$(basename "$dir")
    [ "$test_id" = "index.html" ] && continue

    local slide_count=$(ls "$dir"slide-*.png 2>/dev/null | wc -l | tr -d ' ')
    [ "$slide_count" -eq 0 ] && continue

    cat >> "$index_file" << EOF
<div class="test-group">
  <h2>$test_id ($slide_count slides)</h2>
  <div class="slides">
EOF

    for img in "$dir"slide-*.png; do
      local fname=$(basename "$img")
      local num="${fname#slide-}"
      num="${num%.png}"
      cat >> "$index_file" << EOF
    <img src="$test_id/$fname" alt="Slide $num" onclick="document.getElementById('preview').innerHTML='<img src=\\'$test_id/$fname\\'><div class=\\'label\\'>$test_id — Slide $num</div>'" />
EOF
    done

    echo "  </div></div>" >> "$index_file"
  done

  echo "</body></html>" >> "$index_file"
  echo "📄 Index generated: $index_file"
}

# Main
case "${1:-}" in
  pro-obsidian|pro-excalidraw-light|pro-excalidraw-dark|pro-editorial|pro-binary)
    html="$OUTPUTS_DIR/$1/presentation.html"
    [ -f "$html" ] || { echo "❌ Output not found: $html"; echo "Generate it first with the test agent."; exit 1; }
    screenshot_presentation "$1" "$html" 13
    generate_index
    ;;
  pro-charts)
    html="$OUTPUTS_DIR/pro-charts/presentation.html"
    [ -f "$html" ] || { echo "❌ Output not found: $html"; exit 1; }
    screenshot_presentation "pro-charts" "$html" 6
    generate_index
    ;;
  pro-mermaid)
    html="$OUTPUTS_DIR/pro-mermaid/presentation.html"
    [ -f "$html" ] || { echo "❌ Output not found: $html"; exit 1; }
    screenshot_presentation "pro-mermaid" "$html" 6
    generate_index
    ;;
  vibe-bold-signal|vibe-dark-botanical|vibe-swiss-modern|vibe-creative-voltage|vibe-notebook-tabs|vibe-vintage-editorial)
    html="$OUTPUTS_DIR/$1/presentation.html"
    [ -f "$html" ] || { echo "❌ Output not found: $html"; exit 1; }
    screenshot_presentation "$1" "$html" 6
    generate_index
    ;;
  vibe-excalidraw)
    html="$OUTPUTS_DIR/vibe-excalidraw/presentation.html"
    [ -f "$html" ] || { echo "❌ Output not found: $html"; exit 1; }
    screenshot_presentation "vibe-excalidraw" "$html" 8
    generate_index
    ;;
  pro-all)
    for theme in pro-obsidian pro-excalidraw-light pro-excalidraw-dark pro-editorial pro-binary; do
      html="$OUTPUTS_DIR/$theme/presentation.html"
      [ -f "$html" ] && screenshot_presentation "$theme" "$html" 13
    done
    for extra in pro-charts pro-mermaid; do
      html="$OUTPUTS_DIR/$extra/presentation.html"
      slides=6
      [ -f "$html" ] && screenshot_presentation "$extra" "$html" "$slides"
    done
    generate_index
    ;;
  vibe-all)
    for preset in vibe-bold-signal vibe-dark-botanical vibe-swiss-modern vibe-creative-voltage vibe-notebook-tabs vibe-vintage-editorial; do
      html="$OUTPUTS_DIR/$preset/presentation.html"
      [ -f "$html" ] && screenshot_presentation "$preset" "$html" 6
    done
    html="$OUTPUTS_DIR/vibe-excalidraw/presentation.html"
    [ -f "$html" ] && screenshot_presentation "vibe-excalidraw" "$html" 8
    generate_index
    ;;
  all)
    bash "$0" pro-all
    bash "$0" vibe-all
    ;;
  *)
    echo "Usage: bash eval/screenshot.sh <test-id|pro-all|vibe-all|all>"
    echo ""
    echo "Test IDs:"
    echo "  Pro:  pro-obsidian, pro-excalidraw-light, pro-excalidraw-dark, pro-editorial, pro-binary"
    echo "        pro-charts, pro-mermaid"
    echo "  Vibe: vibe-bold-signal, vibe-dark-botanical, vibe-swiss-modern, vibe-creative-voltage"
    echo "        vibe-notebook-tabs, vibe-vintage-editorial, vibe-excalidraw"
    echo "  All:  pro-all, vibe-all, all"
    exit 1
    ;;
esac
