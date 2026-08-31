#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILL_NAME="html-slides"

echo ""
echo "  HTML Slides Installer"
echo "  ====================="
echo "  Repo: $REPO_DIR"
echo ""

# --- Detect available agents ---
agents=()
command -v claude &>/dev/null && agents+=("claude")
command -v gemini &>/dev/null && agents+=("gemini")
command -v gh &>/dev/null     && agents+=("copilot")
command -v codex &>/dev/null  && agents+=("codex")

if [ ${#agents[@]} -eq 0 ]; then
  echo "  No supported agents detected."
  echo "  You can install manually — see README.md"
  exit 0
fi

echo "  Detected: ${agents[*]}"
echo ""
echo "  Choose install scope:"
echo ""
echo "    1) User-level  — available in all projects (recommended)"
echo "    2) Project-level — available only in current project"
echo "    3) Both"
echo ""
read -rp "  Enter choice [1]: " choice
choice="${choice:-1}"

installed=0

install_skill() {
  local target_dir="$1"
  local label="$2"
  local link="$target_dir/$SKILL_NAME"

  mkdir -p "$target_dir"
  if [ -L "$link" ]; then
    current="$(readlink "$link" 2>/dev/null || true)"
    if [ "$current" = "$REPO_DIR" ]; then
      echo "  ✓ $label — already installed"
    else
      rm -f "$link"
      ln -s "$REPO_DIR" "$link"
      echo "  ✓ $label — updated symlink"
    fi
  elif [ -e "$link" ]; then
    echo "  ✗ $label — path exists but is not a symlink, skipping"
  else
    ln -s "$REPO_DIR" "$link"
    echo "  ✓ $label — installed"
  fi
  installed=$((installed + 1))
}

install_claude_plugin() {
  if claude plugin marketplace list 2>/dev/null | grep -q "html-slides"; then
    echo "  ✓ Claude Code marketplace already added"
    claude plugin marketplace update html-slides 2>/dev/null || true
  else
    claude plugin marketplace add bluedusk/html-slides 2>/dev/null || true
    echo "  ✓ Claude Code marketplace added (bluedusk/html-slides)"
  fi
  claude plugin update "$SKILL_NAME@html-slides" 2>/dev/null \
    || claude plugin install "$SKILL_NAME@html-slides" 2>/dev/null \
    || echo "  Note: Run 'claude plugin install html-slides' manually if needed"
  echo "  ✓ Claude Code plugin installed"
  installed=$((installed + 1))
}

echo ""

# --- User-level installs ---
if [ "$choice" = "1" ] || [ "$choice" = "3" ]; then
  echo "  Installing user-level..."
  echo ""

  for agent in "${agents[@]}"; do
    case "$agent" in
      claude)
        # Claude Code uses plugin marketplace — no skill symlinks needed
        install_claude_plugin
        ;;
      gemini)
        install_skill "$HOME/.gemini/skills" "~/.gemini/skills (Gemini CLI)"
        ;;
      copilot)
        # Copilot has no user-level skills path
        ;;
      codex)
        install_skill "$HOME/.codex/skills" "~/.codex/skills (OpenAI Codex)"
        ;;
    esac
  done
  echo ""
fi

# --- Project-level installs ---
if [ "$choice" = "2" ] || [ "$choice" = "3" ]; then
  echo "  Installing project-level..."
  echo ""

  for agent in "${agents[@]}"; do
    case "$agent" in
      claude)
        # Claude Code uses plugin marketplace — no skill symlinks needed
        install_claude_plugin
        ;;
      gemini)
        install_skill ".gemini/skills" ".gemini/skills (Gemini CLI)"
        ;;
      copilot)
        install_skill ".github/skills" ".github/skills (GitHub Copilot)"
        ;;
      codex)
        install_skill ".codex/skills" ".codex/skills (OpenAI Codex)"
        ;;
    esac
  done
  echo ""
fi

echo "  Done. $installed path(s) configured."
echo "  Restart your agent to pick up the new skill."
echo ""
