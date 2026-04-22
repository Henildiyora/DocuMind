#!/usr/bin/env bash
# DocuMind one-line installer.
#
# Usage:
#     curl -fsSL https://raw.githubusercontent.com/Henildiyora/DocuMind/main/install.sh | bash
#
# What it does:
#   1. Ensures `pipx` is available (auto-installs on macOS / Linux).
#   2. Installs DocuMind from GitHub into its own virtualenv via pipx.
#   3. Runs `pipx ensurepath` so the `documind` command is on PATH in new shells.
#   4. Prints next steps (how to run `documind setup`).
#
# Re-run safe: upgrades an existing install.

set -euo pipefail

REPO_URL="${DOCUMIND_REPO:-https://github.com/Henildiyora/DocuMind.git}"
BRANCH="${DOCUMIND_BRANCH:-main}"

log()  { printf "\033[1;34m[documind]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[documind]\033[0m %s\n" "$*" >&2; }
fail() { printf "\033[1;31m[documind]\033[0m %s\n" "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

detect_python() {
    for candidate in python3.12 python3.11 python3.10 python3; do
        if have "$candidate"; then
            PY="$candidate"
            return 0
        fi
    done
    fail "Python 3.10+ is required. Install Python and re-run this script."
}

ensure_pipx() {
    if have pipx; then
        return 0
    fi

    log "pipx not found, installing..."

    case "$(uname -s)" in
        Darwin)
            if have brew; then
                brew install pipx
            else
                "$PY" -m pip install --user --upgrade pipx
            fi
            ;;
        Linux)
            if have apt-get; then
                sudo apt-get update -y >/dev/null 2>&1 || true
                sudo apt-get install -y pipx >/dev/null 2>&1 \
                    || "$PY" -m pip install --user --upgrade pipx
            elif have dnf; then
                sudo dnf install -y pipx >/dev/null 2>&1 \
                    || "$PY" -m pip install --user --upgrade pipx
            else
                "$PY" -m pip install --user --upgrade pipx
            fi
            ;;
        *)
            "$PY" -m pip install --user --upgrade pipx
            ;;
    esac

    local user_base
    user_base="$("$PY" -m site --user-base 2>/dev/null || echo "$HOME/.local")"
    export PATH="$user_base/bin:$HOME/.local/bin:$PATH"

    if ! have pipx; then
        fail "pipx installation failed. Install it manually and re-run this script."
    fi
}

install_documind() {
    local spec="git+${REPO_URL}@${BRANCH}"
    log "Installing DocuMind from ${spec}"
    # `--force` handles both fresh install and upgrade paths in a single call
    # and works across all pipx versions (older pipx had `--spec`, newer removed it).
    pipx install --force "$spec"
}

ensure_on_path() {
    # Make the pipx apps dir visible to new shells (appends to ~/.zshrc,
    # ~/.bashrc, etc.) and also to this script's remaining session.
    pipx ensurepath >/dev/null 2>&1 || true
    export PATH="$HOME/.local/bin:$PATH"
}

print_next_steps() {
    local bin_dir ollama_hint
    bin_dir="$(pipx environment --value PIPX_BIN_DIR 2>/dev/null || echo "$HOME/.local/bin")"

    case "$(uname -s)" in
        Darwin) ollama_hint="brew install ollama   # or https://ollama.com/download" ;;
        Linux)  ollama_hint="curl -fsSL https://ollama.com/install.sh | sh" ;;
        *)      ollama_hint="See https://ollama.com/download" ;;
    esac

    printf "\n\033[1;32mDocuMind installed.\033[0m  100%% free, 100%% local. No API keys.\n"

    cat <<EOF

The 'documind' command has been installed to:
    ${bin_dir}

IMPORTANT: open a NEW terminal window (or run 'source ~/.zshrc' /
'source ~/.bashrc') so that 'documind' is on your PATH. You can also
apply it to this shell right now:

    export PATH="${bin_dir}:\$PATH"

Use it right away in ANY project (no model, no config required):

    cd ~/path/to/any/project
    documind index
    documind search "your query"

Optional (only if you want natural-language Q&A via 'documind ask' or
'documind chat'):

    # 1) Install Ollama if you don't have it:
    ${ollama_hint}

    # 2) Pick + pull a small local model (~1 GB by default):
    documind setup

DocuMind will auto-start the Ollama daemon for you, so you never need
a second terminal. Search and index work even if Ollama is not installed.

Uninstall later with:
    pipx uninstall documind

EOF
}

main() {
    detect_python
    ensure_pipx
    install_documind
    ensure_on_path
    print_next_steps
}

main "$@"
