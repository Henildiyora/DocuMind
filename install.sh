#!/usr/bin/env bash
# DocuMind one-line installer.
#
# Usage:
#     curl -fsSL https://raw.githubusercontent.com/henildiyora7/DocuMind/main/install.sh | bash
#
# What it does:
#   1. Ensures `pipx` is available (auto-installs on macOS / Linux).
#   2. Installs DocuMind from GitHub into its own virtualenv via pipx.
#   3. Adds the pipx bin directory to PATH for this shell session.
#   4. Prints next steps (how to run `documind setup`).
#
# Re-run safe: upgrades an existing install.

set -euo pipefail

REPO_URL="${DOCUMIND_REPO:-https://github.com/Henildiyora/DocuMind.git}"
BRANCH="${DOCUMIND_BRANCH:-main}"

log()  { printf "\033[1;34m[documind]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[documind]\033[0m %s\n" "$*" >&2; }
fail() { printf "\033[1;31m[documind]\033[0m %s\n" "$*" >&2; exit 1; }

have()  { command -v "$1" >/dev/null 2>&1; }

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

    "$PY" -m pipx ensurepath >/dev/null 2>&1 || true

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

    if pipx list 2>/dev/null | grep -q "package documind"; then
        pipx upgrade documind --spec "$spec" >/dev/null \
            || pipx install --force "$spec"
    else
        pipx install "$spec"
    fi
}

print_next_steps() {
    local bin_dir
    bin_dir="$(pipx environment --value PIPX_BIN_DIR 2>/dev/null || echo "$HOME/.local/bin")"

    cat <<EOF

$(printf "\033[1;32mDocuMind installed.\033[0m")

The 'documind' command is now available globally.

Next steps:

  1. Make sure Ollama is installed and running:
$(
    case "$(uname -s)" in
        Darwin) echo "         brew install ollama   # or https://ollama.com/download" ;;
        Linux)  echo "         curl -fsSL https://ollama.com/install.sh | sh" ;;
        *)      echo "         See https://ollama.com/download" ;;
    esac
)
         ollama serve &

  2. Pick and pull the right model for your project (interactive):
         documind setup

  3. Use it from any project:
         cd ~/path/to/any/project
         documind index
         documind search "your query"
         documind ask "how does X work?"

If 'documind' isn't found, add this to your shell profile and open a new terminal:
    export PATH="$bin_dir:\$PATH"

Uninstall later with:
    pipx uninstall documind

EOF
}

main() {
    detect_python
    ensure_pipx
    install_documind
    print_next_steps
}

main "$@"
