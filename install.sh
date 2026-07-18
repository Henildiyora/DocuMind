#!/usr/bin/env bash
# DocuMind one-line installer (Go build).
#
# Usage:
#     curl -fsSL https://raw.githubusercontent.com/Henildiyora/DocuMind/main/install.sh | bash
#
# What it does:
#   1. Detects your OS/CPU (macOS or Linux; amd64 or arm64).
#   2. Downloads the matching prebuilt binary from the latest GitHub Release.
#   3. Verifies its SHA-256 checksum.
#   4. Installs it to /usr/local/bin (or ~/.local/bin if that is not writable).
#
# No Go and no C compiler required. Re-run safe (upgrades an existing install).
#
# Environment overrides:
#   DOCUMIND_VERSION=vX.Y.Z    install a specific release instead of the latest
#   DOCUMIND_INSTALL_DIR=/path install to a specific directory

set -euo pipefail

REPO="Henildiyora/DocuMind"

log()  { printf "\033[1;34m[documind]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[documind]\033[0m %s\n" "$*" >&2; }
fail() { printf "\033[1;31m[documind]\033[0m %s\n" "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# --- detect platform -------------------------------------------------------
detect_platform() {
    local os arch
    case "$(uname -s)" in
        Darwin) os="darwin" ;;
        Linux)  os="linux" ;;
        *) fail "Unsupported OS: $(uname -s). Build from source: https://github.com/${REPO}" ;;
    esac
    case "$(uname -m)" in
        x86_64|amd64)   arch="amd64" ;;
        arm64|aarch64)  arch="arm64" ;;
        *) fail "Unsupported CPU: $(uname -m). Build from source: https://github.com/${REPO}" ;;
    esac
    PLATFORM="${os}_${arch}"
}

# --- resolve the release tag ----------------------------------------------
resolve_version() {
    if [ -n "${DOCUMIND_VERSION:-}" ]; then
        VERSION="$DOCUMIND_VERSION"
        return
    fi
    # Ask the GitHub API for the latest release tag.
    local api="https://api.github.com/repos/${REPO}/releases/latest"
    local json
    if have curl; then
        json="$(curl -fsSL "$api")" || fail "Could not reach GitHub to find the latest release."
    elif have wget; then
        json="$(wget -qO- "$api")" || fail "Could not reach GitHub to find the latest release."
    else
        fail "Need curl or wget installed."
    fi
    VERSION="$(printf '%s' "$json" | grep -o '"tag_name"[[:space:]]*:[[:space:]]*"[^"]*"' | head -n1 | sed 's/.*"\([^"]*\)"$/\1/')"
    [ -n "$VERSION" ] || fail "No published release found yet. Ask the maintainer to run: git tag v0.1.0 && git push origin v0.1.0"
}

# --- download + verify -----------------------------------------------------
download() {
    local url="$1" out="$2"
    if have curl; then
        curl -fsSL "$url" -o "$out"
    else
        wget -qO "$out" "$url"
    fi
}

verify_checksum() {
    local file="$1" sumfile="$2"
    local expected actual
    expected="$(awk '{print $1}' "$sumfile")"
    if have shasum; then
        actual="$(shasum -a 256 "$file" | awk '{print $1}')"
    elif have sha256sum; then
        actual="$(sha256sum "$file" | awk '{print $1}')"
    else
        warn "No shasum/sha256sum available; skipping checksum verification."
        return 0
    fi
    [ "$expected" = "$actual" ] || fail "Checksum mismatch (expected $expected, got $actual). Aborting."
}

# --- pick an install dir (no forced sudo) ----------------------------------
choose_install_dir() {
    if [ -n "${DOCUMIND_INSTALL_DIR:-}" ]; then
        INSTALL_DIR="$DOCUMIND_INSTALL_DIR"
        mkdir -p "$INSTALL_DIR"
        return
    fi
    if [ -w "/usr/local/bin" ]; then
        INSTALL_DIR="/usr/local/bin"
    else
        INSTALL_DIR="$HOME/.local/bin"
        mkdir -p "$INSTALL_DIR"
    fi
}

path_hint() {
    case ":$PATH:" in
        *":$INSTALL_DIR:"*) : ;; # already on PATH
        *)
            warn "$INSTALL_DIR is not on your PATH. Add this to your shell profile:"
            printf '    export PATH="%s:$PATH"\n' "$INSTALL_DIR" >&2
            ;;
    esac
}

main() {
    detect_platform
    resolve_version
    log "Installing DocuMind ${VERSION} for ${PLATFORM}"

    local asset="documind_${PLATFORM}.tar.gz"
    local base="https://github.com/${REPO}/releases/download/${VERSION}"
    local tmp
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT

    log "Downloading ${asset} ..."
    download "${base}/${asset}" "${tmp}/${asset}" \
        || fail "No prebuilt binary for ${PLATFORM} in ${VERSION}. Build from source: https://github.com/${REPO}"
    if download "${base}/${asset}.sha256" "${tmp}/${asset}.sha256" 2>/dev/null; then
        verify_checksum "${tmp}/${asset}" "${tmp}/${asset}.sha256"
        log "Checksum OK."
    else
        warn "No checksum file published; skipping verification."
    fi

    tar -C "$tmp" -xzf "${tmp}/${asset}"
    [ -f "${tmp}/documind" ] || fail "Archive did not contain a 'documind' binary."
    chmod +x "${tmp}/documind"

    choose_install_dir
    log "Installing to ${INSTALL_DIR}/documind"
    mv -f "${tmp}/documind" "${INSTALL_DIR}/documind" 2>/dev/null \
        || fail "Could not write to ${INSTALL_DIR}. Re-run with DOCUMIND_INSTALL_DIR=~/.local/bin or use sudo."

    path_hint
    log "Done. Try it:"
    printf '    documind --help\n' >&2
    printf '    cd your-project && documind index && documind search "something"\n' >&2
    printf '    documind setup   # optional: pick a local model for ask/chat\n' >&2
}

main "$@"
