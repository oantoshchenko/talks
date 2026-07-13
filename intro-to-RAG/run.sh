#!/usr/bin/env bash
# Run the talk. Verifies + provisions everything, then launches marimo.
# All Python deps live inside each notebook (PEP 723); uv builds an isolated env on first run.
# First cold run downloads a few GB (embedding models ~1 GB + the Chapter 4 contextualizer ~9 GB + notebook env).
# Unattended? set ASSUME_YES=1 to auto-accept every prompt (handy for a pre-talk warm-up).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

NOTEBOOK="${1:-intro-to-RAG.py}"
MODELS=(mxbai-embed-large nomic-embed-text)   # embeddings — required by every chapter
CONTEXT_MODEL="gemma4:e4b-mlx"                 # Chapter 4 contextualizer — optional, swappable in the notebook
MARIMO_VERSION="0.23.13"
OS="$(uname)"

say() { printf '\033[1;34m▸ %s\033[0m\n' "$*"; }
die() { printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }
ask() {
    local _a
    [[ "${ASSUME_YES:-}" == 1 ]] && return 0
    [[ -t 0 ]] || die "Need your OK for: $1 — run in a terminal, or set ASSUME_YES=1."
    read -rp "  $1 [y/N] " _a
    [[ "${_a:-}" =~ ^[Yy] ]]  # any y-prefixed answer (y, Y, yes, Yes, …)
}
model_size() { case "$1" in mxbai-embed-large) echo "~670MB";; nomic-embed-text) echo "~275MB";; gemma4:e4b-mlx) echo "~8.8GB";; *) echo "";; esac; }

[[ -f "$NOTEBOOK" ]] || die "No notebook '$NOTEBOOK' here. Available: $(ls -1 ./*.py 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"

# 1 · uv — the only prerequisite (it provisions Python and every dependency)
if ! command -v uv >/dev/null 2>&1; then
    say "uv is not installed."
    if [[ "$OS" == Darwin ]] && command -v brew >/dev/null 2>&1 && ask "Install uv via Homebrew?"; then
        brew install uv || die "brew install uv failed — https://docs.astral.sh/uv/"
    elif ask "Download and run the official uv installer from astral.sh?"; then
        command -v curl >/dev/null 2>&1 || die "curl is required to install uv — install curl, or install uv manually: https://docs.astral.sh/uv/"
        curl -LsSf https://astral.sh/uv/install.sh | sh || die "uv install failed — https://docs.astral.sh/uv/"
        [[ -f "$HOME/.local/bin/env" ]] && . "$HOME/.local/bin/env"
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"; hash -r
    else
        die "uv is required — https://docs.astral.sh/uv/"
    fi
    command -v uv >/dev/null 2>&1 || die "uv installed but not on PATH — restart your shell and re-run."
fi

# 2 · Ollama — CLI
if ! command -v ollama >/dev/null 2>&1; then
    say "Ollama is not installed."
    if [[ "$OS" == Darwin ]] && command -v brew >/dev/null 2>&1 && ask "Install Ollama via Homebrew?"; then
        brew install ollama || die "brew install ollama failed — https://ollama.com/download"
    elif [[ "$OS" == Linux ]] && ask "Download and run the official Ollama installer?"; then
        command -v curl >/dev/null 2>&1 || die "curl is required — or install Ollama manually: https://ollama.com/download"
        curl -fsSL https://ollama.com/install.sh | sh || die "Ollama install failed — https://ollama.com/download"
    else
        die "Install Ollama, then re-run — https://ollama.com/download"
    fi
    command -v ollama >/dev/null 2>&1 || die "Ollama installed but not on PATH — restart your shell and re-run."
fi

# 2b · Ollama server — prefer the macOS app; otherwise self-host (logged)
if ! ollama list >/dev/null 2>&1; then
    _log="${TMPDIR:-/tmp}/ollama-talk.log"
    if [[ "$OS" == Darwin && -d /Applications/Ollama.app ]]; then
        say "Starting the Ollama app…"; open -a Ollama
    else
        say "Starting the Ollama server (log: $_log)…"; nohup ollama serve >"$_log" 2>&1 &
    fi
    for _ in $(seq 1 30); do ollama list >/dev/null 2>&1 && break; sleep 1; done
    ollama list >/dev/null 2>&1 || die "Ollama server didn't come up on :11434 — Linux: run 'ollama serve' in another terminal; macOS: open the Ollama app. Then re-run."
fi

# 2c · Embedding models — required (every chapter turns text into vectors)
for _m in "${MODELS[@]}"; do
    if ! ollama show "$_m" >/dev/null 2>&1; then
        say "Embedding model '$_m' is missing."
        if ask "Pull $_m ($(model_size "$_m")) now?"; then
            ollama pull "$_m" || die "Pulling '$_m' failed (network?). Retry: ollama pull $_m"
        else
            die "The notebook needs '$_m'."
        fi
    fi
done

# 2d · Contextualizer LLM — Chapter 4 only (an LLM writes a situating blurb per chunk).
#      Optional: skip it and Ch4's contextual demo shows a placeholder; every other chapter runs.
#      The default is an Apple-Silicon (MLX) build — on other hardware, type another model
#      (e.g. llama3.2) into the notebook's "Context-writing model" box.
if ! ollama show "$CONTEXT_MODEL" >/dev/null 2>&1; then
    say "Chapter 4's contextualizer '$CONTEXT_MODEL' is missing."
    if ask "Pull $CONTEXT_MODEL ($(model_size "$CONTEXT_MODEL")) now?"; then
        ollama pull "$CONTEXT_MODEL" || say "Pull failed — Ch4 stays in placeholder mode until you pull it (or pick another model in the notebook)."
    else
        say "Skipping — Ch4's contextual demo shows a placeholder until '$CONTEXT_MODEL' is pulled (or you pick another model in the notebook)."
    fi
fi

# 3 · Launch — the kernel's deps come from the notebook's own PEP 723 block, isolated by uv
say "Launching $NOTEBOOK … (first run builds the notebook env; on slow wifi give it a few minutes)"
exec uvx "marimo@${MARIMO_VERSION}" edit --sandbox --watch "$NOTEBOOK"
