#!/usr/bin/env bash
# Regenerate the static HTML preview served on GitHub Pages.
#
# This RUNS the notebook once to bake in its outputs (plots, diagrams, results), so it needs
# Ollama running with the models pulled — exactly like ./run.sh. The result is a single
# self-contained HTML file written to ../docs/intro-to-rag/index.html; commit it and push, and
# GitHub Pages serves it at https://<user>.github.io/talks/intro-to-rag/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

NOTEBOOK="intro-to-RAG.py"
OUT="../docs/intro-to-rag/index.html"
MARIMO_VERSION="0.23.13"

say() { printf '\033[1;34m▸ %s\033[0m\n' "$*"; }
die() { printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

command -v uv >/dev/null 2>&1 || die "uv is required — run ./run.sh once, or see https://docs.astral.sh/uv/"
ollama list >/dev/null 2>&1 || die "Ollama must be running with the models pulled (the export runs the notebook). Start it via ./run.sh, then re-run."

mkdir -p "$(dirname "$OUT")"
say "Exporting $NOTEBOOK → $OUT  (running the notebook; needs Ollama)…"
uvx "marimo@${MARIMO_VERSION}" export html "$NOTEBOOK" -o "$OUT" --sandbox --force
say "Done. Commit $OUT and push — GitHub Pages will serve it under /talks/intro-to-rag/."
