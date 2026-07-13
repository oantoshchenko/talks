# Intro to RAG

A live, executable talk on **Retrieval-Augmented Generation (RAG)** — built as a
[marimo](https://marimo.io) notebook that runs entirely on your machine via
[Ollama](https://ollama.com). No API keys, no cloud; nothing leaves your laptop.

## What it's about

RAG means giving a language model the *right context at query time* so it answers from
your data instead of guessing. This notebook builds that idea up from scratch — every
chapter runs for real against local models, so you can type your own queries and watch
the results change:

- **RAG you already use** — the everyday version, named.
- **What RAG is** — ingestion → retrieval → augmentation, as hand-drawn diagrams.
- **Vector / semantic search** — what an embedding really *is*, made visual: type words and
  see where their meaning lands, then search a small corpus by meaning rather than keywords.
- **Making it work on real documents** — why naïve chunking loses context, and how
  **Contextual Retrieval** (an LLM writes a short situating blurb for each chunk before it's
  embedded) fixes it — shown side-by-side over a set of look-alike product specs.
- **What to explore next** — graph RAG, other modalities, and the trade-offs that decide
  which techniques you actually need.

## Run

```bash
./run.sh
```

That's the whole setup. `run.sh` checks for — and offers to install — everything it needs
([uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com), and the models), then opens
the notebook in your browser. Python dependencies are declared inside the notebook itself
(PEP 723); uv builds an isolated environment on first run.

**The first run downloads a few GB of models** (two small embedding models, plus one
generative model for the Contextual Retrieval chapter) — on slow wifi, give it a few minutes.

### Good to know

- **Not on a Mac?** The Contextual Retrieval chapter defaults to `gemma4:e4b-mlx`, an
  Apple-Silicon (MLX) build. On other hardware, type a different Ollama model — e.g.
  `llama3.2` — into the notebook's *"Context-writing model"* box.
- **That chapter is slow the first time** (an LLM writes context for each chunk), then caches
  its results to disk, so every reopen after that is instant.
- **Skip the download?** You can decline the generative model at the prompt — that one chapter
  shows a placeholder, and everything else still runs.
