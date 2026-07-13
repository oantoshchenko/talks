# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.23.13",
#     "ollama>=0.4.0",
#     "numpy>=1.26",
#     "pandas>=2.2",
#     "scikit-learn>=1.4",
#     "plotly>=5.20",
#     "langchain-text-splitters==1.1.2",
# ]
# ///

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium", app_title="Introduction to RAG")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import ollama
    import pandas as pd
    import plotly.express as px
    from sklearn.decomposition import PCA

    return PCA, np, ollama, pd, px


@app.cell(hide_code=True)
def _(np, ollama):
    EMBED_MODEL = "mxbai-embed-large"  # 1024-dim, local; separates short concepts more cleanly than nomic

    def embed(texts):
        vecs = np.array(ollama.embed(model=EMBED_MODEL, input=list(texts))["embeddings"], dtype=float)
        # normalize so cosine similarity == dot product
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    return (embed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Introduction to RAG
    *Retrieval-Augmented Generation — what it actually is.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # RAG you already use
    - Right now I'm talking to **Claude Code** — and it answers from my **memory**, my notes (**Obsidian**), my tickets (**Linear**).
    - That *is* RAG: **retrieve** the relevant bits → **feed them to the model** → get a **grounded** answer.
    - You already use it every day. The rest of this talk is *what it is* and *how it works*.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
    # What is RAG?
    Give the model the **right context at query time**, then let it generate.

    - **Ingestion** — parse sources (docs · APIs) → select → process → store & index
    - **Retrieval** — rewrite the query → (agentic) search → pull from the sources
    - **Augmentation** — filter / select findings → push into context → **LLM generates** → result

    ## Formal definition
    """),
        mo.image(str(mo.notebook_dir() / "assets" / "RAG-formal-definition.png"), width=460),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md("## Ingestion &mdash; build the knowledge base *(ahead of time)*"),
        mo.Html(
            """<svg viewBox="0 0 820 256" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" font-family="ui-sans-serif, system-ui, sans-serif">
      <defs>
    <marker id="ig" markerWidth="9" markerHeight="9" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#6366f1"/></marker>
    <marker id="igm" markerWidth="8" markerHeight="8" refX="5" refY="4" orient="auto"><path d="M0,0 L7,4 L0,8 Z" fill="currentColor"/></marker>
      </defs>
      <text x="90" y="22" text-anchor="middle" fill="currentColor" opacity="0.6" font-size="12">sources</text>
      <g font-size="13" fill="currentColor">
    <rect x="14" y="30" width="152" height="34" rx="8" fill="none" stroke="currentColor" stroke-opacity="0.4"/>
    <path d="M28 38 h12 l4 4 v14 h-16 z" fill="none" stroke="currentColor" stroke-opacity="0.55" stroke-width="1.2"/>
    <text x="56" y="52">PDF</text>
    <rect x="14" y="72" width="152" height="34" rx="8" fill="none" stroke="currentColor" stroke-opacity="0.4"/>
    <path d="M28 80 h12 l4 4 v14 h-16 z" fill="none" stroke="currentColor" stroke-opacity="0.55" stroke-width="1.2"/>
    <text x="56" y="94">DOC</text>
    <rect x="14" y="114" width="152" height="34" rx="8" fill="none" stroke="currentColor" stroke-opacity="0.4"/>
    <text x="26" y="136" font-family="ui-monospace, monospace" opacity="0.6">{ }</text>
    <text x="56" y="136">JSON</text>
    <rect x="14" y="156" width="152" height="34" rx="8" fill="none" stroke="currentColor" stroke-opacity="0.4"/>
    <rect x="26" y="164" width="18" height="16" rx="2" fill="none" stroke="currentColor" stroke-opacity="0.55" stroke-width="1.2"/>
    <circle cx="31" cy="169" r="2" fill="currentColor" opacity="0.55"/>
    <path d="M27 179 l6 -6 l4 4 l4 -3" fill="none" stroke="currentColor" stroke-opacity="0.55" stroke-width="1.2"/>
    <text x="56" y="178">Images</text>
      </g>
      <g stroke="#6366f1" stroke-width="1.7" fill="none">
    <path d="M166 47 C 190 47, 196 118, 214 128" marker-end="url(#ig)"/>
    <path d="M166 89 C 192 89, 198 124, 214 132" marker-end="url(#ig)"/>
    <path d="M166 131 L214 136" marker-end="url(#ig)"/>
    <path d="M166 173 C 192 173, 198 148, 214 140" marker-end="url(#ig)"/>
      </g>
      <rect x="216" y="108" width="86" height="54" rx="10" fill="none" stroke="#6366f1" stroke-width="1.5"/>
      <text x="259" y="132" text-anchor="middle" fill="currentColor" font-size="14" font-weight="600">Parse</text>
      <text x="259" y="150" text-anchor="middle" fill="currentColor" opacity="0.65" font-size="11">&#8594; text</text>
      <path d="M302 135 L330 135" stroke="#6366f1" stroke-width="1.7" fill="none" marker-end="url(#ig)"/>
      <rect x="332" y="96" width="150" height="92" rx="10" fill="none" stroke="#6366f1" stroke-width="1.5"/>
      <text x="407" y="114" text-anchor="middle" fill="currentColor" font-size="14" font-weight="600">Chunk</text>
      <rect x="344" y="124" width="34" height="52" rx="2" fill="none" stroke="currentColor" stroke-opacity="0.5"/>
      <line x1="344" y1="141" x2="378" y2="141" stroke="currentColor" stroke-opacity="0.5" stroke-dasharray="3 2"/>
      <line x1="344" y1="158" x2="378" y2="158" stroke="currentColor" stroke-opacity="0.5" stroke-dasharray="3 2"/>
      <path d="M384 148 l14 0" stroke="#6366f1" stroke-width="1.5" fill="none" marker-end="url(#ig)"/>
      <g fill="#6366f1" fill-opacity="0.14" stroke="#6366f1" stroke-opacity="0.7">
    <rect x="414" y="126" width="56" height="14" rx="3"/>
    <rect x="414" y="145" width="56" height="14" rx="3"/>
    <rect x="414" y="164" width="56" height="14" rx="3"/>
      </g>
      <path d="M420 96 C 440 58, 470 52, 500 50" stroke="#6366f1" stroke-width="1.7" fill="none" marker-end="url(#ig)"/>
      <text x="452" y="76" text-anchor="middle" fill="currentColor" opacity="0.7" font-size="11">chunks</text>
      <rect x="504" y="16" width="192" height="66" rx="12" fill="#6366f1" fill-opacity="0.05" stroke="#6366f1" stroke-width="1.6" stroke-dasharray="6 4"/>
      <rect x="520" y="34" width="26" height="26" rx="5" fill="#6366f1" fill-opacity="0.18" stroke="#6366f1"/>
      <text x="533" y="52" text-anchor="middle" fill="currentColor" font-size="12">&#8776;</text>
      <text x="622" y="44" text-anchor="middle" fill="currentColor" font-size="13" font-weight="600">Embedding model</text>
      <text x="622" y="62" text-anchor="middle" fill="#6366f1" font-size="11">external service</text>
      <path d="M654 82 C 676 114, 704 126, 726 138" stroke="#6366f1" stroke-width="1.7" fill="none" marker-end="url(#ig)"/>
      <text x="692" y="110" text-anchor="middle" fill="currentColor" opacity="0.75" font-size="11">vector = key</text>
      <path d="M472 170 C 560 212, 650 208, 712 186" stroke="currentColor" stroke-opacity="0.55" stroke-width="1.6" fill="none" marker-end="url(#igm)"/>
      <text x="586" y="228" text-anchor="middle" fill="currentColor" opacity="0.7" font-size="11">chunk = value</text>
      <path d="M710 148 v44 a44 12 0 0 0 88 0 v-44" fill="#6366f1" fill-opacity="0.1" stroke="#6366f1" stroke-width="1.7"/>
      <ellipse cx="754" cy="148" rx="44" ry="12" fill="#6366f1" fill-opacity="0.2" stroke="#6366f1" stroke-width="1.7"/>
      <g>
    <circle cx="724" cy="166" r="3" fill="#6366f1"/><line x1="729" y1="166" x2="744" y2="166" stroke="#6366f1" stroke-width="1.2"/><rect x="746" y="161" width="12" height="10" rx="2" fill="none" stroke="currentColor" stroke-opacity="0.6"/>
    <circle cx="724" cy="182" r="3" fill="#6366f1"/><line x1="729" y1="182" x2="744" y2="182" stroke="#6366f1" stroke-width="1.2"/><rect x="746" y="177" width="12" height="10" rx="2" fill="none" stroke="currentColor" stroke-opacity="0.6"/>
      </g>
      <text x="754" y="218" text-anchor="middle" fill="currentColor" font-size="13" font-weight="600">Knowledge base</text>
      <text x="754" y="234" text-anchor="middle" fill="currentColor" opacity="0.6" font-size="11">vector &#8594; chunk</text>
    </svg>"""
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md("## Retrieval &mdash; find the relevant bits *(at query time)*"),
        mo.Html(
            """<svg viewBox="0 0 820 320" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" font-family="ui-sans-serif, system-ui, sans-serif">
      <defs>
    <marker id="ga" markerWidth="9" markerHeight="9" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#6366f1"/></marker>
    <marker id="gw" markerWidth="9" markerHeight="9" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#e0813a"/></marker>
    <marker id="gm" markerWidth="8" markerHeight="8" refX="5" refY="4" orient="auto"><path d="M0,0 L7,4 L0,8 Z" fill="currentColor"/></marker>
      </defs>
      <rect x="12" y="28" width="164" height="56" rx="10" fill="none" stroke="currentColor" stroke-opacity="0.4"/>
      <path d="M26 44 h20 v12 h-14 l-6 6 z" fill="none" stroke="currentColor" stroke-opacity="0.5" stroke-width="1.2"/>
      <text x="96" y="52" text-anchor="middle" fill="currentColor" font-size="13">Conversation</text>
      <text x="96" y="70" text-anchor="middle" fill="currentColor" font-size="13">history</text>
      <rect x="12" y="130" width="164" height="56" rx="10" fill="none" stroke="#6366f1"/>
      <text x="96" y="153" text-anchor="middle" fill="currentColor" font-size="14" font-weight="600">User query</text>
      <text x="96" y="172" text-anchor="middle" fill="currentColor" opacity="0.7" font-size="12" font-style="italic">"what is it?"</text>
      <g stroke="#6366f1" stroke-width="1.7" fill="none">
    <path d="M176 60 C 196 60, 200 96, 220 104" marker-end="url(#ga)"/>
    <path d="M176 158 C 196 158, 200 124, 220 116" marker-end="url(#ga)"/>
      </g>
      <rect x="222" y="80" width="150" height="60" rx="10" fill="#6366f1" fill-opacity="0.06" stroke="#6366f1" stroke-width="1.6"/>
      <text x="297" y="105" text-anchor="middle" fill="currentColor" font-size="14" font-weight="600">Contextualize</text>
      <text x="297" y="124" text-anchor="middle" fill="currentColor" opacity="0.65" font-size="11">&#8594; self-contained query</text>
      <path d="M372 110 L402 110" stroke="#6366f1" stroke-width="1.7" fill="none" marker-end="url(#ga)"/>
      <path d="M404 82 h96 a10 10 0 0 1 10 10 v36 a10 10 0 0 1 -10 10 h-96 a10 10 0 0 1 -10 -10 v-36 a10 10 0 0 1 10 -10 z" fill="#6366f1" fill-opacity="0.1" stroke="#6366f1" stroke-width="1.8"/>
      <circle cx="428" cy="112" r="9" fill="none" stroke="#6366f1" stroke-width="1.5"/>
      <circle cx="425" cy="110" r="1.4" fill="#6366f1"/><circle cx="431" cy="110" r="1.4" fill="#6366f1"/>
      <text x="472" y="117" text-anchor="middle" fill="currentColor" font-size="15" font-weight="700">Agent</text>
      <path d="M512 100 C 552 92, 578 92, 600 96" stroke="#6366f1" stroke-width="1.7" fill="none" marker-end="url(#ga)"/>
      <text x="556" y="86" text-anchor="middle" fill="currentColor" opacity="0.75" font-size="11">retrieve</text>
      <path d="M600 132 C 578 138, 552 138, 514 132" stroke="currentColor" stroke-opacity="0.55" stroke-width="1.5" fill="none" marker-end="url(#gm)"/>
      <text x="556" y="152" text-anchor="middle" fill="currentColor" opacity="0.6" font-size="11">results</text>
      <path d="M470 78 C 470 40, 630 40, 636 80" stroke="#6366f1" stroke-width="1.5" stroke-dasharray="5 4" fill="none" marker-end="url(#ga)"/>
      <text x="552" y="34" text-anchor="middle" fill="currentColor" opacity="0.7" font-size="11">explore &#8594; retrieve more  (multi-step)</text>
      <path d="M602 96 v44 a40 12 0 0 0 80 0 v-44" fill="#6366f1" fill-opacity="0.1" stroke="#6366f1" stroke-width="1.7"/>
      <ellipse cx="642" cy="96" rx="40" ry="12" fill="#6366f1" fill-opacity="0.2" stroke="#6366f1" stroke-width="1.7"/>
      <text x="642" y="176" text-anchor="middle" fill="currentColor" font-size="13" font-weight="600">Knowledge base</text>
      <path d="M462 142 C 452 200, 470 220, 512 236" stroke="#e0813a" stroke-width="2" fill="none" marker-end="url(#gw)"/>
      <g stroke="#e0813a" fill="none" stroke-width="1.5">
    <path d="M528 214 h20 l6 6 v30 h-26 z"/>
    <path d="M536 224 h20 l6 6 v30 h-26 z"/>
    <path d="M544 234 h20 l6 6 v30 h-26 z"/>
      </g>
      <text x="560" y="298" text-anchor="middle" fill="#e0813a" font-size="13">relevant pieces</text>
    </svg>"""
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md("## Augmentation &mdash; assemble the context, then generate"),
        mo.Html(
            """<svg viewBox="0 0 780 296" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" font-family="ui-sans-serif, system-ui, sans-serif">
      <defs>
    <marker id="aA" markerWidth="9" markerHeight="9" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#6366f1"/></marker>
    <marker id="aW" markerWidth="9" markerHeight="9" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#e0813a"/></marker>
    <marker id="aI" markerWidth="9" markerHeight="9" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor"/></marker>
      </defs>
      <rect x="12" y="30" width="176" height="44" rx="9" fill="none" stroke="currentColor" stroke-opacity="0.4"/>
      <text x="100" y="57" text-anchor="middle" fill="currentColor" font-size="14">System instructions</text>
      <rect x="12" y="116" width="176" height="44" rx="9" fill="none" stroke="#6366f1"/>
      <text x="100" y="143" text-anchor="middle" fill="currentColor" font-size="14">User query</text>
      <rect x="12" y="202" width="176" height="44" rx="9" fill="none" stroke="#e0813a"/>
      <text x="100" y="229" text-anchor="middle" fill="#e0813a" font-size="14">Retrieved corpus</text>
      <text x="100" y="262" text-anchor="middle" fill="#e0813a" font-size="11">↑ what RAG adds</text>
      <path d="M192 52 L296 68" stroke="currentColor" stroke-opacity="0.5" stroke-width="1.6" fill="none" marker-end="url(#aI)"/>
      <path d="M192 138 L296 130" stroke="#6366f1" stroke-width="1.6" fill="none" marker-end="url(#aA)"/>
      <path d="M192 224 L296 190" stroke="#e0813a" stroke-width="2" fill="none" marker-end="url(#aW)"/>
      <rect x="300" y="34" width="176" height="200" rx="10" fill="#6366f1" fill-opacity="0.06" stroke="#6366f1" stroke-width="2"/>
      <text x="388" y="62" text-anchor="middle" fill="currentColor" font-size="16" font-weight="700">CONTEXT</text>
      <text x="388" y="80" text-anchor="middle" fill="currentColor" opacity="0.6" font-size="11">the prompt the model sees</text>
      <g stroke="currentColor" stroke-opacity="0.25" stroke-width="1.4">
    <line x1="320" y1="102" x2="456" y2="102"/><line x1="320" y1="120" x2="456" y2="120"/>
    <line x1="320" y1="138" x2="456" y2="138"/><line x1="320" y1="156" x2="440" y2="156"/>
    <line x1="320" y1="174" x2="456" y2="174"/><line x1="320" y1="192" x2="430" y2="192"/>
    <line x1="320" y1="210" x2="456" y2="210"/>
      </g>
      <path d="M476 134 L522 134" stroke="#6366f1" stroke-width="1.8" fill="none" marker-end="url(#aA)"/>
      <rect x="524" y="110" width="86" height="48" rx="9" fill="#6366f1" fill-opacity="0.12" stroke="#6366f1"/>
      <text x="567" y="139" text-anchor="middle" fill="currentColor" font-size="15" font-weight="700">LLM</text>
      <path d="M610 134 L650 134" stroke="#6366f1" stroke-width="1.8" fill="none" marker-end="url(#aA)"/>
      <rect x="652" y="113" width="112" height="42" rx="21" fill="none" stroke="currentColor" stroke-opacity="0.4"/>
      <text x="708" y="139" text-anchor="middle" fill="currentColor" font-size="14">Answer</text>
    </svg>"""
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # What *is* a vector (semantic) search?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The toy: axes *I* chose
    Three axes I can name. Similar fruits sit close. This is only the intuition.
    """)
    return


@app.cell(hide_code=True)
def _(px):
    _fruits = ["lemon", "lime", "orange", "apple", "banana", "watermelon"]
    # hand-picked, human-readable axes (0..1): sweetness, sourness, juiciness
    _sweet = [0.10, 0.10, 0.70, 0.65, 0.80, 0.75]
    _sour = [0.95, 0.90, 0.50, 0.35, 0.10, 0.10]
    _juicy = [0.60, 0.55, 0.90, 0.60, 0.30, 0.98]
    _fig = px.scatter_3d(x=_sweet, y=_sour, z=_juicy, text=_fruits)
    _fig.update_traces(marker_size=6, textposition="top center")
    _fig.update_layout(
        scene=dict(xaxis_title="sweetness", yaxis_title="sourness", zaxis_title="juiciness"),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Reality: axes the model chose, and won't explain
    One word → **1024 numbers**, signed, unlabeled. There is no "sweetness" axis, and a
    *different* model would produce different numbers for the same word.
    """)
    return


@app.cell
def _():
    words = [
        "apple", "orange", "banana", "lemon",          # fruit
        "car", "truck", "bicycle", "airplane",         # vehicle
        "guitar", "violin", "piano", "drum",           # instrument
        "democracy", "monarchy", "justice", "freedom",  # concept
    ]
    groups = ["fruit"] * 4 + ["vehicle"] * 4 + ["instrument"] * 4 + ["concept"] * 4
    return groups, words


@app.cell
def _(embed, words):
    E = embed(words)  # (16, 1024), unit vectors
    return (E,)


@app.cell(hide_code=True)
def _(E, mo, words):
    _peek = ", ".join(f"{x:+.3f}" for x in E[0][:8])
    mo.md(f"`{words[0]}` = a point in **{E.shape[1]}-D**. First 8 of its coordinates: `[{_peek}, …]`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Seeing it: a 1024-D *shadow*
    We can't picture 1024-D, so we flatten it onto three axes (PCA) and look at the *shadow*.
    Clusters are real; the axes and exact distances are not. Type any words (comma-separated)
    and watch where their meaning lands — same space, projected the same way.
    """)
    return


@app.cell
def _(E, PCA):
    pca = PCA(n_components=3).fit(E)
    var_kept = float(pca.explained_variance_ratio_.sum())
    return pca, var_kept


@app.cell
def _(mo):
    concept = mo.ui.text(value="strawberry", label="Embed word(s) — comma-separated", full_width=True)
    concept
    return (concept,)


@app.cell(hide_code=True)
def _(E, concept, embed, groups, pca, px, var_kept, words):
    _term = concept.value.strip() or "strawberry"
    _EP = pca.transform(E)
    _QP = pca.transform(embed([_term]))
    _x = list(_EP[:, 0]) + list(_QP[:, 0])
    _y = list(_EP[:, 1]) + list(_QP[:, 1])
    _z = list(_EP[:, 2]) + list(_QP[:, 2])
    _labels = list(words) + [_term[:15]]
    _color = list(groups) + ["★ your word"]
    _fig = px.scatter_3d(
        x=_x, y=_y, z=_z, text=_labels, color=_color,
        title=f"a lossy shadow — 3 axes keep {var_kept:.0%} of the variance",
    )
    _fig.update_traces(marker_size=6, textposition="top center")
    _fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Search: find by meaning, not keywords
    Type a question. We embed it and rank a small corpus by *nearness* — no shared words needed.
    The **★ query** lands next to its answers; the **red-ringed** points are the top-K we return.
    """)
    return


@app.cell
def _():
    corpus = [
        "Knead the dough and let it rise before baking.",
        "A pinch of salt sharpens almost any dish.",
        "Let the loaf cool on a rack before slicing.",
        "The telescope captured a distant spiral galaxy.",
        "Astronauts train for months before a launch.",
        "A black hole's gravity traps even light.",
        "Refactor the function to remove the nested loops.",
        "The unit test failed on a tricky edge case.",
        "Cache the results so you never recompute them.",
        "The fox hunts mice in the tall grass at dusk.",
        "Whales migrate thousands of miles every year.",
        "Bees tell each other where the flowers are.",
    ]
    corpus_topic = ["cooking"] * 3 + ["space"] * 3 + ["code"] * 3 + ["nature"] * 3
    return corpus, corpus_topic


@app.cell(hide_code=True)
def _(PCA, corpus, embed):
    CE = embed(corpus)  # corpus embeddings
    search_pca = PCA(n_components=2).fit(CE)  # 2-D map so "near" is obvious
    return CE, search_pca


@app.cell(hide_code=True)
def _(mo):
    query = mo.ui.text(value="how do I make bread?", label="Search query", full_width=True)
    top_k = mo.ui.slider(1, 6, value=3, label="Top-K results")
    return query, top_k


@app.cell(hide_code=True)
def _(mo, query, top_k):
    mo.vstack([query, top_k])
    return


@app.cell(hide_code=True)
def _(
    CE,
    corpus,
    corpus_topic,
    embed,
    mo,
    np,
    pd,
    px,
    query,
    search_pca,
    top_k,
):
    _q = embed([query.value or "how do I make bread?"])[0]
    _sims = CE @ _q
    _hits = list(np.argsort(-_sims)[: top_k.value])

    _lines = [
        f"{_r + 1}. **{_sims[_i]:.2f}**  ·  {corpus[_i]}  —  _{corpus_topic[_i]}_"
        for _r, _i in enumerate(_hits)
    ]

    _CP = search_pca.transform(CE)
    _qp = search_pca.transform(_q.reshape(1, -1))[0]
    _df = pd.DataFrame({
        "x": _CP[:, 0],
        "y": _CP[:, 1],
        "topic": corpus_topic,
        "text": [c[:20] + "…" for c in corpus],  # hover shows the first 20 chars
    })
    _fig = px.scatter(
        _df, x="x", y="y", color="topic",
        hover_name="text", hover_data={"x": False, "y": False},
    )
    _fig.update_traces(marker_size=11)
    _fig.add_scatter(
        x=_CP[_hits, 0], y=_CP[_hits, 1], mode="markers", name="top-K", hoverinfo="skip",
        marker=dict(size=20, color="rgba(0,0,0,0)", line=dict(width=2.5, color="#d62728")),
    )
    _fig.add_scatter(
        x=[_qp[0]], y=[_qp[1]], mode="markers+text", name="query", hoverinfo="skip",
        marker=dict(size=22, symbol="star", color="#d62728"),
        text=["query"], textposition="bottom center",
    )
    _fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=430, legend_title_text="")

    mo.vstack([mo.md("**Results — nearest by meaning:**\n\n" + "\n".join(_lines)), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The space belongs to the *model*, not the world
    Pick an embedding model and you're married to it: every document you store must be embedded
    with the **same** model, or the numbers don't line up. Switch models → re-embed everything.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, ollama):
    def _vec(model, text):
        v = np.array(ollama.embed(model=model, input=[text])["embeddings"][0], dtype=float)
        return v / np.linalg.norm(v)

    _models = ["mxbai-embed-large", "nomic-embed-text"]
    _rows = []
    for _m in _models:
        _v = _vec(_m, "apple")
        _ao = float(_v @ _vec(_m, "orange"))
        _ad = float(_v @ _vec(_m, "democracy"))
        _rows.append(f"| `{_m}` | {_v.shape[0]} | {_ao:.3f} | {_ad:.3f} |")

    _table = (
        "| model | dims | cos(apple, orange) | cos(apple, democracy) |\n"
        "|---|---|---|---|\n" + "\n".join(_rows)
    )
    _raw = "\n".join(
        f"- `{_m}` → `[{', '.join(f'{x:+.2f}' for x in _vec(_m, 'apple')[:10])}, …]`"
        for _m in _models
    )
    mo.md(
        _table
        + "\n\nSame word **`apple`**, first 10 numbers from each model — different length, "
        "different values, **not interchangeable**:\n\n" + _raw
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Making it work on real documents
    Real documents are long, so we cut them into **chunks**. But a chunk pulled out of its page
    loses the context that told you what it was about — and then search can't find it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The problem: a chunk forgets where it came from
    A spec sheet reads *"up to 34 hours of video playback."* **Which phone?** The sentence never
    says. Embed that chunk on its own and a search for **"Nimbus N1 battery life"** can rank a
    *different* phone's battery section above the right one — the answer is in the corpus, but
    unfindable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    import glob as _glob
    import pathlib as _pathlib
    from langchain_text_splitters import RecursiveCharacterTextSplitter as _Splitter

    CHUNK_SIZE, CHUNK_OVERLAP = 500, 80
    _dir = _pathlib.Path(mo.notebook_dir()) / "docs"
    _splitter = _Splitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    c4_docs = {}
    c4_chunks = []
    c4_origin = []
    for _p in sorted(_glob.glob(str(_dir / "*.md"))):
        _stem = _pathlib.Path(_p).stem
        _text = _pathlib.Path(_p).read_text()
        c4_docs[_stem] = _text
        for _c in _splitter.split_text(_text):
            c4_chunks.append(_c)
            c4_origin.append(_stem)
    return CHUNK_OVERLAP, CHUNK_SIZE, c4_chunks, c4_docs, c4_origin


@app.cell(hide_code=True)
def _(CHUNK_OVERLAP, CHUNK_SIZE, c4_chunks, c4_docs, mo):
    _example = c4_chunks[0].strip()
    _full = {f"📄 {_k}": mo.md(_v) for _k, _v in c4_docs.items()}
    mo.vstack([
        mo.md(f"**{len(c4_chunks)} chunks** from **{len(c4_docs)} documents** "
              f"(size ≈ {CHUNK_SIZE} chars, overlap {CHUNK_OVERLAP}). One chunk looks like:"),
        mo.Html(f'<div style="padding:8px 10px;border:1px solid #6366f1;border-radius:6px;opacity:.9">{_example}</div>'),
        mo.md("*The full documents — click to expand:*"),
        mo.accordion(_full),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multiple ways to ingest the *same* chunks
    **Naïve** — embed each chunk exactly as in the last chapter; nothing added.

    **Contextual Retrieval** *([Anthropic, 2024](https://www.anthropic.com/engineering/contextual-retrieval))* — first an LLM writes a summary that *situates the chunk inside its document*, and we **prepend it before embedding**. The
    embedding model is untouched — **we fix the input**.

    > Prompt (the whole document + the chunk): *"…give a short succinct context to situate this
    > chunk within the overall document for the purposes of improving search retrieval of the
    > chunk. Answer only with the succinct context and nothing else."*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md("## In one picture"),
        mo.Html(
            """<svg viewBox="0 0 840 300" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" font-family="ui-sans-serif, system-ui, sans-serif" role="img">
      <title>Naïve vs Contextual ingestion</title>
      <desc>Naïve embeds the chunk alone; Contextual has an LLM write a situating blurb, prepends it to the chunk, then embeds that — same embedding model.</desc>
      <defs>
    <marker id="ci" markerWidth="9" markerHeight="9" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#6366f1"/></marker>
    <marker id="ca" markerWidth="9" markerHeight="9" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#e0813a"/></marker>
    <marker id="cn" markerWidth="8" markerHeight="8" refX="5" refY="4" orient="auto"><path d="M0,0 L7,4 L0,8 Z" fill="currentColor"/></marker>
      </defs>
      <text x="16" y="60" fill="currentColor" opacity="0.55" font-size="12" font-weight="600">NAÏVE</text>
      <text x="16" y="212" fill="#e0813a" font-size="12" font-weight="600">CONTEXTUAL</text>
      <rect x="72" y="48" width="132" height="48" rx="9" fill="none" stroke="#6366f1" stroke-width="1.5"/>
      <text x="138" y="68" text-anchor="middle" fill="currentColor" font-size="13" font-weight="600">chunk</text>
      <text x="138" y="85" text-anchor="middle" fill="currentColor" opacity="0.6" font-size="10.5" font-style="italic">"…34h video…"</text>
      <path d="M204 72 L548 72" stroke="#6366f1" stroke-width="1.7" fill="none" marker-end="url(#ci)"/>
      <text x="374" y="63" text-anchor="middle" fill="currentColor" opacity="0.5" font-size="10.5">just the chunk</text>
      <rect x="550" y="44" width="150" height="56" rx="10" fill="#6366f1" fill-opacity="0.05" stroke="#6366f1" stroke-width="1.6" stroke-dasharray="6 4"/>
      <text x="625" y="68" text-anchor="middle" fill="currentColor" font-size="13" font-weight="600">Embedding model</text>
      <text x="625" y="85" text-anchor="middle" fill="#6366f1" font-size="10.5">external · unchanged</text>
      <path d="M700 72 L740 72" stroke="#6366f1" stroke-width="1.7" fill="none" marker-end="url(#ci)"/>
      <circle cx="760" cy="72" r="4" fill="#6366f1"/>
      <text x="774" y="60" fill="currentColor" opacity="0.55" font-size="10">which phone?</text>
      <text x="780" y="90" fill="currentColor" opacity="0.55" font-size="10">— lost</text>
      <rect x="24" y="178" width="40" height="52" rx="4" fill="none" stroke="currentColor" stroke-opacity="0.55" stroke-width="1.3"/>
      <line x1="32" y1="190" x2="56" y2="190" stroke="currentColor" stroke-opacity="0.5"/>
      <line x1="32" y1="200" x2="56" y2="200" stroke="currentColor" stroke-opacity="0.5"/>
      <line x1="32" y1="210" x2="56" y2="210" stroke="currentColor" stroke-opacity="0.5"/>
      <line x1="32" y1="220" x2="48" y2="220" stroke="currentColor" stroke-opacity="0.5"/>
      <text x="44" y="246" text-anchor="middle" fill="currentColor" opacity="0.6" font-size="10.5">whole doc</text>
      <rect x="82" y="188" width="72" height="34" rx="7" fill="none" stroke="#6366f1" stroke-width="1.4"/>
      <text x="118" y="209" text-anchor="middle" fill="currentColor" font-size="11.5" font-weight="600">chunk</text>
      <path d="M64 202 C 150 202, 158 205, 214 208" stroke="currentColor" stroke-opacity="0.55" stroke-width="1.4" fill="none" marker-end="url(#cn)"/>
      <path d="M154 205 L214 207" stroke="#6366f1" stroke-width="1.5" fill="none" marker-end="url(#ci)"/>
      <rect x="216" y="182" width="104" height="52" rx="10" fill="#6366f1" fill-opacity="0.1" stroke="#6366f1" stroke-width="1.6"/>
      <text x="268" y="205" text-anchor="middle" fill="currentColor" font-size="13" font-weight="700">LLM</text>
      <text x="268" y="222" text-anchor="middle" fill="currentColor" opacity="0.65" font-size="10">writes context</text>
      <path d="M320 208 L360 208" stroke="#e0813a" stroke-width="1.8" fill="none" marker-end="url(#ca)"/>
      <rect x="362" y="176" width="150" height="64" rx="10" fill="#e0813a" fill-opacity="0.08" stroke="#e0813a" stroke-width="1.6"/>
      <text x="437" y="197" text-anchor="middle" fill="#e0813a" font-size="11" font-weight="600">context (50–100 tok)</text>
      <line x1="376" y1="205" x2="498" y2="205" stroke="#e0813a" stroke-opacity="0.4"/>
      <text x="437" y="224" text-anchor="middle" fill="currentColor" opacity="0.8" font-size="12">⊕ chunk</text>
      <path d="M512 208 L548 208" stroke="#e0813a" stroke-width="1.8" fill="none" marker-end="url(#ca)"/>
      <rect x="550" y="180" width="150" height="56" rx="10" fill="#6366f1" fill-opacity="0.05" stroke="#6366f1" stroke-width="1.6" stroke-dasharray="6 4"/>
      <text x="625" y="204" text-anchor="middle" fill="currentColor" font-size="13" font-weight="600">Embedding model</text>
      <text x="625" y="221" text-anchor="middle" fill="#6366f1" font-size="10.5">external · unchanged</text>
      <path d="M700 208 L740 208" stroke="#e0813a" stroke-width="1.8" fill="none" marker-end="url(#ca)"/>
      <circle cx="760" cy="208" r="4" fill="#e0813a"/>
      <text x="774" y="205" fill="#e0813a" font-size="10">carries the</text>
      <text x="777" y="218" fill="#e0813a" font-size="10">context</text>
      <path d="M712 100 C 728 100, 728 180, 712 180" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.2" fill="none"/>
      <text x="735" y="144" fill="currentColor" opacity="0.6" font-size="10.5">same model</text>
      <text x="410" y="284" text-anchor="middle" fill="currentColor" opacity="0.75" font-size="12">We didn't touch the embedding model — we changed the <tspan fill="#e0813a" font-weight="600">input</tspan>.</text>
    </svg>"""
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    c4_model = mo.ui.text(value="gemma4:e4b-mlx", label="Context-writing model (Ollama)", full_width=True)
    c4_model
    return (c4_model,)


@app.cell(hide_code=True)
def _(c4_chunks, embed):
    c4_naiveE = embed(c4_chunks)
    return (c4_naiveE,)


@app.cell(hide_code=True)
def _(c4_chunks, c4_docs, c4_model, c4_origin, mo, ollama):
    import hashlib as _hashlib
    import json as _json
    import pathlib as _pathlib

    _cache_path = _pathlib.Path(mo.notebook_dir()) / "__marimo__" / "ctx_cache.json"
    _cache_path.parent.mkdir(parents=True, exist_ok=True)
    _cache = _json.loads(_cache_path.read_text()) if _cache_path.exists() else {}

    def _key(model, doc, chunk):
        return _hashlib.sha1(f"{model}\x1f{doc}\x1f{chunk}".encode()).hexdigest()

    def _blurb(model, doc, chunk):
        _k = _key(model, doc, chunk)
        if _k in _cache:
            return _cache[_k]
        _p = (f"<document>\n{doc}\n</document>\n"
              "Here is the chunk we want to situate within the whole document\n"
              f"<chunk>\n{chunk}\n</chunk>\n"
              "Please give a short succinct context to situate this chunk within the overall "
              "document for the purposes of improving search retrieval of the chunk. Answer only "
              "with the succinct context and nothing else.")
        _out = ollama.generate(model=model, prompt=_p, options={"temperature": 0})["response"].strip()
        _cache[_k] = _out
        return _out

    _model = c4_model.value.strip() or "gemma4:e4b-mlx"
    try:
        c4_blurbs = [_blurb(_model, c4_docs[_o], _ch) for _ch, _o in zip(c4_chunks, c4_origin)]
        _cache_path.write_text(_json.dumps(_cache))
        c4_ctx_chunks = [f"{_b}\n\n{_ch}" for _b, _ch in zip(c4_blurbs, c4_chunks)]
        c4_ctx_error = None
    except Exception as _e:
        c4_blurbs = None
        c4_ctx_chunks = None
        c4_ctx_error = str(_e)
    return c4_blurbs, c4_ctx_chunks, c4_ctx_error


@app.cell(hide_code=True)
def _(c4_ctx_chunks, embed):
    c4_ctxE = embed(c4_ctx_chunks) if c4_ctx_chunks else None
    return (c4_ctxE,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Same query, two indexes — watch it move
    One search box. Left ranks the **naïve** chunks, right ranks the **contextualized** ones — the
    *same* chunks, embedded differently. Colour = which phone the chunk came from.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    c4_query = mo.ui.text(value="Does the Nimbus N1 have a telephoto camera?", label="Search query", full_width=True)
    c4_topk = mo.ui.slider(1, 5, value=3, label="Top-K")
    mo.vstack([c4_query, c4_topk])
    return c4_query, c4_topk


@app.cell(hide_code=True)
def _(
    c4_blurbs,
    c4_chunks,
    c4_ctxE,
    c4_ctx_error,
    c4_naiveE,
    c4_origin,
    c4_query,
    c4_topk,
    embed,
    mo,
    np,
):
    _q = c4_query.value.strip() or "Does the Nimbus N1 have a telephoto camera?"
    _qv = embed([_q])[0]

    _names = sorted(set(c4_origin))
    _palette = ["#6366f1", "#e0813a", "#14b8a6", "#a855f7", "#ec4899"]
    _color = {_n: _palette[_i % len(_palette)] for _i, _n in enumerate(_names)}

    def _pretty(_stem):
        return _stem.replace("-", " ").title()

    def _clean(_text):
        return " ".join(_text.replace("#", " ").split())

    def _clamp2(_inner, _extra):
        return (f'<div style="{_extra}display:-webkit-box;-webkit-line-clamp:2;'
                f'-webkit-box-orient:vertical;overflow:hidden">{_inner}</div>')

    def _rows(_E, _blurbs):
        _sims = _E @ _qv
        _idx = list(np.argsort(-_sims)[: c4_topk.value])
        _html = []
        for _r, _i in enumerate(_idx):
            _n = c4_origin[_i]
            _ctx = ""
            if _blurbs is not None:
                _ctx = _clamp2("+ context: " + _clean(_blurbs[_i]),
                               "color:#e0813a;font-size:.8em;margin:3px 0;")
            _html.append(
                f'<div style="margin:6px 0;padding:8px 10px;border:1px solid {_color[_n]}44;'
                f'border-left:3px solid {_color[_n]};border-radius:6px">'
                f'<span style="opacity:.5">{_r + 1}.</span> '
                f'<b>{_sims[_i]:.2f}</b> &nbsp;'
                f'<span style="color:{_color[_n]};font-weight:600">{_pretty(_n)}</span>'
                f'{_ctx}'
                + _clamp2(_clean(c4_chunks[_i]), "opacity:.75;font-size:.9em;margin-top:2px;")
                + "</div>"
            )
        return mo.Html("".join(_html))

    _left = mo.vstack([mo.md("### Naïve chunks"), _rows(c4_naiveE, None)])
    if c4_ctx_error:
        _right = mo.vstack([mo.md("### Contextualized chunks"),
                            mo.callout(mo.md("Contextual index unavailable — see the model box above."), kind="warn")])
    else:
        _right = mo.vstack([mo.md("### Contextualized chunks"), _rows(c4_ctxE, c4_blurbs)])
    mo.vstack([
        mo.hstack([_left, _right], widths="equal", gap=1.5),
        mo.md("*Score = cosine similarity (1.0 = identical direction). The right column shows the "
              "LLM-added context in amber — the blurb **and** the chunk together are what we embed.*"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Semantic search alone isn't the whole answer
    Contextual **Embeddings** are only step one. Real systems stack more:

    - **Contextual BM25** — keyword / exact-match retrieval (catches `error code TS-999` that
      embeddings gloss over), using the *same* prepended context.
    - **Reranking** — vector search is fast but rough, so it grabs a wide net of candidates (say
      the top 150). A slower, more accurate **cross-encoder** then re-reads each candidate
      *together with the query* — not as separate vectors — and keeps only the best handful to
      hand to the LLM.

    Anthropic's measured drop in retrieval failures: **−35%** (contextual embeddings) →
    **−49%** (+ contextual BM25) → **−67%** (+ reranking). *We fixed the **input**, not the model.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Final words — what to explore next
    Everything so far — semantic search, then Contextual Retrieval — is **one slice**. Two
    directions worth clicking into:

    ### Retrieval beyond chunks → **graph RAG**
    **[LightRAG](https://arxiv.org/abs/2410.05779)** · [code](https://github.com/HKUDS/LightRAG) —
    an LLM builds a **knowledge graph** (entities + how they relate) over the corpus, so retrieval
    reasons over how things *connect*, not just which chunks look alike. Two levels: precise entity
    lookups **and** broad themes. *(Its own line of work — not Microsoft's GraphRAG.)*

    ### Retrieval beyond text → **other modalities**
    **[VideoRAG](https://arxiv.org/abs/2501.05874)** · [code](https://github.com/starsuzi/VideoRAG) —
    RAG over a **video** corpus: retrieve the relevant clips and feed the model both the **visual**
    frames *and* the text. Every modality needs its **own ingestion**. Same problem, new medium.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## …and it depends
    We only scratched the surface. There's a whole menu we skipped — hybrid (keyword + vector)
    search, reranking, agentic / multi-step retrieval, multimodal embeddings, graph RAG.

    Which of these you actually need is a **trade-off**: use case, requirements, scale, cost,
    latency, freshness, data types. **Start simple; add only what your evals tell you to.**
    """)
    return


if __name__ == "__main__":
    app.run()
