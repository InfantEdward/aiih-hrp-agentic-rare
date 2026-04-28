# HW7: Phenotype-guided retrieval for rare-disease differential diagnosis

Code for the Spring 2026 *AI in Healthcare* high-risk project. The benchmark covers
18 rare diseases and 144 simulated patients, and compares four ranking
pipelines (heuristic, static LLM, text RAG, text + graph) plus an agentic
tool-selection extension built on LangGraph. There's also a Streamlit demo for
poking at single cases.


## Requirements

- Python 3.11 or 3.12.
- [uv](https://docs.astral.sh/uv/) for the environment. Install with
  `curl -LsSf https://astral.sh/uv/install.sh | sh` if you don't have it.
- An OpenAI API key. The retriever uses `text-embedding-3-small` and the
  reranker / agent use `gpt-4o-mini` by default.

## Setup

```bash
git clone https://github.com/InfantEdward/aiih-hrp-agentic-rare.git hw7
cd hw7
uv sync
```

### `.env` file

You need a `.env` at the repo root before anything else runs. The scripts
(`fetch_pubmed.py`, `build_chroma_index.py`, both evaluators) load it via
`python-dotenv`, so without it the OpenAI calls will fail.

Minimum contents:

```
# Required : used by the embedder, the reranker, and the agent.
OPENAI_API_KEY=sk-...
```

Optional variables, all with defaults : leave them unset unless you're tuning:
`OPENAI_CHAT_MODEL` (default `gpt-4o-mini`), `OPENAI_EMBEDDING_MODEL`
(default `text-embedding-3-small`), `AGENT_RETRIEVAL_K`, `AGENTIC_TOOL_BUDGET`,
`AGENT_EVAL_MAX_PATIENTS`, `PUBMED_RETMAX`, `CHROMA_COLLECTION`,
`CHROMA_RESET`. The full list lives at the top of
`scripts/run_agent_eval.py` and `scripts/run_agentic_eval.py`.

`.env` is read at import time, so if you change a value mid-session you'll
need to re-run the script (or restart the Streamlit server).

## Downloading the data

Four sources, in this order. The first three are static downloads pinned to
specific releases; the fourth depends on the locked disease list and runs
later.

### HPO (`hp.obo` + `phenotype.hpoa`)

```bash
mkdir -p data/{hpo,mondo,medquad,pubmed}

HPO_TAG="v2025-11-24"
HPO_BASE="https://github.com/obophenotype/human-phenotype-ontology/releases/download/${HPO_TAG}"
curl -L "${HPO_BASE}/hp.obo"        -o data/hpo/hp.obo
curl -L "${HPO_BASE}/phenotype.hpoa" -o data/hpo/phenotype.hpoa
```

### Mondo (`mondo.obo`)

```bash
MONDO_TAG="v2025-11-04"
MONDO_BASE="https://github.com/monarch-initiative/mondo/releases/download/${MONDO_TAG}"
curl -L "${MONDO_BASE}/mondo.obo" -o data/mondo/mondo.obo
```

### MedQuAD (HuggingFace)

```bash
uv run python -c "from datasets import load_dataset; \
  load_dataset('lavita/MedQuAD').save_to_disk('data/medquad')"
```


### PubMed abstracts

This one needs `outputs/locked_diseases.csv` to exist first, so it runs after
the disease ranking step in the next section.

```bash
uv run python scripts/fetch_pubmed.py
```

Defaults to 35 abstracts per disease and writes them to
`data/pubmed/abstracts.jsonl`.

## Building the artefacts

Each step writes into `outputs/` (or `data/chroma/` for the index). They
have to run in this order because each one feeds the next.

```bash
# 1. Rank candidate diseases and lock the 18-disease set.
uv run python scripts/rank_diseases.py

# 2. PubMed pull (now that the locked list exists).
uv run python scripts/fetch_pubmed.py

# 3. Slice HPOA annotations down to the locked diseases + summary stats.
uv run python scripts/build_locked_hpoa_subset.py

# 4. Build the HPO + HPOA + Mondo subgraph used for the graph condition.
uv run python -m src.graph.build_graph

# 5. Generate the 144 simulated patients (frequency-weighted, with noise).
uv run python -m src.patients.hpoa_simulator

# 6. Build the MedQuAD + PubMed retrieval corpus.
uv run python scripts/build_retrieval_corpus.py

# 7. Embed the corpus into a persistent Chroma index.
uv run python scripts/build_chroma_index.py
```

Step 7 makes one OpenAI embedding call per chunk. With the default settings
that's a few thousand calls, all batched, and finishes in a couple of minutes.

If you want to re-embed from a clean slate, set `CHROMA_RESET=true` before
running step 7.

## Running the experiments

Two evaluators. The first runs the three fixed-pipeline conditions, the second
runs the agentic extension.

```bash
# Fixed pipelines: static LLM, text RAG, text + graph.
uv run python scripts/run_agent_eval.py

# Agentic tool-selection extension (LangGraph).
uv run python scripts/run_agentic_eval.py
```

Pick a subset with `AGENT_EXPERIMENTS=rag` (or `static`, `rag_graph`, comma-
separated, or `all`). Cap to a handful of patients while iterating with
`AGENT_EVAL_MAX_PATIENTS=10`.

Each run drops three files into `outputs/`:

- `*_predictions.jsonl`: per-patient prediction with the evidence trace.
- `*_summary.csv`: flat per-patient summary.
- `*_metrics.json`: Top-1 / Top-3 / MRR plus tool-use stats for the agent.

## Running the demo

```bash
uv run streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`. You can:

- pick one of the 144 simulated patients,
- paste a free-text vignette and let the LLM extract HPO terms,
- or hand-pick HPO terms for a what-if case.

For each input you get the ranked differential under each condition, the
exact-match phenotypes, the retrieved snippets, and (for the agentic mode) the
sequence of tool calls. It's the same code path as the evaluators, so anything
you see in the demo matches what the metrics were computed on.

## Layout

```
app/              Streamlit demo.
data/             Raw downloads + Chroma index.
outputs/          Everything generated by the pipeline.
scripts/          One-shot pipeline scripts (run in the order above).
src/              Library code: graph builder, patient simulator, agent.
```