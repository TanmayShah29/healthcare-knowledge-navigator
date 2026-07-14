# Healthcare Knowledge Navigator

A medical RAG assistant for healthcare professionals: retrieves and synthesizes
evidence-based answers from clinical guidelines, research papers, and treatment
protocols, with inline **citations** back to source documents and a **confidence
score** reflecting how well-supported the answer is.

**Works with any AI API.** Same provider-agnostic `src/llm.py` pattern as
[`../multi-agent-code-review`](../multi-agent-code-review) and
[`../graphrag-knowledge-assistant`](../graphrag-knowledge-assistant) — swap providers
via `.env`.

> **This is a decision-support tool, not a clinical authority.** Every answer ends with
> a reminder to verify against current full-text sources and apply clinical judgment for
> the individual patient. Sample data shipped in `samples/` is fictional and for
> pipeline-testing only — see the disclaimer at the top of each sample file.

## Why hybrid retrieval (graph + vector), not just one or the other?

Clinical questions come in two shapes that plain vector search or a pure knowledge
graph each handle poorly alone:

- **"Is drug X contraindicated with condition Y?"** — often stated as a direct fact in
  one sentence somewhere. Vector similarity search finds this well, but doesn't chain
  facts together.
- **"What's a good second-line option for a T2DM patient with heart failure?"** — this
  requires connecting *treats*, *contraindicated with*, and *reduces risk of* facts that
  may live in different documents. That's what graph traversal is for.

So retrieval runs **both** in parallel — vector similarity search over embedded chunks
(catches nuanced passages, dosing caveats, anything that doesn't reduce cleanly to a
triple), and multi-hop traversal over an entity graph (catches multi-document reasoning
chains) — and the Synthesizer agent combines both into one cited answer.

## Architecture

```
INGESTION (per document — clinical_guideline | research_paper | treatment_protocol):
  document text/pdf ─▶ section-aware chunking ─▶ embed each chunk ─▶ Neo4j Chunk node
                                              └─▶ Extractor Agent ─▶ triples + dosages + contraindications ─▶ Neo4j graph (MERGE, idempotent)

QUERY (hybrid):
  question ─▶ embed question ─▶ Neo4j vector index search ─▶ vector_hits (passages)
            ─▶ Entity Linker Agent ─▶ candidate names ─▶ fuzzy-match ─▶ seed entities
                                                        ─▶ multi-hop Cypher traversal ─▶ subgraph (facts)
                                                        ─▶ dosage/contraindication lookup
            ─▶ Synthesizer Agent (vector_hits + subgraph + dosages + contras) ─▶ cited answer, grounded flag
            ─▶ Confidence Scorer (similarity + source agreement + graph corroboration + recency) ─▶ score + label
```

Schema (entities and chunks are separate node families, linked implicitly via shared
`source_doc` rather than a direct relationship — a chunk can mention several entities,
an entity can appear in several chunks):

```
(:Entity {name, type})-[:RELATES_TO {predicate, source_doc, doc_type}]->(:Entity)
(:Entity {name})-[:HAS_DOSAGE {dose, frequency, route, notes, source_doc, doc_type}]->(:Entity)
(:Entity {name})-[:CONTRAINDICATED_FOR {reason, severity, source_doc, doc_type}]->(:Entity)
(:Chunk {chunk_id, text, source_doc, doc_type, section, embedding, publication_date})   -- vector-indexed
```

## Agents

- **Extractor** — clinical text chunk → JSON with `{triples, dosages, contraindications}`.
  Triples are `(subject, predicate, object)` facts (entity types: Condition, Drug, Symptom,
  Treatment, Procedure, Guideline, RiskFactor). Dosages are structured `(drug, dose,
  frequency, route)` facts. Contraindications are structured `(drug, condition, reason,
  severity)` facts. All extracted in a single LLM call. Explicitly instructed not to infer
  dosages, indications, or contraindications beyond what the text states.
- **Entity Linker** — question → candidate clinical entity names, expanding common
  medical abbreviations (T2DM, MI, ...) so either form matches the graph.
- **Synthesizer** — question + retrieved evidence (facts + passages + dosages +
  contraindications) → answer with inline `[Source, doc_type]` citations after each
  claim. Emits an internal "insufficient evidence" flag (used by the confidence scorer)
  rather than silently filling gaps from general medical knowledge.

Multi-hop traversal and vector similarity search themselves (`src/graph_db.py`) are
plain Cypher — no LLM involved, keeping retrieval fast and deterministic. Vector search
prefers the Cypher `SEARCH` clause (Neo4j 2026.01+) and falls back automatically to the
`db.index.vector.queryNodes` procedure on older versions, so this repo works across
Neo4j versions without a manual migration when the procedure is eventually removed.
Confidence scoring (`src/confidence.py`) is likewise deterministic, not an LLM call —
see that file's docstring for the exact blend of signals.

## Setup

**1. Get a Neo4j instance with vector index support (5.11+).** Either:
- [Neo4j Desktop](https://neo4j.com/download/) (local, free, current versions qualify), or
- [Neo4j Aura Free](https://neo4j.com/cloud/aura-free/) (hosted, free tier, no local install)

**2. Install dependencies and configure:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in NEO4J_*, your LLM provider/key, and EMBEDDING_DIMENSIONS
python -m src.main check   # verify Neo4j connection + vector index + LLM config
```

`EMBEDDING_DIMENSIONS` must match your embedding model's actual output size (the
`.env.example` default, 768, matches the default `EMBEDDING_MODEL` — Gemini
`text-embedding-004`). Get this wrong and the vector index creation will fail or
similarity search will return garbage — check the table in `.env.example` if you switch
embedding providers.

## Run

```bash
# Ingest the sample documents (two sources, so hybrid retrieval + source
# agreement in the confidence score both have something to demonstrate)
python -m src.main ingest samples/t2dm_management_guideline.txt clinical_guideline --date 2024-01-01
python -m src.main ingest samples/cv_risk_t2dm_research_summary.txt research_paper --date 2023-06-15

# PDF ingestion is supported natively
python -m src.main ingest path/to/guideline.pdf clinical_guideline --date 2024-06-01

# A question answerable from one document (high vector similarity, one source)
python -m src.main query "What is first-line therapy for Type 2 Diabetes Mellitus?"

# A question that needs both documents — proves hybrid retrieval + source
# agreement are doing real work
python -m src.main query "What second-line diabetes drug should I consider for a patient with heart failure, and are there monitoring concerns?"

# A question with no supporting evidence — Synthesizer should flag it and
# confidence should come back Low, not a fluent guess
python -m src.main query "What is the recommended dosage of insulin glargine?"

python -m src.main stats

# Launch the Gradio web UI (http://127.0.0.1:7860)
python -m src.main web

# Run the eval harness (8 labeled test cases)
python -m eval.run_eval --verbose
```

## Bring your own AI API

Identical scheme to `multi-agent-code-review` and `graphrag-knowledge-assistant` — see
either project's README for the full provider table (Anthropic, OpenAI, Gemini, Groq,
Mistral, xAI, Ollama, etc.). Set `LLM_PROVIDER` / `AGENT_MODEL` and, separately if
desired, `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS`, plus the
matching API key(s), in `.env`.

## Tuning

- `TOP_K_CHUNKS` (default 5) — how many chunks vector search returns per query.
- `MAX_HOPS` (default 2) — how far graph traversal reaches from seed entities.
- `MAX_TRIPLES` (default 40) — hard cap on structured facts returned per query.
- `CHUNK_SIZE` (default 1200 chars) — how documents are split before embedding/extraction.
- `CONFIDENCE_HIGH_THRESHOLD` / `CONFIDENCE_MEDIUM_THRESHOLD` — score cutoffs for the
  High / Medium / Low label.

## Roadmap

- [x] Hybrid Neo4j store — entity graph + native vector index over chunks
- [x] Section-aware chunking (headings tracked for finer-grained citations)
- [x] Extractor, Entity Linker, and Synthesizer agents (clinical-tuned)
- [x] Deterministic confidence scoring (similarity + source agreement + graph corroboration)
- [x] CLI (`check` / `ingest` / `query` / `stats` / `web`)
- [x] Sample documents spanning two doc types for immediate testing
- [x] PDF ingestion via PyMuPDF (guidelines are often PDF in practice)
- [x] Structured dosage/contraindication schema (HAS_DOSAGE, CONTRAINDICATED_FOR typed
      relationship properties for machine-queryable clinical safety data)
- [x] Recency weighting in confidence scoring (publication_date on chunks, linear penalty
      for documents older than 3 years)
- [x] Web UI via Gradio (Ingest/Query/Stats tabs at http://127.0.0.1:7860)
- [x] Eval harness — 8 labeled clinical Q&A test cases with automated validation
      (source citations, confidence range, keyword presence)
