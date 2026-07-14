# CLAUDE.md

Guidance for Claude (via Claude Code or any agentic session) working in this repo.

## What this project is

A hybrid (graph + vector) medical RAG assistant. Clinical documents are ingested into a
Neo4j store that's simultaneously an entity graph (LLM-extracted structured facts) and
a vector index (embedded text chunks). Questions are answered by running both retrieval
modes in parallel and combining their results into a cited, confidence-scored answer.
See `README.md` for the architecture diagram and `PLAN.md` for the full build plan and
phase status.

**It's provider-agnostic for BOTH chat and embeddings**, extending the pattern from the
sibling projects `../graphrag-knowledge-assistant` and `../multi-agent-code-review`:
every agent calls `src/llm.py`'s `get_llm()`, and every embedding call goes through
`src/embeddings.py`'s `get_embeddings()`/`embed_text()`/`embed_texts()`.

## Project structure

```
src/
  config.py          # env/config: Neo4j, LLM_PROVIDER/AGENT_MODEL, EMBEDDING_PROVIDER/MODEL/DIMENSIONS, retrieval + confidence tuning
  llm.py              # get_llm(temperature) — chat model factory
  embeddings.py        # get_embeddings() / embed_text() / embed_texts() — embedding factory
  state.py               # Triple, DosageFact, ContraindicationFact, Chunk, VectorHit, IngestResult, RetrievalResult dataclasses
  graph_db.py              # Neo4j wrapper: entity upsert/traversal (graph) + chunk upsert/vector search (vector) + dosage/contraindication queries
  ingest.py                  # section-aware chunking + hybrid ingestion pipeline (txt + PDF)
  retrieval.py                 # runs vector search + graph traversal + dosage/contraindication lookup legs
  confidence.py                  # deterministic score_confidence() with recency weighting — NOT an LLM call
  main.py                          # CLI: check / ingest / query / stats / web
  web.py                            # Gradio web UI (Ingest/Query/Stats tabs)
  agents/
    extractor.py                    # clinical text chunk -> triples + dosages + contraindications (ingestion time)
    entity_linker.py                  # question -> candidate clinical entity names (query time)
    synthesizer.py                      # question + evidence -> cited answer + grounded flag (query time)
eval/
  test_cases.json          # 8 labeled Q&A test cases with expected sources/confidence/keywords
  run_eval.py              # automated eval harness runner
samples/
  t2dm_management_guideline.txt         # sample clinical_guideline, fictional/illustrative
  cv_risk_t2dm_research_summary.txt       # sample research_paper, fictional/illustrative
```

## Core invariants — do not break these

- **No agent or pipeline file imports a provider SDK directly** — chat goes through
  `from src.llm import get_llm`, embeddings go through `from src.embeddings import
  get_embeddings` (or `embed_text`/`embed_texts`). Same rule as `graphrag-knowledge-assistant`;
  keep all provider logic confined to `src/llm.py`, `src/embeddings.py`, and `src/config.py`.
- **The Extractor must never invent a dosage, indication, or contraindication that
  isn't explicitly in the source text.** This is the single highest-risk failure mode
  in this domain — a hallucinated structured fact looks just as authoritative in the
  graph as a real one. If you touch `extractor.py`'s prompt, keep that instruction
  intact and keep the entity type vocabulary restricted to the clinical set (Condition,
  Drug, Symptom, Treatment, Procedure, Guideline, RiskFactor).
- **The Synthesizer must only use retrieved evidence, never fall back to general
  medical knowledge silently**, and must cite every claim back to a source. If you
  touch its prompt, keep the `INSUFFICIENT_EVIDENCE_MARKER` sentinel and the per-claim
  citation instruction — `confidence.py` depends on that marker to cap scores correctly,
  and citations are the whole point of the tool (a clinician needs to verify against the
  original source before acting on anything). **Detection must stay a `startswith` check
  on the stripped answer (`answer.strip().startswith(INSUFFICIENT_EVIDENCE_MARKER)`),
  not a substring check (`marker in answer`).** The prompt tells the model to open the
  answer with the marker; a substring check false-positives whenever the model echoes
  or paraphrases the marker text later in an otherwise well-evidenced answer, which
  wrongly caps confidence for a good answer.
- **Confidence scoring (`confidence.py`) stays a deterministic function, never an LLM
  call.** Same rationale as `graphrag-knowledge-assistant`'s multi-hop traversal:
  a score meant to communicate trustworthiness must itself be auditable and not another
  thing that can hallucinate.
- **`graph_db.py`'s `upsert_triples` and `upsert_chunk` must stay idempotent (MERGE, not
  CREATE).** Re-ingesting the same document should never duplicate nodes, relationships,
  or chunks.
- **`EMBEDDING_DIMENSIONS` in config must match the real embedding model's output
  size.** The Neo4j vector index is created with this fixed dimensionality
  (`ensure_constraints`); a mismatch either fails index creation or silently corrupts
  similarity search. If you change `EMBEDDING_MODEL` to a different provider/model,
  update `EMBEDDING_DIMENSIONS` in the same `.env` change.
- **`MAX_TRIPLES` must always cap the graph traversal Cypher `LIMIT`, and
  `TOP_K_CHUNKS` must always cap the vector search `k`.** Without these, a densely
  connected seed entity or a large corpus can blow out the Synthesizer's context.
- **`vector_search`'s SEARCH-clause/procedure fallback must stay in sync.** Both query
  strings in `graph_db.py` return the same columns (`chunk_id`, `text`, `source_doc`,
  `doc_type`, `section`, `score`, `publication_date`) so `VectorHit` construction doesn't
  need to care which one ran. If you add a field to `Chunk`, add it to both queries.
- **The Extractor's `extract_all()` must return three lists in one LLM call** (triples,
  dosages, contraindications). Do not split this into three separate calls — the single
  call is both faster and gives the model better context for consistent extraction.
- **`upsert_dosages` and `upsert_contraindications` must stay idempotent (MERGE, not
  CREATE).** Same invariant as `upsert_triples` — re-ingesting the same document should
  never duplicate dosage or contraindication relationships.
- **Recency weighting is a soft signal, not a hard cutoff.** The `_recency_penalty()`
  function in `confidence.py` applies a linear penalty for documents older than
  `RECENCY_FRESH_YEARS` (3 years), capped at `RECENCY_MAX_PENALTY` (0.1). Older
  documents are still useful — they're just slightly less authoritative when newer ones
  exist. Never make recency a hard filter that excludes older sources entirely.

## Conventions

- Extractor temperature is `0.1` (extraction should be close to deterministic), Entity
  Linker is `0.0`, Synthesizer is `0.2` (a little more room for well-written, cited prose).
- Relationship type in Neo4j is always the generic `RELATES_TO`, with actual semantics
  carried in the `predicate` property string — same reasoning as
  `graphrag-knowledge-assistant`: a fixed predicate vocabulary (e.g. `:TREATS`,
  `:CONTRAINDICATED_WITH` as distinct relationship types) would defeat the point of
  LLM-driven extraction and make schema evolution harder.
- `source_doc` and `doc_type` are stored on relationships (not just chunks) for
  provenance, since a clinical fact's trustworthiness depends on what kind of document
  it came from (guideline vs. single research paper vs. protocol).
- Every sample document under `samples/` starts with an explicit "SAMPLE DATA — not
  real clinical guidance" disclaimer line. Keep that convention for any new sample data
  you add — this repo is a demonstration of the pipeline, not a source of clinical fact.

## When adding a new agent or pipeline step

1. Add any new fields it needs to the relevant dataclass in `state.py`.
2. Write the function to take and return plain data (a dataclass or list of them) —
   there's no LangGraph state machine here (see `state.py`'s docstring for why); don't
   introduce one unless a step actually needs a retry loop.
3. Wire it into `ingest.py` or `retrieval.py`'s pipeline function, and expose it via a
   new `main.py` subcommand if it's user-facing.
4. If it changes what evidence the Synthesizer sees or what confidence depends on,
   update `confidence.py`'s docstring (the signal weights are documented there) and
   the architecture diagram in `README.md`.

## Testing changes

Run the automated eval harness (8 labeled Q&A test cases) to catch regressions:

```bash
python -m eval.run_eval --verbose
```

Or validate manually:

```bash
python -m src.main check
python -m src.main ingest samples/t2dm_management_guideline.txt clinical_guideline --date 2024-01-01
python -m src.main ingest samples/cv_risk_t2dm_research_summary.txt research_paper --date 2023-06-15
python -m src.main query "What second-line diabetes drug should I consider for a patient with heart failure, and are there monitoring concerns?"
python -m src.main query "What is the recommended dosage of insulin glargine?"
```

The first query is the important multi-document one to re-check after any change to
`extractor.py`, `graph_db.py`, or `retrieval.py` — it only answers well (citing both
sources) if canonicalization, upsert, vector search, and graph traversal are all still
working together. The second query (no ingested evidence exists for it) is the
important one to re-check after any change to `synthesizer.py` or `confidence.py` — it
must trigger the insufficient-evidence marker and a Low confidence score, not a
confident-sounding fabrication.

## Things NOT to do

- Don't let the Extractor or Synthesizer write directly to Neo4j — all writes go
  through `graph_db.py`'s `upsert_triples`/`upsert_chunk`/`upsert_dosages`/`upsert_contraindications`,
  so there's one place that enforces the idempotent-MERGE invariant.
- Don't let the Synthesizer see raw Cypher, embeddings, or graph internals — it only
  ever gets plain-language fact strings (`Triple.as_fact_string()` + `.citation()`,
  `DosageFact.as_fact_string()`, `ContraindicationFact.as_fact_string()`) and
  chunk text with citations, so its prompt stays portable across schema changes.
- Don't hardcode a provider or model name anywhere outside `src/config.py`,
  `src/llm.py`, and `src/embeddings.py`.
- Don't remove or soften the decision-support disclaimer in the Synthesizer's system
  prompt, and don't let confidence scoring report High for an answer the Synthesizer
  itself flagged as insufficiently evidenced — the whole point of this tool is that its
  confidence signal is trustworthy enough for a busy clinician to act on at a glance.
