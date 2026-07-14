# Build Plan — Healthcare Knowledge Navigator

A hybrid (vector + graph) medical RAG assistant: ingest clinical guidelines, research
papers, and treatment protocols into a Neo4j store that's both an entity graph
(structured facts, multi-hop traversal) and a vector index (embedded chunks, similarity
search); answer clinician questions with cited, confidence-scored answers.

## Phase 0 — Environment & Scaffolding
- Python 3.10+, virtualenv, `requirements.txt` (neo4j driver, langchain core,
  python-dotenv, plus one LLM/embeddings provider package — reusing the
  provider-agnostic pattern from `graphrag-knowledge-assistant` and extending it to
  embeddings)
- Neo4j instance with vector index support (5.11+): Neo4j Desktop or Aura free tier;
  `.env` needs `NEO4J_URI/USER/PASSWORD`, `LLM_PROVIDER`/`AGENT_MODEL`,
  `EMBEDDING_PROVIDER`/`EMBEDDING_MODEL`/`EMBEDDING_DIMENSIONS`, plus matching API key(s)
- Repo skeleton: `src/agents/`, `src/graph_db.py`, `src/state.py`, `src/embeddings.py`,
  `src/ingest.py`, `src/retrieval.py`, `src/confidence.py`, `src/main.py`, `src/llm.py`

**Deliverable:** project boots, Neo4j connection + vector index creation succeeds
(`python -m src.main check`).

## Phase 1 — Schema & State Design
- Node families: `Entity {name, type}` (type ∈ Condition, Drug, Symptom, Treatment,
  Procedure, Guideline, RiskFactor — a clinical vocabulary, not free-text like the
  general-purpose graphrag project) and `Chunk {chunk_id, text, source_doc, doc_type,
  section, embedding}` (vector-indexed)
- Relationship: generic `RELATES_TO {predicate, source_doc, doc_type}` — predicate
  string carries the semantics (e.g. "treats", "contraindicated with", "monitor for")
- Define `Triple`, `Chunk`, `VectorHit`, `IngestResult`, `RetrievalResult` dataclasses

**Deliverable:** `state.py`, `graph_db.py` (Neo4j wrapper: entity upsert/traversal +
Chunk upsert/vector search, `ensure_constraints` creates both node constraints and the
vector index).

## Phase 2 — Provider-Agnostic Embeddings Layer
- `src/embeddings.py`: `get_embeddings()` factory mirroring `llm.py`'s `get_llm()`, via
  LangChain's `init_embeddings`; `embed_text` / `embed_texts` convenience wrappers
- `EMBEDDING_PROVIDER` defaults to `LLM_PROVIDER` but can be split out independently;
  `EMBEDDING_DIMENSIONS` must match the model's real output size (used to size the
  Neo4j vector index)

**Deliverable:** embeddings factory, tested standalone against a couple of sample strings.

## Phase 3 — Clinical Extractor Agent
- System prompt: clinical text chunk → JSON list of `{subject, predicate, object,
  subject_type, object_type}` triples, entity types restricted to the clinical
  vocabulary
- Explicit instruction: never infer a dosage, indication, or contraindication that
  isn't stated in the text — hallucinated structured facts are the highest-risk failure
  mode in this domain
- Test in isolation against sample guideline paragraphs — check triples are sensible
  and nothing is invented

**Deliverable:** `agents/extractor.py`, tested standalone.

## Phase 4 — Hybrid Ingestion Pipeline
- `src/ingest.py`: section-aware chunking (tracks heading lines so citations can
  reference e.g. "Dosing" or "Contraindications" within a doc, not just the doc as a
  whole) → for each chunk: embed it + write a `Chunk` node, AND run the Extractor +
  upsert triples
- `doc_type` (clinical_guideline | research_paper | treatment_protocol) tagged on both
  Chunks and relationships for provenance and citation labeling

**Deliverable:** `python -m src.main ingest <file> [doc_type]` populates both halves of
the hybrid store.

## Phase 5 — Entity Linker (query time)
- System prompt: clinical question → candidate entity names, with common medical
  abbreviation expansion (T2DM, MI, ...) so either form matches the graph
- Fuzzy-match against existing `Entity` nodes (case-insensitive substring, both
  directions) to find real seed nodes

**Deliverable:** `agents/entity_linker.py`, tested against sample clinical questions.

## Phase 6 — Hybrid Retriever
- Vector leg: embed the question, `CALL db.index.vector.queryNodes(...)` over
  `chunk_embeddings` index, return top `TOP_K_CHUNKS` passages with similarity scores
- Graph leg: seed entities → Cypher variable-length path traversal up to `MAX_HOPS`,
  capped at `MAX_TRIPLES`
- Both legs run independently (no dependency between them) and are combined at
  synthesis time, not merged into one query — keeps each retrieval mode auditable on
  its own

**Deliverable:** `retrieval.py`'s `retrieve()`, tested against the seeded sample store.

## Phase 7 — Cited Synthesizer Agent
- System prompt: question + structured facts (with citations) + retrieved passages
  (with citations) → answer with an inline `[Source, doc_type]` citation after each
  claim
- Must emit a detectable "insufficient evidence" signal (a sentinel marker string) when
  evidence is thin, rather than filling gaps from general medical knowledge — this
  flag feeds directly into confidence scoring
- Every answer ends with a decision-support disclaimer

**Deliverable:** `agents/synthesizer.py`, tested against evidence sets of varying
completeness (strong single-source, multi-source agreement, and empty).

## Phase 8 — Deterministic Confidence Scorer
- Blend, NOT an LLM call: average vector similarity (0.55 weight) + distinct-source
  agreement capped at 3 sources (0.35 weight) + small graph-corroboration bonus (0.1),
  hard-capped low if the Synthesizer flagged insufficient evidence
- Maps final 0-1 score to High/Medium/Low via configurable thresholds

**Deliverable:** `confidence.py`'s `score_confidence()`, tested against hand-constructed
evidence sets (strong multi-source, weak single-source, empty) to confirm scores and
labels land where expected.

## Phase 9 — CLI
- `python -m src.main check` — Neo4j connection, vector index, and LLM config
- `python -m src.main ingest <file> [doc_type]` — runs the hybrid ingestion pipeline
- `python -m src.main query "<question>"` — prints vector hits, linked entities + graph
  facts, then the cited answer with its confidence label and score
- `python -m src.main stats` — quick entity/relationship/chunk counts

**Deliverable:** all four commands work end-to-end against a live Neo4j instance.

## Phase 10 — Validation Pass
1. Ingest both sample documents (different `doc_type`s, overlapping entities like
   "Type 2 Diabetes Mellitus" and "SGLT2 inhibitors") — confirm entities merge into one
   node rather than duplicating across documents
2. Ask a single-document question — confirm high similarity, one source, sensible
   confidence
3. Ask a question that only a genuine multi-hop chain across BOTH documents can answer
   — confirms both retrieval legs are contributing, not just one
4. Ask a question with no supporting evidence in the ingested set — confirms the
   Synthesizer flags it AND confidence comes back Low, not a fluent unsupported guess

## Phase 11 — Stretch Goals (post-MVP, priority order)
1. **PDF ingestion** — guidelines are commonly distributed as PDF in practice; add a
   PDF → text extraction step ahead of chunking
2. **Structured dosage/contraindication schema** — move beyond free-text predicates for
   dosage and contraindication facts specifically, enabling safer machine-readable
   queries (e.g. "list all drugs contraindicated with renal impairment")
3. **Recency weighting** — factor document publication date into confidence scoring, so
   a superseded 2015 guideline doesn't outweigh a 2024 update
4. **Web UI** — current interface is CLI-only; a clinician-facing UI would surface
   citations and confidence more legibly than terminal output
5. **Eval harness** — a labeled set of clinical Q&A pairs with expected
   citations/confidence bands, to catch retrieval or synthesis regressions
   automatically instead of manual spot-checking

## Status
- [x] Phase 0 — scaffolding
- [x] Phase 1 — schema + state design
- [x] Phase 2 — embeddings layer
- [x] Phase 3 — Extractor agent
- [x] Phase 4 — hybrid ingestion pipeline
- [x] Phase 5 — Entity Linker agent
- [x] Phase 6 — hybrid retriever
- [x] Phase 7 — cited Synthesizer agent
- [x] Phase 8 — confidence scorer
- [x] Phase 9 — CLI
- [x] Phase 10 — validation pass (both sample docs ingested; multi-document hybrid
      query correctly cited both sources; no-evidence query correctly triggered the
      insufficient-evidence marker and Low confidence without hallucinating; fixed two
      bugs found in the process — Neo4j SEARCH-clause migration and a substring-vs-
      startswith marker-detection bug — see git history / CLAUDE.md invariants)
- [ ] Phase 11 — stretch goals
