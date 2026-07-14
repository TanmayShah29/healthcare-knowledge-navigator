"""Thin Neo4j driver wrapper: connection management, idempotent upserts, and
the Cypher queries used by ingestion and hybrid (vector + graph) retrieval.

Schema (deliberately simple, domain-tuned for clinical text):
  (:Entity {name, type})-[:RELATES_TO {predicate, source_doc, doc_type}]->(:Entity)
  (:Chunk {chunk_id, text, source_doc, doc_type, section, embedding})

Entities and Chunks are separate node families — Entities carry structured
clinical facts (for multi-hop traversal), Chunks carry raw text + embeddings
(for vector similarity search). They're linked implicitly through shared
source_doc, not a direct relationship, since a chunk can mention several
entities and an entity can appear in several chunks.
"""
from typing import List
from neo4j import GraphDatabase
from src.config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, MAX_TRIPLES, TOP_K_CHUNKS,
    EMBEDDING_DIMENSIONS,
)
from src.state import Triple, Chunk, VectorHit

_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def close():
    _driver.close()


def check_connection() -> bool:
    with _driver.session() as session:
        session.run("RETURN 1").single()
    return True


def ensure_constraints():
    """Unique constraints so MERGE behaves as a real upsert for both node
    families, plus the vector index over Chunk.embedding."""
    with _driver.session() as session:
        session.run(
            "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE"
        )
        session.run(
            "CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS "
            "FOR (c:Chunk) ON (c.embedding) "
            "OPTIONS {indexConfig: {"
            "`vector.dimensions`: $dims, "
            "`vector.similarity_function`: 'cosine'}}",
            dims=EMBEDDING_DIMENSIONS,
        )


# ── Entity graph (structured facts, multi-hop traversal) ─────────────────

def upsert_triples(triples: List[Triple]) -> int:
    """MERGE each triple's two entity nodes and the relationship between
    them. Idempotent: re-running with the same triples doesn't duplicate."""
    if not triples:
        return 0

    query = """
    UNWIND $rows AS row
    MERGE (s:Entity {name: row.subject})
      ON CREATE SET s.type = row.subject_type
    MERGE (o:Entity {name: row.object})
      ON CREATE SET o.type = row.object_type
    MERGE (s)-[r:RELATES_TO {predicate: row.predicate}]->(o)
      ON CREATE SET r.source_doc = row.source_doc, r.doc_type = row.doc_type
    """
    rows = [
        {
            "subject": t.subject,
            "subject_type": t.subject_type,
            "object": t.object,
            "object_type": t.object_type,
            "predicate": t.predicate,
            "source_doc": t.source_doc,
            "doc_type": t.doc_type,
        }
        for t in triples
    ]
    with _driver.session() as session:
        session.run(query, rows=rows)
    return len(triples)


def find_entities_like(names: List[str]) -> List[str]:
    """Fuzzy-match candidate entity names (from the Entity Linker agent)
    against actual node names, case-insensitive substring match both ways.
    Returns real graph node names to use as multi-hop traversal seeds."""
    if not names:
        return []

    query = """
    UNWIND $candidates AS candidate
    MATCH (e:Entity)
    WHERE toLower(e.name) CONTAINS toLower(candidate)
       OR toLower(candidate) CONTAINS toLower(e.name)
    RETURN DISTINCT e.name AS name
    """
    with _driver.session() as session:
        result = session.run(query, candidates=names)
        return [record["name"] for record in result]


def multi_hop_neighbors(seed_names: List[str], max_hops: int) -> List[Triple]:
    """Variable-length path traversal out from seed entities, up to max_hops
    relationships away — e.g. Drug -> treats -> Condition -> has_symptom ->
    Symptom, chaining facts that no single relationship states directly."""
    if not seed_names:
        return []

    query = f"""
    MATCH (seed:Entity)
    WHERE seed.name IN $seed_names
    MATCH path = (seed)-[:RELATES_TO*1..{max_hops}]-(neighbor:Entity)
    UNWIND relationships(path) AS rel
    WITH DISTINCT rel, startNode(rel) AS s, endNode(rel) AS o
    RETURN s.name AS subject, s.type AS subject_type,
           rel.predicate AS predicate,
           o.name AS object, o.type AS object_type,
           rel.source_doc AS source_doc, rel.doc_type AS doc_type
    LIMIT $limit
    """
    with _driver.session() as session:
        result = session.run(query, seed_names=seed_names, limit=MAX_TRIPLES)
        return [
            Triple(
                subject=r["subject"],
                subject_type=r["subject_type"] or "Unknown",
                predicate=r["predicate"],
                object=r["object"],
                object_type=r["object_type"] or "Unknown",
                source_doc=r["source_doc"] or "",
                doc_type=r["doc_type"] or "",
            )
            for r in result
        ]


# ── Chunk vector store (unstructured text, similarity search) ────────────

def upsert_chunk(chunk: Chunk) -> None:
    query = """
    MERGE (c:Chunk {chunk_id: $chunk_id})
    SET c.text = $text, c.source_doc = $source_doc, c.doc_type = $doc_type,
        c.section = $section, c.embedding = $embedding
    """
    with _driver.session() as session:
        session.run(
            query,
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            source_doc=chunk.source_doc,
            doc_type=chunk.doc_type,
            section=chunk.section,
            embedding=chunk.embedding,
        )


_SEARCH_CLAUSE_QUERY = """
CYPHER 25
MATCH (c:Chunk)
  SEARCH c IN (
    VECTOR INDEX chunk_embeddings
    FOR $embedding
    LIMIT $k
  ) SCORE AS score
RETURN c.chunk_id AS chunk_id, c.text AS text,
       c.source_doc AS source_doc, c.doc_type AS doc_type,
       c.section AS section, score
"""

_QUERY_NODES_PROCEDURE_QUERY = """
CALL db.index.vector.queryNodes('chunk_embeddings', $k, $embedding)
YIELD node, score
RETURN node.chunk_id AS chunk_id, node.text AS text,
       node.source_doc AS source_doc, node.doc_type AS doc_type,
       node.section AS section, score
"""

_use_search_clause = True  # flips to False after the first fallback, so we
                            # don't retry the unsupported syntax on every call


def vector_search(query_embedding: List[float], top_k: int = None) -> List[VectorHit]:
    """Cosine similarity search over Chunk.embedding via Neo4j's native
    vector index. This is the piece that makes retrieval "hybrid" — it finds
    textually-relevant passages the entity graph might miss (e.g. nuanced
    dosing caveats that don't reduce cleanly to a (subject, predicate,
    object) triple).

    Prefers the Cypher `SEARCH` clause (the preferred syntax as of Neo4j
    2026.01; `db.index.vector.queryNodes` is deprecated as of 2026.04) and
    transparently falls back to the procedure on older Neo4j versions that
    don't yet support `SEARCH`, so this repo keeps working across Neo4j
    versions without a manual migration.
    """
    global _use_search_clause
    k = top_k or TOP_K_CHUNKS

    with _driver.session() as session:
        records = None
        if _use_search_clause:
            try:
                records = list(session.run(_SEARCH_CLAUSE_QUERY, k=k, embedding=query_embedding))
            except Exception:
                _use_search_clause = False  # older Neo4j — don't retry SEARCH again this run

        if records is None:
            records = list(session.run(_QUERY_NODES_PROCEDURE_QUERY, k=k, embedding=query_embedding))

        return [
            VectorHit(
                chunk_id=r["chunk_id"],
                text=r["text"],
                source_doc=r["source_doc"] or "",
                doc_type=r["doc_type"] or "",
                section=r["section"] or "",
                score=r["score"],
            )
            for r in records
        ]


def graph_stats() -> dict:
    query = """
    MATCH (e:Entity) WITH count(e) AS node_count
    OPTIONAL MATCH ()-[r:RELATES_TO]->()
    WITH node_count, count(r) AS rel_count
    OPTIONAL MATCH (c:Chunk)
    RETURN node_count, rel_count, count(c) AS chunk_count
    """
    with _driver.session() as session:
        record = session.run(query).single()
        if record is None:
            return {"node_count": 0, "rel_count": 0, "chunk_count": 0}
        return {
            "node_count": record["node_count"],
            "rel_count": record["rel_count"],
            "chunk_count": record["chunk_count"],
        }
