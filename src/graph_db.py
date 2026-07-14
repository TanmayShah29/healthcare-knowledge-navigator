"""Thin Neo4j driver wrapper: connection management, idempotent upserts, and
the Cypher queries used by ingestion and hybrid (vector + graph) retrieval.

Schema (deliberately simple, domain-tuned for clinical text):
  (:Entity {name, type})-[:RELATES_TO {predicate, source_doc, doc_type}]->(:Entity)
  (:Chunk {chunk_id, text, source_doc, doc_type, section, embedding})

Structured clinical safety data (Phase 11):
  (:Entity {name})-[:HAS_DOSAGE {dose, frequency, route, notes, source_doc, doc_type}]->(:Entity)
  (:Entity {name})-[:CONTRAINDICATED_FOR {reason, severity, source_doc, doc_type}]->(:Entity)

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
from src.state import Triple, Chunk, VectorHit, DosageFact, ContraindicationFact

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
        c.section = $section, c.embedding = $embedding,
        c.publication_date = $publication_date
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
            publication_date=chunk.publication_date,
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
       c.section AS section, score, c.publication_date AS publication_date
"""

_QUERY_NODES_PROCEDURE_QUERY = """
CALL db.index.vector.queryNodes('chunk_embeddings', $k, $embedding)
YIELD node, score
RETURN node.chunk_id AS chunk_id, node.text AS text,
       node.source_doc AS source_doc, node.doc_type AS doc_type,
       node.section AS section, score, node.publication_date AS publication_date
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
                publication_date=r["publication_date"] or "",
            )
            for r in records
        ]


def graph_stats() -> dict:
    query = """
    MATCH (e:Entity) WITH count(e) AS node_count
    OPTIONAL MATCH ()-[r:RELATES_TO]->()
    WITH node_count, count(r) AS rel_count
    OPTIONAL MATCH (c:Chunk)
    OPTIONAL MATCH ()-[d:HAS_DOSAGE]->()
    WITH node_count, rel_count, count(c) AS chunk_count, count(d) AS dosage_count
    OPTIONAL MATCH ()-[ci:CONTRAINDICATED_FOR]->()
    RETURN node_count, rel_count, chunk_count, dosage_count, count(ci) AS contra_count
    """
    with _driver.session() as session:
        record = session.run(query).single()
        if record is None:
            return {"node_count": 0, "rel_count": 0, "chunk_count": 0, "dosage_count": 0, "contra_count": 0}
        return {
            "node_count": record["node_count"],
            "rel_count": record["rel_count"],
            "chunk_count": record["chunk_count"],
            "dosage_count": record["dosage_count"],
            "contra_count": record["contra_count"],
        }


# ── Structured dosage facts (Phase 11) ──────────────────────────────────

def upsert_dosages(dosages: List[DosageFact]) -> int:
    """MERGE dosage relationships as typed properties on HAS_DOSAGE edges.
    Idempotent: re-running doesn't duplicate."""
    if not dosages:
        return 0

    query = """
    UNWIND $rows AS row
    MERGE (d:Entity {name: row.drug})
      ON CREATE SET d.type = 'Drug'
    MERGE (d)-[r:HAS_DOSAGE {dose: row.dose, source_doc: row.source_doc}]->(d)
      ON CREATE SET r.frequency = row.frequency, r.route = row.route,
                    r.notes = row.notes, r.doc_type = row.doc_type
    """
    rows = [
        {
            "drug": dos.drug,
            "dose": dos.dose,
            "frequency": dos.frequency,
            "route": dos.route,
            "notes": dos.notes,
            "source_doc": dos.source_doc,
            "doc_type": dos.doc_type,
        }
        for dos in dosages
    ]
    with _driver.session() as session:
        session.run(query, rows=rows)
    return len(dosages)


def upsert_contraindications(contras: List[ContraindicationFact]) -> int:
    """MERGE contraindication relationships as typed properties on
    CONTRAINDICATED_FOR edges. Idempotent."""
    if not contras:
        return 0

    query = """
    UNWIND $rows AS row
    MERGE (d:Entity {name: row.drug})
      ON CREATE SET d.type = 'Drug'
    MERGE (c:Entity {name: row.condition})
      ON CREATE SET c.type = 'Condition'
    MERGE (d)-[r:CONTRAINDICATED_FOR {condition: row.condition, source_doc: row.source_doc}]->(c)
      ON CREATE SET r.reason = row.reason, r.severity = row.severity,
                    r.doc_type = row.doc_type
    """
    rows = [
        {
            "drug": ci.drug,
            "condition": ci.condition,
            "reason": ci.reason,
            "severity": ci.severity,
            "source_doc": ci.source_doc,
            "doc_type": ci.doc_type,
        }
        for ci in contras
    ]
    with _driver.session() as session:
        session.run(query, rows=rows)
    return len(contras)


def query_dosages(drug_names: List[str] = None) -> List[DosageFact]:
    """Query dosage facts, optionally filtered by drug names."""
    if drug_names:
        query = """
        MATCH (d:Entity)-[r:HAS_DOSAGE]->(d)
        WHERE d.name IN $drug_names
        RETURN d.name AS drug, r.dose AS dose, r.frequency AS frequency,
               r.route AS route, r.notes AS notes,
               r.source_doc AS source_doc, r.doc_type AS doc_type
        """
        with _driver.session() as session:
            result = session.run(query, drug_names=drug_names)
    else:
        query = """
        MATCH (d:Entity)-[r:HAS_DOSAGE]->(d)
        RETURN d.name AS drug, r.dose AS dose, r.frequency AS frequency,
               r.route AS route, r.notes AS notes,
               r.source_doc AS source_doc, r.doc_type AS doc_type
        """
        with _driver.session() as session:
            result = session.run(query)

    return [
        DosageFact(
            drug=r["drug"],
            dose=r["dose"],
            frequency=r["frequency"] or "",
            route=r["route"] or "",
            notes=r["notes"] or "",
            source_doc=r["source_doc"] or "",
            doc_type=r["doc_type"] or "",
        )
        for r in result
    ]


def query_contraindications(condition_names: List[str] = None) -> List[ContraindicationFact]:
    """Query contraindication facts, optionally filtered by condition."""
    if condition_names:
        query = """
        MATCH (d:Entity)-[r:CONTRAINDICATED_FOR]->(c:Entity)
        WHERE c.name IN $condition_names
        RETURN d.name AS drug, c.name AS condition, r.reason AS reason,
               r.severity AS severity, r.source_doc AS source_doc,
               r.doc_type AS doc_type
        """
        with _driver.session() as session:
            result = session.run(query, condition_names=condition_names)
    else:
        query = """
        MATCH (d:Entity)-[r:CONTRAINDICATED_FOR]->(c:Entity)
        RETURN d.name AS drug, c.name AS condition, r.reason AS reason,
               r.severity AS severity, r.source_doc AS source_doc,
               r.doc_type AS doc_type
        """
        with _driver.session() as session:
            result = session.run(query)

    return [
        ContraindicationFact(
            drug=r["drug"],
            condition=r["condition"],
            reason=r["reason"] or "",
            severity=r["severity"] or "",
            source_doc=r["source_doc"] or "",
            doc_type=r["doc_type"] or "",
        )
        for r in result
    ]
