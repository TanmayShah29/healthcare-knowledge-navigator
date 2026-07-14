"""CLI entrypoint.

Usage:
    python -m src.main check
    python -m src.main ingest path/to/document.txt [doc_type]
    python -m src.main query "your clinical question"
    python -m src.main stats

doc_type (optional, defaults to clinical_guideline) is one of:
    clinical_guideline | research_paper | treatment_protocol
"""
import sys
from src.graph_db import check_connection, graph_stats, close
from src.ingest import ingest_file
from src.retrieval import retrieve
from src.agents.synthesizer import synthesize_answer
from src.confidence import score_confidence

VALID_DOC_TYPES = {"clinical_guideline", "research_paper", "treatment_protocol"}


def cmd_check():
    print("Checking Neo4j connection (entity graph + vector index)...")
    check_connection()
    stats = graph_stats()
    print(
        f"Connected. Graph currently has {stats['node_count']} entities, "
        f"{stats['rel_count']} relationships, {stats['chunk_count']} indexed chunks."
    )


def cmd_ingest(path: str, doc_type: str = "clinical_guideline"):
    if doc_type not in VALID_DOC_TYPES:
        print(f"Unknown doc_type {doc_type!r}. Valid: {', '.join(sorted(VALID_DOC_TYPES))}")
        sys.exit(1)

    print(f"Ingesting {path} as {doc_type} ...")
    result = ingest_file(path, doc_type=doc_type)
    print(
        f"Processed {result.chunks_processed} chunk(s), "
        f"extracted {len(result.triples_extracted)} triple(s), "
        f"loaded {result.triples_loaded} into the graph."
    )
    if result.errors:
        print("Errors:")
        for err in result.errors:
            print(f"  - {err}")
    stats = graph_stats()
    print(
        f"Graph now has {stats['node_count']} entities, {stats['rel_count']} "
        f"relationships, {stats['chunk_count']} indexed chunks."
    )


def cmd_query(question: str):
    result = retrieve(question)

    print("=" * 70)
    print(f"VECTOR-MATCHED PASSAGES ({len(result.vector_hits)})")
    print("=" * 70)
    for hit in result.vector_hits:
        print(f"  [{hit.score:.2f}] {hit.citation()}")
    if not result.vector_hits:
        print("  (none)")

    print("\n" + "=" * 70)
    print(f"LINKED ENTITIES: {', '.join(result.linked_entities) or '(none found)'}")
    print(f"GRAPH FACTS ({len(result.subgraph)})")
    print("=" * 70)
    for triple in result.subgraph:
        print(f"  {triple.as_fact_string()}  [{triple.citation()}]")
    if not result.subgraph:
        print("  (none)")

    answer, grounded = synthesize_answer(question, result.subgraph, result.vector_hits)
    score, label = score_confidence(result.subgraph, result.vector_hits, grounded)

    print("\n" + "=" * 70)
    print(f"ANSWER  —  Confidence: {label} ({score})")
    print("=" * 70)
    print(answer)


def cmd_stats():
    stats = graph_stats()
    print(
        f"{stats['node_count']} entities, {stats['rel_count']} relationships, "
        f"{stats['chunk_count']} indexed chunks."
    )


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    try:
        if command == "check":
            cmd_check()
        elif command == "ingest" and len(sys.argv) >= 3:
            doc_type = sys.argv[3] if len(sys.argv) >= 4 else "clinical_guideline"
            cmd_ingest(sys.argv[2], doc_type)
        elif command == "query" and len(sys.argv) >= 3:
            cmd_query(" ".join(sys.argv[2:]))
        elif command == "stats":
            cmd_stats()
        else:
            print(__doc__)
            sys.exit(1)
    finally:
        close()


if __name__ == "__main__":
    main()
