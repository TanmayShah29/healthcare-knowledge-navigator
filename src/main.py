"""CLI entrypoint.

Usage:
    python -m src.main check
    python -m src.main ingest path/to/document.{txt,pdf} [doc_type] [--date YYYY-MM-DD]
    python -m src.main query "your clinical question"
    python -m src.main stats
    python -m src.main web   (launch Gradio web UI)

doc_type (optional, defaults to clinical_guideline) is one of:
    clinical_guideline | research_paper | treatment_protocol

--date (optional) publication date for recency weighting in confidence scoring.
    Accepts YYYY-MM-DD, YYYY-MM, or YYYY format.

PDF files are supported natively — text is extracted via PyMuPDF before
chunking and ingestion.
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
        f"{stats['rel_count']} relationships, {stats['chunk_count']} indexed chunks, "
        f"{stats.get('dosage_count', 0)} dosage facts, "
        f"{stats.get('contra_count', 0)} contraindication facts."
    )


def cmd_ingest(path: str, doc_type: str = "clinical_guideline", publication_date: str = ""):
    if doc_type not in VALID_DOC_TYPES:
        print(f"Unknown doc_type {doc_type!r}. Valid: {', '.join(sorted(VALID_DOC_TYPES))}")
        sys.exit(1)

    date_str = f" (published {publication_date})" if publication_date else ""
    print(f"Ingesting {path} as {doc_type}{date_str} ...")
    result = ingest_file(path, doc_type=doc_type, publication_date=publication_date)
    print(
        f"Processed {result.chunks_processed} chunk(s), "
        f"extracted {len(result.triples_extracted)} triple(s), "
        f"{len(result.dosages_extracted)} dosage(s), "
        f"{len(result.contraindications_extracted)} contraindication(s)."
    )
    print(
        f"Loaded into graph: {result.triples_loaded} triples, "
        f"{result.dosages_loaded} dosages, "
        f"{result.contraindications_loaded} contraindications."
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

    if result.dosages:
        print(f"\n  DOSAGE FACTS ({len(result.dosages)}):")
        for d in result.dosages:
            print(f"  {d.as_fact_string()}  [{d.citation()}]")

    if result.contraindications:
        print(f"\n  CONTRAINDICATIONS ({len(result.contraindications)}):")
        for c in result.contraindications:
            print(f"  {c.as_fact_string()}  [{c.citation()}]")

    answer, grounded = synthesize_answer(
        question, result.subgraph, result.vector_hits,
        result.dosages, result.contraindications,
    )
    score, label = score_confidence(result.subgraph, result.vector_hits, grounded)

    print("\n" + "=" * 70)
    print(f"ANSWER  —  Confidence: {label} ({score})")
    print("=" * 70)
    print(answer)


def cmd_stats():
    stats = graph_stats()
    print(
        f"{stats['node_count']} entities, {stats['rel_count']} relationships, "
        f"{stats['chunk_count']} indexed chunks, "
        f"{stats.get('dosage_count', 0)} dosage facts, "
        f"{stats.get('contra_count', 0)} contraindication facts."
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
            path = sys.argv[2]
            doc_type = "clinical_guideline"
            pub_date = ""
            for arg in sys.argv[3:]:
                if arg == "--date" and sys.argv.index(arg) + 1 < len(sys.argv):
                    pub_date = sys.argv[sys.argv.index(arg) + 1]
                elif arg != "--date" and not arg.startswith("--"):
                    doc_type = arg
            if doc_type not in VALID_DOC_TYPES:
                print(f"Unknown doc_type {doc_type!r}. Valid: {', '.join(sorted(VALID_DOC_TYPES))}")
                sys.exit(1)
            cmd_ingest(path, doc_type, pub_date)
        elif command == "query" and len(sys.argv) >= 3:
            cmd_query(" ".join(sys.argv[2:]))
        elif command == "stats":
            cmd_stats()
        elif command == "web":
            from src.web import launch_web
            close()  # close the default driver; web UI manages its own
            launch_web()
        else:
            print(__doc__)
            sys.exit(1)
    finally:
        close()


if __name__ == "__main__":
    main()
