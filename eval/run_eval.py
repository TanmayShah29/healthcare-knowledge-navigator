"""Eval harness for the Healthcare Knowledge Navigator.

Ingests sample documents, runs each test case from eval/test_cases.json
through the full retrieval + synthesis pipeline, and checks:
  1. Whether expected source documents appear in citations.
  2. Whether the confidence label falls within the expected range.
  3. Whether expected keywords appear in the synthesized answer.
  4. Whether no-evidence cases correctly trigger Low confidence.

Usage:
    python -m eval.run_eval            # full run (ingest + query)
    python -m eval.run_eval --query    # query-only (assumes docs already ingested)
    python -m eval.run_eval --verbose  # print full answers for each test case
"""
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_db import check_connection, graph_stats, close
from src.ingest import ingest_file
from src.retrieval import retrieve
from src.agents.synthesizer import synthesize_answer
from src.confidence import score_confidence

CONFIDENCE_ORDER = {"Low": 0, "Medium": 1, "High": 2}
SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
TEST_CASES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_cases.json")


@dataclass
class TestCase:
    id: str
    question: str
    expected_sources: List[str]
    expected_min_confidence: str
    expected_max_confidence: str
    expected_keywords: List[str]
    description: str = ""


@dataclass
class EvalResult:
    test_id: str
    question: str
    passed: bool
    checks: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    confidence_label: str = ""
    answer_length: int = 0
    grounded: bool = True


def _load_test_cases() -> List[TestCase]:
    with open(TEST_CASES_PATH, "r") as f:
        raw = json.load(f)
    return [TestCase(**tc) for tc in raw]


def _ingest_samples():
    """Ingest both sample documents if not already present."""
    stats = graph_stats()
    if stats["chunk_count"] > 0:
        print(f"Graph already has {stats['chunk_count']} chunks — skipping ingestion.")
        return

    docs = [
        ("t2dm_management_guideline.txt", "clinical_guideline", "2024-01-01"),
        ("cv_risk_t2dm_research_summary.txt", "research_paper", "2023-06-15"),
    ]
    for filename, doc_type, pub_date in docs:
        path = os.path.join(SAMPLES_DIR, filename)
        if os.path.exists(path):
            print(f"Ingesting {filename} as {doc_type}...")
            result = ingest_file(path, doc_type=doc_type, publication_date=pub_date)
            print(
                f"  {result.chunks_processed} chunks, "
                f"{result.triples_loaded} triples, "
                f"{result.dosages_loaded} dosages, "
                f"{result.contraindications_loaded} contraindications."
            )
        else:
            print(f"WARNING: {path} not found — skipping.")


def _run_test(tc: TestCase, verbose: bool = False) -> EvalResult:
    """Run a single test case through the full pipeline and validate."""
    result = EvalResult(test_id=tc.id, question=tc.question, passed=True)

    try:
        retrieval = retrieve(tc.question)
        answer, grounded = synthesize_answer(
            tc.question, retrieval.subgraph, retrieval.vector_hits,
            retrieval.dosages, retrieval.contraindications,
        )
        score, label = score_confidence(retrieval.subgraph, retrieval.vector_hits, grounded)
    except Exception as e:
        result.passed = False
        result.failures.append(f"Pipeline error: {e}")
        return result

    result.confidence_score = score
    result.confidence_label = label
    result.answer_length = len(answer)
    result.grounded = grounded

    # Check 1: Source citations
    if tc.expected_sources:
        cited_sources = set()
        for h in retrieval.vector_hits:
            if h.source_doc:
                cited_sources.add(h.source_doc)
        for t in retrieval.subgraph:
            if t.source_doc:
                cited_sources.add(t.source_doc)

        found_any = any(src in cited_sources for src in tc.expected_sources)
        if found_any:
            result.checks.append(f"SOURCES: found expected source(s) in citations")
        else:
            result.failures.append(
                f"SOURCES: expected {tc.expected_sources}, got {list(cited_sources)}"
            )
            result.passed = False
    else:
        result.checks.append("SOURCES: no expected sources (no-evidence case)")

    # Check 2: Confidence range
    min_level = CONFIDENCE_ORDER.get(tc.expected_min_confidence, 0)
    max_level = CONFIDENCE_ORDER.get(tc.expected_max_confidence, 2)
    actual_level = CONFIDENCE_ORDER.get(label, 0)

    if min_level <= actual_level <= max_level:
        result.checks.append(f"CONFIDENCE: {label} ({score}) within [{tc.expected_min_confidence}, {tc.expected_max_confidence}]")
    else:
        result.failures.append(
            f"CONFIDENCE: got {label} ({score}), expected [{tc.expected_min_confidence}, {tc.expected_max_confidence}]"
        )
        result.passed = False

    # Check 3: Keywords in answer (if expected)
    if tc.expected_keywords:
        answer_lower = answer.lower()
        found_kws = [kw for kw in tc.expected_keywords if kw.lower() in answer_lower]
        if len(found_kws) >= len(tc.expected_keywords) // 2:  # at least half the keywords
            result.checks.append(f"KEYWORDS: found {found_kws}")
        else:
            missing = [kw for kw in tc.expected_keywords if kw.lower() not in answer_lower]
            result.failures.append(f"KEYWORDS: missing {missing}")
            result.passed = False

    if verbose:
        print(f"\n{'='*60}")
        print(f"Q: {tc.question}")
        print(f"A ({label}, {score}): {answer[:500]}...")
        print(f"Grounded: {grounded}")

    return result


def run_eval(verbose: bool = False, query_only: bool = False):
    """Run the full evaluation suite."""
    print("=" * 60)
    print("Healthcare Knowledge Navigator — Eval Harness")
    print("=" * 60)

    check_connection()
    print("Connected to Neo4j.\n")

    if not query_only:
        _ingest_samples()
        print()

    test_cases = _load_test_cases()
    print(f"Running {len(test_cases)} test cases...\n")

    results = []
    for tc in test_cases:
        print(f"  [{tc.id}] ", end="", flush=True)
        result = _run_test(tc, verbose=verbose)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"{status}  confidence={result.confidence_label}({result.confidence_score})")
        time.sleep(0.5)  # be nice to the LLM

    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{len(results)} passed, {failed} failed")
    print(f"{'='*60}")

    if failed:
        print("\nFailed tests:")
        for r in results:
            if not r.passed:
                print(f"\n  [{r.test_id}]")
                for f in r.failures:
                    print(f"    - {f}")

    close()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    query_only = "--query" in sys.argv
    sys.exit(run_eval(verbose=verbose, query_only=query_only))
