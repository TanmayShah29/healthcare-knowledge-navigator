"""Unit tests for src.confidence — deterministic confidence scoring."""

from src.confidence import score_confidence, _recency_penalty, _parse_year
from src.state import Triple, VectorHit


class TestRecencyPenalty:
    def test_recent_document_no_penalty(self):
        penalty = _recency_penalty("2024-01-01")
        assert penalty == 0.0

    def test_old_document_has_penalty(self):
        penalty = _recency_penalty("2015-01-01")
        assert penalty > 0.0

    def test_empty_date_no_penalty(self):
        penalty = _recency_penalty("")
        assert penalty == 0.0

    def test_invalid_date_no_penalty(self):
        penalty = _recency_penalty("not-a-date")
        assert penalty == 0.0


class TestParseYear:
    def test_full_date(self):
        assert _parse_year("2024-01-15") == 2024

    def test_year_month(self):
        assert _parse_year("2024-06") == 2024

    def test_year_only(self):
        assert _parse_year("2024") == 2024

    def test_empty_string(self):
        assert _parse_year("") is None

    def test_invalid(self):
        assert _parse_year("abc") is None


class TestScoreConfidence:
    def test_no_evidence(self):
        score, label = score_confidence(subgraph=[], vector_hits=[], grounded=True)
        assert score == 0.0
        assert label == "Low"

    def test_with_vector_hits(self):
        hits = [
            VectorHit(chunk_id="1", text="a", source_doc="doc1", doc_type="guideline",
                      section="intro", score=0.9, publication_date="2024-01-01"),
        ]
        score, label = score_confidence(subgraph=[], vector_hits=hits, grounded=True)
        assert score > 0.0
        assert label in ("Low", "Medium", "High")

    def test_with_subgraph(self):
        triples = [
            Triple(subject="A", subject_type="Drug", predicate="treats",
                   object="B", object_type="Condition", source_doc="doc1",
                   doc_type="guideline"),
        ]
        hits = [
            VectorHit(chunk_id="1", text="a", source_doc="doc1", doc_type="guideline",
                      section="intro", score=0.8, publication_date="2024-01-01"),
        ]
        score, label = score_confidence(subgraph=triples, vector_hits=hits, grounded=True)
        assert score > 0.0

    def test_ungrounded_caps_score(self):
        hits = [
            VectorHit(chunk_id="1", text="a", source_doc="doc1", doc_type="guideline",
                      section="intro", score=0.95, publication_date="2024-01-01"),
        ]
        score_grounded, _ = score_confidence(subgraph=[], vector_hits=hits, grounded=True)
        score_ungrounded, _ = score_confidence(subgraph=[], vector_hits=hits, grounded=False)
        assert score_ungrounded <= score_grounded

    def test_score_range(self):
        hits = [
            VectorHit(chunk_id="1", text="a", source_doc="doc1", doc_type="guideline",
                      section="intro", score=0.9, publication_date="2024-01-01"),
        ]
        score, _ = score_confidence(subgraph=[], vector_hits=hits, grounded=True)
        assert 0.0 <= score <= 1.0
