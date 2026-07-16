"""Unit tests for src.state — clinical dataclasses."""

from src.state import Triple, DosageFact, ContraindicationFact, Chunk, VectorHit


class TestTriple:
    def test_creation(self):
        t = Triple(
            subject="Metformin",
            subject_type="Drug",
            predicate="treats",
            object="Type 2 Diabetes",
            object_type="Condition",
            source_doc="guideline.txt",
            doc_type="clinical_guideline",
        )
        assert t.subject == "Metformin"
        assert t.predicate == "treats"

    def test_as_fact_string(self):
        t = Triple(
            subject="Metformin",
            subject_type="Drug",
            predicate="treats",
            object="Type 2 Diabetes",
            object_type="Condition",
            source_doc="guideline.txt",
            doc_type="clinical_guideline",
        )
        fact = t.as_fact_string()
        assert "Metformin" in fact
        assert "Type 2 Diabetes" in fact

    def test_citation(self):
        t = Triple(
            subject="A",
            subject_type="Drug",
            predicate="treats",
            object="B",
            object_type="Condition",
            source_doc="test.txt",
            doc_type="research_paper",
        )
        citation = t.citation()
        assert "test.txt" in citation
        assert "research paper" in citation


class TestDosageFact:
    def test_creation(self):
        d = DosageFact(
            drug="Metformin",
            dose="500mg",
            frequency="twice daily",
            route="oral",
            source_doc="guideline.txt",
            doc_type="clinical_guideline",
        )
        assert d.drug == "Metformin"
        assert d.dose == "500mg"

    def test_as_fact_string(self):
        d = DosageFact(
            drug="Metformin",
            dose="500mg",
            frequency="twice daily",
            route="oral",
            source_doc="test.txt",
            doc_type="clinical_guideline",
        )
        fact = d.as_fact_string()
        assert "Metformin" in fact
        assert "500mg" in fact


class TestContraindicationFact:
    def test_creation(self):
        c = ContraindicationFact(
            drug="Metformin",
            condition="Renal impairment",
            severity="absolute",
            source_doc="guideline.txt",
            doc_type="clinical_guideline",
        )
        assert c.drug == "Metformin"
        assert c.condition == "Renal impairment"

    def test_as_fact_string(self):
        c = ContraindicationFact(
            drug="Metformin",
            condition="Renal impairment",
            severity="absolute",
            reason="risk of lactic acidosis",
            source_doc="test.txt",
            doc_type="clinical_guideline",
        )
        fact = c.as_fact_string()
        assert "Metformin" in fact
        assert "Renal impairment" in fact
        assert "absolute" in fact


class TestChunk:
    def test_creation(self):
        chunk = Chunk(
            chunk_id="c1",
            text="Some clinical text",
            source_doc="guideline.txt",
            doc_type="clinical_guideline",
            section="Treatment",
        )
        assert chunk.chunk_id == "c1"
        assert chunk.text == "Some clinical text"


class TestVectorHit:
    def test_creation(self):
        hit = VectorHit(
            chunk_id="1",
            text="a",
            source_doc="doc1",
            doc_type="guideline",
            section="intro",
            score=0.9,
        )
        assert hit.score == 0.9

    def test_citation(self):
        hit = VectorHit(
            chunk_id="1",
            text="a",
            source_doc="doc1",
            doc_type="clinical_guideline",
            section="Treatment",
            score=0.9,
            publication_date="2024-01-01",
        )
        citation = hit.citation()
        assert "doc1" in citation
        assert "Treatment" in citation
