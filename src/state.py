"""Shared data structures for the ingestion and query pipelines.

Plain dataclasses rather than a LangGraph AgentState — same rationale as
../graphrag-knowledge-assistant/src/state.py: ingest and query are linear
pipelines, not retry loops, so a full state machine is more machinery than the
problem needs.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Triple:
    """One (subject) -[predicate]-> (object) clinical fact extracted from a
    document — e.g. (Metformin) -[first-line treatment for]-> (Type 2 Diabetes)."""
    subject: str
    subject_type: str          # Condition | Drug | Symptom | Treatment | Procedure | Guideline
    predicate: str
    object: str
    object_type: str
    source_doc: str = ""
    doc_type: str = ""         # clinical_guideline | research_paper | treatment_protocol

    def as_fact_string(self) -> str:
        return f"({self.subject}) -[{self.predicate}]-> ({self.object})"

    def citation(self) -> str:
        label = self.doc_type.replace("_", " ") if self.doc_type else "source"
        return f"{self.source_doc} ({label})"


@dataclass
class Chunk:
    """One embedded text chunk from an ingested document, stored as a Neo4j
    node with a vector index over `embedding` for similarity search."""
    chunk_id: str
    text: str
    source_doc: str
    doc_type: str = ""
    section: str = ""
    embedding: Optional[List[float]] = None


@dataclass
class VectorHit:
    """A chunk retrieved via vector similarity search at query time."""
    chunk_id: str
    text: str
    source_doc: str
    doc_type: str
    section: str
    score: float                # cosine similarity, 0-1 (higher = more similar)

    def citation(self) -> str:
        label = self.doc_type.replace("_", " ") if self.doc_type else "source"
        loc = f", {self.section}" if self.section else ""
        return f"{self.source_doc} ({label}{loc})"


@dataclass
class IngestResult:
    source_doc: str
    doc_type: str = ""
    chunks_processed: int = 0
    triples_extracted: List[Triple] = field(default_factory=list)
    triples_loaded: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    question: str
    linked_entities: List[str] = field(default_factory=list)
    subgraph: List[Triple] = field(default_factory=list)         # graph hits
    vector_hits: List[VectorHit] = field(default_factory=list)   # similarity hits
    answer: str = ""
    citations: List[str] = field(default_factory=list)
    grounded: bool = True       # False if the Synthesizer had to say "not enough evidence"
    confidence_score: float = 0.0
    confidence_label: str = "Low"
