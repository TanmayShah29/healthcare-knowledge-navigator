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
    subject_type: str          # Condition | Drug | Symptom | Treatment | Procedure | Guideline | RiskFactor
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
class DosageFact:
    """A structured dosage fact: (Drug) -[HAS_DOSAGE]-> metadata. Stored as
    typed relationship properties so dosages are machine-queryable (e.g. "list
    all drugs with starting dose 500mg") rather than buried in free-text
    predicates."""
    drug: str
    dose: str
    frequency: str = ""
    route: str = ""
    notes: str = ""
    source_doc: str = ""
    doc_type: str = ""

    def as_fact_string(self) -> str:
        parts = [f"({self.drug}) -[HAS_DOSAGE]-> dose: {self.dose}"]
        if self.frequency:
            parts.append(f"frequency: {self.frequency}")
        if self.route:
            parts.append(f"route: {self.route}")
        if self.notes:
            parts.append(f"notes: {self.notes}")
        return ", ".join(parts)

    def citation(self) -> str:
        label = self.doc_type.replace("_", " ") if self.doc_type else "source"
        return f"{self.source_doc} ({label})"


@dataclass
class ContraindicationFact:
    """A structured contraindication: (Drug) -[CONTRAINDICATED_FOR]-> (Condition)
    with a reason. Stored as typed properties so clinicians can query "list all
    drugs contraindicated with renal impairment" safely from the graph."""
    drug: str
    condition: str
    reason: str = ""
    severity: str = ""         # absolute | relative | precaution
    source_doc: str = ""
    doc_type: str = ""

    def as_fact_string(self) -> str:
        severity_str = f" [{self.severity}]" if self.severity else ""
        reason_str = f" — {self.reason}" if self.reason else ""
        return f"({self.drug}) -[CONTRAINDICATED_FOR{severity_str}]-> ({self.condition}){reason_str}"

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
    publication_date: str = ""  # ISO format YYYY-MM-DD or YYYY-MM or YYYY


@dataclass
class VectorHit:
    """A chunk retrieved via vector similarity search at query time."""
    chunk_id: str
    text: str
    source_doc: str
    doc_type: str
    section: str
    score: float                # cosine similarity, 0-1 (higher = more similar)
    publication_date: str = ""

    def citation(self) -> str:
        label = self.doc_type.replace("_", " ") if self.doc_type else "source"
        loc = f", {self.section}" if self.section else ""
        date_str = f" [{self.publication_date}]" if self.publication_date else ""
        return f"{self.source_doc} ({label}{loc}){date_str}"


@dataclass
class IngestResult:
    source_doc: str
    doc_type: str = ""
    publication_date: str = ""
    chunks_processed: int = 0
    triples_extracted: List[Triple] = field(default_factory=list)
    dosages_extracted: List[DosageFact] = field(default_factory=list)
    contraindications_extracted: List[ContraindicationFact] = field(default_factory=list)
    triples_loaded: int = 0
    dosages_loaded: int = 0
    contraindications_loaded: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    question: str
    linked_entities: List[str] = field(default_factory=list)
    subgraph: List[Triple] = field(default_factory=list)         # graph hits
    dosages: List[DosageFact] = field(default_factory=list)      # dosage facts from graph
    contraindications: List[ContraindicationFact] = field(default_factory=list)
    vector_hits: List[VectorHit] = field(default_factory=list)   # similarity hits
    answer: str = ""
    citations: List[str] = field(default_factory=list)
    grounded: bool = True       # False if the Synthesizer had to say "not enough evidence"
    confidence_score: float = 0.0
    confidence_label: str = "Low"
