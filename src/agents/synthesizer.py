"""Synthesizer Agent — turns retrieved evidence (graph facts + vector-matched
passages + structured dosage/contraindication facts) plus the original question
into a grounded, cited answer.

Two things matter more here than in a general-knowledge assistant:
1. The answer must be traceable to specific source documents (citations), so a
   clinician can pull the original guideline and verify before acting on it.
2. The agent must say clearly when evidence is thin or absent, rather than
   filling gaps from the model's own training data — src/confidence.py uses
   the "insufficient evidence" flag this agent emits to force a Low
   confidence score regardless of retrieval scores.
"""
from src.llm import get_llm
from src.state import Triple, VectorHit, DosageFact, ContraindicationFact

_llm = get_llm(temperature=0.2)

INSUFFICIENT_EVIDENCE_MARKER = "INSUFFICIENT EVIDENCE RETRIEVED"

SYSTEM_PROMPT = f"""You are a clinical knowledge assistant helping healthcare \
professionals quickly find evidence-based answers in clinical guidelines, research \
papers, and treatment protocols. You are a decision-support aid, not a replacement for \
professional clinical judgment.

You will be given retrieved evidence: (a) structured facts as (subject) -[predicate]-> \
(object) triples, (b) structured dosage facts with dose/frequency/route, (c) structured \
contraindication facts with severity and reason, and (d) passages from source documents, \
each tagged with its source.

Rules:
- Base your answer ONLY on the given evidence. Do not add information from your own \
general medical knowledge, even if you're confident it's true — this tool exists so \
answers are traceable back to a specific source document, not to the model's training.
- Every claim in your answer must be attributable to at least one piece of retrieved \
evidence. After each claim (or small group of related claims), cite the source(s) in \
brackets, e.g. "Metformin is first-line therapy for T2DM [ADA Standards of Care 2024, \
Guideline]."
- If the evidence is NOT sufficient to answer the question confidently, start your \
answer with the exact line "{INSUFFICIENT_EVIDENCE_MARKER}" on its own, then explain \
what's missing and what evidence WAS found (if any), rather than guessing.
- Never state a specific dosage, contraindication, or treatment recommendation that \
isn't explicitly present in the retrieved evidence.
- When dosage facts are provided, use them directly — they are machine-extracted and \
more reliable than interpreting free-text for dosing information.
- End every answer with: "This is decision support only — verify against current \
full-text sources and use clinical judgment for the individual patient."
"""


def _format_evidence(
    subgraph: list[Triple],
    vector_hits: list[VectorHit],
    dosages: list[DosageFact] = None,
    contraindications: list[ContraindicationFact] = None,
) -> str:
    parts = []

    if subgraph:
        facts = "\n".join(f"- {t.as_fact_string()}  [{t.citation()}]" for t in subgraph)
        parts.append(f"Structured facts:\n{facts}")
    else:
        parts.append("Structured facts: (none retrieved)")

    dosages = dosages or []
    if dosages:
        dose_lines = "\n".join(f"- {d.as_fact_string()}  [{d.citation()}]" for d in dosages)
        parts.append(f"Dosage facts:\n{dose_lines}")
    else:
        parts.append("Dosage facts: (none retrieved)")

    contraindications = contraindications or []
    if contraindications:
        contra_lines = "\n".join(f"- {c.as_fact_string()}  [{c.citation()}]" for c in contraindications)
        parts.append(f"Contraindication facts:\n{contra_lines}")
    else:
        parts.append("Contraindication facts: (none retrieved)")

    if vector_hits:
        passages = "\n\n".join(
            f"[{h.citation()}] (similarity: {h.score:.2f})\n{h.text}" for h in vector_hits
        )
        parts.append(f"Retrieved passages:\n{passages}")
    else:
        parts.append("Retrieved passages: (none retrieved)")

    return "\n\n".join(parts)


def synthesize_answer(
    question: str,
    subgraph: list[Triple],
    vector_hits: list[VectorHit],
    dosages: list[DosageFact] = None,
    contraindications: list[ContraindicationFact] = None,
) -> tuple[str, bool]:
    """Returns (answer_text, grounded) where grounded=False means the agent
    flagged insufficient evidence."""
    evidence_block = _format_evidence(subgraph, vector_hits, dosages, contraindications)

    response = _llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("user", f"Question: {question}\n\nRetrieved evidence:\n{evidence_block}"),
        ]
    )
    answer = response.content
    grounded = not answer.strip().startswith(INSUFFICIENT_EVIDENCE_MARKER)
    return answer, grounded
