"""Extractor Agent — turns a raw clinical text chunk into a list of
(subject, predicate, object) triples for loading into the knowledge graph.

Tuned for clinical documents specifically (vs. the domain-agnostic extractor
in ../graphrag-knowledge-assistant): the entity type vocabulary is clinical,
and the prompt is explicit about not inventing dosages, indications, or
contraindications that aren't stated in the source text — hallucinated
structured facts are far more dangerous here than in a general-knowledge graph.
"""
import json
import re
from src.llm import get_llm
from src.state import Triple

_llm = get_llm(temperature=0.1)

SYSTEM_PROMPT = """You are a clinical information extraction system. Read the given \
text — drawn from a clinical guideline, research paper, or treatment protocol — and \
extract factual relationships between clinical entities.

Output ONLY a JSON array, nothing else, where each element is:
{
  "subject": "<entity name, canonical form>",
  "subject_type": "<one of: Condition, Drug, Symptom, Treatment, Procedure, Guideline, RiskFactor>",
  "predicate": "<short verb phrase, e.g. \\"treats\\", \\"contraindicated with\\", \\"first-line for\\", \\"recommended dosage\\", \\"monitor for\\", \\"risk factor for\\">",
  "object": "<entity name, canonical form>",
  "object_type": "<one of: Condition, Drug, Symptom, Treatment, Procedure, Guideline, RiskFactor>"
}

Rules:
- Use canonical, consistent names for the same entity across triples (e.g. always
  "Type 2 Diabetes Mellitus", not sometimes "T2DM" and sometimes "diabetes") so the
  graph merges references to the same entity into one node.
- ONLY extract relationships the text actually states — this is clinical information
  that may inform real decisions, so do not infer, generalize, or fill in from general
  medical knowledge. If the text doesn't explicitly state a dosage, contraindication,
  or indication, don't extract one.
- Keep predicates short and consistent (prefer "treats" over "is used in the treatment of").
- If the text contains no clear clinical relationships, output an empty array: []
"""


def _extract_json_array(text: str) -> str:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    return match.group(0) if match else "[]"


def extract_triples(chunk_text: str, source_doc: str, doc_type: str = "") -> list[Triple]:
    response = _llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("user", f"Text:\n{chunk_text}"),
        ]
    )
    raw = _extract_json_array(response.content)

    try:
        rows = json.loads(raw)
    except json.JSONDecodeError:
        return []

    triples = []
    for row in rows:
        try:
            triples.append(
                Triple(
                    subject=row["subject"].strip(),
                    subject_type=row.get("subject_type", "Unknown").strip(),
                    predicate=row["predicate"].strip(),
                    object=row["object"].strip(),
                    object_type=row.get("object_type", "Unknown").strip(),
                    source_doc=source_doc,
                    doc_type=doc_type,
                )
            )
        except (KeyError, AttributeError):
            continue  # skip malformed rows rather than failing the whole batch

    return triples
