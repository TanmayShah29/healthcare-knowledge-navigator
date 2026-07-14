"""Extractor Agent — turns a raw clinical text chunk into a list of
(subject, predicate, object) triples for loading into the knowledge graph,
plus structured dosage and contraindication facts for machine-queryable
clinical safety data.

Tuned for clinical documents specifically (vs. the domain-agnostic extractor
in ../graphrag-knowledge-assistant): the entity type vocabulary is clinical,
and the prompt is explicit about not inventing dosages, indications, or
contraindications that aren't stated in the source text — hallucinated
structured facts are far more dangerous here than in a general-knowledge graph.
"""
import json
import re
from src.llm import get_llm
from src.state import Triple, DosageFact, ContraindicationFact

_llm = get_llm(temperature=0.1)

SYSTEM_PROMPT = """You are a clinical information extraction system. Read the given \
text — drawn from a clinical guideline, research paper, or treatment protocol — and \
extract factual relationships between clinical entities.

Output ONLY a JSON object with two keys, nothing else:
{
  "triples": [
    {
      "subject": "<entity name, canonical form>",
      "subject_type": "<one of: Condition, Drug, Symptom, Treatment, Procedure, Guideline, RiskFactor>",
      "predicate": "<short verb phrase, e.g. \\"treats\\", \\"contraindicated with\\", \\"first-line for\\", \\"monitor for\\", \\"risk factor for\\">",
      "object": "<entity name, canonical form>",
      "object_type": "<one of: Condition, Drug, Symptom, Treatment, Procedure, Guideline, RiskFactor>"
    }
  ],
  "dosages": [
    {
      "drug": "<drug name>",
      "dose": "<dose, e.g. \\"500mg\\">",
      "frequency": "<frequency, e.g. \\"once or twice daily\\", or empty string if not stated>",
      "route": "<route, e.g. \\"oral\\", or empty string if not stated>",
      "notes": "<any additional context, or empty string>"
    }
  ],
  "contraindications": [
    {
      "drug": "<drug name>",
      "condition": "<condition or factor, e.g. \\"severe renal impairment\\">",
      "reason": "<brief reason if stated, or empty string>",
      "severity": "<absolute | relative | precaution, or empty string if not stated>"
    }
  ]
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
- For dosages: only extract if the text explicitly mentions a specific dose value (number + unit).
- For contraindications: only extract if the text explicitly states a drug is contraindicated
  with a condition or factor. Include severity if the text uses words like "absolute",
  "relative", or "precaution".
- If the text contains no clinical relationships of a given type, use an empty array for
  that key.
"""


def _extract_json_object(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else '{"triples":[],"dosages":[],"contraindications":[]}'


def extract_triples(chunk_text: str, source_doc: str, doc_type: str = "") -> list[Triple]:
    """Extract general clinical triples from a text chunk."""
    response = _llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("user", f"Text:\n{chunk_text}"),
        ]
    )
    raw = _extract_json_object(response.content)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    triples = []
    for row in data.get("triples", []):
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
            continue

    return triples


def extract_dosages(chunk_text: str, source_doc: str, doc_type: str = "") -> list[DosageFact]:
    """Extract structured dosage facts from a text chunk."""
    response = _llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("user", f"Text:\n{chunk_text}"),
        ]
    )
    raw = _extract_json_object(response.content)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    dosages = []
    for row in data.get("dosages", []):
        try:
            dosages.append(
                DosageFact(
                    drug=row["drug"].strip(),
                    dose=row["dose"].strip(),
                    frequency=row.get("frequency", "").strip(),
                    route=row.get("route", "").strip(),
                    notes=row.get("notes", "").strip(),
                    source_doc=source_doc,
                    doc_type=doc_type,
                )
            )
        except (KeyError, AttributeError):
            continue

    return dosages


def extract_contraindications(chunk_text: str, source_doc: str, doc_type: str = "") -> list[ContraindicationFact]:
    """Extract structured contraindication facts from a text chunk."""
    response = _llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("user", f"Text:\n{chunk_text}"),
        ]
    )
    raw = _extract_json_object(response.content)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    contras = []
    for row in data.get("contraindications", []):
        try:
            contras.append(
                ContraindicationFact(
                    drug=row["drug"].strip(),
                    condition=row["condition"].strip(),
                    reason=row.get("reason", "").strip(),
                    severity=row.get("severity", "").strip(),
                    source_doc=source_doc,
                    doc_type=doc_type,
                )
            )
        except (KeyError, AttributeError):
            continue

    return contras


def extract_all(chunk_text: str, source_doc: str, doc_type: str = "") -> tuple[list[Triple], list[DosageFact], list[ContraindicationFact]]:
    """Single LLM call that extracts triples, dosages, and contraindications
    together — more efficient than three separate calls."""
    response = _llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("user", f"Text:\n{chunk_text}"),
        ]
    )
    raw = _extract_json_object(response.content)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [], [], []

    triples = []
    for row in data.get("triples", []):
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
            continue

    dosages = []
    for row in data.get("dosages", []):
        try:
            dosages.append(
                DosageFact(
                    drug=row["drug"].strip(),
                    dose=row["dose"].strip(),
                    frequency=row.get("frequency", "").strip(),
                    route=row.get("route", "").strip(),
                    notes=row.get("notes", "").strip(),
                    source_doc=source_doc,
                    doc_type=doc_type,
                )
            )
        except (KeyError, AttributeError):
            continue

    contras = []
    for row in data.get("contraindications", []):
        try:
            contras.append(
                ContraindicationFact(
                    drug=row["drug"].strip(),
                    condition=row["condition"].strip(),
                    reason=row.get("reason", "").strip(),
                    severity=row.get("severity", "").strip(),
                    source_doc=source_doc,
                    doc_type=doc_type,
                )
            )
        except (KeyError, AttributeError):
            continue

    return triples, dosages, contras
