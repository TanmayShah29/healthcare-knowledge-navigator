"""Ingestion pipeline: raw document text -> chunks -> (embeddings + extracted
triples) -> Neo4j. Each chunk is written to both halves of the hybrid store:
as a Chunk node with a vector embedding, and as source material for the
Extractor agent's structured triples.

Chunking is paragraph-aware (same approach as
../graphrag-knowledge-assistant/src/ingest.py) with lightweight section
tracking: a short line that looks like a heading (markdown `#`/`##` or an
ALL-CAPS line) updates the "current section" label attached to subsequent
chunks, so citations can point to e.g. "Dosing" or "Contraindications"
within a document, not just the document as a whole.
"""
import os
import re
import uuid
from src.config import CHUNK_SIZE
from src.state import IngestResult, Chunk
from src.embeddings import embed_texts
from src.agents.extractor import extract_triples
from src.graph_db import upsert_triples, upsert_chunk, ensure_constraints

_HEADING_RE = re.compile(r"^(#{1,6}\s+.+|[A-Z][A-Z0-9 /&,\-]{3,60})$")


def _is_heading(line: str) -> bool:
    line = line.strip()
    return bool(line) and len(line) <= 80 and bool(_HEADING_RE.match(line))


def _clean_heading(line: str) -> str:
    return line.strip().lstrip("#").strip()


def _chunk_with_sections(text: str, chunk_size: int) -> list[tuple[str, str]]:
    """Returns list of (chunk_text, section_label) tuples. Paragraphs
    accumulate into a chunk until adding the next would exceed chunk_size, so
    entities aren't split mid-sentence as often as a naive char-count split."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[tuple[str, str]] = []
    current, current_section = "", ""

    for para in paragraphs:
        if _is_heading(para):
            current_section = _clean_heading(para)
            continue

        if current and len(current) + len(para) > chunk_size:
            chunks.append((current, current_section))
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para

    if current:
        chunks.append((current, current_section))

    return chunks or [(text, "")]


def ingest_text(text: str, source_doc: str, doc_type: str = "clinical_guideline") -> IngestResult:
    ensure_constraints()
    result = IngestResult(source_doc=source_doc, doc_type=doc_type)

    chunk_pairs = _chunk_with_sections(text, CHUNK_SIZE)
    chunk_texts = [c for c, _ in chunk_pairs]

    try:
        vectors = embed_texts(chunk_texts)
    except Exception as e:
        result.errors.append(f"Embedding failed for document: {e}")
        vectors = [None] * len(chunk_texts)

    for (chunk_text, section), vector in zip(chunk_pairs, vectors):
        try:
            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                text=chunk_text,
                source_doc=source_doc,
                doc_type=doc_type,
                section=section,
                embedding=vector,
            )
            upsert_chunk(chunk)

            triples = extract_triples(chunk_text, source_doc, doc_type)
            result.triples_extracted.extend(triples)
            result.triples_loaded += upsert_triples(triples)
            result.chunks_processed += 1
        except Exception as e:  # keep going on a single bad chunk
            result.errors.append(f"Chunk failed: {e}")

    return result


def ingest_file(path: str, doc_type: str = "clinical_guideline") -> IngestResult:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return ingest_text(text, source_doc=os.path.basename(path), doc_type=doc_type)
