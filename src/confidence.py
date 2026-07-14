"""Confidence scoring: turns retrieval + synthesis signals into a single 0-1
score and a High/Medium/Low label, so a clinician can tell at a glance how
much to trust an answer before reading the full citation trail.

Deliberately NOT an LLM call — same rationale as graph_db.py's multi-hop
traversal in ../graphrag-knowledge-assistant: confidence scoring should be
fast, deterministic, and auditable, not one more thing that can hallucinate.

The blend, in order of weight:
  1. Groundedness — if the Synthesizer flagged insufficient evidence, the
     score is capped low regardless of everything else. This is the most
     important signal: a fluent-sounding answer built on thin evidence is
     the failure mode this scorer exists to catch.
  2. Vector similarity — average cosine similarity of retrieved chunks.
     Higher means the retrieved text is more directly on-topic.
  3. Source agreement — how many DISTINCT source documents contributed
     evidence. One guideline saying something is weaker support than two or
     three independent sources agreeing.
  4. Graph corroboration — a small bonus if the entity graph *also* has
     structured facts backing the answer, not just similarity-matched text.
"""
from src.config import CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_MEDIUM_THRESHOLD
from src.state import Triple, VectorHit

GROUNDED_CAP_SCORE = 0.35  # ceiling applied when the Synthesizer flags insufficient evidence


def _distinct_sources(subgraph: list[Triple], vector_hits: list[VectorHit]) -> int:
    sources = {t.source_doc for t in subgraph if t.source_doc}
    sources |= {h.source_doc for h in vector_hits if h.source_doc}
    return len(sources)


def score_confidence(
    subgraph: list[Triple], vector_hits: list[VectorHit], grounded: bool
) -> tuple[float, str]:
    if not vector_hits and not subgraph:
        return 0.0, "Low"

    avg_similarity = (
        sum(h.score for h in vector_hits) / len(vector_hits) if vector_hits else 0.0
    )

    n_sources = _distinct_sources(subgraph, vector_hits)
    source_agreement = min(n_sources / 3, 1.0)  # 3+ independent sources = full credit

    graph_bonus = 0.1 if subgraph else 0.0

    score = (0.55 * avg_similarity) + (0.35 * source_agreement) + graph_bonus
    score = min(score, 1.0)

    if not grounded:
        score = min(score, GROUNDED_CAP_SCORE)

    if score >= CONFIDENCE_HIGH_THRESHOLD:
        label = "High"
    elif score >= CONFIDENCE_MEDIUM_THRESHOLD:
        label = "Medium"
    else:
        label = "Low"

    return round(score, 2), label
