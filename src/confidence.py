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
  5. Recency — newer documents get a small boost, older ones a small penalty.
     A 2015 guideline shouldn't outweigh a 2024 update. This is a soft signal,
     not a hard cutoff — an older source is still valuable, just slightly
     less authoritative when a newer one exists.
"""
import datetime
from src.config import CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_MEDIUM_THRESHOLD
from src.state import Triple, VectorHit

GROUNDED_CAP_SCORE = 0.35  # ceiling applied when the Synthesizer flags insufficient evidence

# Recency: documents within this many years of today get no penalty;
# documents older than this get a linear penalty up to RECENCY_MAX_PENALTY.
RECENCY_FRESH_YEARS = 3
RECENCY_MAX_PENALTY = 0.1  # maximum penalty for very old documents

_TODAY = datetime.date.today()


def _parse_year(date_str: str) -> int | None:
    """Extract a year from an ISO-ish date string (YYYY-MM-DD, YYYY-MM, or YYYY)."""
    if not date_str:
        return None
    try:
        return int(date_str[:4])
    except (ValueError, IndexError):
        return None


def _recency_penalty(date_str: str) -> float:
    """Return a penalty (0.0 = no penalty, positive = penalty) based on how
    old the document is. Newer documents get 0, older ones get up to
    RECENCY_MAX_PENALTY."""
    year = _parse_year(date_str)
    if year is None:
        return 0.0  # no date = no penalty (don't penalize unknown)
    age = _TODAY.year - year
    if age <= RECENCY_FRESH_YEARS:
        return 0.0
    excess = age - RECENCY_FRESH_YEARS
    # Linear ramp, capped at RECENCY_MAX_PENALTY
    return min(excess * 0.02, RECENCY_MAX_PENALTY)


def _distinct_sources(subgraph: list[Triple], vector_hits: list[VectorHit]) -> int:
    sources = {t.source_doc for t in subgraph if t.source_doc}
    sources |= {h.source_doc for h in vector_hits if h.source_doc}
    return len(sources)


def _avg_recency_penalty(vector_hits: list[VectorHit]) -> float:
    """Average recency penalty across all vector hits."""
    if not vector_hits:
        return 0.0
    penalties = [_recency_penalty(h.publication_date) for h in vector_hits]
    return sum(penalties) / len(penalties)


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

    recency_penalty = _avg_recency_penalty(vector_hits)

    score = (0.55 * avg_similarity) + (0.35 * source_agreement) + graph_bonus - recency_penalty
    score = max(score, 0.0)
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
