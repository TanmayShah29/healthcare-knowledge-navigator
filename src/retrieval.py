"""Query-time hybrid retrieval pipeline:

  question ─▶ embed question ─▶ vector search over Chunks ─▶ vector_hits
            ─▶ link entities ─▶ fuzzy-match against graph ─▶ multi-hop traversal ─▶ subgraph
                                   └─▶ dosage/contraindication lookup

Vector search and graph traversal run independently and their results are
combined by the Synthesizer — this is what makes it "hybrid" rather than
picking one retrieval strategy. Vector search catches nuanced passages that
don't reduce cleanly to a triple (dosing caveats, patient-population
qualifiers); graph traversal catches multi-hop chains that pure similarity
search would miss (e.g. drug -> treats -> condition -> contraindicated-with
-> drug, connecting two facts stated in different documents).
"""
from src.config import MAX_HOPS
from src.state import RetrievalResult
from src.embeddings import embed_text
from src.agents.entity_linker import link_entities
from src.graph_db import (
    find_entities_like, multi_hop_neighbors, vector_search,
    query_dosages, query_contraindications,
)


def retrieve(question: str) -> RetrievalResult:
    result = RetrievalResult(question=question)

    # Vector similarity leg
    question_embedding = embed_text(question)
    result.vector_hits = vector_search(question_embedding)

    # Graph traversal leg
    candidate_names = link_entities(question)
    seed_names = find_entities_like(candidate_names)
    result.linked_entities = seed_names
    if seed_names:
        result.subgraph = multi_hop_neighbors(seed_names, max_hops=MAX_HOPS)
        result.dosages = query_dosages(seed_names)
        result.contraindications = query_contraindications(seed_names)

    return result
