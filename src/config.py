"""Central config for the Healthcare Knowledge Navigator. Reads from environment / .env.

LLM and embeddings layers are both provider-agnostic (same pattern as
../graphrag-knowledge-assistant and ../multi-agent-code-review) — every agent gets its
model via src/llm.py's get_llm(), and all vector embedding goes through
src/embeddings.py's get_embeddings(). See README.md for the full provider table.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Neo4j (hybrid store: entity graph + vector index over chunks) ────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NEO4J_PASSWORD:
    raise RuntimeError(
        "NEO4J_PASSWORD is not set. Copy .env.example to .env and fill in your "
        "Neo4j connection details (Neo4j Desktop or Aura free tier both work). "
        "Note: the vector index feature requires Neo4j 5.11+ (Aura free tier "
        "and current Desktop versions both qualify)."
    )

# ── LLM provider (same scheme as graphrag-knowledge-assistant) ───────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google_genai")
AGENT_MODEL = os.getenv("AGENT_MODEL", "gemini-2.5-flash")

_PROVIDER_KEY_ENV_VARS = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "google_genai": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "google_vertexai": ["GOOGLE_API_KEY"],
    "groq": ["GROQ_API_KEY"],
    "mistralai": ["MISTRAL_API_KEY"],
    "xai": ["XAI_API_KEY"],
    "fireworks": ["FIREWORKS_API_KEY"],
    "together": ["TOGETHER_API_KEY"],
    "cohere": ["COHERE_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "ollama": [],
    "bedrock": [],
}
_KEYLESS_PROVIDERS = {"ollama", "bedrock"}


def _resolve_api_key(provider: str):
    override = os.getenv("LLM_API_KEY")
    if override:
        return override
    for var in _PROVIDER_KEY_ENV_VARS.get(provider, []):
        value = os.getenv(var)
        if value:
            return value
    return None


LLM_API_KEY = _resolve_api_key(LLM_PROVIDER)

if LLM_PROVIDER not in _KEYLESS_PROVIDERS and not LLM_API_KEY:
    expected_vars = _PROVIDER_KEY_ENV_VARS.get(LLM_PROVIDER) or ["LLM_API_KEY"]
    raise RuntimeError(
        f"No API key found for LLM_PROVIDER={LLM_PROVIDER!r}. Set one of "
        f"{expected_vars} in your .env (or the universal LLM_API_KEY)."
    )

# ── Embeddings provider (defaults to the same provider as the LLM, but can ──
# be split out independently — e.g. Gemini for chat, OpenAI for embeddings) ─
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", LLM_PROVIDER)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")

# Must match the actual output dimensionality of EMBEDDING_MODEL — used to size
# the Neo4j vector index. Common values: 768 (Gemini text-embedding-004),
# 1536 (OpenAI text-embedding-3-small), 3072 (OpenAI text-embedding-3-large).
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))

EMBEDDING_API_KEY = (
    LLM_API_KEY if EMBEDDING_PROVIDER == LLM_PROVIDER else _resolve_api_key(EMBEDDING_PROVIDER)
)

if EMBEDDING_PROVIDER not in _KEYLESS_PROVIDERS and not EMBEDDING_API_KEY:
    expected_vars = _PROVIDER_KEY_ENV_VARS.get(EMBEDDING_PROVIDER) or ["LLM_API_KEY"]
    raise RuntimeError(
        f"No API key found for EMBEDDING_PROVIDER={EMBEDDING_PROVIDER!r}. Set one of "
        f"{expected_vars} in your .env (or the universal LLM_API_KEY)."
    )

# ── Retrieval tuning (hybrid: vector similarity + graph traversal) ───────
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "5"))     # vector-search hits per query
MAX_HOPS = int(os.getenv("MAX_HOPS", "2"))              # graph traversal depth
MAX_TRIPLES = int(os.getenv("MAX_TRIPLES", "40"))       # cap on facts per query

# ── Ingestion tuning ───────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))        # chars per chunk before embedding

# ── Confidence scoring thresholds ─────────────────────────────────────────
# Final score is a 0-1 blend of vector similarity, source agreement, and graph
# corroboration (see src/confidence.py). These thresholds map it to a label.
CONFIDENCE_HIGH_THRESHOLD = float(os.getenv("CONFIDENCE_HIGH_THRESHOLD", "0.75"))
CONFIDENCE_MEDIUM_THRESHOLD = float(os.getenv("CONFIDENCE_MEDIUM_THRESHOLD", "0.5"))
