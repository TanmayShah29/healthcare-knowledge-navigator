"""Provider-agnostic chat model factory — identical pattern to
../graphrag-knowledge-assistant/src/llm.py and ../multi-agent-code-review/src/llm.py.
Every agent gets its LLM client from get_llm() instead of importing a provider SDK
directly, so swapping providers is a .env change (LLM_PROVIDER + AGENT_MODEL + matching
key).
"""
from langchain.chat_models import init_chat_model
from src.config import LLM_PROVIDER, AGENT_MODEL, LLM_API_KEY


def get_llm(temperature: float = 0.2):
    kwargs = {"temperature": temperature}
    if LLM_API_KEY:
        kwargs["api_key"] = LLM_API_KEY
    return init_chat_model(AGENT_MODEL, model_provider=LLM_PROVIDER, **kwargs)
