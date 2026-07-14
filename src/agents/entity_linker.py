"""Entity Linker Agent — pulls candidate clinical entity names/aliases out of
a user's question, which retrieval.py then fuzzy-matches against real graph
node names to find seed entities to traverse from.
"""
import json
import re
from src.llm import get_llm

_llm = get_llm(temperature=0.0)

SYSTEM_PROMPT = """Read the user's clinical question and list the named entities it \
refers to — conditions, drugs, symptoms, treatments, procedures, or risk factors that \
might exist as nodes in a clinical knowledge graph. Expand common medical abbreviations \
to their full form when unambiguous (e.g. "T2DM" -> "Type 2 Diabetes Mellitus", "MI" -> \
"Myocardial Infarction"), and include both the abbreviation and expansion if the \
question uses the abbreviation, so either form can match.

Output ONLY a JSON array of strings, nothing else, e.g.:
["Type 2 Diabetes Mellitus", "Metformin"]

If the question doesn't clearly reference any named clinical entity, output an empty
array: []
"""


def _extract_json_array(text: str) -> str:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    return match.group(0) if match else "[]"


def link_entities(question: str) -> list[str]:
    response = _llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("user", question),
        ]
    )
    raw = _extract_json_array(response.content)
    try:
        names = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [n.strip() for n in names if isinstance(n, str) and n.strip()]
