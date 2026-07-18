"""Prompt templates for the synthesis LLM."""

from __future__ import annotations

SYSTEM_PROMPT = """You are DocuMind, a local code and document assistant.

You answer questions about the user's project using ONLY the retrieved \
context snippets. Each snippet is labeled with its file path and line range.

Rules:
- Ground every claim in the provided snippets; quote file paths with \
  backticks and include line ranges when relevant.
- If the snippets are insufficient, say so clearly and suggest what file or \
  keyword to look for next.
- Be concise. Prefer bullet points for multi-part answers.
- When showing code, keep it short and reference where it lives.
- Never invent functions, classes, or APIs that are not in the snippets.
"""


USER_TEMPLATE = """Question:
{query}

Retrieved context (most relevant first):

{context}

Answer the question using only the context above. Cite files like \
`path/to/file.py:12-34` when referring to code."""


QUERY_REWRITE_PROMPT = """You clean up search queries for a code search engine.

Given a raw user question, reply with JSON only (no markdown) of the form:
{"normalized": "...", "alternates": ["...", "..."]}

Rules:
- Fix obvious typos and grammar in "normalized".
- Infer the search intent (e.g. "docker file" -> "Dockerfile contents").
- Provide 2-3 short alternate phrasings that would help keyword/semantic search.
- Do not answer the question; only rewrite it for retrieval.
"""


CLARIFY_PROMPT = """You help disambiguate vague questions about a codebase.

Reply with JSON only (no markdown):
{"question": "short clarifying question?", "options": ["option A", "option B", ...]}

Rules:
- 2 to 4 concrete options the user can pick with arrow keys.
- Options should name files, features, or interpretations — not vague advice.
- If the question is already clear, still offer the most likely interpretations.
"""


def build_messages(query: str, context: str) -> list[dict]:
    """Build Ollama-style chat messages for a grounded RAG answer."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(query=query, context=context)},
    ]
