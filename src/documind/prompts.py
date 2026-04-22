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


def build_messages(query: str, context: str) -> list[dict]:
    """Build Ollama-style chat messages for a grounded RAG answer."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(query=query, context=context)},
    ]
