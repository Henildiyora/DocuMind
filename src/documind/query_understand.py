"""Query rewriting, multi-query retrieval, relevance filtering, clarification."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

from rich.console import Console

from .config import Config
from .index import DocuMindIndex
from .llm import LLMError, OllamaClient
from .prompts import CLARIFY_PROMPT, QUERY_REWRITE_PROMPT
from .search import SearchHit, filter_relevant, search, search_multi

_FILE_HINT_RE = re.compile(
    r"\b("
    r"docker\s*file|dockerfile|"
    r"make\s*file|makefile|"
    r"read\s*me|readme|"
    r"[\w.-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|md|yml|yaml|toml|json|sh)"
    r")\b",
    re.IGNORECASE,
)

_OVERVIEW_RE = re.compile(
    r"\b("
    r"explain\s+(me\s+)?(this\s+)?project|"
    r"what\s+is\s+this(\s+project)?|"
    r"how\s+does\s+this\s+project\s+work|"
    r"overview|summarize\s+(this\s+)?project|"
    r"tell\s+me\s+about\s+(this\s+)?project|"
    r"how\s+many\s+files|"
    r"project\s+structure|"
    r"what\s+does\s+this\s+(app|repo|codebase)\s+do"
    r")\b",
    re.IGNORECASE,
)

_ENTRYPOINT_NAMES = frozenset({
    "readme.md",
    "readme",
    "readme.rst",
    "readme.txt",
    "main.py",
    "app.py",
    "index.py",
    "index.js",
    "index.ts",
    "pyproject.toml",
    "package.json",
    "cargo.toml",
    "go.mod",
    "dockerfile",
    "makefile",
})

_ARTIFACT_PATH_MARKERS = (
    "generated_reports/",
    "/generated_reports/",
    "generated/",
    "/htmlcov/",
    "htmlcov/",
)

_OVERVIEW_ALTERNATES = (
    "README project overview",
    "main.py entrypoint",
    "application architecture",
)


def is_overview_intent(query: str) -> bool:
    """True for broad project-overview / meta questions."""
    return bool(_OVERVIEW_RE.search(query or ""))


def is_artifact_path(rel_path: str) -> bool:
    """True for generated report dumps and similar pollutants."""
    p = (rel_path or "").replace("\\", "/").lower()
    name = Path(p).name
    if name == "report.json":
        return True
    return any(m in p for m in _ARTIFACT_PATH_MARKERS)


def _extract_json_object(text: str) -> dict | None:
    """Best-effort extract of a JSON object from an LLM reply."""
    text = (text or "").strip()
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def rewrite_query(llm: OllamaClient, query: str, cfg: Config) -> tuple[str, list[str]]:
    """Return (normalized_query, alternate_phrasings).

    On failure, returns the original query and an empty alternate list.
    Overview queries always get README/entrypoint alternates appended.
    """
    messages = [
        {"role": "system", "content": QUERY_REWRITE_PROMPT},
        {"role": "user", "content": query},
    ]
    try:
        raw = llm.chat(messages, keep_alive=cfg.keep_alive)
    except LLMError:
        normalized, alternates = query, []
    else:
        data = _extract_json_object(raw)
        if not data:
            normalized, alternates = query, []
        else:
            normalized = str(data.get("normalized") or query).strip() or query
            alts_raw = data.get("alternates") or data.get("alternatives") or []
            alternates = []
            if isinstance(alts_raw, list):
                for item in alts_raw:
                    s = str(item).strip()
                    if s and s.lower() != normalized.lower() and s not in alternates:
                        alternates.append(s)

    if is_overview_intent(query):
        for alt in _OVERVIEW_ALTERNATES:
            if alt.lower() != normalized.lower() and alt not in alternates:
                alternates.append(alt)

    return normalized, alternates[:4]


def _filename_hints(query: str) -> list[str]:
    """Extract likely filename tokens from a natural-language query."""
    hints: list[str] = []
    for m in _FILE_HINT_RE.finditer(query):
        token = re.sub(r"\s+", "", m.group(1)).lower()
        if token in {"dockerfile", "makefile", "readme"}:
            hints.append(token)
        else:
            hints.append(token)
    q = query.lower()
    if "docker" in q and "file" in q and "dockerfile" not in hints:
        hints.append("dockerfile")
    return hints


def _rescore(hits: list[SearchHit], score_fn) -> list[SearchHit]:
    rescored = [
        SearchHit(
            chunk_id=h.chunk_id,
            rel_path=h.rel_path,
            language=h.language,
            start_line=h.start_line,
            end_line=h.end_line,
            text=h.text,
            score=score_fn(h),
            bm25_rank=h.bm25_rank,
            vector_rank=h.vector_rank,
        )
        for h in hits
    ]
    rescored.sort(key=lambda h: h.score, reverse=True)
    return rescored


def demote_artifact_hits(hits: list[SearchHit]) -> list[SearchHit]:
    """Push generated_reports / report.json chunks below real source hits."""
    if not hits:
        return hits

    def score_fn(hit: SearchHit) -> float:
        if is_artifact_path(hit.rel_path):
            return hit.score * 0.05
        return hit.score

    return _rescore(hits, score_fn)


def boost_filename_matches(hits: list[SearchHit], query: str) -> list[SearchHit]:
    """Boost / re-rank hits whose basename matches a filename mentioned in the query."""
    hints = _filename_hints(query)
    if not hints or not hits:
        return hits

    def score_boost(hit: SearchHit) -> float:
        if is_artifact_path(hit.rel_path):
            return hit.score
        name = Path(hit.rel_path).name.lower()
        stem = Path(hit.rel_path).stem.lower()
        for h in hints:
            if h in (name, stem) or name.startswith(h) or h in name:
                return hit.score + 1.0
            compact = name.replace("-", "").replace("_", "")
            if h.replace("-", "") in compact:
                return hit.score + 0.5
        return hit.score

    return _rescore(hits, score_boost)


def boost_entrypoint_hits(hits: list[SearchHit], query: str) -> list[SearchHit]:
    """For overview intents, prefer README / main / app entrypoints."""
    if not hits or not is_overview_intent(query):
        return hits

    def score_fn(hit: SearchHit) -> float:
        if is_artifact_path(hit.rel_path):
            return hit.score
        name = Path(hit.rel_path).name.lower()
        parts = hit.rel_path.replace("\\", "/").split("/")
        bonus = 0.0
        if name in _ENTRYPOINT_NAMES or name.startswith("readme"):
            bonus += 1.5
        # Prefer top-level app package over deep noise.
        if len(parts) >= 1 and parts[0] in {"app", "src"} and name.endswith(
            (".py", ".ts", ".js", ".go", ".rs")
        ):
            bonus += 0.4
        if len(parts) == 1 and name.endswith((".py", ".md", ".toml", ".json")):
            bonus += 0.3
        return hit.score + bonus

    return _rescore(hits, score_fn)


def refine_hits(hits: list[SearchHit], query: str, cfg: Config) -> list[SearchHit]:
    """Demote artifacts, boost filenames/entrypoints, then relevance-filter."""
    hits = demote_artifact_hits(hits)
    hits = boost_filename_matches(hits, query)
    hits = boost_entrypoint_hits(hits, query)
    return filter_relevant(hits, cfg) or hits


def hits_look_weak(hits: list[SearchHit], cfg: Config) -> bool:
    """True when retrieval confidence is too low to answer confidently."""
    if not hits:
        return True
    filtered = filter_relevant(hits, cfg)
    if not filtered:
        return True
    return len(filtered) == 1 and filtered[0].score < 0.02


def detect_ambiguity(hits: list[SearchHit], query: str) -> bool:
    """True only for real basename collisions (or named-file miss).

    Overview / meta questions never count as ambiguous here — clarifier
    must not interrupt \"explain this project\".
    """
    if not hits or is_overview_intent(query):
        return False

    basenames = [Path(h.rel_path).name.lower() for h in hits[:8]]
    counts = Counter(basenames)
    if any(c > 1 for c in counts.values()):
        return True
    hints = _filename_hints(query)
    if not hints:
        return False
    top = basenames[0] if basenames else ""
    return not any(h in top or top.startswith(h) for h in hints)


def request_clarification(
    llm: OllamaClient,
    query: str,
    hits: list[SearchHit],
    cfg: Config,
) -> dict | None:
    """Ask the LLM for a short MCQ clarification structure.

    Returns None when the model says the question is clear
    (``ambiguous: false``) or the reply is unusable.
    """
    preview = "\n".join(
        f"- {h.rel_path}:{h.start_line}-{h.end_line}" for h in hits[:6]
    ) or "(no strong matches)"
    user = (
        f"User question: {query}\n\n"
        f"Top retrieved paths:\n{preview}\n\n"
        "Reply with JSON only."
    )
    messages = [
        {"role": "system", "content": CLARIFY_PROMPT},
        {"role": "user", "content": user},
    ]
    try:
        raw = llm.chat(messages, keep_alive=cfg.keep_alive)
    except LLMError:
        return None
    data = _extract_json_object(raw)
    if not data:
        return None
    if data.get("ambiguous") is False:
        return None
    options = data.get("options") or []
    if not isinstance(options, list) or len(options) < 2:
        return None
    question = str(data.get("question") or "Which of these did you mean?").strip()
    clean_opts = [str(o).strip() for o in options if str(o).strip()][:4]
    if len(clean_opts) < 2:
        return None
    return {"question": question, "options": clean_opts}


def pick_clarification(
    clarification: dict,
    *,
    console: Console | None = None,
) -> str | None:
    """Interactive arrow-key (or numbered) picker. Returns chosen option text."""
    console = console or Console()
    question = clarification["question"]
    options: list[str] = clarification["options"]

    if sys.stdin.isatty():
        try:
            import questionary
            from questionary import Style

            style = Style([
                ("question", "bold"),
                ("selected", "fg:cyan bold"),
            ])
            choice = questionary.select(
                question,
                choices=options,
                style=style,
            ).ask()
            return choice
        except Exception:
            pass

    console.print(f"[yellow]{question}[/yellow]")
    for i, opt in enumerate(options, start=1):
        console.print(f"  {i}. {opt}")
    console.print(
        "[dim]Re-run with a clearer question, or use an interactive terminal "
        "to pick an option.[/dim]"
    )
    return None


def retrieve_for_question(
    idx: DocuMindIndex,
    query: str,
    cfg: Config,
    *,
    llm: OllamaClient | None = None,
    allow_clarify: bool = True,
    console: Console | None = None,
    prior_user_turns: list[str] | None = None,
) -> tuple[list[SearchHit], str]:
    """Full ask/chat retrieval path with optional rewrite + clarify.

    Returns ``(hits, effective_query)``.
    """
    console = console or Console()
    search_query = query
    if prior_user_turns:
        search_query = f"{prior_user_turns[-1]} {query}".strip()

    overview = is_overview_intent(query)

    if llm is not None:
        normalized, alts = rewrite_query(llm, search_query, cfg)
        queries = [normalized] + [a for a in alts if a not in {normalized}]
        effective = normalized
        hits = search_multi(idx, queries, cfg)
    else:
        effective = query
        hits = search(idx, search_query, cfg)

    hits = refine_hits(hits, effective if llm else search_query, cfg)

    # Clarify only on real basename collisions — never for overview/meta Qs,
    # and never solely because hits look weak on a broad question.
    needs_clarify = (
        allow_clarify
        and not overview
        and detect_ambiguity(hits, effective)
    )
    if needs_clarify and llm is not None:
        clarification = request_clarification(llm, query, hits, cfg)
        if clarification:
            choice = pick_clarification(clarification, console=console)
            if choice:
                effective = f"{effective} — focusing on: {choice}"
                hits = search_multi(idx, [effective, choice], cfg)
                hits = refine_hits(hits, effective, cfg)

    return hits, effective
