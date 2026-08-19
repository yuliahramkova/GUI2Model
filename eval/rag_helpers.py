"""Общие хелперы RAG для eval-скриптов (multistep, grounding)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.client import RagSession

DEFAULT_RAG_MAX_CHARS = 6000


def build_rag_query(
    *,
    task: str,
    screen_id: str | None = None,
    url: str | None = None,
    title: str | None = None,
) -> str:
    lines = [f"Task: {task}"]
    if screen_id:
        lines.append(f"Screen: {screen_id}")
    if url:
        lines.append(f"Current page URL: {url}")
    if title:
        lines.append(f"Current page title: {title}")
    lines.append(
        "Which GUI elements, navigation paths, labels, roles, and actions on this storefront "
        "are most relevant?"
    )
    return "\n".join(lines)


def truncate_rag_context(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def fetch_rag_context(
    session: RagSession,
    *,
    task: str,
    screen_id: str | None = None,
    url: str | None = None,
    title: str | None = None,
    max_chars: int = DEFAULT_RAG_MAX_CHARS,
) -> str:
    query = build_rag_query(task=task, screen_id=screen_id, url=url, title=title)
    raw = session.query(query)
    return truncate_rag_context(raw, max_chars)


def prior_knowledge_label(rag_mode: str | None) -> str:
    if rag_mode:
        return f"rag ({rag_mode})"
    return "none (no RAG / no LoRA)"


def prior_knowledge_meta(
    *,
    rag_mode: str | None,
    rag_query_mode: str,
    rag_max_chars: int,
    rag_stores_root: Path,
) -> dict[str, Any] | None:
    if not rag_mode:
        return None
    return {
        "type": "rag",
        "mode": rag_mode,
        "query_mode": rag_query_mode,
        "max_chars": rag_max_chars,
        "stores_root": str(rag_stores_root).replace("\\", "/"),
    }
