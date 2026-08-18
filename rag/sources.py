"""Сериализация KB exploration в текстовые документы для LightRAG.

Каждая запись KB → одна логическая строка-документ. LightRAG при ainsert()
дополнительно режет длинные строки на token-chunks; короткие записи обычно = 1 chunk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag.modes import SOURCES, SourceName

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PATHS: dict[SourceName, Path] = {
    "a11y": REPO_ROOT / "data/a11y_explore/knowledge_base.json",
    "cua": REPO_ROOT / "data/cua_explore/screenshot_plus_som/knowledge_base.json",
    "docs": REPO_ROOT / "data/docs_explore/knowledge_base_cleaned.json",
}


def _fmt_action(action: dict[str, Any] | None) -> str:
    if not action:
        return ""
    parts = [str(action.get("action") or "")]
    if action.get("target_mark_id") is not None:
        parts.append(f"target_mark_id={action['target_mark_id']}")
    if action.get("value"):
        parts.append(f"value={action['value']!r}")
    return " ".join(p for p in parts if p)


def load_a11y_documents(kb_path: Path) -> list[str]:
    kb = json.loads(kb_path.read_text(encoding="utf-8"))
    docs: list[str] = []

    for screen in kb.get("screens") or []:
        docs.append(
            "[source=a11y] [type=screen] "
            f"Screen id={screen.get('id')!r} title={screen.get('title')!r} "
            f"description={screen.get('description')!r} "
            f"url={screen.get('url')!r}"
        )

    for el in kb.get("elements") or []:
        docs.append(
            "[source=a11y] [type=element] "
            f"screen_id={el.get('screen_id')!r} "
            f"instruction={el.get('instruction')!r} "
            f"role={el.get('role')!r} name={el.get('name')!r} "
            f"bbox_px={el.get('bbox_px')}"
        )

    for tr in kb.get("transitions") or []:
        docs.append(
            "[source=a11y] [type=transition] "
            f"screen_id={tr.get('screen_id')!r} step={tr.get('step')} "
            f"action={tr.get('action')!r} target={tr.get('target')!r} "
            f"url_before={tr.get('url_before')!r} url_after={tr.get('url_after')!r}"
        )

    return docs


def load_cua_documents(kb_path: Path) -> list[str]:
    kb = json.loads(kb_path.read_text(encoding="utf-8"))
    docs: list[str] = []

    goals_by_task: dict[str, str] = {}
    for task in kb.get("tasks") or []:
        task_id = task.get("id") or ""
        goal = task.get("goal") or ""
        goals_by_task[task_id] = goal
        docs.append(
            "[source=cua] [type=task] "
            f"task_id={task_id!r} success={task.get('success')} "
            f"n_steps={task.get('n_steps_executed')} goal={goal!r} "
            f"start_url={task.get('start_url')!r}"
        )

    for state in kb.get("states") or []:
        docs.append(
            "[source=cua] [type=state] "
            f"state_id={state.get('state_id')!r} key={state.get('key')!r}"
        )

    for proc in kb.get("procedures") or []:
        task_id = proc.get("task_id") or proc.get("id") or ""
        goal = goals_by_task.get(task_id, "")
        docs.append(
            "[source=cua] [type=procedure] "
            f"task_id={task_id!r} goal={goal!r} "
            f"procedure={proc.get('procedure')!r} n_steps={proc.get('n_steps')}"
        )

    for tr in kb.get("transitions") or []:
        task_id = tr.get("task_id") or ""
        goal = goals_by_task.get(task_id, "")
        docs.append(
            "[source=cua] [type=transition] "
            f"task_id={task_id!r} goal={goal!r} step={tr.get('step')} "
            f"from_state={tr.get('from_state_id')!r} to_state={tr.get('to_state_id')!r} "
            f"action={_fmt_action(tr.get('action'))} success={tr.get('success')} "
            f"url_after={tr.get('url_after')!r} title_after={tr.get('title_after')!r}"
        )

    return docs


def load_docs_documents(kb_path: Path) -> list[str]:
    kb = json.loads(kb_path.read_text(encoding="utf-8"))
    docs: list[str] = []

    for section in kb.get("sections") or []:
        section_id = section.get("id") or ""
        docs.append(
            "[source=docs] [type=section] "
            f"section_id={section_id!r} name={section.get('name')!r} "
            f"summary={section.get('summary')!r} "
            f"typical_actions={section.get('typical_actions')} "
            f"related_urls={section.get('related_urls')} "
            f"aliases={section.get('aliases')}"
        )
        for intent in section.get("example_intents") or []:
            docs.append(
                "[source=docs] [type=intent] "
                f"section_id={section_id!r} intent={intent!r}"
            )

    return docs


_LOADERS = {
    "a11y": load_a11y_documents,
    "cua": load_cua_documents,
    "docs": load_docs_documents,
}


def load_documents(
    sources: tuple[SourceName, ...],
    *,
    paths: dict[SourceName, Path] | None = None,
) -> tuple[list[str], dict[str, int]]:
    """Загружает и дедуплицирует документы для указанных источников."""
    paths = paths or DEFAULT_PATHS
    docs: list[str] = []
    counts: dict[str, int] = {}

    for source in sources:
        if source not in SOURCES:
            raise ValueError(f"Unknown source {source!r}")
        path = paths[source]
        if not path.exists():
            raise FileNotFoundError(f"Missing {source} KB: {path}")
        loaded = _LOADERS[source](path)
        counts[source] = len(loaded)
        docs.extend(loaded)

    seen: set[str] = set()
    unique: list[str] = []
    for doc in docs:
        if doc in seen:
            continue
        seen.add(doc)
        unique.append(doc)

    return unique, counts
