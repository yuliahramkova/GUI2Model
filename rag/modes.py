"""Режимы RAG-хранилищ: какие источники exploration входят в каждый индекс."""

from __future__ import annotations

SourceName = str
ModeName = str

SOURCES: tuple[SourceName, ...] = ("a11y", "cua", "docs")

MODE_SOURCES: dict[ModeName, tuple[SourceName, ...]] = {
    "a11y": ("a11y",),
    "cua": ("cua",),
    "a11y_cua": ("a11y", "cua"),
    "a11y_docs": ("a11y", "docs"),
    "cua_docs": ("cua", "docs"),
    "all": ("a11y", "cua", "docs"),
    "docs": ("docs",),
}

ALL_MODES: tuple[ModeName, ...] = tuple(MODE_SOURCES.keys())


def validate_mode(mode: str) -> ModeName:
    if mode not in MODE_SOURCES:
        known = ", ".join(ALL_MODES)
        raise ValueError(f"Unknown RAG mode {mode!r}. Expected one of: {known}")
    return mode
