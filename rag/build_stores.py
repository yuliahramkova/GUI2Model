"""Сборка LightRAG-индексов из KB exploration (a11y / CUA / веб-доки).

Готовые индексы: rag/stores/<mode>/.
LLM (индексация + query): LNSIGO из .env (LNSIGO_API_KEY, LNSIGO_BASE_URL, LNSIGO_MODEL).
Embeddings: локально sentence-transformers/all-MiniLM-L6-v2.

Примеры:
  python rag/build_stores.py --list-modes
  python rag/build_stores.py --mode a11y
  python rag/build_stores.py --all
  python rag/build_stores.py --query "search for shirt on homepage" --mode a11y_cua
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag.client import (  # noqa: E402
    DEFAULT_STORES_ROOT,
    build_rag_store,
    default_query_param,
    query_store,
    set_verbose,
    write_manifest,
)
from rag.modes import ALL_MODES, MODE_SOURCES  # noqa: E402


def _print_modes() -> None:
    print("Доступные режимы RAG:")
    for mode in ALL_MODES:
        sources = ", ".join(MODE_SOURCES[mode])
        print(f"  {mode:<12}  источники: {sources}")


async def _build_one(mode: str, stores_root: Path, fresh: bool) -> dict:
    print(f"\n=== Сборка RAG store: {mode} ===")
    meta = await build_rag_store(mode, stores_root=stores_root, fresh=fresh)
    print(
        f"Готово: {meta['n_documents']} документов "
        f"({meta['source_doc_counts']}) -> {meta['working_dir']}"
    )
    return meta


def _setup_lightrag_logging() -> None:
    try:
        from lightrag.utils import setup_logger

        setup_logger("lightrag", level=logging.INFO)
    except Exception:
        pass


async def _run(args: argparse.Namespace) -> None:
    set_verbose(not args.quiet)
    if not args.quiet:
        _setup_lightrag_logging()

    stores_root = args.stores_root.resolve()

    if args.list_modes:
        _print_modes()
        return

    if args.query:
        if not args.mode:
            raise SystemExit("--query требует --mode")
        result = await query_store(
            args.mode,
            args.query,
            stores_root=stores_root,
            param=default_query_param(mode=args.query_mode),
        )
        print(result)
        return

    modes = list(ALL_MODES) if args.all else [args.mode]
    if not modes or modes == [None]:
        raise SystemExit("Укажите --mode <имя>, --all или --list-modes")

    built = []
    for mode in modes:
        built.append(await _build_one(mode, stores_root, fresh=not args.no_fresh))

    if len(built) > 1 or args.all:
        manifest = write_manifest(stores_root, built)
        print(f"\nManifest: {manifest}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сборка LightRAG-индексов из KB exploration (GUI)"
    )
    parser.add_argument("--stores-root", type=Path, default=DEFAULT_STORES_ROOT)
    parser.add_argument(
        "--mode",
        choices=ALL_MODES,
        default=None,
        help="Один режим для сборки или query",
    )
    parser.add_argument("--all", action="store_true", help="Собрать все режимы")
    parser.add_argument(
        "--list-modes",
        action="store_true",
        help="Показать режим → источники",
    )
    parser.add_argument(
        "--no-fresh",
        action="store_true",
        help="Не удалять существующую папку store перед пересборкой",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Smoke-test query к уже собранному store",
    )
    parser.add_argument(
        "--query-mode",
        default="hybrid",
        choices=["naive", "local", "global", "hybrid", "mix"],
        help="Режим retrieval LightRAG для --query",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Без пошагового вывода [rag]",
    )
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
