"""Клиент LightRAG: LLM через LNSIGO, локальные embeddings, сборка индекса и query."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from sentence_transformers import SentenceTransformer

from rag.modes import MODE_SOURCES, ModeName, validate_mode
from rag.sources import load_documents

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STORES_ROOT = REPO_ROOT / "rag" / "stores"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEFAULT_CHUNK_TOKEN_SIZE = 800
# Ниже дефолтов LightRAG (4 LLM / 8 embed / 3 insert) — меньше WinError 5 на Windows
# при атомарном rename kv_store_*.json и vdb_*.json.
DEFAULT_LLM_MAX_ASYNC = 2
DEFAULT_EMBEDDING_MAX_ASYNC = 2
DEFAULT_MAX_PARALLEL_INSERT = 1
DEFAULT_MAX_PARALLEL_ANALYZE = 2

_progress_t0: float | None = None
_llm_calls = 0
_embed_calls = 0
_embed_texts_total = 0
_verbose = True


def set_verbose(enabled: bool) -> None:
    global _verbose
    _verbose = enabled


def _log(msg: str) -> None:
    if not _verbose:
        return
    elapsed = ""
    if _progress_t0 is not None:
        elapsed = f" +{time.monotonic() - _progress_t0:.1f}s"
    print(f"[rag{elapsed}] {msg}", flush=True)


def _reset_progress() -> None:
    global _progress_t0, _llm_calls, _embed_calls, _embed_texts_total
    _progress_t0 = time.monotonic()
    _llm_calls = 0
    _embed_calls = 0
    _embed_texts_total = 0


@lru_cache(maxsize=1)
def _embed_model() -> SentenceTransformer:
    _log(f"Загрузка embedding-модели {EMBEDDING_MODEL} (первый раз может качать с HuggingFace)...")
    t0 = time.monotonic()
    model = SentenceTransformer(EMBEDDING_MODEL)
    _log(f"Embedding-модель готова за {time.monotonic() - t0:.1f}s")
    return model


def preload_embedding_model() -> None:
    """Явно грузит MiniLM до ainsert, чтобы видеть зависание на HF отдельно."""
    _embed_model()


async def embedding_func(texts: list[str]) -> np.ndarray:
    """NanoVectorDB ждёт np.ndarray с .size, не Python list."""
    global _embed_calls, _embed_texts_total
    if isinstance(texts, str):
        texts = [texts]
    n = len(texts)
    _embed_calls += 1
    _embed_texts_total += n
    _log(f"embed #{_embed_calls}: batch={n} texts (всего embed-текстов: {_embed_texts_total})")
    t0 = time.monotonic()
    model = _embed_model()
    out = model.encode(texts, convert_to_numpy=True)
    if not isinstance(out, np.ndarray):
        out = np.asarray(out, dtype=np.float32)
    _log(
        f"embed #{_embed_calls}: готово за {time.monotonic() - t0:.1f}s "
        f"shape={getattr(out, 'shape', None)}"
    )
    return out


def _require_lnsigo_env() -> tuple[str, str, str]:
    load_dotenv()
    missing = [k for k in ("LNSIGO_API_KEY", "LNSIGO_BASE_URL", "LNSIGO_MODEL") if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing in .env: {', '.join(missing)}")
    return (
        os.environ["LNSIGO_API_KEY"],
        os.environ["LNSIGO_BASE_URL"],
        os.environ["LNSIGO_MODEL"],
    )


async def lnsigo_llm_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, str]] | None = None,
    keyword_extraction: bool = False,
    **kwargs: Any,
) -> str:
    """OpenAI-compatible LNSIGO для индексации (KG) и синтеза ответа при query."""
    global _llm_calls
    from lightrag.llm.openai import openai_complete_if_cache

    _llm_calls += 1
    _, base_url, model = _require_lnsigo_env()
    prompt_len = len(prompt or "")
    sys_len = len(system_prompt or "")
    _log(
        f"LLM #{_llm_calls} -> {model} @ {base_url} "
        f"(prompt={prompt_len} chars, system={sys_len} chars, "
        f"keyword_extraction={keyword_extraction})"
    )
    t0 = time.monotonic()
    try:
        result = await openai_complete_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=os.environ["LNSIGO_API_KEY"],
            base_url=base_url,
            keyword_extraction=keyword_extraction,
            **kwargs,
        )
    except Exception as exc:
        _log(f"LLM #{_llm_calls}: ОШИБКА за {time.monotonic() - t0:.1f}s — {exc}")
        raise
    _log(f"LLM #{_llm_calls}: ответ за {time.monotonic() - t0:.1f}s ({len(result or '')} chars)")
    return result


async def create_rag(working_dir: Path) -> LightRAG:
    working_dir.mkdir(parents=True, exist_ok=True)
    _log(
        f"LightRAG(...): working_dir={working_dir} "
        f"(llm_max_async={DEFAULT_LLM_MAX_ASYNC}, "
        f"embed_max_async={DEFAULT_EMBEDDING_MAX_ASYNC}, "
        f"max_parallel_insert={DEFAULT_MAX_PARALLEL_INSERT})"
    )
    t0 = time.monotonic()
    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=lnsigo_llm_func,
        llm_model_max_async=DEFAULT_LLM_MAX_ASYNC,
        embedding_func_max_async=DEFAULT_EMBEDDING_MAX_ASYNC,
        max_parallel_insert=DEFAULT_MAX_PARALLEL_INSERT,
        max_parallel_analyze=DEFAULT_MAX_PARALLEL_ANALYZE,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=8192,
            model_name=EMBEDDING_MODEL,
            func=embedding_func,
        ),
        addon_params={
            "language": "English",
            "chunker": {
                "chunk_token_size": DEFAULT_CHUNK_TOKEN_SIZE,
                "recursive_character": {
                    "separators": ["\n\n", "\n", ". ", " "],
                },
            },
        },
    )
    _log(f"LightRAG(...) создан за {time.monotonic() - t0:.1f}s, вызываю initialize_storages()...")
    t1 = time.monotonic()
    await rag.initialize_storages()
    _log(f"initialize_storages() готово за {time.monotonic() - t1:.1f}s")
    return rag


def default_query_param(*, mode: str = "hybrid") -> QueryParam:
    """Параметры retrieval по умолчанию; rerank встроен в LightRAG (enable_rerank=True)."""
    return QueryParam(
        mode=mode,
        top_k=40,
        chunk_top_k=15,
        enable_rerank=True,
    )


async def build_rag_store(
    mode: ModeName,
    *,
    stores_root: Path = DEFAULT_STORES_ROOT,
    fresh: bool = True,
) -> dict[str, Any]:
    _reset_progress()
    _log(f"=== build_rag_store mode={mode} ===")

    mode = validate_mode(mode)
    sources = MODE_SOURCES[mode]

    _log(f"load_documents({list(sources)})...")
    t0 = time.monotonic()
    documents, source_counts = load_documents(sources)
    _log(
        f"load_documents: {len(documents)} docs {source_counts} "
        f"за {time.monotonic() - t0:.1f}s"
    )

    store_dir = stores_root / mode
    if fresh and store_dir.exists():
        _log(f"Удаляю старый store: {store_dir}")
        shutil.rmtree(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    _log(f"store_dir: {store_dir}")

    _, base_url, llm_model = _require_lnsigo_env()
    _log(f"LNSIGO: model={llm_model} base_url={base_url}")

    preload_embedding_model()

    rag = await create_rag(store_dir)
    try:
        _log(
            f"ainsert({len(documents)} docs): chunk+embed локально, "
            f"затем LLM (KG) — смотри строки embed # / LLM #"
        )
        t_insert = time.monotonic()
        await rag.ainsert(documents)
        _log(
            f"ainsert готово за {time.monotonic() - t_insert:.1f}s "
            f"(LLM вызовов: {_llm_calls}, embed батчей: {_embed_calls})"
        )
    finally:
        _log("finalize_storages()...")
        t_fin = time.monotonic()
        await rag.finalize_storages()
        _log(f"finalize_storages() за {time.monotonic() - t_fin:.1f}s")

    meta = {
        "mode": mode,
        "sources": list(sources),
        "source_doc_counts": source_counts,
        "n_documents": len(documents),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "llm_model": llm_model,
        "llm_base_url": base_url,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "chunk_token_size": DEFAULT_CHUNK_TOKEN_SIZE,
        "working_dir": str(store_dir).replace("\\", "/"),
        "build_stats": {
            "llm_calls": _llm_calls,
            "embed_batches": _embed_calls,
            "embed_texts": _embed_texts_total,
        },
    }
    meta_path = store_dir / "build_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"build_meta.json записан")
    return meta


class RagSession:
    """Держит LightRAG открытым между query — для eval-цикла (Playwright sync)."""

    def __init__(
        self,
        mode: ModeName,
        *,
        stores_root: Path = DEFAULT_STORES_ROOT,
        query_mode: str = "hybrid",
        verbose: bool = False,
    ) -> None:
        self.mode = validate_mode(mode)
        self.stores_root = stores_root
        self.query_param = default_query_param(mode=query_mode)
        self._rag: LightRAG | None = None
        self._loop = asyncio.new_event_loop()
        self._prev_verbose = _verbose
        set_verbose(verbose)

        meta_path = stores_root / self.mode / "build_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"RAG store for mode {self.mode!r} not built yet: {stores_root / self.mode}. "
                f"Run: python rag/build_stores.py --mode {self.mode}"
            )

    def query(self, text: str) -> str:
        return self._loop.run_until_complete(self._aquery(text))

    async def _aquery(self, text: str) -> str:
        _reset_progress()
        _log(f"=== RagSession query mode={self.mode} ===")
        _log(f"query: {text!r}")
        if self._rag is None:
            preload_embedding_model()
            store_dir = self.stores_root / self.mode
            self._rag = await create_rag(store_dir)
        t0 = time.monotonic()
        result = await self._rag.aquery(text, param=self.query_param)
        _log(f"aquery готово за {time.monotonic() - t0:.1f}s (LLM вызовов: {_llm_calls})")
        return result or ""

    def close(self) -> None:
        if self._rag is not None:
            self._loop.run_until_complete(self._rag.finalize_storages())
            self._rag = None
        self._loop.close()
        set_verbose(self._prev_verbose)

    def __enter__(self) -> RagSession:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


async def query_store(
    mode: ModeName,
    query: str,
    *,
    stores_root: Path = DEFAULT_STORES_ROOT,
    param: QueryParam | None = None,
) -> str:
    _reset_progress()
    _log(f"=== query_store mode={mode} ===")
    _log(f"query: {query!r}")

    mode = validate_mode(mode)
    store_dir = stores_root / mode
    meta_path = store_dir / "build_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"RAG store for mode {mode!r} not built yet: {store_dir}. "
            f"Run: python rag/build_stores.py --mode {mode}"
        )

    preload_embedding_model()
    rag = await create_rag(store_dir)
    try:
        _log("aquery()...")
        t0 = time.monotonic()
        result = await rag.aquery(query, param=param or default_query_param())
        _log(f"aquery готово за {time.monotonic() - t0:.1f}s (LLM вызовов: {_llm_calls})")
        return result
    finally:
        await rag.finalize_storages()


def write_manifest(stores_root: Path, built: list[dict[str, Any]]) -> Path:
    manifest = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "stores_root": str(stores_root).replace("\\", "/"),
        "modes": {meta["mode"]: meta for meta in built},
    }
    path = stores_root / "manifest.json"
    stores_root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
