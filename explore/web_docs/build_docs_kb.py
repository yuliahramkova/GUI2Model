"""Собирает KB веб-доков (3-й источник exploration) из:

1) configs/doc_sources.json — секции + Magento summary + typical_actions
2) data/docs_explore/shopping_intents.raw.json — WebArena shopping intents

Сначала keyword-раскладка, затем unassigned доразбивает LLM через HF
(Qwen/Qwen2.5-7B-Instruct по умолчанию, токен HF_TOKEN из .env).

Запуск из корня репо:
  python explore/web_docs/extract_wa_shopping_intents.py
  python explore/web_docs/build_docs_kb.py
  python explore/web_docs/build_docs_kb.py --no-llm   # только keywords

Выход:
  data/docs_explore/knowledge_base.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

DEFAULT_DOC_SOURCES = Path("configs/doc_sources.json")
DEFAULT_INTENTS = Path("data/docs_explore/shopping_intents.raw.json")
DEFAULT_OUT = Path("data/docs_explore/knowledge_base.json")
DEFAULT_HF_BASE = "https://router.huggingface.co/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _extract_json(text: str) -> Any | None:
    """Достаёт JSON-объект или массив из ответа модели."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        i = text.find(opener)
        j = text.rfind(closer)
        if i >= 0 and j > i:
            try:
                return json.loads(text[i : j + 1])
            except Exception:
                continue
    return None


def assign_section_keywords(intent: str, sections: list[dict[str, Any]]) -> str | None:
    """Возвращает id секции с наибольшим числом совпавших keyword, иначе None."""
    text = _norm(intent)
    if not text:
        return None
    best_id: str | None = None
    best_score = 0.0
    for sec in sections:
        kws = sec.get("intent_keywords") or []
        score = float(sum(1 for kw in kws if _norm(str(kw)) in text))
        score += sum(0.5 for kw in kws if len(_norm(str(kw))) >= 12 and _norm(str(kw)) in text)
        if score > best_score:
            best_score = score
            best_id = str(sec["id"])
    return best_id if best_score > 0 else None


def classify_unassigned_with_llm(
    unassigned: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    *,
    client: Any,
    model: str,
    batch_size: int,
) -> dict[int, str]:
    """Классифицирует unassigned intents → section_id. Ключ — индекс в unassigned."""
    allowed = [str(s["id"]) for s in sections]
    section_lines = "\n".join(
        f"- {s['id']}: {s.get('name')} — {(s.get('summary') or '')[:180]}" for s in sections
    )
    mapping: dict[int, str] = {}

    for start in range(0, len(unassigned), batch_size):
        batch = unassigned[start : start + batch_size]
        items = [
            {"idx": start + i, "intent": (it.get("intent") or it.get("intent_template") or "").strip()}
            for i, it in enumerate(batch)
        ]
        system = (
            "You classify shopping-website user intents into ONE section id. "
            "Reply with ONLY a JSON array of objects: "
            '[{"idx": <int>, "section_id": "<id>"}]. '
            f"Allowed section_id values: {allowed}. "
            "Pick the closest section; never invent new ids."
        )
        user = (
            f"Sections:\n{section_lines}\n\n"
            f"Classify these intents:\n{json.dumps(items, ensure_ascii=False, indent=2)}"
        )
        print(f"  LLM batch {start}-{start + len(batch) - 1} / {len(unassigned) - 1} ...")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=800,
        )
        content = resp.choices[0].message.content or ""
        parsed = _extract_json(content)
        if not isinstance(parsed, list):
            print(f"  ! bad LLM JSON, skip batch: {content[:200]!r}")
            continue
        for row in parsed:
            if not isinstance(row, dict):
                continue
            try:
                idx = int(row.get("idx"))
            except Exception:
                continue
            sid = str(row.get("section_id") or "").strip()
            if sid not in allowed:
                continue
            if start <= idx < start + len(batch):
                mapping[idx] = sid

    return mapping


def build_kb(
    *,
    doc_sources: dict[str, Any],
    intents_payload: dict[str, Any],
    max_examples_per_section: int,
    doc_sources_path: Path,
    intents_path: Path,
    llm_client: Any | None,
    llm_model: str,
    llm_batch_size: int,
) -> dict[str, Any]:
    """Мержит Magento summaries с интентами WA (keywords + опционально LLM)."""
    sections_cfg = doc_sources.get("sections") or []
    intents = intents_payload.get("intents") or []
    section_ids = {str(s["id"]) for s in sections_cfg}

    buckets: dict[str, list[dict[str, Any]]] = {str(s["id"]): [] for s in sections_cfg}
    unassigned: list[dict[str, Any]] = []
    n_keyword = 0
    n_llm = 0

    for item in intents:
        intent_text = item.get("intent") or item.get("intent_template") or ""
        sid = assign_section_keywords(intent_text, sections_cfg)
        if sid is None:
            unassigned.append(item)
            continue
        buckets[sid].append({**item, "_assign": "keyword"})
        n_keyword += 1

    still_unassigned: list[dict[str, Any]] = []
    if unassigned and llm_client is not None:
        print(f"Keyword left unassigned={len(unassigned)}; classifying with {llm_model} ...")
        mapping = classify_unassigned_with_llm(
            unassigned,
            sections_cfg,
            client=llm_client,
            model=llm_model,
            batch_size=llm_batch_size,
        )
        for idx, item in enumerate(unassigned):
            sid = mapping.get(idx)
            if sid and sid in section_ids:
                buckets[sid].append({**item, "_assign": "llm"})
                n_llm += 1
            else:
                still_unassigned.append(item)
    else:
        still_unassigned = unassigned

    sections_out: list[dict[str, Any]] = []
    for sec in sections_cfg:
        sid = str(sec["id"])
        examples = buckets.get(sid) or []
        seen: set[str] = set()
        example_intents: list[str] = []
        for ex in examples:
            text = (ex.get("intent") or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            example_intents.append(text)
            if len(example_intents) >= max_examples_per_section:
                break

        sections_out.append(
            {
                "id": sid,
                "name": sec.get("name"),
                "summary": sec.get("summary"),
                "typical_actions": sec.get("typical_actions") or [],
                "related_urls": sec.get("related_urls") or [],
                "aliases": sec.get("aliases") or [],
                "example_intents": example_intents,
                "n_intents_matched": len(examples),
                "source_urls": sec.get("source_urls") or [],
            }
        )

    return {
        "app": doc_sources.get("app", "webarena_shopping"),
        "source": "web_docs",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "doc_sources": str(doc_sources_path).replace("\\", "/"),
            "shopping_intents": str(intents_path).replace("\\", "/"),
            "n_shopping_intents": len(intents),
            "n_assigned_keyword": n_keyword,
            "n_assigned_llm": n_llm,
            "n_unassigned_intents": len(still_unassigned),
            "llm_model": llm_model if llm_client is not None else None,
        },
        "sections": sections_out,
        "unassigned_intents_sample": [
            {"task_id": u.get("task_id"), "intent": u.get("intent")} for u in still_unassigned[:40]
        ],
        "summary": {
            "n_sections": len(sections_out),
            "n_intents_assigned": sum(s["n_intents_matched"] for s in sections_out),
            "n_intents_assigned_keyword": n_keyword,
            "n_intents_assigned_llm": n_llm,
            "n_intents_unassigned": len(still_unassigned),
        },
    }


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build web_docs knowledge base")
    parser.add_argument("--doc-sources", type=Path, default=DEFAULT_DOC_SOURCES)
    parser.add_argument("--intents", type=Path, default=DEFAULT_INTENTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--no-llm", action="store_true", help="Только keywords, без HF")
    parser.add_argument(
        "--model",
        default=os.environ.get("DOCS_LLM_MODEL", DEFAULT_MODEL),
        help="HF model id (default Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--hf-base-url",
        default=os.environ.get("HF_BASE_URL", DEFAULT_HF_BASE),
    )
    parser.add_argument("--batch-size", type=int, default=15)
    args = parser.parse_args()

    if not args.intents.exists():
        raise SystemExit(
            f"Нет {args.intents}. Сначала:\n"
            f"  python explore/web_docs/extract_wa_shopping_intents.py"
        )

    llm_client = None
    if not args.no_llm:
        if OpenAI is None:
            raise SystemExit("Нужен пакет openai: pip install openai")
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if not token:
            raise SystemExit("Нет HF_TOKEN в .env / окружении (или запусти с --no-llm)")
        llm_client = OpenAI(api_key=token, base_url=args.hf_base_url)

    doc_sources = json.loads(args.doc_sources.read_text(encoding="utf-8"))
    intents_payload = json.loads(args.intents.read_text(encoding="utf-8"))
    kb = build_kb(
        doc_sources=doc_sources,
        intents_payload=intents_payload,
        max_examples_per_section=args.max_examples,
        doc_sources_path=args.doc_sources,
        intents_path=args.intents,
        llm_client=llm_client,
        llm_model=args.model,
        llm_batch_size=args.batch_size,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(kb, ensure_ascii=False, indent=2), encoding="utf-8")
    s = kb["summary"]
    print(
        f"KB -> {args.out}\n"
        f"  sections={s['n_sections']} "
        f"assigned={s['n_intents_assigned']} "
        f"(keyword={s['n_intents_assigned_keyword']}, llm={s['n_intents_assigned_llm']}) "
        f"unassigned={s['n_intents_unassigned']}"
    )


if __name__ == "__main__":
    main()
