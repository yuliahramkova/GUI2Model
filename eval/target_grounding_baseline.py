"""Baseline grounding на target GUI — Qwen2.5-VL (zero-shot или LightRAG prior knowledge).

Читает data/target_app/eval/grounding.json.
Метрики: click accuracy (точка в GT bbox), токены на пример.

Пример:
  python eval/target_grounding_baseline.py
  python eval/target_grounding_baseline.py --limit 5
  python eval/target_grounding_baseline.py --rag-mode cua --out-dir reports/grounding_rag_cua
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.rag_helpers import (  # noqa: E402
    DEFAULT_RAG_MAX_CHARS,
    fetch_rag_context,
    prior_knowledge_label,
    prior_knowledge_meta,
)
from rag.client import DEFAULT_STORES_ROOT, RagSession  # noqa: E402
from rag.modes import ALL_MODES  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "reports" / "grounding"
DEFAULT_GROUNDING = REPO_ROOT / "data/target_app/eval/grounding.json"


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


@dataclass
class GroundingSample:
    id: str
    screen_id: str
    image: Image.Image
    image_path: str
    instruction: str
    bbox_px: tuple[int, int, int, int]
    tags: list[str]
    image_width: int
    image_height: int


@dataclass
class EvalRow:
    id: str
    screen_id: str
    image_path: str
    instruction: str
    tags: str
    bbox_px: tuple[int, int, int, int]
    raw_response: str
    pred_px: int | None
    pred_py: int | None
    hit: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    steps: int
    image_width: int
    image_height: int
    parse_ok: bool
    rag_context: str | None = None


def load_model_client() -> tuple[OpenAI, str]:
    missing = [k for k in ("LNSIGO_API_KEY", "LNSIGO_BASE_URL", "LNSIGO_MODEL") if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing in .env: {', '.join(missing)}")
    client = OpenAI(
        api_key=os.environ["LNSIGO_API_KEY"],
        base_url=os.environ["LNSIGO_BASE_URL"],
    )
    return client, os.environ["LNSIGO_MODEL"]


def image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_grounding_prompt(instruction: str, width: int, height: int) -> str:
    return (
        f"What is the location of the element corresponding to the instruction: '{instruction}'? "
        f"The image size is {width}x{height} pixels. "
        f"Provide the click coordinates as (x, y) in pixels of THIS image "
        f"(x from 0 to {width - 1}, y from 0 to {height - 1}). "
        f"Reply with only the coordinates, e.g. (120, 340)."
    )


def extract_raw_point(text: str) -> tuple[int, int] | None:
    if not text:
        return None
    box_match = re.search(
        r"<\|box_start\|>\s*\(?(-?\d+)\s*,\s*(-?\d+)\)?\s*<\|box_end\|>",
        text,
    )
    if box_match:
        return int(box_match.group(1)), int(box_match.group(2))
    paren_match = re.search(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", text)
    if paren_match:
        return int(paren_match.group(1)), int(paren_match.group(2))
    nums = [int(n) for n in re.findall(r"-?\d+", text)]
    if len(nums) < 2:
        return None
    return nums[0], nums[1]


def is_hit(point: tuple[int, int] | None, bbox: tuple[int, int, int, int]) -> bool:
    if point is None:
        return False
    px, py = point
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2


def load_grounding_dataset(
    path: Path,
    limit: int | None = None,
) -> list[GroundingSample]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    samples: list[GroundingSample] = []
    for row in raw:
        image_path = resolve_repo_path(Path(row["image"]))
        if not image_path.is_file():
            raise FileNotFoundError(f"Screenshot missing: {image_path}")
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        bbox = row["bbox"]
        samples.append(
            GroundingSample(
                id=row["id"],
                screen_id=row.get("screen_id", ""),
                image=image,
                image_path=str(image_path).replace("\\", "/"),
                instruction=row["instruction"],
                bbox_px=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                tags=list(row.get("tags") or []),
                image_width=w,
                image_height=h,
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def predict_point(
    client: OpenAI,
    model: str,
    sample: GroundingSample,
    *,
    rag_context: str | None = None,
) -> tuple[str, int, int, int]:
    messages: list[dict[str, Any]] = []
    if rag_context:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Prior knowledge about this GUI (from a knowledge base — hints only; "
                    "the screenshot is authoritative):\n"
                    f"{rag_context}"
                ),
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_to_data_url(sample.image)}},
                {
                    "type": "text",
                    "text": build_grounding_prompt(
                        sample.instruction,
                        sample.image_width,
                        sample.image_height,
                    ),
                },
            ],
        }
    )

    response = client.chat.completions.create(model=model, messages=messages)
    text = response.choices[0].message.content or ""
    usage = response.usage
    return (
        text,
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
        usage.total_tokens if usage else 0,
    )


def run_eval(
    client: OpenAI,
    model: str,
    samples: list[GroundingSample],
    *,
    rag_session: RagSession | None = None,
    rag_max_chars: int = DEFAULT_RAG_MAX_CHARS,
    artifacts_dir: Path | None = None,
) -> list[EvalRow]:
    rows: list[EvalRow] = []
    desc = "Target grounding (RAG)" if rag_session else "Target grounding baseline"
    for sample in tqdm(samples, desc=desc):
        rag_context: str | None = None
        if rag_session is not None:
            rag_context = fetch_rag_context(
                rag_session,
                task=sample.instruction,
                screen_id=sample.screen_id,
                max_chars=rag_max_chars,
            )
            if artifacts_dir is not None:
                sample_dir = artifacts_dir / sample.id
                sample_dir.mkdir(parents=True, exist_ok=True)
                (sample_dir / "rag_context.txt").write_text(rag_context, encoding="utf-8")

        raw, prompt_tokens, completion_tokens, total_tokens = predict_point(
            client,
            model,
            sample,
            rag_context=rag_context,
        )
        point = extract_raw_point(raw)
        rows.append(
            EvalRow(
                id=sample.id,
                screen_id=sample.screen_id,
                image_path=sample.image_path,
                instruction=sample.instruction,
                tags=",".join(sample.tags),
                bbox_px=sample.bbox_px,
                raw_response=raw,
                pred_px=point[0] if point else None,
                pred_py=point[1] if point else None,
                hit=is_hit(point, sample.bbox_px),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                steps=1,
                image_width=sample.image_width,
                image_height=sample.image_height,
                parse_ok=point is not None,
                rag_context=rag_context,
            )
        )
    return rows


def summarize(rows: list[EvalRow]) -> dict[str, dict[str, float | int]]:
    by_screen: dict[str, list[EvalRow]] = defaultdict(list)
    for row in rows:
        by_screen[row.screen_id].append(row)

    def pack(subset: list[EvalRow]) -> dict[str, float | int]:
        n = len(subset)
        hits = sum(r.hit for r in subset)
        return {
            "n": n,
            "hits": hits,
            "accuracy": hits / n if n else 0.0,
            "total_tokens": sum(r.total_tokens for r in subset),
            "avg_tokens": sum(r.total_tokens for r in subset) / n if n else 0.0,
            "total_prompt_tokens": sum(r.prompt_tokens for r in subset),
            "total_completion_tokens": sum(r.completion_tokens for r in subset),
            "parse_fail": sum(1 for r in subset if not r.parse_ok),
        }

    summary: dict[str, dict[str, float | int]] = {"overall": pack(rows)}
    for key in sorted(by_screen):
        summary[f"screen:{key}"] = pack(by_screen[key])
    return summary


def print_summary(summary: dict[str, dict[str, float | int]], *, prior_knowledge: str) -> None:
    print(f"\n=== Target GUI grounding ({prior_knowledge}) ===")
    print(
        f"{'slice':<28} {'n':>4} {'acc':>8} {'tokens':>10} {'avg_tok':>10} {'parse_fail':>10}"
    )
    for name, s in summary.items():
        print(
            f"{name:<28} {s['n']:>4} {s['accuracy']:>8.1%} "
            f"{s['total_tokens']:>10} {s['avg_tokens']:>10.1f} {s['parse_fail']:>10}"
        )


def save_csv(rows: list[EvalRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "screen_id",
                "image_path",
                "instruction",
                "tags",
                "image_width",
                "image_height",
                "bbox_px",
                "pred_px",
                "pred_py",
                "hit",
                "parse_ok",
                "raw_response",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "steps",
                "rag_context",
            ],
            delimiter=",",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row.id,
                    "screen_id": row.screen_id,
                    "image_path": row.image_path,
                    "instruction": row.instruction,
                    "tags": row.tags,
                    "image_width": row.image_width,
                    "image_height": row.image_height,
                    "bbox_px": list(row.bbox_px),
                    "pred_px": row.pred_px,
                    "pred_py": row.pred_py,
                    "hit": row.hit,
                    "parse_ok": row.parse_ok,
                    "raw_response": row.raw_response,
                    "prompt_tokens": row.prompt_tokens,
                    "completion_tokens": row.completion_tokens,
                    "total_tokens": row.total_tokens,
                    "steps": row.steps,
                    "rag_context": row.rag_context or "",
                }
            )


def save_markdown(
    summary: dict[str, dict[str, float | int]],
    path: Path,
    *,
    model: str,
    grounding_path: Path,
    n_samples: int,
    prior_knowledge: str,
    prior_knowledge_detail: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Target GUI grounding baseline",
        "",
        f"- created_at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- model: `{model}`",
        f"- prior_knowledge: **{prior_knowledge}**",
        f"- dataset: `{grounding_path.as_posix()}`",
        f"- samples: {n_samples}",
    ]
    if prior_knowledge_detail:
        lines.append(f"- rag_config: `{json.dumps(prior_knowledge_detail, ensure_ascii=False)}`")
    lines.extend(
        [
            "",
            "| slice | n | accuracy | total_tokens | avg_tokens | parse_fail |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, s in summary.items():
        if name.startswith("screen:"):
            continue
        lines.append(
            f"| {name} | {s['n']} | {s['accuracy']:.1%} | {s['total_tokens']} "
            f"| {s['avg_tokens']:.1f} | {s['parse_fail']} |"
        )
    lines.extend(
        [
            "",
            "## By screen",
            "",
            "| screen | n | accuracy | avg_tokens |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, s in summary.items():
        if not name.startswith("screen:"):
            continue
        screen = name.removeprefix("screen:")
        lines.append(
            f"| {screen} | {s['n']} | {s['accuracy']:.1%} | {s['avg_tokens']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_results_json(
    rows: list[EvalRow],
    summary: dict[str, dict[str, float | int]],
    path: Path,
    meta: dict[str, Any],
) -> None:
    payload = {
        **meta,
        "summary": summary,
        "results": [
            {
                "id": r.id,
                "screen_id": r.screen_id,
                "instruction": r.instruction,
                "hit": r.hit,
                "parse_ok": r.parse_ok,
                "pred_px": r.pred_px,
                "pred_py": r.pred_py,
                "bbox_px": list(r.bbox_px),
                "total_tokens": r.total_tokens,
                "rag_context": r.rag_context,
            }
            for r in rows
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen grounding on target GUI (optional LightRAG prior knowledge)"
    )
    parser.add_argument("--grounding", type=Path, default=DEFAULT_GROUNDING)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None, help="Optional: only first N samples")
    parser.add_argument(
        "--test-model",
        action="store_true",
        help="Debug one sample: print raw response / hit, no CSV",
    )
    parser.add_argument(
        "--rag-mode",
        choices=ALL_MODES,
        default=None,
        help="Включить prior knowledge через LightRAG (store rag/stores/<mode>/)",
    )
    parser.add_argument(
        "--rag-query-mode",
        default="hybrid",
        choices=["naive", "local", "global", "hybrid", "mix"],
        help="Режим retrieval LightRAG при --rag-mode",
    )
    parser.add_argument(
        "--rag-stores-root",
        type=Path,
        default=DEFAULT_STORES_ROOT,
        help="Корень готовых RAG stores",
    )
    parser.add_argument(
        "--rag-max-chars",
        type=int,
        default=DEFAULT_RAG_MAX_CHARS,
        help="Обрезка RAG-ответа перед подачей в VLM",
    )
    parser.add_argument(
        "--rag-quiet",
        action="store_true",
        help="Без пошагового вывода [rag] во время eval",
    )
    args = parser.parse_args()

    load_dotenv()

    grounding_path = resolve_repo_path(args.grounding)
    out_dir = resolve_repo_path(args.out_dir)
    rag_stores_root = resolve_repo_path(args.rag_stores_root)
    csv_path = out_dir / "detailed.csv"
    md_path = out_dir / "baseline.md"
    json_path = out_dir / "results.json"

    if not grounding_path.exists():
        raise SystemExit(
            f"Grounding set not found: {grounding_path}. "
            "Run: python eval/build_eval_dataset.py"
        )

    samples = load_grounding_dataset(grounding_path, limit=args.limit)
    print(f"Loaded {len(samples)} grounding samples from {grounding_path}")
    if not samples:
        raise SystemExit("No samples to evaluate")

    client, model = load_model_client()
    pk_label = prior_knowledge_label(args.rag_mode)
    pk_meta = prior_knowledge_meta(
        rag_mode=args.rag_mode,
        rag_query_mode=args.rag_query_mode,
        rag_max_chars=args.rag_max_chars,
        rag_stores_root=rag_stores_root,
    )

    print(f"Model: {model}")
    print(f"Prior knowledge: {pk_label}")
    if args.rag_mode:
        print(f"RAG store: {rag_stores_root / args.rag_mode}")
        print(f"RAG query mode: {args.rag_query_mode}")

    rag_session: RagSession | None = None
    if args.rag_mode:
        rag_session = RagSession(
            args.rag_mode,
            stores_root=rag_stores_root,
            query_mode=args.rag_query_mode,
            verbose=not args.rag_quiet,
        )

    artifacts_dir = out_dir / "artifacts" if args.rag_mode else None

    try:
        if args.test_model:
            s = samples[0]
            rag_context: str | None = None
            if rag_session is not None:
                rag_context = fetch_rag_context(
                    rag_session,
                    task=s.instruction,
                    screen_id=s.screen_id,
                    max_chars=args.rag_max_chars,
                )
                print(f"RAG context ({len(rag_context or '')} chars):\n{rag_context[:500]}...")
            raw, pt, ct, tt = predict_point(client, model, s, rag_context=rag_context)
            point = extract_raw_point(raw)
            print(f"id: {s.id}")
            print(f"instruction: {s.instruction!r}")
            print(f"image: {s.image_path} ({s.image_width}x{s.image_height})")
            print(f"GT bbox_px: {s.bbox_px}")
            print(f"raw: {raw!r}")
            print(f"parsed: {point}")
            print(f"hit: {is_hit(point, s.bbox_px)}")
            print(f"tokens: prompt={pt} completion={ct} total={tt}")
            return

        rows = run_eval(
            client,
            model,
            samples,
            rag_session=rag_session,
            rag_max_chars=args.rag_max_chars,
            artifacts_dir=artifacts_dir,
        )
        summary = summarize(rows)
        print_summary(summary, prior_knowledge=pk_label)

        out_dir.mkdir(parents=True, exist_ok=True)
        save_csv(rows, csv_path)
        save_markdown(
            summary,
            md_path,
            model=model,
            grounding_path=grounding_path,
            n_samples=len(rows),
            prior_knowledge=pk_label,
            prior_knowledge_detail=pk_meta,
        )
        save_results_json(
            rows,
            summary,
            json_path,
            meta={
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "grounding_path": str(grounding_path).replace("\\", "/"),
                "prior_knowledge": pk_meta,
            },
        )
        print(f"\nSaved: {csv_path}")
        print(f"Saved: {md_path}")
        print(f"Saved: {json_path}")
        if artifacts_dir is not None:
            print(f"Artifacts: {artifacts_dir}")
    finally:
        if rag_session is not None:
            rag_session.close()


if __name__ == "__main__":
    main()
