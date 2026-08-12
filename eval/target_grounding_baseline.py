"""Baseline grounding на target GUI (без RAG/LoRA) — Qwen2.5-VL zero-shot.

Читает data/target_app/eval/grounding.json.
Метрики: click accuracy (точка в GT bbox), токены на пример.

Пример:
  python eval/target_grounding_baseline.py
  python eval/target_grounding_baseline.py --out-dir reports/target_baseline_run
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

DEFAULT_GROUNDING = Path("data/target_app/eval/grounding.json")
DEFAULT_OUT = Path("reports/target_grounding_baseline")


@dataclass
class GroundingSample:
    id: str
    screen_id: str
    image: Image.Image
    image_path: str
    instruction: str
    bbox_px: tuple[int, int, int, int]
    difficulty: str
    tags: list[str]
    image_width: int
    image_height: int


@dataclass
class EvalRow:
    id: str
    screen_id: str
    image_path: str
    instruction: str
    difficulty: str
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
    difficulties: set[str] | None = None,
) -> list[GroundingSample]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    samples: list[GroundingSample] = []
    for row in raw:
        difficulty = row.get("difficulty", "medium")
        if difficulties and difficulty not in difficulties:
            continue
        image_path = Path(row["image"])
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
                difficulty=difficulty,
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
) -> tuple[str, int, int, int]:
    response = client.chat.completions.create(
        model=model,
        messages=[
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
        ],
    )
    text = response.choices[0].message.content or ""
    usage = response.usage
    return (
        text,
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
        usage.total_tokens if usage else 0,
    )


def run_eval(client: OpenAI, model: str, samples: list[GroundingSample]) -> list[EvalRow]:
    rows: list[EvalRow] = []
    for sample in tqdm(samples, desc="Target grounding baseline"):
        raw, prompt_tokens, completion_tokens, total_tokens = predict_point(client, model, sample)
        point = extract_raw_point(raw)
        rows.append(
            EvalRow(
                id=sample.id,
                screen_id=sample.screen_id,
                image_path=sample.image_path,
                instruction=sample.instruction,
                difficulty=sample.difficulty,
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
            )
        )
    return rows


def summarize(rows: list[EvalRow]) -> dict[str, dict[str, float | int]]:
    by_diff: dict[str, list[EvalRow]] = defaultdict(list)
    by_screen: dict[str, list[EvalRow]] = defaultdict(list)
    for row in rows:
        by_diff[row.difficulty].append(row)
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
    for key in sorted(by_diff):
        summary[f"diff:{key}"] = pack(by_diff[key])
    for key in sorted(by_screen):
        summary[f"screen:{key}"] = pack(by_screen[key])
    return summary


def print_summary(summary: dict[str, dict[str, float | int]]) -> None:
    print("\n=== Target GUI grounding baseline (zero-shot, no prior knowledge) ===")
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
                "difficulty",
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
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row.id,
                    "screen_id": row.screen_id,
                    "image_path": row.image_path,
                    "instruction": row.instruction,
                    "difficulty": row.difficulty,
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
                }
            )


def save_markdown(
    summary: dict[str, dict[str, float | int]],
    path: Path,
    *,
    model: str,
    grounding_path: Path,
    n_samples: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Target GUI grounding baseline",
        "",
        f"- created_at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- model: `{model}`",
        "- prior_knowledge: **none** (no RAG / no LoRA)",
        f"- dataset: `{grounding_path.as_posix()}`",
        f"- samples: {n_samples}",
        "",
        "| slice | n | accuracy | total_tokens | avg_tokens | parse_fail |",
        "|---|---:|---:|---:|---:|---:|",
    ]
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot Qwen baseline on target GUI grounding set"
    )
    parser.add_argument("--grounding", type=Path, default=DEFAULT_GROUNDING)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None, help="Optional: only first N samples")
    parser.add_argument(
        "--difficulty",
        action="append",
        choices=("easy", "medium", "hard"),
        help="Filter by difficulty (repeatable)",
    )
    parser.add_argument(
        "--test-model",
        action="store_true",
        help="Debug one sample: print raw response / hit, no CSV",
    )
    args = parser.parse_args()

    load_dotenv()

    if not args.grounding.exists():
        raise SystemExit(
            f"Grounding set not found: {args.grounding}. "
            "Run: python eval/build_eval_dataset.py"
        )

    difficulties = set(args.difficulty) if args.difficulty else None
    samples = load_grounding_dataset(args.grounding, limit=args.limit, difficulties=difficulties)
    print(f"Loaded {len(samples)} grounding samples from {args.grounding}")
    if not samples:
        raise SystemExit("No samples to evaluate")

    client, model = load_model_client()

    if args.test_model:
        s = samples[0]
        raw, pt, ct, tt = predict_point(client, model, s)
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

    rows = run_eval(client, model, samples)
    summary = summarize(rows)
    print_summary(summary)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "detailed.csv"
    md_path = out_dir / "baseline.md"
    save_csv(rows, csv_path)
    save_markdown(
        summary,
        md_path,
        model=model,
        grounding_path=args.grounding,
        n_samples=len(rows),
    )
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
