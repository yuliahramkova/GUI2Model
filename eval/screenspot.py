"""Расчет метрик для ScreenSpot без предзнания Qwen2.5-VL-7B-Instruct (baseline)."""

from __future__ import annotations

import argparse
import base64
import csv
import io
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

DEFAULT_DATASET = "bevaya/ScreenSpot"
DEFAULT_SPLIT = "test"
CATEGORIES = ("desktop", "mobile", "web")
REPORTS_DIR = Path("reports")

SOURCE_TO_CATEGORY = {
    "windows": "desktop",
    "macos": "desktop",
    "ios": "mobile",
    "android": "mobile",
    "gitlab": "web",
    "shop": "web",
    "forum": "web",
    "tool": "web",
}

@dataclass
class ScreenSpotSample:
    """Один пример из датасета: скриншот, инструкция и GT bbox."""

    image: Image.Image
    instruction: str
    bbox_norm_1000: tuple[int, int, int, int]
    category: str
    data_type: str
    data_source: str
    file_name: str


@dataclass
class EvalRow:
    """Результат eval для одного примера: предсказание, hit/miss, токены."""

    file_name: str
    instruction: str
    category: str
    data_type: str
    bbox: tuple[int, int, int, int]
    raw_response: str
    pred_x: int | None
    pred_y: int | None
    hit: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    steps: int


def load_screenspot_dataset(
    dataset_repo: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    limit: int | None = None,
) -> list[ScreenSpotSample]:
    """Загружает ScreenSpot с HuggingFace, нормализует bbox в [0, 1000]."""
    dataset = load_dataset(dataset_repo, split=split, streaming=True)
    samples: list[ScreenSpotSample] = []

    for row in dataset:
        image = row["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        x1, y1, x2, y2 = row["bbox"]
        bbox = (round(x1 * 1000), round(y1 * 1000), round(x2 * 1000), round(y2 * 1000))

        data_source = row["data_source"]
        samples.append(
            ScreenSpotSample(
                image=image,
                instruction=row["instruction"],
                bbox_norm_1000=bbox,
                category=SOURCE_TO_CATEGORY[data_source],
                data_type=row["data_type"],
                data_source=data_source,
                file_name=row.get("file_name", ""),
            )
        )
        if limit is not None and len(samples) >= limit:
            break

    return samples


def load_model_client() -> tuple[OpenAI, str]:
    """Создаёт OpenAI-клиент и id модели из переменных .env."""
    missing = [k for k in ("LNSIGO_API_KEY", "LNSIGO_BASE_URL", "LNSIGO_MODEL") if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing in .env: {', '.join(missing)}")

    client = OpenAI(
        api_key=os.environ["LNSIGO_API_KEY"],
        base_url=os.environ["LNSIGO_BASE_URL"],
    )
    return client, os.environ["LNSIGO_MODEL"]


def image_to_data_url(image: Image.Image) -> str:
    """Кодирует PIL-изображение в data URL для multimodal API."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_grounding_prompt(instruction: str) -> str:
    """Собирает текстовый промпт: найти элемент и вернуть (x, y) в [0, 1000]."""
    return (
        f"What is the location of the element corresponding to the instruction: "
        f"'{instruction}'? Provide the coordinates as (x, y) in the range [0, 1000]."
    )


def parse_point(text: str) -> tuple[int, int] | None:
    """Извлекает (x, y) из ответа модели. None, если распарсить не удалось."""
    if not text:
        return None

    box_match = re.search(
        r"<\|box_start\|>\s*\(?(\d+)\s*,\s*(\d+)\)?\s*<\|box_end\|>",
        text,
    )
    if box_match:
        x, y = int(box_match.group(1)), int(box_match.group(2))
    else:
        paren_match = re.search(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
        if paren_match:
            x, y = int(paren_match.group(1)), int(paren_match.group(2))
        else:
            nums = [int(n) for n in re.findall(r"\d+", text)]
            if len(nums) < 2:
                return None
            x, y = nums[0], nums[1]

    if 0 <= x <= 1000 and 0 <= y <= 1000:
        return x, y
    return None


def is_hit(point: tuple[int, int] | None, bbox: tuple[int, int, int, int]) -> bool:
    """True, если точка попала в GT bbox (ScreenSpot click accuracy)."""
    if point is None:
        return False
    px, py = point
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2


def predict_point(client: OpenAI, model: str, sample: ScreenSpotSample) -> tuple[str, int, int, int]:
    """Один запрос к модели: image + instruction → текст ответа и usage-токены."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_url(sample.image)}},
                    {"type": "text", "text": build_grounding_prompt(sample.instruction)},
                ],
            }
        ],
    )
    text = response.choices[0].message.content or ""
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    total_tokens = usage.total_tokens if usage else 0
    return text, prompt_tokens, completion_tokens, total_tokens


def run_eval(client: OpenAI, model: str, samples: list[ScreenSpotSample]) -> list[EvalRow]:
    """Прогоняет все примеры: infer → parse → hit/miss."""
    rows: list[EvalRow] = []
    for sample in tqdm(samples, desc="ScreenSpot eval"):
        raw, prompt_tokens, completion_tokens, total_tokens = predict_point(client, model, sample)
        point = parse_point(raw)
        rows.append(
            EvalRow(
                file_name=sample.file_name,
                instruction=sample.instruction,
                category=sample.category,
                data_type=sample.data_type,
                bbox=sample.bbox_norm_1000,
                raw_response=raw,
                pred_x=point[0] if point else None,
                pred_y=point[1] if point else None,
                hit=is_hit(point, sample.bbox_norm_1000),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                steps=1,
            )
        )
    return rows


def summarize(rows: list[EvalRow]) -> dict[str, dict[str, float | int]]:
    """Считает accuracy, token/step budget и parse_fail по overall и категориям."""
    by_cat: dict[str, list[EvalRow]] = defaultdict(list)
    for row in rows:
        by_cat[row.category].append(row)

    summary: dict[str, dict[str, float | int]] = {}
    for cat in ("overall", *CATEGORIES):
        subset = rows if cat == "overall" else by_cat.get(cat, [])
        n = len(subset)
        hits = sum(r.hit for r in subset)
        summary[cat] = {
            "n": n,
            "hits": hits,
            "accuracy": hits / n if n else 0.0,
            "total_tokens": sum(r.total_tokens for r in subset),
            "avg_tokens": sum(r.total_tokens for r in subset) / n if n else 0.0,
            "total_steps": sum(r.steps for r in subset),
            "avg_steps": sum(r.steps for r in subset) / n if n else 0.0,
            "parse_fail": sum(1 for r in subset if r.pred_x is None),
        }
    return summary


def print_summary(summary: dict[str, dict[str, float | int]]) -> None:
    """Печатает сводную таблицу метрик в консоль."""
    print("\n=== ScreenSpot results ===")
    print(f"{'category':<10} {'n':>5} {'acc':>8} {'tokens':>10} {'avg_tok':>10} {'steps':>8} {'avg_step':>10} {'parse_fail':>10}")
    for cat in ("overall", *CATEGORIES):
        s = summary[cat]
        print(
            f"{cat:<10} {s['n']:>5} {s['accuracy']:>8.1%} "
            f"{s['total_tokens']:>10} {s['avg_tokens']:>10.1f} "
            f"{s['total_steps']:>8} {s['avg_steps']:>10.1f} {s['parse_fail']:>10}"
        )


def save_csv(rows: list[EvalRow], path: Path) -> None:
    """Сохраняет построчные результаты eval в CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        f.write("sep=,\n")
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_name", "instruction", "category", "data_type",
                "bbox", "pred_x", "pred_y", "hit", "raw_response",
                "prompt_tokens", "completion_tokens", "total_tokens", "steps",
            ],
            delimiter=",",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "file_name": row.file_name,
                    "instruction": row.instruction,
                    "category": row.category,
                    "data_type": row.data_type,
                    "bbox": row.bbox,
                    "pred_x": row.pred_x,
                    "pred_y": row.pred_y,
                    "hit": row.hit,
                    "raw_response": row.raw_response,
                    "prompt_tokens": row.prompt_tokens,
                    "completion_tokens": row.completion_tokens,
                    "total_tokens": row.total_tokens,
                    "steps": row.steps,
                }
            )


def save_markdown(summary: dict[str, dict[str, float | int]], path: Path, n_samples: int) -> None:
    """Сохраняет сводку baseline (accuracy + budget) в Markdown."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ScreenSpot baseline",
        "",
        f"Samples evaluated: {n_samples}",
        "",
        "| category | n | accuracy | total_tokens | avg_tokens | total_steps | avg_steps | parse_fail |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cat in ("overall", *CATEGORIES):
        s = summary[cat]
        lines.append(
            f"| {cat} | {s['n']} | {s['accuracy']:.1%} | {s['total_tokens']} "
            f"| {s['avg_tokens']:.1f} | {s['total_steps']} | {s['avg_steps']:.1f} | {s['parse_fail']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """CLI: загрузка датасета, eval или --test-model на одном примере."""
    parser = argparse.ArgumentParser(description="ScreenSpot grounding eval")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--test-model",
        action="store_true",
        help="Debug: one sample, print raw response and parsed point",
    )
    args = parser.parse_args()

    load_dotenv()

    print(f"Loading {args.dataset} [{args.split}], limit={args.limit}")
    samples = load_screenspot_dataset(args.dataset, args.split, args.limit)
    print(f"Loaded {len(samples)} samples")

    if not samples:
        return

    client, model = load_model_client()

    if args.test_model:
        s = samples[0]
        raw, *_ = predict_point(client, model, s)
        point = parse_point(raw)
        print(f"\nModel: {model}")
        print(f"Instruction: {s.instruction!r}")
        print(f"GT bbox: {s.bbox_norm_1000}")
        print(f"Raw response: {raw!r}")
        print(f"Parsed point: {point}")
        print(f"Hit: {is_hit(point, s.bbox_norm_1000)}")
        return

    rows = run_eval(client, model, samples)
    summary = summarize(rows)
    print_summary(summary)

    save_csv(rows, REPORTS_DIR / "screenspot_detailed.csv")
    save_markdown(summary, REPORTS_DIR / "baseline.md", len(rows))
    print(f"\nSaved: {REPORTS_DIR / 'screenspot_detailed.csv'}")
    print(f"Saved: {REPORTS_DIR / 'baseline.md'}")


if __name__ == "__main__":
    main()
