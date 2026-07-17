"""Анализ reports/screenspot_detailed.csv после ScreenSpot eval."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from PIL import Image

CSV_PATH = Path("reports/screenspot_detailed.csv")
EXAMPLES_DIR = Path("reports/parse_fail_examples")
ANALYSIS_MD = Path("reports/screenspot_analysis.md")
DATASET = "bevaya/ScreenSpot"

# Два ответа из no_coords_other (Gmail / full screen) считаем как not_found_refuse.
MANUAL_MOVE_OTHER_TO_REFUSE = 2

REFUSE_KEYWORDS = (
    "not visible",
    "not present",
    "not found",
    "does not contain",
    "no element",
    "not directly visible",
    "cannot find",
    "unable to",
    "not shown",
    "there is no",
    "no button",
    "not labeled",
    "could not find",
    "i cannot",
    "i'm unable",
)

def load_csv(path: Path) -> list[dict[str, str]]:
    """Читает CSV."""
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    if lines and lines[0].lower().startswith("sep="):
        lines = lines[1:]

    header = lines[0]
    delimiter = ";" if header.count(";") > header.count(",") else ","
    return list(csv.DictReader(lines, delimiter=delimiter))


def extract_point(text: str) -> tuple[int, int] | None:
    """Достаёт первую пару чисел из ответа (как в eval, но без фильтра [0,1000])."""
    if not text:
        return None
    box = re.search(
        r"<\|box_start\|>\s*\(?(-?\d+)\s*,\s*(-?\d+)\)?\s*<\|box_end\|>",
        text,
    )
    if box:
        return int(box.group(1)), int(box.group(2))
    paren = re.search(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", text)
    if paren:
        return int(paren.group(1)), int(paren.group(2))
    nums = [int(n) for n in re.findall(r"-?\d+", text)]
    if len(nums) >= 2:
        return nums[0], nums[1]
    return None


def classify_parse_fail(raw: str) -> str:
    """Тип неудачного парсинга."""
    if not raw or not raw.strip():
        return "empty_response"

    point = extract_point(raw)
    if point is not None:
        x, y = point
        if not (0 <= x <= 1000 and 0 <= y <= 1000):
            return "out_of_range"  # координата есть, но не в [0, 1000]
        return "in_range_unparsed"  # формат странный, хотя числа в диапазоне

    low = raw.lower()
    if any(k in low for k in REFUSE_KEYWORDS):
        return "not_found_refuse"  # модель говорит, что элемента нет

    return "no_coords_other"  # текст без координат, без явного отказа


def print_parse_fail_stats(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """Считает и печатает типы parse_fail. Возвращает примеры по типам."""
    fails = [r for r in rows if not r.get("pred_x")]
    by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in fails:
        by_type[classify_parse_fail(row.get("raw_response", ""))].append(row)

    print("\n=== Parse fail by type ===")
    print(f"{'type':<22} {'n':>6} {'% of fails':>10} {'% of all':>10}")
    n_fail, n_all = len(fails), len(rows)
    for fail_type, examples in sorted(by_type.items(), key=lambda x: -len(x[1])):
        n = len(examples)
        print(
            f"{fail_type:<22} {n:>6} "
            f"{n / n_fail if n_fail else 0:>9.1%} "
            f"{n / n_all if n_all else 0:>9.1%}"
        )
    print(f"{'TOTAL fails':<22} {n_fail:>6} {1.0 if n_fail else 0:>9.1%} {n_fail / n_all if n_all else 0:>9.1%}")

    others = by_type.get("no_coords_other", [])
    if others:
        print(f"\n--- raw_response for no_coords_other ({len(others)}) ---")
        for ex in others:
            print(ex["raw_response"])
            print()

    return by_type


def data_type_stats(rows: list[dict[str, str]]) -> list[tuple[str, int, float, int, float, int]]:
    """Возвращает строки метрик по data_type: (dt, n, acc, tokens, avg_tok, parse_fail)."""
    by_dt: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_dt[row.get("data_type", "unknown")].append(row)

    result = []
    for dt in sorted(by_dt):
        subset = by_dt[dt]
        n = len(subset)
        hits = sum(1 for r in subset if str(r.get("hit", "")).lower() in ("true", "1"))
        tokens = sum(int(r.get("total_tokens") or 0) for r in subset)
        fails = sum(1 for r in subset if not r.get("pred_x"))
        result.append((dt, n, hits / n if n else 0.0, tokens, tokens / n if n else 0.0, fails))
    return result


def print_data_type_stats(rows: list[dict[str, str]]) -> None:
    """Accuracy и токены по data_type (text / icon)."""
    print("\n=== Metrics by data_type ===")
    print(
        f"{'data_type':<10} {'n':>6} {'acc':>8} {'tokens':>10} {'avg_tok':>10} "
        f"{'parse_fail':>10}"
    )
    for dt, n, acc, tokens, avg_tok, fails in data_type_stats(rows):
        print(f"{dt:<10} {n:>6} {acc:>8.1%} {tokens:>10} {avg_tok:>10.1f} {fails:>10}")


def adjusted_fail_counts(
    by_type: dict[str, list[dict[str, str]]],
) -> dict[str, int]:
    """Счётчики parse_fail с ручным переносом 2 штук other → refuse."""
    counts = {k: len(v) for k, v in by_type.items()}
    move = min(MANUAL_MOVE_OTHER_TO_REFUSE, counts.get("no_coords_other", 0))
    counts["no_coords_other"] = counts.get("no_coords_other", 0) - move
    counts["not_found_refuse"] = counts.get("not_found_refuse", 0) + move
    if counts.get("no_coords_other", 0) == 0:
        counts.pop("no_coords_other", None)
    return counts


def save_analysis_markdown(
    rows: list[dict[str, str]],
    by_type: dict[str, list[dict[str, str]]],
    path: Path = ANALYSIS_MD,
) -> None:
    """Пишет две таблицы в md (без raw_response); 2 other перенесены в refuse."""
    n_all = len(rows)
    n_fail = sum(len(v) for v in by_type.values())
    counts = adjusted_fail_counts(by_type)

    lines = [
        "# ScreenSpot analysis",
        "",
        f"Samples: {n_all}. Parse fails: {n_fail}.",
        "",
        "Note: 2 responses from `no_coords_other` (Gmail / full screen) "
        "are counted as `not_found_refuse`.",
        "",
        "## Parse fail by type",
        "",
        "| type | n | % of fails | % of all |",
        "|---|---:|---:|---:|",
    ]
    for fail_type, n in sorted(counts.items(), key=lambda x: -x[1]):
        if n == 0:
            continue
        lines.append(
            f"| {fail_type} | {n} | {n / n_fail if n_fail else 0:.1%} "
            f"| {n / n_all if n_all else 0:.1%} |"
        )
    lines.append(
        f"| TOTAL fails | {n_fail} | {1.0 if n_fail else 0:.1%} "
        f"| {n_fail / n_all if n_all else 0:.1%} |"
    )

    lines += [
        "",
        "## Metrics by data_type",
        "",
        "| data_type | n | accuracy | total_tokens | avg_tokens | parse_fail |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dt, n, acc, tokens, avg_tok, fails in data_type_stats(rows):
        lines.append(
            f"| {dt} | {n} | {acc:.1%} | {tokens} | {avg_tok:.1f} | {fails} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved: {path}")


def load_image_by_filename(file_name: str) -> Image.Image | None:
    """Достаёт скриншот из ScreenSpot по file_name."""
    load_dotenv()
    ds = load_dataset(DATASET, split="test", streaming=True)
    for row in ds:
        if row.get("file_name") == file_name:
            image = row["image"]
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")
            return image
    return None


def save_one_image_per_fail_type(by_type: dict[str, list[dict[str, str]]]) -> None:
    """Сохраняет по одной картинке на тип parse_fail в reports/parse_fail_examples/."""
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for fail_type, examples in sorted(by_type.items()):
        row = examples[0]
        file_name = row.get("file_name")
        if not file_name:
            continue
        image = load_image_by_filename(file_name)
        if image is None:
            continue
        image.save(EXAMPLES_DIR / f"{fail_type}__{row['file_name']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ScreenSpot eval CSV")
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip saving example images (stats only)",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    rows = load_csv(args.csv)
    print(f"Loaded {len(rows)} rows from {args.csv}")

    by_type = print_parse_fail_stats(rows)
    print_data_type_stats(rows)
    save_analysis_markdown(rows, by_type)

    if not args.skip_images:
        save_one_image_per_fail_type(by_type)


if __name__ == "__main__":
    main()
