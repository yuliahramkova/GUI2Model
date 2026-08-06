"""WebArena shopping intents — часть exploration через веб-доки (3-й источник).

Скачивает test.raw.json и пишет только shopping-задачи.

Запуск из корня репо:
  python explore/web_docs/extract_wa_shopping_intents.py
  python explore/web_docs/extract_wa_shopping_intents.py --skip-download

Файлы:
  data/docs_explore/webarena_test.raw.json
  data/docs_explore/shopping_intents.raw.json
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_URL = (
    "https://raw.githubusercontent.com/web-arena-x/webarena/main/config_files/test.raw.json"
)
DEFAULT_OUT_DIR = Path("data/docs_explore")
DEFAULT_RAW_FILE = DEFAULT_OUT_DIR / "webarena_test.raw.json"
DEFAULT_OUT = DEFAULT_OUT_DIR / "shopping_intents.raw.json"


def _load_tasks(raw: Any) -> list[dict[str, Any]]:
    """Нормализует test.raw.json к списку task-словарей."""
    if isinstance(raw, list):
        return [t for t in raw if isinstance(t, dict)]
    if isinstance(raw, dict):
        if "tasks" in raw and isinstance(raw["tasks"], list):
            return [t for t in raw["tasks"] if isinstance(t, dict)]
        values = list(raw.values())
        if values and all(isinstance(v, dict) for v in values):
            return values  # type: ignore[return-value]
    raise SystemExit(f"Непонятный формат test.raw.json: {type(raw)}")


def download_raw(url: str, dest: Path) -> None:
    """Качает test.raw.json в dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Качаю {url}")
    print(f"  → {dest.resolve()}")
    urllib.request.urlretrieve(url, dest) 
    print(f"Готово, размер {dest.stat().st_size} bytes")


def extract_shopping(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Оставляет задачи, где среди sites есть shopping."""
    out: list[dict[str, Any]] = []
    for t in tasks:
        sites = t.get("sites") or []
        if isinstance(sites, str):
            sites = [sites]
        if "shopping" not in sites:
            continue
        out.append(
            {
                "task_id": t.get("task_id"),
                "sites": sites,
                "intent": (t.get("intent") or "").strip(),
                "intent_template": (t.get("intent_template") or "").strip(),
                "start_url": t.get("start_url"),
                "require_login": t.get("require_login"),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract WebArena shopping intents (web_docs)")
    parser.add_argument("--url", default=DEFAULT_URL, help="URL test.raw.json")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_FILE, help="Путь к webarena_test.raw.json")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Куда писать shopping_intents.raw.json",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Не качать: использовать уже лежащий --raw",
    )
    args = parser.parse_args()

    if not args.skip_download:
        download_raw(args.url, args.raw)
    elif not args.raw.exists():
        raise SystemExit(f"Нет файла {args.raw}. Убери --skip-download или положи raw вручную.")

    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    tasks = _load_tasks(raw)
    shopping = extract_shopping(tasks)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "webarena_test.raw.json",
        "raw_file": str(args.raw).replace("\\", "/"),
        "n_tasks_total": len(tasks),
        "n_shopping": len(shopping),
        "intents": shopping,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Shopping intents: {len(shopping)} / {len(tasks)} → {args.out}")


if __name__ == "__main__":
    main()
