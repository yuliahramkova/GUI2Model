"""Сборка held-out eval датасета для target GUI (WebArena Shopping).

Grounding: paraphrased instructions из configs/eval_dataset.json + bbox
из a11y knowledge_base (match по screen_id + name/role).
Multi-step tasks: отдельные цели, не копии CUA train.

Пример:
  python eval/build_eval_dataset.py
  python eval/build_eval_dataset.py --config configs/eval_dataset.json

Результат: data/target_app/eval/
  grounding.json, tasks.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path("configs/eval_dataset.json")
DEFAULT_OUT = Path("data/target_app/eval")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_screenshot(states_dir: Path, screen_id: str, kb_image: str | None) -> Path | None:
    """Ищет screenshot экрана: states/<id>/screenshot.png или путь из KB."""
    candidates: list[Path] = [
        states_dir / screen_id / "screenshot.png",
        Path("data/a11y_explore") / "states" / screen_id / "screenshot.png",
    ]
    if kb_image:
        candidates.append(Path("data/a11y_explore") / kb_image)
        # KB иногда пишет screens/, фактически лежит states/
        candidates.append(Path("data/a11y_explore") / kb_image.replace("screens/", "states/", 1))
    for path in candidates:
        if path.is_file():
            return path
    return None


def index_a11y(kb: dict[str, Any]) -> tuple[dict[str, dict], list[dict]]:
    screens = {s["id"]: s for s in kb.get("screens", [])}
    elements = kb.get("elements", [])
    return screens, elements


def find_element(
    elements: list[dict[str, Any]],
    screen_id: str,
    match: dict[str, Any],
) -> dict[str, Any] | None:
    name = match.get("name")
    role = match.get("role")
    element_id = match.get("element_id")
    name_contains = match.get("name_contains")

    candidates = [e for e in elements if e.get("screen_id") == screen_id]
    if element_id:
        for e in candidates:
            if e.get("element_id") == element_id:
                return e
        return None

    scored: list[tuple[int, dict[str, Any]]] = []
    for e in candidates:
        score = 0
        ename = e.get("name") or ""
        erole = e.get("role") or ""
        if name is not None and ename != name:
            if name_contains and name_contains in ename:
                score += 1
            else:
                continue
        elif name is not None:
            score += 2
        if role is not None:
            if erole != role:
                continue
            score += 2
        if name_contains and name is None:
            if name_contains not in ename:
                continue
            score += 1
        scored.append((score, e))

    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    return scored[0][1]


def assert_paraphrase(instruction: str, a11y_instruction: str | None, strict: bool) -> None:
    if not strict or not a11y_instruction:
        return
    if instruction.strip().lower() == a11y_instruction.strip().lower():
        raise ValueError(
            f"Eval instruction must not copy a11y instruction verbatim: {instruction!r}"
        )


def build_grounding(
    cfg: dict[str, Any],
    screens: dict[str, dict],
    elements: list[dict[str, Any]],
    states_dir: Path,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    strict = bool(cfg.get("train_exclusion", {}).get("forbid_a11y_instruction_as_eval_prompt", True))
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    shots_dir = out_dir / "screenshots"
    shots_dir.mkdir(parents=True, exist_ok=True)

    for item in cfg.get("grounding", []):
        gid = item["id"]
        screen_id = item["screen_id"]
        match = item.get("match") or {}
        instruction = item["instruction"]

        el = find_element(elements, screen_id, match)
        if el is None:
            errors.append(f"[grounding:{gid}] element not found for {screen_id=} {match=}")
            continue

        try:
            assert_paraphrase(instruction, el.get("instruction"), strict=strict)
        except ValueError as exc:
            errors.append(f"[grounding:{gid}] {exc}")
            continue

        kb_image = (screens.get(screen_id) or {}).get("image")
        src = resolve_screenshot(states_dir, screen_id, kb_image)
        if src is None:
            errors.append(f"[grounding:{gid}] screenshot missing for screen {screen_id}")
            continue

        dest_name = f"{screen_id}.png"
        dest = shots_dir / dest_name
        if not dest.exists() or dest.stat().st_mtime < src.stat().st_mtime:
            shutil.copy2(src, dest)

        bbox = el.get("bbox_px")
        if not bbox or len(bbox) != 4:
            errors.append(f"[grounding:{gid}] bad bbox_px: {bbox}")
            continue

        viewport = (screens.get(screen_id) or {}).get("viewport") or {}
        rows.append(
            {
                "id": gid,
                "type": "grounding",
                "screen_id": screen_id,
                "image": str(Path("data/target_app/eval/screenshots") / dest_name).replace("\\", "/"),
                "instruction": instruction,
                "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                "bbox_norm": el.get("bbox_norm"),
                "viewport": viewport,
                "element_id": el.get("element_id"),
                "element_role": el.get("role"),
                "element_name": el.get("name"),
                "a11y_instruction_ref": el.get("instruction"),
                "tags": item.get("tags", []),
            }
        )

    return rows, errors


def build_tasks(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    excluded = set(cfg.get("train_exclusion", {}).get("cua_task_ids", []))
    tasks: list[dict[str, Any]] = []
    errors: list[str] = []

    for item in cfg.get("tasks", []):
        tid = item["id"]
        if tid in excluded:
            errors.append(f"[task:{tid}] id collides with train CUA task id")
            continue
        if not item.get("goal"):
            errors.append(f"[task:{tid}] missing goal")
            continue
        if not item.get("expected_end_url_contains") and not item.get("expected_end_title_contains"):
            errors.append(f"[task:{tid}] need at least one success criterion")
            continue

        tasks.append(
            {
                "id": tid,
                "type": "multi_step",
                "goal": item["goal"],
                "start_url": item.get("start_url", "/"),
                "max_steps": item.get("max_steps", 8),
                "credentials_ref": item.get("credentials_ref"),
                "expected_end_url_contains": item.get("expected_end_url_contains"),
                "expected_end_title_contains": item.get("expected_end_title_contains"),
                "expected_end_title_not_contains": item.get("expected_end_title_not_contains"),
                "setup_hint": item.get("setup_hint"),
                "expected_steps": item.get("expected_steps"),
                "tags": item.get("tags", []),
                "notes": item.get("notes"),
            }
        )

    return tasks, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Build held-out target-GUI eval dataset")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit non-zero if any grounding/task failed to resolve",
    )
    args = parser.parse_args()

    if not args.config.exists():
        raise SystemExit(f"Config not found: {args.config}")

    cfg = load_json(args.config)
    kb_path = Path(cfg.get("a11y_kb", "data/a11y_explore/knowledge_base.json"))
    states_dir = Path(cfg.get("a11y_states_dir", "data/a11y_explore/states"))
    if not kb_path.exists():
        raise SystemExit(f"A11y KB not found: {kb_path}")

    kb = load_json(kb_path)
    screens, elements = index_a11y(kb)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    grounding, g_errors = build_grounding(cfg, screens, elements, states_dir, out_dir)
    tasks, t_errors = build_tasks(cfg)
    errors = g_errors + t_errors

    for row in grounding:
        print(f"  grounding OK  {row['id']}: {row['instruction'][:60]}... @ {row['bbox']}")
    for task in tasks:
        print(f"  task OK       {task['id']}: {task['goal'][:60]}...")
    for err in errors:
        print(f"  ERROR {err}")

    (out_dir / "grounding.json").write_text(
        json.dumps(grounding, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    tasks_doc = {
        "app": cfg.get("app", "webarena_shopping"),
        "source": "eval_held_out",
        "credentials": cfg.get("credentials", {}),
        "tasks": tasks,
    }
    (out_dir / "tasks.json").write_text(
        json.dumps(tasks_doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nDone.")
    print(f"  grounding: {len(grounding)}")
    print(f"  tasks:     {len(tasks)}")
    print(f"  errors:    {len(errors)}")
    print(f"  out:       {out_dir}")

    if errors and args.fail_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
