"""Обход GUI через teacher-VLM / CUA с записью трейсов и базы знаний.

Скрипт является прототипом exploration-пайплайна:
- читает задачи из `configs/cua_tasks.json`;
- поддерживает два режима наблюдения:
    1) screenshot_only
    2) screenshot_plus_som
- пишет пошаговые трейсы (JSONL) и собирает агрегированный knowledge base.

Важные идеи:
- в режиме screenshot_plus_som список элементов строится заново на каждом шаге
  по текущему состоянию страницы;
- knowledge_base.json дописывается между запусками (повтор задачи заменяет запись);
- `transitions` собираются из реально выполненных шагов: from_state -> action -> to_state.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urljoin

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import Page, sync_playwright

try:
    # Библиотека нужна для OpenAI-compatible API.
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


ObservationMode = Literal["screenshot_only", "screenshot_plus_som"]


DEFAULT_CONFIG = Path("configs/cua_tasks.json")
DEFAULT_OUT_DIR = Path("data/cua_explore")
DEFAULT_STATES_DIR = "states"
DEFAULT_TRACES_DIR = "traces"


INTERACTIVE_TAGS = {"a", "button", "input", "select", "textarea", "summary"}
INTERACTIVE_ROLES = {
    "button",
    "link",
    "textbox",
    "searchbox",
    "combobox",
    "checkbox",
    "radio",
    "menuitem",
    "tab",
    "option",
    "switch",
    "slider",
}


ACTION_SCHEMA_NOTE = (
    "Верни ТОЛЬКО JSON-объект без markdown. "
    "Обязательный ключ: `action`. "
    "Для click/type/press укажи `target_mark_id`, "
    "а для screenshot_only как запасной вариант можно вернуть `click_x` и `click_y` "
    "в нормализованных координатах [0,1]. "
    "Для action=`type` положи вводимый текст в `value`. "
    "Для action=`press` положи клавишу в `value` (например, 'Enter' или 'Escape'). "
    "Для action=`scroll` положи целое смещение dy в `value`: "
    "строго >0 вниз, <0 вверх. "
    "Минус НЕ значит «вниз». Если экран не изменился после scroll — не повторяй то же value. "
    "Для action=`wait` положи время в миллисекундах в `value`. "
    "Для action=`done` положи пустую строку в `value`."
)


@dataclass(frozen=True)
class Mark:
    """Описывает один размеченный элемент в Set-of-Mark представлении."""

    mark_id: int
    role: str
    name: str
    bbox_px: list[int]
    bbox_norm: list[float]


def _sha_state_id(url: str, title: str) -> str:
    """Строит короткий стабильный идентификатор состояния по URL и title."""
    norm_url = url.split("#", 1)[0]
    raw = f"{norm_url}::{title}".encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()[:12]


def _abs_url(base_url: str, path: str) -> str:
    """Собирает абсолютный URL из base_url и относительного path."""
    return urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def _read_image_as_b64(path: Path) -> str:
    """Читает изображение и кодирует его в base64 для API teacher-модели."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _extract_json_obj(text: str) -> dict[str, Any] | None:
    """Пытается извлечь JSON-объект из текстового ответа teacher-модели."""
    text = text.strip()
    # Быстрый путь: ответ уже чистый JSON.
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass
    # Иначе ищем первый JSON-подобный блок в тексте.
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def collect_elements_dom(page: Page) -> list[dict[str, Any]]:
    """Собирает видимые интерактивные элементы и их bbox на текущей странице."""
    return page.evaluate(
        """({ interactiveTags, interactiveRoles }) => {
          const tagSet = new Set(interactiveTags);
          const roleSet = new Set(interactiveRoles);
          const out = [];
          const seen = new Set();

          const implicitRole = (el) => {
            const tag = el.tagName.toLowerCase();
            if (tag === 'a' && el.hasAttribute('href')) return 'link';
            if (tag === 'button') return 'button';
            if (tag === 'input') {
              const t = (el.getAttribute('type') || 'text').toLowerCase();
              if (t === 'checkbox') return 'checkbox';
              if (t === 'radio') return 'radio';
              if (t === 'submit' || t === 'button') return 'button';
              if (t === 'search') return 'searchbox';
              return 'textbox';
            }
            if (tag === 'select') return 'combobox';
            if (tag === 'textarea') return 'textbox';
            return el.getAttribute('role') || tag;
          };

          const accessibleName = (el) => {
            const aria = el.getAttribute('aria-label');
            if (aria && aria.trim()) return aria.trim();
            const labelled = el.getAttribute('aria-labelledby');
            if (labelled) {
              const parts = labelled.split(/\\s+/).map(id => {
                const n = document.getElementById(id);
                return n ? n.innerText.trim() : '';
              }).filter(Boolean);
              if (parts.length) return parts.join(' ');
            }
            if (el.tagName.toLowerCase() === 'input') {
              const ph = el.getAttribute('placeholder');
              if (ph && ph.trim()) return ph.trim();
              const t = el.getAttribute('title');
              if (t && t.trim()) return t.trim();
            }
            const text = (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim();
            if (text) return text.slice(0, 120);
            const title = (el.getAttribute('title') || '').trim();
            return title ? title.slice(0, 120) : '';
          };

          // 0 = ключевые CTA (товар / View and Edit Cart), 1 = обычные, 2 = шум сайдбара.
          const priorityOf = (el) => {
            const nameLower = ((el.innerText || el.textContent || el.getAttribute('title') || '')
              .replace(/\\s+/g, ' ').trim().toLowerCase());
            if (el.classList.contains('product-item-link')) return 0;
            if (el.closest('.product-item-name')) return 0;
            // Magento minicart: иначе ссылка получает большой mark_id и модель кликает соседний 8/9.
            if (el.classList.contains('viewcart') || nameLower.includes('view and edit cart')) return 0;
            if (el.closest('.minicart-wrapper') && (el.classList.contains('action') || el.tagName === 'A')) {
              if (nameLower.includes('view and edit') || nameLower.includes('checkout')) return 0;
            }
            if (el.closest('.sidebar, .sidebar-main, .filter, .filter-options, .block-filter')) return 2;
            return 1;
          };

          const pushEl = (el) => {
            if (!(el instanceof HTMLElement)) return;
            const style = window.getComputedStyle(el);
            if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
              return;
            }
            const tag = el.tagName.toLowerCase();
            const role = (el.getAttribute('role') || implicitRole(el) || '').toLowerCase();
            const interesting =
              tagSet.has(tag) ||
              roleSet.has(role) ||
              el.onclick != null ||
              el.getAttribute('tabindex') === '0' ||
              el.classList.contains('product-item-link');
            if (!interesting) return;

            const r = el.getBoundingClientRect();
            if (r.width < 2 || r.height < 2) return;
            if (r.bottom < 0 || r.right < 0 || r.top > window.innerHeight || r.left > window.innerWidth) {
              return;
            }

            const name = accessibleName(el);
            if (!name) return;
            const key = [role, name, Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)].join('|');
            if (seen.has(key)) return;
            seen.add(key);

            out.push({
              role,
              name,
              tag,
              priority: priorityOf(el),
              bbox_px: [
                Math.round(r.x),
                Math.round(r.y),
                Math.round(r.x + r.width),
                Math.round(r.y + r.height),
              ],
              bbox_norm: [
                r.x / window.innerWidth,
                r.y / window.innerHeight,
                (r.x + r.width) / window.innerWidth,
                (r.y + r.height) / window.innerHeight,
              ],
            });
          };

          // Сначала явно названия товаров на листинге/выдаче.
          document.querySelectorAll('a.product-item-link, .product-item-name a').forEach((el) => pushEl(el));
          Array.from(document.querySelectorAll('body *')).forEach((el) => pushEl(el));
          return out;
        }""",
        {"interactiveTags": list(INTERACTIVE_TAGS), "interactiveRoles": list(INTERACTIVE_ROLES)},
    )


def build_marks(
    elements: list[dict[str, Any]],
    *,
    max_marks: int,
) -> list[Mark]:
    """Преобразует элементы в пронумерованные marks для teacher-модели."""
    filtered: list[dict[str, Any]] = []
    for el in elements:
        role = (el.get("role") or "").lower().strip()
        name = (el.get("name") or "").strip()
        bbox_px = el.get("bbox_px") or []
        bbox_norm = el.get("bbox_norm") or []
        if not name:
            continue
        if "{" in name or "}" in name:
            continue
        if role not in INTERACTIVE_ROLES and role not in {"menu", "menuitem"}:
            continue
        if len(bbox_norm) != 4 or any(x is None for x in bbox_norm):
            continue
        x1, y1, x2, y2 = bbox_norm
        if not (0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1):
            continue
        if (x2 - x1) * (y2 - y1) > 0.25:
            continue
        if len(bbox_px) != 4:
            continue
        filtered.append(
            {
                "role": role,
                "name": name,
                "bbox_px": bbox_px,
                "bbox_norm": bbox_norm,
                "priority": int(el.get("priority", 1)),
            }
        )

    # Сначала product title links, потом остальное; сайдбар-фильтры в конце.
    filtered.sort(key=lambda e: (e["priority"], e["bbox_norm"][1], e["bbox_norm"][0], e["name"]))
    out: list[Mark] = []
    for idx, el in enumerate(filtered[:max_marks], start=1):
        out.append(
            Mark(
                mark_id=idx,
                role=el["role"],
                name=el["name"],
                bbox_px=[int(el["bbox_px"][0]), int(el["bbox_px"][1]), int(el["bbox_px"][2]), int(el["bbox_px"][3])],
                bbox_norm=[float(el["bbox_norm"][0]), float(el["bbox_norm"][1]), float(el["bbox_norm"][2]), float(el["bbox_norm"][3])],
            )
        )
    return out


def draw_som_overlay(
    screenshot_path: Path,
    marks: list[Mark],
    overlay_path: Path,
) -> None:
    """Рисует поверх скриншота нумерованные рамки для Set-of-Mark."""
    img = Image.open(screenshot_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Пытаемся взять системный шрифт, иначе используем встроенный.
    try:
        font = ImageFont.truetype("arial.ttf", size=18)
    except Exception:  # pragma: no cover
        font = ImageFont.load_default()

    for m in marks:
        x1, y1, x2, y2 = m.bbox_px
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        label = str(m.mark_id)
        text_w, text_h = draw.textbbox((0, 0), label, font=font)[2:]
        pad = 4
        draw.rectangle([x1, max(0, y1 - (text_h + pad)), x1 + text_w + pad * 2, y1], fill=(255, 0, 0))
        draw.text((x1 + pad, max(0, y1 - (text_h + pad / 2))), label, fill=(255, 255, 255), font=font)

    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(overlay_path)


class OpenAITeacher:
    """Teacher для OpenAI-compatible API, совместимый с Qwen2.5-VL по HTTP."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
    ):
        """Инициализирует клиента teacher-модели через OpenAI-compatible API."""
        if OpenAI is None:
            raise RuntimeError("Пакет openai не установлен или недоступен.")
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def decide_action(
        self,
        *,
        goal: str,
        url: str,
        title: str,
        observation_mode: ObservationMode,
        screenshot_b64: str,
        som_marks: list[Mark] | None,
        action_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Отправляет текущее наблюдение в teacher-модель и парсит действие из ответа."""
        marks_payload = None
        if observation_mode == "screenshot_plus_som":
            marks_payload = [
                {
                    "mark_id": m.mark_id,
                    "role": m.role,
                    "name": m.name,
                    "bbox_norm": m.bbox_norm,
                }
                for m in (som_marks or [])
            ]

        system_prompt = (
            "Ты teacher-модель для GUI-агента. "
            "На каждом шаге выбери ровно одно следующее действие, которое продвигает к цели. "
            "Учитывай action_history: не повторяй то же действие, если URL/title не изменились. "
            "Для click выбирай mark по полю name (точное совпадение с целью), "
            "а не по маленькому mark_id и не по соседней рамке на оверлее. "
            "Ссылки в футере (Contact Us, Orders and Returns): сначала scroll вниз "
            "с value>0 (например 1500), value<0 только вверх; затем click по нужному mark. "
            "Для поиска: сначала click по полю (textbox/combobox/searchbox), "
            "затем type с value (например 'tea'), затем press Enter или click по кнопке Search. "
            "Кнопка Search часто disabled, пока поле пустое — сначала type. "
            "Когда цель достигнута (нужный URL/экран), верни action=done. "
            "Никаких пояснений и markdown: только корректный JSON. "
            f"{ACTION_SCHEMA_NOTE}"
        )

        user_text = {
            "goal": goal,
            "url": url,
            "title": title,
            "observation_mode": observation_mode,
            "action_history": action_history or [],
            "marks": marks_payload,
            "allowed_actions": ["click", "type", "press", "scroll", "wait", "done"],
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(user_text, ensure_ascii=False, indent=2)},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                    },
                ],
            },
        ]

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.2,
            max_tokens=200,
        )
        content = resp.choices[0].message.content or ""
        parsed = _extract_json_obj(content)
        if not parsed:
            return {
                "action": "done",
                "value": "",
                "notes": f"Не удалось распарсить JSON teacher-модели: {content[:200]}",
            }
        return parsed


def execute_action(
    page: Page,
    *,
    action: dict[str, Any],
    marks: list[Mark],
    observation_mode: ObservationMode,
) -> tuple[bool, str | None]:
    """Исполняет одно действие агента в браузере и возвращает статус."""
    action_type = (action.get("action") or "").strip().lower()
    value = action.get("value") or ""

    viewport = page.viewport_size or {"width": 1280, "height": 720}

    try:
        if action_type == "done":
            return True, None

        if action_type == "click":
            if observation_mode == "screenshot_plus_som" and action.get("target_mark_id") is not None:
                target_id = int(action["target_mark_id"])
                mark = next((m for m in marks if m.mark_id == target_id), None)
                if mark is None:
                    return False, f"unknown target_mark_id={target_id}"
                x1, y1, x2, y2 = mark.bbox_px
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                page.mouse.click(cx, cy)
            elif action.get("click_x") is not None and action.get("click_y") is not None:
                # Запасной вариант для режима без marks.
                x_norm = float(action["click_x"])
                y_norm = float(action["click_y"])
                cx = int(x_norm * viewport["width"])
                cy = int(y_norm * viewport["height"])
                page.mouse.click(cx, cy)
            else:
                return False, "click requires target_mark_id or click_x/click_y"
            return True, None

        if action_type == "type":
            if not isinstance(value, str) or not value:
                return False, "type requires non-empty value"
            if observation_mode == "screenshot_plus_som" and action.get("target_mark_id") is not None:
                target_id = int(action["target_mark_id"])
                mark = next((m for m in marks if m.mark_id == target_id), None)
                if mark is None:
                    return False, f"unknown target_mark_id={target_id}"
                x1, y1, x2, y2 = mark.bbox_px
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                page.mouse.click(cx, cy)
            elif action.get("click_x") is not None and action.get("click_y") is not None:
                x_norm = float(action["click_x"])
                y_norm = float(action["click_y"])
                cx = int(x_norm * viewport["width"])
                cy = int(y_norm * viewport["height"])
                page.mouse.click(cx, cy)
            else:
                return False, "type requires target_mark_id or click_x/click_y"

            # Сначала очищаем текущее поле, затем вводим текст.
            page.keyboard.press("Control+A")
            page.keyboard.press("Backspace")
            page.keyboard.type(value)
            return True, None

        if action_type == "press":
            key = str(value) if value else "Enter"
            page.keyboard.press(key)
            return True, None

        if action_type == "scroll":
            dy = int(value)
            page.mouse.wheel(0, dy)
            return True, None

        if action_type == "wait":
            ms = int(value)
            page.wait_for_timeout(ms)
            return True, None

        return False, f"unknown action_type={action_type}"
    except Exception as exc:
        return False, str(exc)


def wait_for_magento_ready(page: Page) -> None:
    """Ждёт, пока Magento в основном дорисует интерфейс после перехода."""
    try:
        page.wait_for_load_state("networkidle", timeout=20_000)
    except Exception:
        page.wait_for_timeout(1000)
    try:
        page.wait_for_selector(".page-wrapper, #maincontent", timeout=15_000)
    except Exception:
        pass
    page.wait_for_timeout(300)


def apply_page_zoom(page: Page, zoom: float) -> None:
    """Масштаб страницы (zoom < 1 = дальше / больше контента в кадре)."""
    if zoom is None or abs(float(zoom) - 1.0) < 1e-6:
        return
    z = float(zoom)
    page.evaluate("""(zoom) => { document.documentElement.style.zoom = String(zoom); }""", z)


def task_goal_reached(page: Page, task: dict[str, Any]) -> bool:
    """Проверяет ожидаемые URL/title из конфига задачи (если заданы)."""
    url = page.url
    title = page.title()
    checks: list[bool] = []

    if "expected_end_url_contains" in task:
        checks.append(str(task["expected_end_url_contains"]).lower() in url.lower())
    if "expected_end_title_contains" in task:
        checks.append(str(task["expected_end_title_contains"]).lower() in title.lower())
    if "expected_end_url_not_contains" in task:
        checks.append(str(task["expected_end_url_not_contains"]).lower() not in url.lower())
    if "expected_end_title_not_contains" in task:
        checks.append(str(task["expected_end_title_not_contains"]).lower() not in title.lower())

    return bool(checks) and all(checks)


def select_tasks(
    tasks: list[dict[str, Any]],
    *,
    from_task: int | None,
    to_task: int | None,
    task_ids: str | None,
    limit_tasks: int | None,
) -> list[dict[str, Any]]:
    """Фильтрует задачи по диапазону номеров (1-based) и/или списку id."""
    if not tasks:
        return []

    selected = list(tasks)

    if task_ids:
        wanted = {x.strip() for x in task_ids.split(",") if x.strip()}
        selected = [t for t in selected if t.get("id") in wanted]
        missing = wanted - {t.get("id") for t in selected}
        if missing:
            raise SystemExit(f"Unknown task id(s): {', '.join(sorted(missing))}")
    elif from_task is not None or to_task is not None:
        n = len(tasks)
        start = from_task if from_task is not None else 1
        end = to_task if to_task is not None else n
        if start < 1 or end < 1 or start > n or end > n or start > end:
            raise SystemExit(
                f"Invalid task range: from={start} to={end} (valid: 1..{n}, inclusive)"
            )
        # Нумерация для человека: 1 = первая задача в конфиге.
        selected = tasks[start - 1 : end]

    if limit_tasks is not None:
        selected = selected[:limit_tasks]

    return selected


def build_procedures(transition_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Строит короткие процедурные рецепты по transitions."""
    procedures: list[dict[str, Any]] = []
    by_task: dict[str, list[dict[str, Any]]] = {}
    for tr in transition_records:
        by_task.setdefault(tr["task_id"], []).append(tr)
    for task_id, trs in by_task.items():
        recipe = []
        for tr in trs:
            a = tr["action"] or {}
            at = str(a.get("action") or "")
            if at == "click":
                tgt = a.get("target_mark_id") if "target_mark_id" in a else None
                recipe.append(f"{at}(target_mark_id={tgt})")
            elif at == "type":
                recipe.append(f"type('{(a.get('value') or '')[:30]}')")
            elif at == "press":
                recipe.append(f"press('{a.get('value') or 'Enter'}')")
            else:
                recipe.append(at or "action")
        procedures.append(
            {
                "id": task_id,
                "task_id": task_id,
                "procedure": " -> ".join(recipe[:15]),
                "n_steps": len(trs),
            }
        )
    return procedures


def recompute_kb_summary(kb: dict[str, Any], *, n_tasks_total: int) -> None:
    """Пересчитывает счётчики summary по фактическим спискам в KB."""
    tasks = kb.get("tasks") or []
    kb["summary"] = {
        "n_tasks_total": n_tasks_total,
        "n_tasks_run": len(tasks),
        "n_tasks_success": sum(1 for t in tasks if t.get("success")),
        "n_tasks_failed": sum(1 for t in tasks if not t.get("success")),
        "n_states": len(kb.get("states") or []),
        "n_transitions": len(kb.get("transitions") or []),
        "n_procedures": len(kb.get("procedures") or []),
    }


def merge_knowledge_base(
    existing: dict[str, Any] | None,
    *,
    run_tasks: list[dict[str, Any]],
    run_transitions: list[dict[str, Any]],
    run_states: list[dict[str, Any]],
    meta: dict[str, Any],
    n_tasks_total: int,
    replace_task_ids: set[str],
) -> dict[str, Any]:
    """Дописывает результаты текущего запуска в KB; повторный прогон задачи заменяет старую запись."""
    if existing is None:
        kb: dict[str, Any] = {
            **meta,
            "created_at": meta["updated_at"],
            "tasks": [],
            "transitions": [],
            "procedures": [],
            "states": [],
            "summary": {},
        }
    else:
        kb = dict(existing)
        for k, v in meta.items():
            if k != "updated_at":
                kb[k] = v
        kb["updated_at"] = meta["updated_at"]
        kb.setdefault("created_at", meta["updated_at"])
        kb.setdefault("tasks", [])
        kb.setdefault("transitions", [])
        kb.setdefault("states", [])

    kb["tasks"] = [t for t in kb["tasks"] if t.get("id") not in replace_task_ids] + run_tasks
    kb["transitions"] = [
        tr for tr in kb["transitions"] if tr.get("task_id") not in replace_task_ids
    ] + run_transitions

    # States: объединяем по key, id из текущего прогона побеждает при совпадении key.
    state_by_key: dict[str, str] = {}
    for s in kb.get("states") or []:
        key = s.get("key")
        sid = s.get("state_id")
        if key and sid:
            state_by_key[key] = sid
    for s in run_states:
        key = s.get("key")
        sid = s.get("state_id")
        if key and sid:
            state_by_key[key] = sid
    kb["states"] = [{"state_id": sid, "key": key} for key, sid in state_by_key.items()]

    kb["procedures"] = build_procedures(kb["transitions"])
    recompute_kb_summary(kb, n_tasks_total=n_tasks_total)
    return kb


def main() -> None:
    """Запускает обход задач CUA и сохраняет трейсы, состояния и knowledge base."""
    parser = argparse.ArgumentParser(description="Обход GUI-задач через teacher-VLM / CUA-подобного агента.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Путь к configs/cua_tasks.json")
    parser.add_argument("--mode", default="screenshot_plus_som", choices=["screenshot_only", "screenshot_plus_som"])
    parser.add_argument(
        "--from-task",
        type=int,
        default=None,
        help="Номер первой задачи (1-based, включительно). Пример: --from-task 2 --to-task 4",
    )
    parser.add_argument(
        "--to-task",
        type=int,
        default=None,
        help="Номер последней задачи (1-based, включительно)",
    )
    parser.add_argument(
        "--task-ids",
        default=None,
        help="Список id через запятую, например: search_tea_results,open_advanced_search",
    )
    parser.add_argument(
        "--limit-tasks",
        type=int,
        default=None,
        help="Взять только первые N из уже отфильтрованного списка",
    )
    parser.add_argument(
        "--fresh-kb",
        action="store_true",
        help="Не мержить с существующим knowledge_base.json, начать KB с нуля",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Показать нумерованный список задач из конфига и выйти",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-marks", type=int, default=50)
    parser.add_argument(
        "--teacher-model",
        default=os.environ.get("TEACHER_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct"),
    )
    parser.add_argument(
        "--teacher-api-key",
        default=os.environ.get("TEACHER_API_KEY") or os.environ.get("OPENAI_API_KEY"),
    )
    parser.add_argument(
        "--teacher-base-url",
        default=os.environ.get("TEACHER_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
    )

    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    mode: ObservationMode = args.mode  # type: ignore[assignment]

    all_tasks = cfg.get("tasks", [])
    if args.list_tasks:
        for i, t in enumerate(all_tasks, start=1):
            print(f"{i:2d}. {t.get('id')} — {t.get('goal', '')}")
        return

    tasks = select_tasks(
        all_tasks,
        from_task=args.from_task,
        to_task=args.to_task,
        task_ids=args.task_ids,
        limit_tasks=args.limit_tasks,
    )
    if not tasks:
        raise SystemExit("No tasks found in config / empty selection")

    print("Задачи к запуску:")
    for t in tasks:
        # Показываем исходный номер в полном конфиге.
        idx = next(i for i, x in enumerate(all_tasks, start=1) if x.get("id") == t.get("id"))
        print(f"  [{idx}] {t.get('id')}")

    out_dir = DEFAULT_OUT_DIR / mode
    traces_dir = out_dir / DEFAULT_TRACES_DIR
    states_root = out_dir / DEFAULT_STATES_DIR
    out_kb_path = out_dir / "knowledge_base.json"
    traces_dir.mkdir(parents=True, exist_ok=True)
    states_root.mkdir(parents=True, exist_ok=True)

    existing_kb: dict[str, Any] | None = None
    if out_kb_path.exists() and not args.fresh_kb:
        existing_kb = json.loads(out_kb_path.read_text(encoding="utf-8"))
        print(f"Дописываю в существующий KB: {out_kb_path}")
    elif args.fresh_kb and out_kb_path.exists():
        print(f"--fresh-kb: перезаписываю KB с нуля ({out_kb_path})")

    # Инициализируем teacher-модель (OpenAI-compatible API).
    teacher = OpenAITeacher(
        model=args.teacher_model,
        api_key=args.teacher_api_key,
        base_url=args.teacher_base_url,
    )

    max_steps_default = cfg.get("defaults", {}).get("max_steps", 12)
    viewport = cfg.get("defaults", {}).get("viewport", {"width": 1440, "height": 1100})
    page_zoom = float(cfg.get("defaults", {}).get("page_zoom", 1.0))
    base_url = cfg.get("base_url", "http://localhost:7770")

    # Результаты только текущего запуска; в конце смержим с existing_kb.
    run_task_records: list[dict[str, Any]] = []
    transition_records: list[dict[str, Any]] = []
    state_map: dict[str, str] = {}
    if existing_kb:
        for s in existing_kb.get("states") or []:
            key = s.get("key")
            sid = s.get("state_id")
            if key and sid:
                state_map[str(key)] = str(sid)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # ngrok free показывает interstitial без этого заголовка — иначе Colab
        # получает HTML-заглушку вместо Magento.
        context = browser.new_context(
            viewport=viewport,
            extra_http_headers={"ngrok-skip-browser-warning": "true"},
        )
        if abs(page_zoom - 1.0) >= 1e-6:
            # На каждое navigation, чтобы zoom не сбрасывался.
            context.add_init_script(
                f"document.documentElement.style.zoom = '{page_zoom}';"
            )
        page = context.new_page()

        try:
            for task in tasks:
                task_id = task["id"]
                goal = task["goal"]
                start_url_path = task.get("start_url", cfg.get("defaults", {}).get("start_url", "/"))
                start_url = _abs_url(base_url, start_url_path) if start_url_path else base_url
                task_max_steps = int(args.max_steps or task.get("max_steps", max_steps_default))

                task_dir = states_root / f"{task_id}"
                task_dir.mkdir(parents=True, exist_ok=True)
                trace_path = traces_dir / f"{task_id}__{mode}.jsonl"
                # Каждый запуск перезаписывает trace выбранной задачи.
                trace_fp = trace_path.open("w", encoding="utf-8")

                # Поддерживаем задачи, которым нужны тестовые креды.
                credentials: dict[str, Any] = {}
                if "credentials_ref" in task:
                    ref = task["credentials_ref"]
                    cred_obj = cfg.get("credentials", {}).get(ref, {})
                    credentials = {
                        "username": os.environ.get(
                            cred_obj.get("username_env", ""),
                            cred_obj.get("username_fallback"),
                        ),
                        "password": os.environ.get(
                            cred_obj.get("password_env", ""),
                            cred_obj.get("password_fallback"),
                        ),
                    }
                elif task.get("credentials"):
                    credentials = dict(task["credentials"])

                def maybe_prep_fill_if_login() -> None:
                    """При необходимости подставляет тестовые креды на логин-странице."""
                    if not credentials.get("username") or not credentials.get("password"):
                        return
                    if "login" not in task_id and "account" not in task_id:
                        return
                    try:
                        page.locator("#email").first.fill(str(credentials["username"]))
                        page.locator("#pass").first.fill(str(credentials["password"]))
                    except Exception:
                        return

                page.goto(start_url, wait_until="domcontentloaded", timeout=90_000)
                wait_for_magento_ready(page)
                apply_page_zoom(page, page_zoom)
                maybe_prep_fill_if_login()

                title = page.title()
                state_key = f"{page.url.split('#', 1)[0]}::{title}"
                state_id = state_map.setdefault(state_key, _sha_state_id(page.url, title))

                run_steps = 0
                success = False
                action_history: list[dict[str, Any]] = []
                for step_idx in range(task_max_steps):
                    step_dir = task_dir / f"step_{step_idx:03d}"
                    step_dir.mkdir(parents=True, exist_ok=True)
                    url_before = page.url

                    screenshot_path = step_dir / "screenshot.png"
                    page.screenshot(path=str(screenshot_path), full_page=False)

                    elements: list[dict[str, Any]] = []
                    marks: list[Mark] = []
                    som_path: Path | None = None
                    som_overlay_path: Path | None = None

                    if mode == "screenshot_plus_som":
                        elements = collect_elements_dom(page)
                        marks = build_marks(elements, max_marks=args.max_marks)

                        som_path = step_dir / "som.json"
                        som_overlay_path = step_dir / "som_overlay.png"
                        som_path.write_text(
                            json.dumps(
                                [
                                    {
                                        "mark_id": m.mark_id,
                                        "role": m.role,
                                        "name": m.name,
                                        "bbox_norm": m.bbox_norm,
                                    }
                                    for m in marks
                                ],
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        draw_som_overlay(screenshot_path, marks, som_overlay_path)

                    screenshot_b64 = _read_image_as_b64(screenshot_path)
                    action = teacher.decide_action(
                        goal=goal,
                        url=page.url,
                        title=title,
                        observation_mode=mode,
                        screenshot_b64=screenshot_b64,
                        som_marks=marks if mode == "screenshot_plus_som" else None,
                        action_history=action_history,
                    )

                    action_success, action_error = execute_action(
                        page,
                        action=action,
                        marks=marks,
                        observation_mode=mode,
                    )

                    # После действия фиксируем новое состояние страницы.
                    wait_for_magento_ready(page)
                    apply_page_zoom(page, page_zoom)
                    title_after = page.title()
                    state_key_after = f"{page.url.split('#', 1)[0]}::{title_after}"
                    state_id_after = state_map.setdefault(state_key_after, _sha_state_id(page.url, title_after))

                    trace_record = {
                        "trace_id": f"{task_id}__{mode}",
                        "task_id": task_id,
                        "observation_mode": mode,
                        "step": step_idx,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "state_id_before": state_id,
                        "state_id_after": state_id_after,
                        "goal": goal,
                        "url_before": url_before.split("#", 1)[0],
                        "url_after": page.url,
                        "title_before": title,
                        "title_after": title_after,
                        "artifacts": {
                            "screenshot": str(screenshot_path.relative_to(DEFAULT_OUT_DIR)),
                            "som_overlay": str(som_overlay_path.relative_to(DEFAULT_OUT_DIR)) if som_overlay_path else None,
                            "som_json": str(som_path.relative_to(DEFAULT_OUT_DIR)) if som_path else None,
                        },
                        "action": action,
                        "execution": {
                            "success": bool(action_success),
                            "error": action_error,
                        },
                    }
                    trace_fp.write(json.dumps(trace_record, ensure_ascii=False) + "\n")
                    trace_fp.flush()

                    transition_records.append(
                        {
                            "task_id": task_id,
                            "step": step_idx,
                            "from_state_id": state_id,
                            "to_state_id": state_id_after,
                            "action": action,
                            "success": bool(action_success),
                            "error": action_error,
                            "url_after": page.url,
                            "title_after": title_after,
                        }
                    )

                    action_history.append(
                        {
                            "step": step_idx,
                            "action": action,
                            "execution_success": bool(action_success),
                            "execution_error": action_error,
                            "url_before": url_before.split("#", 1)[0],
                            "url_after": page.url.split("#", 1)[0],
                            "title_after": title_after,
                            "url_changed": url_before.split("#", 1)[0] != page.url.split("#", 1)[0],
                        }
                    )

                    run_steps += 1
                    state_id = state_id_after
                    title = title_after

                    if task_goal_reached(page, task):
                        success = True
                        break

                    if (action.get("action") or "").strip().lower() == "done":
                        has_expected = any(k.startswith("expected_end_") for k in task)
                        success = task_goal_reached(page, task) if has_expected else bool(action_success)
                        break
                trace_fp.close()

                run_task_records.append(
                    {
                        "id": task_id,
                        "goal": goal,
                        "start_url": start_url,
                        "observation_mode": mode,
                        "n_steps_executed": run_steps,
                        "success": success,
                        "trace_file": str(trace_path.relative_to(DEFAULT_OUT_DIR)),
                    }
                )
                print(
                    f"[{task_id}] steps={run_steps} success={success}"
                )
        finally:
            browser.close()

    now = datetime.now(timezone.utc).isoformat()
    run_states = [{"state_id": sid, "key": key} for key, sid in state_map.items()]
    kb = merge_knowledge_base(
        existing_kb,
        run_tasks=run_task_records,
        run_transitions=transition_records,
        run_states=run_states,
        meta={
            "app": cfg.get("app", "target_app"),
            "base_url": base_url,
            "updated_at": now,
            "source": "cua_explore",
            "observation_mode": mode,
            "task_config": args.config,
        },
        n_tasks_total=len(all_tasks),
        replace_task_ids={t["id"] for t in run_task_records},
    )

    out_kb_path.parent.mkdir(parents=True, exist_ok=True)
    out_kb_path.write_text(json.dumps(kb, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Готово. KB: {out_kb_path} "
        f"(tasks_run={kb['summary']['n_tasks_run']}/{kb['summary']['n_tasks_total']}, "
        f"success={kb['summary']['n_tasks_success']}, failed={kb['summary']['n_tasks_failed']})"
    )

if __name__ == "__main__":
    main()
