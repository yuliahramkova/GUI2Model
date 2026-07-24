"""Сбор скриншотов + elements/bbox + aria snapshot для target app (WebArena Shopping).

Ожидает, что сайт уже доступен (Docker / любой URL), например:
  http://localhost:7770

Экраны задаются в configs/shopping_screens.json.
У экрана может быть поле setup — последовательность действий до съёмки
(goto / click / fill / wait_for / wait_ms). После setup делается screenshot.

Пример:
  python explore/crawl_a11y.py --base-url http://localhost:7770
  python explore/crawl_a11y.py --base-url http://localhost:7770 --headed

Результат: data/target_app/
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from playwright.sync_api import Page, sync_playwright

DEFAULT_CONFIG = Path("configs/shopping_screens.json")
DEFAULT_OUT = Path("data/target_app")

# Теги/роли, которые обычно кликабельны или важны для grounding.
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


def load_screens_config(path: Path) -> dict[str, Any]:
    """Читает JSON со списком экранов для обхода (id, path, description, setup?)."""
    return json.loads(path.read_text(encoding="utf-8"))


def abs_url(base_url: str, path: str) -> str:
    """Абсолютный URL из base + path (path может быть с query)."""
    return urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def collect_elements_dom(page: Page) -> list[dict[str, Any]]:
    """Собирает видимые интерактивные элементы с bbox из DOM (+ aria)."""
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
            return text.slice(0, 120);
          };

          const nodes = Array.from(document.querySelectorAll('body *'));
          for (const el of nodes) {
            if (!(el instanceof HTMLElement)) continue;
            const style = window.getComputedStyle(el);
            if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
              continue;
            }
            const tag = el.tagName.toLowerCase();
            const role = (el.getAttribute('role') || implicitRole(el) || '').toLowerCase();
            const interesting =
              tagSet.has(tag) ||
              roleSet.has(role) ||
              el.onclick != null ||
              el.getAttribute('tabindex') === '0';
            if (!interesting) continue;

            const r = el.getBoundingClientRect();
            if (r.width < 2 || r.height < 2) continue;
            if (r.bottom < 0 || r.right < 0 || r.top > window.innerHeight || r.left > window.innerWidth) {
              continue;
            }

            const name = accessibleName(el);
            const key = [role, name, Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)].join('|');
            if (seen.has(key)) continue;
            seen.add(key);

            out.push({
              role,
              name,
              tag,
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
          }
          return out;
        }""",
        {"interactiveTags": list(INTERACTIVE_TAGS), "interactiveRoles": list(INTERACTIVE_ROLES)},
    )


def dump_aria_snapshot(page: Page) -> str:
    """Компактный aria snapshot (YAML-текст Playwright)."""
    try:
        return page.locator("body").aria_snapshot()
    except Exception as exc:
        return f"(aria_snapshot failed: {exc})"


def element_to_instruction(el: dict[str, Any]) -> str:
    """Простая referring expression для будущего grounding."""
    name = (el.get("name") or "").strip()
    role = (el.get("role") or "element").strip()
    if name:
        return f"click the {role} '{name}'"
    return f"click the {role}"


def wait_for_magento_ready(page: Page) -> None:
    """Ждёт отрисовки Magento: networkidle с fallback + page-wrapper."""
    try:
        page.wait_for_load_state("networkidle", timeout=20_000)
    except Exception:
        page.wait_for_timeout(2000)
    try:
        page.wait_for_selector(".page-wrapper, #maincontent", timeout=15_000)
    except Exception:
        pass
    page.wait_for_timeout(500)


def run_setup(
    page: Page,
    *,
    steps: list[dict[str, Any]],
    base_url: str,
    screen_id: str,
) -> list[dict[str, Any]]:
    """Выполняет setup-шаги экрана; возвращает transitions (рёбра навигации).

    Поддерживаемые шаги:
      {"goto": "/path"}
      {"click": "selector"}
      {"fill": {"selector": "...", "value": "..."}}  или {"fill": "...", "value": "..."}
      {"wait_for": "selector"}
      {"wait_ms": 1000}
    """
    transitions: list[dict[str, Any]] = []

    for i, step in enumerate(steps):
        url_before = page.url

        if "goto" in step:
            target = abs_url(base_url, step["goto"])
            print(f"[{screen_id}] setup[{i}] goto {target}")
            page.goto(target, wait_until="domcontentloaded", timeout=90_000)
            wait_for_magento_ready(page)
            transitions.append(
                {
                    "screen_id": screen_id,
                    "step": i,
                    "action": "goto",
                    "target": step["goto"],
                    "url_before": url_before,
                    "url_after": page.url,
                }
            )
            continue

        if "click" in step:
            selector = step["click"]
            print(f"[{screen_id}] setup[{i}] click {selector!r}")
            loc = page.locator(selector).first
            loc.wait_for(state="visible", timeout=20_000)
            loc.click(timeout=20_000)
            wait_for_magento_ready(page)
            transitions.append(
                {
                    "screen_id": screen_id,
                    "step": i,
                    "action": "click",
                    "selector": selector,
                    "url_before": url_before,
                    "url_after": page.url,
                }
            )
            continue

        if "fill" in step:
            fill_spec = step["fill"]
            if isinstance(fill_spec, dict):
                selector = fill_spec["selector"]
                value = fill_spec["value"]
            else:
                selector = fill_spec
                value = step["value"]
            print(f"[{screen_id}] setup[{i}] fill {selector!r}")
            loc = page.locator(selector).first
            loc.wait_for(state="visible", timeout=20_000)
            loc.fill(str(value), timeout=20_000)
            transitions.append(
                {
                    "screen_id": screen_id,
                    "step": i,
                    "action": "fill",
                    "selector": selector,
                    "url_before": url_before,
                    "url_after": page.url,
                }
            )
            continue

        if "wait_for" in step:
            selector = step["wait_for"]
            print(f"[{screen_id}] setup[{i}] wait_for {selector!r}")
            page.locator(selector).first.wait_for(state="visible", timeout=30_000)
            continue

        if "wait_ms" in step:
            ms = int(step["wait_ms"])
            print(f"[{screen_id}] setup[{i}] wait_ms {ms}")
            page.wait_for_timeout(ms)
            continue

        raise ValueError(f"Unknown setup step on screen {screen_id}: {step}")

    return transitions


def crawl_screen(
    page: Page,
    *,
    screen: dict[str, Any],
    base_url: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Готовит состояние (setup), снимает экран: screenshot + aria + elements."""
    screen_id = screen["id"]
    final_path = screen.get("path")
    setup = screen.get("setup") or []

    print(f"[{screen_id}] begin")
    transitions = run_setup(page, steps=setup, base_url=base_url, screen_id=screen_id)

    if final_path is not None:
        url = abs_url(base_url, final_path)
        # Не дублируем goto, если setup уже на нужном URL.
        if page.url.rstrip("/") != url.rstrip("/"):
            print(f"[{screen_id}] open {url}")
            response = page.goto(url, wait_until="domcontentloaded", timeout=90_000)
            status = response.status if response is not None else None
            wait_for_magento_ready(page)
        else:
            status = 200
            wait_for_magento_ready(page)
    else:
        url = page.url
        status = 200
        wait_for_magento_ready(page)

    title = page.title()
    print(f"[{screen_id}] status={status} title={title!r} url={page.url}")
    if status is not None and status >= 400:
        raise RuntimeError(f"HTTP {status} for {url}")

    screen_dir = out_dir / "screens" / screen_id
    screen_dir.mkdir(parents=True, exist_ok=True)

    screenshot_rel = f"screens/{screen_id}/screenshot.png"
    page.screenshot(path=str(out_dir / screenshot_rel), full_page=False)

    viewport = page.viewport_size or {"width": 1280, "height": 720}
    aria_text = dump_aria_snapshot(page)
    elements = collect_elements_dom(page)

    (screen_dir / "aria_snapshot.yml").write_text(aria_text, encoding="utf-8")

    enriched = []
    for i, el in enumerate(elements):
        item = {
            "element_id": f"{screen_id}_{i:04d}",
            "screen_id": screen_id,
            "role": el.get("role"),
            "name": el.get("name"),
            "tag": el.get("tag"),
            "bbox_px": el.get("bbox_px"),
            "bbox_norm": el.get("bbox_norm"),
            "instruction": element_to_instruction(el),
        }
        enriched.append(item)

    (screen_dir / "elements.json").write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[{screen_id}] elements={len(enriched)} transitions={len(transitions)} -> {screen_dir}")

    return {
        "id": screen_id,
        "url": page.url,
        "path": final_path,
        "description": screen.get("description", ""),
        "title": title,
        "http_status": status,
        "image": screenshot_rel,
        "viewport": viewport,
        "n_elements": len(enriched),
        "elements_file": f"screens/{screen_id}/elements.json",
        "aria_snapshot_file": f"screens/{screen_id}/aria_snapshot.yml",
        "elements": enriched,
        "transitions": transitions,
    }


def build_knowledge_base(
    *,
    app: str,
    base_url: str,
    screens_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Собирает сводный knowledge_base: screens[], elements[], transitions[]."""
    screens_meta = []
    all_elements = []
    all_transitions = []
    for s in screens_data:
        screens_meta.append(
            {
                "id": s["id"],
                "url": s["url"],
                "description": s["description"],
                "title": s.get("title", ""),
                "http_status": s.get("http_status"),
                "image": s["image"],
                "viewport": s["viewport"],
                "n_elements": s["n_elements"],
                "aria_snapshot_file": s["aria_snapshot_file"],
                "elements_file": s["elements_file"],
            }
        )
        all_elements.extend(s["elements"])
        all_transitions.extend(s.get("transitions") or [])

    return {
        "app": app,
        "base_url": base_url,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "playwright_a11y_dom",
        "screens": screens_meta,
        "elements": all_elements,
        "transitions": all_transitions,
    }


def build_grounding_jsonl(elements: list[dict[str, Any]], screens_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """ScreenSpot-like строки: image + instruction + bbox."""
    rows = []
    for el in elements:
        name = (el.get("name") or "").strip()
        if not name:
            continue
        screen = screens_by_id[el["screen_id"]]
        rows.append(
            {
                "image": screen["image"],
                "instruction": el["instruction"],
                "bbox": el["bbox_norm"],
                "bbox_px": el["bbox_px"],
                "screen_id": el["screen_id"],
                "element_id": el["element_id"],
                "role": el.get("role"),
                "name": name,
            }
        )
    return rows


def main() -> None:
    """CLI: обход экранов из конфига, запись KB и grounding.jsonl в data/target_app/."""
    parser = argparse.ArgumentParser(description="Crawl target app: screenshots + a11y + bbox JSON")
    parser.add_argument(
        "--base-url",
        default="http://localhost:7770",
        help="Root URL of Shopping (or other) container",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Show browser window (default: headless)",
    )
    parser.add_argument(
        "--limit-screens",
        type=int,
        default=None,
        help="Crawl only first N screens from config (smoke test)",
    )
    args = parser.parse_args()

    if not DEFAULT_CONFIG.exists():
        raise SystemExit(f"Config not found: {DEFAULT_CONFIG}")

    cfg = load_screens_config(DEFAULT_CONFIG)
    screens = cfg.get("screens", [])
    if args.limit_screens is not None:
        screens = screens[: args.limit_screens]
    if not screens:
        raise SystemExit("No screens in config")

    out_dir = DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    screens_data: list[dict[str, Any]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headed)
        context = browser.new_context(viewport={"width": 1280, "height": 720})
        page = context.new_page()
        try:
            for screen in screens:
                try:
                    screens_data.append(
                        crawl_screen(
                            page,
                            screen=screen,
                            base_url=args.base_url,
                            out_dir=out_dir,
                        )
                    )
                except Exception as exc:
                    print(f"[{screen.get('id')}] FAILED: {exc}")
        finally:
            browser.close()

    if not screens_data:
        raise SystemExit("No screens crawled successfully")

    kb = build_knowledge_base(
        app=cfg.get("app", "target_app"),
        base_url=args.base_url,
        screens_data=screens_data,
    )
    kb_path = out_dir / "knowledge_base.json"
    kb_path.write_text(json.dumps(kb, ensure_ascii=False, indent=2), encoding="utf-8")

    screens_by_id = {s["id"]: s for s in screens_data}
    grounding = build_grounding_jsonl(kb["elements"], screens_by_id)
    grounding_path = out_dir / "grounding.jsonl"
    with grounding_path.open("w", encoding="utf-8") as f:
        for row in grounding:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nDone.")
    print(f"  screens:     {len(screens_data)}")
    print(f"  elements:    {len(kb['elements'])}")
    print(f"  transitions: {len(kb['transitions'])}")
    print(f"  grounding:   {len(grounding)} rows (with non-empty name)")
    print(f"  KB:          {kb_path}")
    print(f"  grounding:   {grounding_path}")


if __name__ == "__main__":
    main()
