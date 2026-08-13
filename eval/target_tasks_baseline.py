"""Multi-step baseline на target GUI — Qwen2.5-VL agent loop.

Читает data/target_app/eval/tasks.json.
Метрики: success rate, шаги, токены на задачу.

Пример:
  python eval/target_tasks_baseline.py --base-url http://localhost:7770
  python eval/target_tasks_baseline.py --task-ids eval_search_shirt_results --headed
  python eval/target_tasks_baseline.py --limit 2
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from playwright.sync_api import Page, sync_playwright
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TASKS = REPO_ROOT / "data/target_app/eval/tasks.json"
DEFAULT_OUT = REPO_ROOT / "reports" / "multistep"
DEFAULT_BASE_URL = "http://localhost:7770"
DEFAULT_VIEWPORT = {"width": 1440, "height": 1100}
DEFAULT_PAGE_ZOOM = 0.75
DEFAULT_STUCK_LIMIT = 3
CART_SETUP_PRODUCT = "/briess-dme-pilsen-light-1-lb-bag.html"


def _load_crawl_cua():
    path = REPO_ROOT / "explore" / "crawl_cua.py"
    spec = importlib.util.spec_from_file_location("crawl_cua", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cua = _load_crawl_cua()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_model_client() -> tuple[OpenAI, str]:
    missing = [k for k in ("LNSIGO_API_KEY", "LNSIGO_BASE_URL", "LNSIGO_MODEL") if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing in .env: {', '.join(missing)}")
    client = OpenAI(
        api_key=os.environ["LNSIGO_API_KEY"],
        base_url=os.environ["LNSIGO_BASE_URL"],
    )
    return client, os.environ["LNSIGO_MODEL"]


@dataclass
class StepRecord:
    step: int
    url_before: str
    url_after: str
    title_before: str
    title_after: str
    action: dict[str, Any]
    execution_success: bool
    execution_error: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    screenshot: str | None = None


@dataclass
class TaskResult:
    id: str
    goal: str
    success: bool
    n_steps: int
    expected_steps: int | None
    max_steps: int
    start_url: str
    end_url: str
    end_title: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tags: str
    error: str | None = None
    steps: list[StepRecord] = field(default_factory=list)

    @property
    def steps_delta(self) -> int | None:
        if self.expected_steps is None:
            return None
        return self.n_steps - self.expected_steps


class VlmAgent:
    """Qwen VLM agent: screenshot + SoM marks → next GUI action + token usage."""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def decide_action(
        self,
        *,
        goal: str,
        url: str,
        title: str,
        screenshot_b64: str,
        som_marks: list[Any],
        action_history: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], int, int, int]:
        marks_payload = [
            {
                "mark_id": m.mark_id,
                "role": m.role,
                "name": m.name,
                "bbox_norm": m.bbox_norm,
            }
            for m in som_marks
        ]

        system_prompt = (
            "You are a GUI agent. "
            "At each step choose exactly one next action that advances toward the goal. "
            "Consider action_history: do not repeat the same action if URL/title did not change. "
            "For click, pick the mark by the name field (exact match with the target), "
            "not by a small mark_id or a neighboring box on the overlay. "
            "Footer links (Contact Us, Orders and Returns): scroll down first "
            "with value>0 (e.g. 1500), value<0 only upward; then click the correct mark. "
            "For search: first click the field (textbox/combobox/searchbox), "
            "then type with value (e.g. 'tea'), then press Enter or click the Search button. "
            "The Search button is often disabled while the field is empty — type first. "
            "When the goal is reached (expected URL/screen), return action=done. "
            "No explanations or markdown: only valid JSON. "
            f"{cua.ACTION_SCHEMA_NOTE}"
        )

        user_text = {
            "goal": goal,
            "url": url,
            "title": title,
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
            messages=messages,
            temperature=0.2,
            max_tokens=200,
        )
        content = resp.choices[0].message.content or ""
        usage = resp.usage
        pt = usage.prompt_tokens if usage else 0
        ct = usage.completion_tokens if usage else 0
        tt = usage.total_tokens if usage else 0

        parsed = cua._extract_json_obj(content)
        if not parsed:
            parsed = {
                "action": "done",
                "value": "",
                "notes": f"Failed to parse model response JSON: {content[:200]}",
            }
        return parsed, pt, ct, tt


def resolve_credentials(doc: dict[str, Any], task: dict[str, Any]) -> dict[str, str]:
    ref = task.get("credentials_ref")
    if not ref:
        return {}
    cred_obj = (doc.get("credentials") or {}).get(ref, {})
    username = os.environ.get(
        cred_obj.get("username_env", ""),
        cred_obj.get("username_fallback") or "",
    )
    password = os.environ.get(
        cred_obj.get("password_env", ""),
        cred_obj.get("password_fallback") or "",
    )
    out: dict[str, str] = {}
    if username:
        out["username"] = str(username)
    if password:
        out["password"] = str(password)
    return out


def enrich_goal(goal: str, credentials: dict[str, str]) -> str:
    if not credentials:
        return goal
    return (
        f"{goal} "
        f"Use email `{credentials['username']}` and password `{credentials['password']}` "
        f"when signing in (type them into the login form yourself)."
    )


def ensure_cart_has_item(page: Page, base_url: str, page_zoom: float) -> None:
    """Precondition for minicart tasks: add a known product to cart."""
    product_url = cua._abs_url(base_url, CART_SETUP_PRODUCT)
    page.goto(product_url, wait_until="domcontentloaded", timeout=90_000)
    cua.wait_for_magento_ready(page)
    cua.apply_page_zoom(page, page_zoom)
    try:
        page.locator("#product-addtocart-button").first.click(timeout=10_000)
        page.wait_for_selector(".message-success, .counter-number", timeout=15_000)
        page.wait_for_timeout(800)
    except Exception as exc:
        raise RuntimeError(f"Failed to seed cart with item: {exc}") from exc


def _normalize_url(url: str) -> str:
    return url.split("#", 1)[0]


def _resolve_mark(marks: list[Any], mark_id: Any) -> tuple[str, str]:
    if mark_id is None:
        return "", ""
    try:
        wanted = int(mark_id)
    except (TypeError, ValueError):
        return "", ""
    for m in marks:
        if m.mark_id == wanted:
            return m.role, m.name
    return "", ""


def _click_stuck_key(
    *,
    url: str,
    title: str,
    role: str,
    name: str,
) -> tuple[str, str, str, str]:
    return (_normalize_url(url), title, role, name)


def select_tasks(
    tasks: list[dict[str, Any]],
    *,
    task_ids: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = list(tasks)
    if task_ids:
        wanted = {x.strip() for x in task_ids.split(",") if x.strip()}
        selected = [t for t in selected if t.get("id") in wanted]
        missing = wanted - {t.get("id") for t in selected}
        if missing:
            raise SystemExit(f"Unknown task id(s): {', '.join(sorted(missing))}")
    if limit is not None:
        selected = selected[:limit]
    return selected


def run_task(
    *,
    page: Page,
    agent: VlmAgent,
    task: dict[str, Any],
    doc: dict[str, Any],
    base_url: str,
    page_zoom: float,
    max_marks: int,
    max_steps_override: int | None,
    stuck_limit: int,
    artifacts_dir: Path,
) -> TaskResult:
    task_id = task["id"]
    credentials = resolve_credentials(doc, task)
    goal = enrich_goal(task["goal"], credentials)
    start_path = task.get("start_url") or "/"
    start_url = cua._abs_url(base_url, start_path)
    max_steps = int(max_steps_override or task.get("max_steps") or 8)
    expected_steps = task.get("expected_steps")

    task_dir = artifacts_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    # Seed cart before tasks that need an item already present.
    tags = set(task.get("tags") or [])
    if "minicart" in tags or (task.get("setup_hint") or "").lower().find("cart with item") >= 0:
        ensure_cart_has_item(page, base_url, page_zoom)

    page.goto(start_url, wait_until="domcontentloaded", timeout=90_000)
    cua.wait_for_magento_ready(page)
    cua.apply_page_zoom(page, page_zoom)

    title = page.title()
    action_history: list[dict[str, Any]] = []
    step_records: list[StepRecord] = []
    success = False
    error: str | None = None
    total_pt = total_ct = total_tt = 0
    stuck_repeat_count = 0
    last_stuck_key: tuple[str, str, str, str] | None = None

    try:
        for step_idx in range(max_steps):
            if cua.task_goal_reached(page, task):
                success = True
                break

            step_dir = task_dir / f"step_{step_idx:03d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            url_before = page.url
            title_before = title

            screenshot_path = step_dir / "screenshot.png"
            page.screenshot(path=str(screenshot_path), full_page=False)

            elements = cua.collect_elements_dom(page)
            marks = cua.build_marks(elements, max_marks=max_marks)
            som_path = step_dir / "som.json"
            overlay_path = step_dir / "som_overlay.png"
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
            cua.draw_som_overlay(screenshot_path, marks, overlay_path)

            screenshot_b64 = cua._read_image_as_b64(screenshot_path)
            action, pt, ct, tt = agent.decide_action(
                goal=goal,
                url=page.url,
                title=title,
                screenshot_b64=screenshot_b64,
                som_marks=marks,
                action_history=action_history,
            )
            total_pt += pt
            total_ct += ct
            total_tt += tt

            action_ok, action_err = cua.execute_action(
                page,
                action=action,
                marks=marks,
            )
            cua.wait_for_magento_ready(page)
            cua.apply_page_zoom(page, page_zoom)
            title_after = page.title()

            step_rec = StepRecord(
                step=step_idx,
                url_before=url_before.split("#", 1)[0],
                url_after=page.url.split("#", 1)[0],
                title_before=title_before,
                title_after=title_after,
                action=action,
                execution_success=bool(action_ok),
                execution_error=action_err,
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=tt,
                screenshot=str(screenshot_path.relative_to(artifacts_dir.parent)).replace("\\", "/"),
            )
            step_records.append(step_rec)

            action_history.append(
                {
                    "step": step_idx,
                    "action": action,
                    "execution_success": bool(action_ok),
                    "execution_error": action_err,
                    "url_before": step_rec.url_before,
                    "url_after": step_rec.url_after,
                    "title_after": title_after,
                    "url_changed": step_rec.url_before != step_rec.url_after,
                }
            )
            title = title_after

            (step_dir / "action.json").write_text(
                json.dumps(
                    {
                        "action": action,
                        "execution": {"success": bool(action_ok), "error": action_err},
                        "tokens": {"prompt": pt, "completion": ct, "total": tt},
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            if cua.task_goal_reached(page, task):
                success = True
                break

            if (action.get("action") or "").strip().lower() == "done":
                success = cua.task_goal_reached(page, task)
                break

            if stuck_limit > 0 and action_ok:
                screen_unchanged = (
                    step_rec.url_before == step_rec.url_after
                    and step_rec.title_before == step_rec.title_after
                )
                action_type = (action.get("action") or "").strip().lower()
                if screen_unchanged and action_type == "click":
                    role, name = _resolve_mark(marks, action.get("target_mark_id"))
                    stuck_key = _click_stuck_key(
                        url=step_rec.url_before,
                        title=step_rec.title_before,
                        role=role,
                        name=name,
                    )
                    if stuck_key == last_stuck_key:
                        stuck_repeat_count += 1
                    else:
                        stuck_repeat_count = 1
                        last_stuck_key = stuck_key
                    if stuck_repeat_count >= stuck_limit:
                        target = f"{role} '{name}'".strip() or f"mark_id={action.get('target_mark_id')}"
                        error = (
                            f"stuck_loop: repeated click on {target} "
                            f"{stuck_repeat_count}x without leaving "
                            f"{step_rec.url_before!r} / {step_rec.title_before!r}"
                        )
                        success = False
                        break
                else:
                    stuck_repeat_count = 0
                    last_stuck_key = None
    except Exception as exc:
        error = str(exc)
        success = False

    return TaskResult(
        id=task_id,
        goal=task["goal"],
        success=success,
        n_steps=len(step_records),
        expected_steps=int(expected_steps) if expected_steps is not None else None,
        max_steps=max_steps,
        start_url=start_url,
        end_url=page.url,
        end_title=page.title(),
        prompt_tokens=total_pt,
        completion_tokens=total_ct,
        total_tokens=total_tt,
        tags=",".join(task.get("tags") or []),
        error=error,
        steps=step_records,
    )


def summarize(results: list[TaskResult]) -> dict[str, Any]:
    n = len(results)
    successes = sum(1 for r in results if r.success)
    with_expected = [r for r in results if r.expected_steps is not None]
    return {
        "n_tasks": n,
        "n_success": successes,
        "success_rate": successes / n if n else 0.0,
        "total_steps": sum(r.n_steps for r in results),
        "avg_steps": sum(r.n_steps for r in results) / n if n else 0.0,
        "avg_steps_success": (
            sum(r.n_steps for r in results if r.success) / successes if successes else 0.0
        ),
        "avg_expected_steps": (
            sum(r.expected_steps for r in with_expected) / len(with_expected)
            if with_expected
            else 0.0
        ),
        "avg_steps_delta": (
            sum(r.steps_delta for r in with_expected if r.steps_delta is not None)
            / len(with_expected)
            if with_expected
            else 0.0
        ),
        "total_tokens": sum(r.total_tokens for r in results),
        "avg_tokens": sum(r.total_tokens for r in results) / n if n else 0.0,
        "total_prompt_tokens": sum(r.prompt_tokens for r in results),
        "total_completion_tokens": sum(r.completion_tokens for r in results),
    }


def print_summary(results: list[TaskResult], summary: dict[str, Any]) -> None:
    print("\n=== Target GUI multi-step baseline (zero-shot, no prior knowledge) ===")
    print(
        f"success_rate={summary['success_rate']:.1%} "
        f"({summary['n_success']}/{summary['n_tasks']})  "
        f"avg_steps={summary['avg_steps']:.1f}  "
        f"avg_expected={summary['avg_expected_steps']:.1f}  "
        f"avg_delta={summary['avg_steps_delta']:+.1f}  "
        f"avg_tokens={summary['avg_tokens']:.1f}"
    )
    print(f"{'id':<36} {'ok':>3} {'steps':>5} {'exp':>4} {'delta':>6} {'tokens':>8}")
    for r in results:
        ok = "Y" if r.success else "N"
        exp = str(r.expected_steps) if r.expected_steps is not None else "-"
        delta = f"{r.steps_delta:+d}" if r.steps_delta is not None else "-"
        print(f"{r.id:<36} {ok:>3} {r.n_steps:>5} {exp:>4} {delta:>6} {r.total_tokens:>8}")
        if r.error:
            print(f"  error: {r.error}")


def save_csv(results: list[TaskResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "success",
                "n_steps",
                "expected_steps",
                "steps_delta",
                "max_steps",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "tags",
                "start_url",
                "end_url",
                "end_title",
                "goal",
                "error",
            ],
            delimiter=",",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "id": r.id,
                    "success": r.success,
                    "n_steps": r.n_steps,
                    "expected_steps": r.expected_steps if r.expected_steps is not None else "",
                    "steps_delta": r.steps_delta if r.steps_delta is not None else "",
                    "max_steps": r.max_steps,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "total_tokens": r.total_tokens,
                    "tags": r.tags,
                    "start_url": r.start_url,
                    "end_url": r.end_url,
                    "end_title": r.end_title,
                    "goal": r.goal,
                    "error": r.error or "",
                }
            )


def save_markdown(
    results: list[TaskResult],
    summary: dict[str, Any],
    path: Path,
    *,
    model: str,
    tasks_path: Path,
    base_url: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Target GUI multi-step baseline",
        "",
        f"- created_at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- model: `{model}`",
        "- prior_knowledge: **none** (no RAG / no LoRA)",
        f"- dataset: `{tasks_path.as_posix()}`",
        f"- base_url: `{base_url}`",
        f"- tasks: {summary['n_tasks']}",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| success_rate | {summary['success_rate']:.1%} |",
        f"| n_success | {summary['n_success']} / {summary['n_tasks']} |",
        f"| avg_steps | {summary['avg_steps']:.1f} |",
        f"| avg_expected_steps | {summary['avg_expected_steps']:.1f} |",
        f"| avg_steps_delta | {summary['avg_steps_delta']:+.1f} |",
        f"| avg_steps (success only) | {summary['avg_steps_success']:.1f} |",
        f"| total_tokens | {summary['total_tokens']} |",
        f"| avg_tokens | {summary['avg_tokens']:.1f} |",
        "",
        "## Per task",
        "",
        "| id | success | steps | expected | delta | tokens |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in results:
        exp = r.expected_steps if r.expected_steps is not None else "-"
        delta = f"{r.steps_delta:+d}" if r.steps_delta is not None else "-"
        lines.append(
            f"| {r.id} | {'yes' if r.success else 'no'} | {r.n_steps} | {exp} | {delta} | {r.total_tokens} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_results_json(results: list[TaskResult], path: Path, meta: dict[str, Any]) -> None:
    payload = {
        **meta,
        "results": [
            {
                "id": r.id,
                "goal": r.goal,
                "success": r.success,
                "n_steps": r.n_steps,
                "expected_steps": r.expected_steps,
                "steps_delta": r.steps_delta,
                "max_steps": r.max_steps,
                "start_url": r.start_url,
                "end_url": r.end_url,
                "end_title": r.end_title,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "total_tokens": r.total_tokens,
                "tags": r.tags,
                "error": r.error,
                "steps": [
                    {
                        "step": s.step,
                        "url_before": s.url_before,
                        "url_after": s.url_after,
                        "title_before": s.title_before,
                        "title_after": s.title_after,
                        "action": s.action,
                        "execution_success": s.execution_success,
                        "execution_error": s.execution_error,
                        "prompt_tokens": s.prompt_tokens,
                        "completion_tokens": s.completion_tokens,
                        "total_tokens": s.total_tokens,
                        "screenshot": s.screenshot,
                    }
                    for s in r.steps
                ],
            }
            for r in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot multi-step Qwen baseline on target GUI tasks.json"
    )
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base-url", default=os.environ.get("SHOPPING_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--task-ids", default=None, help="Comma-separated task ids")
    parser.add_argument("--limit", type=int, default=None, help="Only first N tasks")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max_steps for all")
    parser.add_argument("--max-marks", type=int, default=50)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--page-zoom", type=float, default=DEFAULT_PAGE_ZOOM)
    parser.add_argument(
        "--stuck-limit",
        type=int,
        default=DEFAULT_STUCK_LIMIT,
        help="Fail task after N identical clicks on the same screen with no navigation (0=off)",
    )
    parser.add_argument("--list-tasks", action="store_true")
    args = parser.parse_args()

    load_dotenv()

    tasks_path = resolve_repo_path(args.tasks)
    out_dir = resolve_repo_path(args.out_dir)
    if not tasks_path.exists():
        raise SystemExit(
            f"Tasks not found: {tasks_path}. Run: python eval/build_eval_dataset.py"
        )

    doc = json.loads(tasks_path.read_text(encoding="utf-8"))
    all_tasks = doc.get("tasks") or []
    if args.list_tasks:
        for i, t in enumerate(all_tasks, start=1):
            exp = t.get("expected_steps", "?")
            print(f"{i:2d}. [{exp} steps] {t.get('id')} — {t.get('goal', '')}")
        return

    tasks = select_tasks(all_tasks, task_ids=args.task_ids, limit=args.limit)
    if not tasks:
        raise SystemExit("No tasks to run")

    client, model = load_model_client()
    agent = VlmAgent(client, model)

    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model}")
    print(f"Base:  {args.base_url}")
    if args.stuck_limit:
        print(f"Stuck limit: {args.stuck_limit} identical clicks on same screen")
    print(f"Tasks: {len(tasks)}")
    for t in tasks:
        print(f"  - {t['id']}")

    results: list[TaskResult] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headed)
        context = browser.new_context(
            viewport=DEFAULT_VIEWPORT,
            extra_http_headers={"ngrok-skip-browser-warning": "true"},
        )
        if abs(float(args.page_zoom) - 1.0) >= 1e-6:
            context.add_init_script(
                f"document.documentElement.style.zoom = '{float(args.page_zoom)}';"
            )
        page = context.new_page()
        try:
            for task in tqdm(tasks, desc="Multi-step baseline"):
                result = run_task(
                    page=page,
                    agent=agent,
                    task=task,
                    doc=doc,
                    base_url=args.base_url,
                    page_zoom=float(args.page_zoom),
                    max_marks=args.max_marks,
                    max_steps_override=args.max_steps,
                    stuck_limit=args.stuck_limit,
                    artifacts_dir=artifacts_dir,
                )
                results.append(result)
                print(
                    f"[{result.id}] success={result.success} "
                    f"steps={result.n_steps} tokens={result.total_tokens}"
                )
        finally:
            browser.close()

    summary = summarize(results)
    print_summary(results, summary)

    csv_path = out_dir / "detailed.csv"
    md_path = out_dir / "baseline.md"
    json_path = out_dir / "results.json"
    save_csv(results, csv_path)
    save_markdown(
        results,
        summary,
        md_path,
        model=model,
        tasks_path=tasks_path,
        base_url=args.base_url,
    )
    save_results_json(
        results,
        json_path,
        meta={
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "base_url": args.base_url,
            "stuck_limit": args.stuck_limit,
            "tasks_path": str(tasks_path).replace("\\", "/"),
            "prior_knowledge": None,
            "summary": summary,
        },
    )
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {json_path}")
    print(f"Artifacts: {artifacts_dir}")


if __name__ == "__main__":
    main()
