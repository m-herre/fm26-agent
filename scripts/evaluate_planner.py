"""How well does the planner turn free-form requests into objectives? Runs a fixed set of requests
twice on the set-up save (use --demo for the sample):

1. interactive, first turn only: does it ask (and what), or propose straight away?
2. one-shot: the objective it builds without asking, and what running it gives.

Writes runs/planner-eval-<time>.json for review. Needs the DeepSeek key; TabPFN runs as set up.

    python scripts/evaluate_planner.py --demo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime

from fm26_agent.backend import OpenAICompatibleBackend
from fm26_agent.config import demo_settings, load_settings
from fm26_agent.keys import load_env_file
from fm26_agent.planner import PlanningSession
from fm26_agent.runtime import open_lab
from fm26_agent.tabpfn_backend import use_project_weights
from fm26_agent.visible_db import VisibleStore

REQUESTS = [
    # potential / ability
    "five wonderkid central midfielders under 21, max €20M",
    "a striker in his prime for less than 10m",
    "a centre-back who is ready to start now, under 25",
    # hidden attributes and personality, alone and combined
    "a left back who won't get injured all the time",
    "a reliable goalkeeper who performs in big games",
    "a model professional midfielder under 23 who handles pressure",
    "a loyal defender who won't ask to leave",
    # value
    "an undervalued striker, 25, who could become world class, max €40M",
    "the biggest bargains among wingers under 24",
    "cheap strikers compared to players of the same level",
    # lookalikes and follow-up style
    "a cheaper version of Pedro G. Ferreira",
    # things the save can't filter
    "a Brazilian winger under 21",
    "a tall centre-back who is good in the air",
    "a fast winger with pace at least 16",
    # blends no single target covers (agent-defined targets)
    "a centre-back with a strong mentality",
    "a natural leader for my midfield who won't crack in big games",
    # vague / not a request
    "I need a new number 10",
    "hello, what can you do?",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--only", type=int, nargs="*", help="indices of REQUESTS to run")
    args = parser.parse_args()
    settings = load_settings(args.config)
    if args.demo:
        settings = demo_settings(settings)
    load_env_file(settings.project_root)
    use_project_weights(settings.project_root)
    if not settings.deepseek_api_key:
        print("DEEPSEEK_API_KEY is missing (put it in .env).", file=sys.stderr)
        return 1
    store = VisibleStore(settings.data.visible_database)
    lab = open_lab(settings, store)
    backend = OpenAICompatibleBackend(settings.llm, settings.deepseek_api_key)
    cases = []
    for index, request in enumerate(REQUESTS):
        if args.only and index not in args.only:
            continue
        case: dict = {"index": index, "request": request}
        started = time.monotonic()
        try:
            first = PlanningSession(backend, store, lab, currency=settings.currency).send(request)
            case["interactive"] = {"kind": first.kind, "text": first.text}
        except Exception as exc:
            case["interactive"] = {"kind": "crash", "text": f"{type(exc).__name__}: {exc}"}
        try:
            session = PlanningSession(backend, store, lab, currency=settings.currency, auto=True)
            reply = session.send(request)
            case["one_shot"] = {
                "kind": reply.kind,
                "objective": reply.objective.to_dict() if reply.objective else None,
                "funnel": [
                    f"{stage.remaining} {stage.label}"
                    + (f" ({stage.no_data} unjudged)" if stage.no_data else "")
                    for stage in (reply.result.funnel if reply.result else [])
                ],
                "shortlisted": len(reply.result.shortlist) if reply.result else None,
                "suggestions": reply.result.suggestions if reply.result else [],
                "text": reply.text if reply.kind in ("chat", "error") else None,
                "tools": [entry.get("tool") for entry in session.log if "tool" in entry],
            }
        except Exception as exc:
            case["one_shot"] = {"kind": "crash", "text": f"{type(exc).__name__}: {exc}"}
        case["seconds"] = round(time.monotonic() - started, 1)
        cases.append(case)
        shot = case["one_shot"]
        print(
            f"[{index:2}] {case['interactive']['kind']:>9} | {shot['kind']:>9} "
            f"| {shot.get('shortlisted')} | {request}",
            flush=True,
        )
    settings.data.runs_directory.mkdir(parents=True, exist_ok=True)
    path = settings.data.runs_directory / f"planner-eval-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
