"""A small browser front end for the scouting chat: `fm26-agent web` (or `fm26-agent web --demo`).

Standard library only. One worker thread owns the database, model and LLM backend (SQLite
connections stay on the thread that opened them); requests queue up and stream progress back as
newline-delimited JSON.
"""

from __future__ import annotations

import json
import math
import queue
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import Settings, demo_settings

STATIC = Path(__file__).with_name("static")
DEMO_SAMPLE = Path("sample") / "players.csv.gz"
DROP = ("traces", "prediction_operations", "validation_events", "finish_reasons", "usage")
TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript",
    ".css": "text/css",
    ".svg": "image/svg+xml",
    ".png": "image/png",
}


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def present(result) -> dict[str, Any]:
    """The agent result as the page needs it; every number and label is formatted by code."""
    from .agent import (
        POTENTIAL_CAVEAT,
        RANKING_NOTES,
        format_season_stats,
        price,
        render_shortlist,
        task_line,
    )

    data = {key: value for key, value in result.to_dict().items() if key not in DROP}
    for row in data["recommendations"]:
        row["price"] = price(row)
        row["season"] = format_season_stats(row.get("season_stats"), row.get("goalkeeper", False))
        line = task_line(row, result.task) if result.task else None
        row["task_line"] = line.strip() if line else None
    data["message"] = render_shortlist(result) if result.error else None
    data["ranking_note"] = RANKING_NOTES.get(
        result.ranking, "Ranked by estimate: the middle of each player's range."
    )
    data["caveat"] = POTENTIAL_CAVEAT
    return _clean(data)


def reliability(project_root: Path) -> dict[str, Any] | None:
    """The measured quality shown on the page, read from the evaluation's docs/results.json so
    the page never shows numbers that drifted from the latest run (None if it hasn't been run)."""
    try:
        report = json.loads((project_root / "docs" / "results.json").read_text(encoding="utf-8"))
        return {
            "range_coverage": report["ranges"]["actual_coverage"],
            "average_error": report["overall"]["mean_abs_error"],
            "within_10": report["overall"]["within_10"],
            "star_auc": report["star_chances"]["auc"],
            "scored_players": report["scored_players"],
        }
    except (OSError, ValueError, KeyError, TypeError):
        return None


def present_reply(reply, store, scale: float = 1.0, replay: str | None = None) -> dict[str, Any]:
    """A planning-mode reply as the page needs it; every number and label is formatted by code."""
    from .present import objective_view, result_view

    data: dict[str, Any] = {"kind": reply.kind, "text": reply.text}
    if reply.kind == "questions":
        data["questions"] = reply.questions
    if reply.objective is not None:
        data["objective"] = objective_view(
            reply.objective,
            reply.data.get("quality"),
            reply.data.get("pool", reply.result.pool_size if reply.result else None),
        )
    if reply.kind == "shortlist" and reply.result is not None:
        data["result"] = result_view(
            reply.result,
            store,
            explanations=reply.data.get("explanations"),
            note=reply.data.get("note", ""),
            fair_values=reply.data.get("fair_values"),
            scale=scale,
        )
        data["replay"] = replay
    return _clean(data)


class Worker(threading.Thread):
    """Owns the runtime; runs one request at a time.

    With a TabPFN lab (the normal case) the page gets planning mode: questions, an objective card,
    and the result after "go". Without one it falls back to the classic direct agent.
    """

    def __init__(self, settings: Settings, demo: bool):
        super().__init__(daemon=True)
        self.settings, self.demo = settings, demo
        self.jobs: queue.Queue = queue.Queue()
        self.ready = threading.Event()
        self.status: dict[str, Any] = {}
        self.failure: str | None = None

    def run(self) -> None:
        from .app import progress_message
        from .runtime import open_lab, open_runtime, open_session, scout

        session = None
        try:
            backend, store, predictor = open_runtime(self.settings)
            lab = open_lab(self.settings, store)
            if lab is not None:
                session = open_session(self.settings, backend, store, lab)
            self.status = _clean(
                {
                    "demo": self.demo,
                    "planning": session is not None,
                    "summary": store.summary(),
                    "model_ready": predictor is not None,
                    "calibrated": self.settings.currency_calibrated,
                    "reliability": reliability(self.settings.project_root),
                }
            )
        except Exception as exc:  # shown on the page instead of crashing the server
            self.failure = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        self.ready.set()
        while True:
            action, payload, out = self.jobs.get()
            if self.failure:
                out.put({"type": "error", "text": self.failure})
                out.put(None)
                continue
            shown: list[str] = []

            def trace(message: str, shown=shown, out=out) -> None:
                text = progress_message(message)
                if text and text not in shown[-1:]:
                    shown.append(text)
                    out.put({"type": "progress", "text": text})

            try:
                if session is None:
                    if action != "ask":
                        raise ValueError("Only questions work without planning mode")
                    result, _ = scout(
                        self.settings, backend, store, predictor, payload, trace=trace, lab=lab
                    )
                    out.put({"type": "result", "result": present(result)})
                else:
                    out.put(
                        {
                            "type": "reply",
                            "reply": self._plan(session, store, action, payload, trace),
                        }
                    )
            except ValueError as exc:
                out.put({"type": "error", "text": str(exc)})
            except Exception as exc:
                out.put({"type": "error", "text": f"Something went wrong ({type(exc).__name__})."})
            out.put(None)

    def _plan(self, session, store, action: str, payload: Any, trace) -> dict[str, Any]:
        from .runtime import write_report, write_session_report

        session.progress = session.lab.progress = trace
        if action == "reset":
            reply = session.send("new search")
        elif action == "rerank":
            reply = session.rerank(payload)
        else:
            reply = session.send(payload)
        replay = None
        if reply.kind == "shortlist":
            write_session_report(self.settings, session)
            saved = write_report(self.settings, "objective", reply.objective.to_dict())
            replay = (
                f"fm26-agent find{' --demo' if self.demo else ''} --objective "
                f"{saved.relative_to(self.settings.project_root)}"
            )
        scale = self.settings.currency.eur_per_internal_unit
        return present_reply(reply, store, scale, replay)


def make_handler(worker: Worker):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # keep the terminal quiet
            pass

        def _send(self, status: int, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, status: int = 200) -> None:
            self._send(status, json.dumps(payload, ensure_ascii=False).encode(), "application/json")

        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            if path == "/api/status":
                worker.ready.wait(timeout=60)
                return self._json(worker.status | {"error": worker.failure})
            name = "index.html" if path in ("/", "/index.html") else path.removeprefix("/static/")
            file = (STATIC / name).resolve()
            if not file.is_file() or STATIC.resolve() not in file.parents:
                return self._send(404, b"Not found", "text/plain")
            self._send(200, file.read_bytes(), TYPES.get(file.suffix, "application/octet-stream"))

        def do_POST(self) -> None:
            routes = {"/api/scout": "ask", "/api/rerank": "rerank", "/api/reset": "reset"}
            action = routes.get(self.path)
            if action is None:
                return self._send(404, b"Not found", "text/plain")
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(min(length, 10_000)) or b"{}")
                payload = None
                if action == "ask":
                    payload = str(body["query"]).strip()
                    if not payload:
                        return self._json({"error": "Empty question"}, HTTPStatus.BAD_REQUEST)
                elif action == "rerank":
                    payload = str(body["mode"])
                    if payload not in ("expected", "ceiling", "safe", "chance"):
                        raise ValueError(payload)
            except (ValueError, KeyError, TypeError, AttributeError):
                return self._json({"error": 'Send {"query": "..."}'}, HTTPStatus.BAD_REQUEST)
            out: queue.Queue = queue.Queue()
            worker.jobs.put((action, payload, out))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            while (event := out.get()) is not None:
                self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode())
                self.wfile.flush()

    return Handler


def serve(
    settings: Settings,
    *,
    demo: bool = False,
    save: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8626,
    open_browser: bool = True,
) -> int:
    """Set up (same steps as the terminal chat), then serve the page until Ctrl+C."""
    from .app import Console, choose_save, ensure_keys
    from .prepare import add_missing_estimates, prepare

    console = Console()
    ensure_keys(settings, console, sys.stdin.isatty())
    if demo:
        source = settings.project_root / DEMO_SAMPLE
        if not source.exists():
            raise ValueError("The demo data is missing (sample/players.csv.gz)")
        settings = demo_settings(settings)
    else:
        source = choose_save(settings, console, save)
        if source is None:
            console.say("Nothing to do without a save file. Bye!")
            return 1
    prepare(settings, source, emit=console.say)
    add_missing_estimates(settings, console.say)
    if host not in ("127.0.0.1", "localhost", "::1"):
        console.say(
            f"Warning: listening on {host} lets anyone on your network use this page and your "
            "keys. There is no login."
        )
    worker = Worker(settings, demo)
    worker.start()
    server = ThreadingHTTPServer((host, port), make_handler(worker))
    url = f"http://{host}:{port}/"
    console.say(f"\nScout Paul is ready at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.say("\nBye!")
    finally:
        server.server_close()
    return 0
