"""The browser front end: what reaches the page, and the HTTP endpoints (stub worker, real port)."""

from __future__ import annotations

import json
import queue
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from test_agent import FakeBackend, call, final

from fm26_agent.agent import ScoutingAgent
from fm26_agent.tools import ScoutingTools
from fm26_agent.web import make_handler, present, reliability


def test_present_formats_numbers_by_code_and_drops_internals(store, fake_predictor):
    backend = FakeBackend(
        [
            call("search_players", {"age_max": 19, "value_max_eur": 8_000_000, "position": "MC"}),
            call("predict_player_potential", {"search_id": "search-1"}, index=2),
            final((1, 2, 4, 5, 6)),
        ]
    )
    result = ScoutingAgent(backend, ScoutingTools(store, fake_predictor)).run("midfielders")
    page = present(result)
    assert result.error is None and len(page["recommendations"]) == 5
    assert page["recommendations"][0]["price"].startswith("€")
    assert "Potential is an estimate" in page["caveat"]
    for internal in ("traces", "usage", "prediction_operations", "validation_events"):
        assert internal not in page
    json.dumps(page, allow_nan=False)  # the page always gets valid JSON


def test_reliability_comes_from_the_latest_evaluation(tmp_path):
    assert reliability(tmp_path) is None
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "results.json").write_text(
        json.dumps(
            {
                "ranges": {"actual_coverage": 0.8},
                "overall": {"mean_abs_error": 9.1, "within_10": 0.6},
                "star_chances": {"auc": 0.9},
                "scored_players": 50,
            }
        )
    )
    assert reliability(tmp_path)["average_error"] == 9.1


class StubWorker:
    """Answers every question with one progress line and a fixed result."""

    def __init__(self, failure=None):
        self.jobs: queue.Queue = queue.Queue()
        self.ready = threading.Event()
        self.ready.set()
        self.status = {"demo": True, "summary": {"player_count": 3}}
        self.failure = failure
        threading.Thread(target=self._answer, daemon=True).start()

    def _answer(self):
        while True:
            action, payload, out = self.jobs.get()
            out.put({"type": "progress", "text": "Searching your save..."})
            out.put({"type": "result", "result": {"action": action, "query": payload}})
            out.put(None)


@pytest.fixture
def server():
    worker = StubWorker()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(worker))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield SimpleNamespace(url=f"http://127.0.0.1:{httpd.server_address[1]}", worker=worker)
    httpd.shutdown()
    httpd.server_close()


def get(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, response.headers.get("Content-Type"), response.read()


def post(url, body):
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, response.read()


def test_the_page_and_status_are_served(server):
    status, kind, body = get(server.url + "/")
    assert status == 200 and kind.startswith("text/html") and b"Scout Paul" in body
    status, _, body = get(server.url + "/api/status")
    assert json.loads(body) == {"demo": True, "summary": {"player_count": 3}, "error": None}


def test_files_outside_the_static_folder_are_never_served(server):
    for path in ("/static/../web.py", "/static/%2e%2e/web.py", "/nothing.html"):
        with pytest.raises(urllib.error.HTTPError) as error:
            get(server.url + path)
        assert error.value.code == 404


def test_a_question_streams_progress_then_the_result(server):
    status, body = post(server.url + "/api/scout", json.dumps({"query": "  strikers  "}).encode())
    events = [json.loads(line) for line in body.decode().splitlines()]
    assert status == 200 and [event["type"] for event in events] == ["progress", "result"]
    assert events[1]["result"]["query"] == "strikers"


@pytest.mark.parametrize(
    "path,body",
    [
        ("/api/scout", b"not json"),
        ("/api/scout", b'{"q": 1}'),
        ("/api/scout", b'{"query": "   "}'),
        ("/api/scout", b'["query"]'),
        ("/api/rerank", b'{"mode": "random"}'),
    ],
)
def test_bad_requests_are_refused(server, path, body):
    with pytest.raises(urllib.error.HTTPError) as error:
        post(server.url + path, body)
    assert error.value.code == 400


def test_rerank_and_reset_reach_the_worker(server):
    _, body = post(server.url + "/api/rerank", b'{"mode": "ceiling"}')
    assert json.loads(body.decode().splitlines()[-1])["result"] == {
        "action": "rerank",
        "query": "ceiling",
    }
    _, body = post(server.url + "/api/reset", b"")
    assert json.loads(body.decode().splitlines()[-1])["result"]["action"] == "reset"


def test_planning_replies_carry_every_field_the_page_reads(store):
    from test_objective import FakeLab
    from test_planner import GOAL, explained

    from fm26_agent.planner import PlanningSession
    from fm26_agent.web import present_reply

    session = PlanningSession(
        FakeBackend(
            [
                call(
                    "ask_user",
                    {
                        "questions": [
                            {"question": "How likely?", "options": ["25% (recommended)", "50%"]}
                        ]
                    },
                ),
                call("propose_objective", {"objective": GOAL}, index=2),
                explained([100, 99]),
            ]
        ),
        store,
        FakeLab(),
    )
    asked = present_reply(session.send("world class midfielders"), store)
    assert (
        asked["kind"] == "questions" and asked["questions"][0]["options"][0] == "25% (recommended)"
    )
    card = present_reply(session.send("1a"), store)
    objective = card["objective"]
    assert card["kind"] == "objective" and objective["pool"] == 100
    for key in ("filters", "conditions", "rank_by", "rank_mode", "count", "readings", "quality"):
        assert key in objective
    done = present_reply(session.send("go"), store, replay="fm26-agent find --objective x.json")
    result = done["result"]
    assert done["kind"] == "shortlist" and done["replay"].endswith("x.json")
    assert [stage["count"] for stage in result["funnel"]] == [100, 41]
    player = result["players"][0]
    for key in (
        "rank",
        "name",
        "age",
        "club",
        "price",
        "positions",
        "lines",
        "explanation",
        "season",
    ):
        assert key in player
    assert player["lines"][0]["chance"] is not None and "Potential" in player["lines"][0]["text"]
    json.dumps(done, allow_nan=False)
    rerun = present_reply(session.rerank("ceiling"), store)  # no LLM call needed
    assert rerun["kind"] == "shortlist" and rerun["objective"]["rank_mode"] == "ceiling"
