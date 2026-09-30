"""Planning mode: ask, propose, edit, go — with a scripted LLM and deterministic predictions."""

from __future__ import annotations

import json

from test_agent import FakeBackend, call
from test_objective import FakeLab

from fm26_agent.backend import ChatReply
from fm26_agent.planner import PlanningSession, is_go

GOAL = {
    "filters": {"position": "MC"},
    "conditions": [{"target": "potential_ability", "at_least": 160, "min_chance": 0.5}],
    "rank_by": {"target": "potential_ability"},
    "count": 2,
    "readings": [{"phrase": "world class", "meaning": "potential 160 or higher"}],
}


def text(content):
    return ChatReply({"role": "assistant", "content": content}, {"total_tokens": 5}, "stop")


def explained(ids, note="Few players qualified."):
    return text(
        json.dumps(
            {
                "explanations": [{"player_id": i, "explanation": f"Profile {i}."} for i in ids],
                "note": note,
            }
        )
    )


def session(store, replies, auto=False):
    backend = FakeBackend(replies)
    return PlanningSession(backend, store, FakeLab(), auto=auto), backend


def test_ask_then_propose_then_go_runs_the_objective(store):
    planner, backend = session(
        store,
        [
            call("count_matches", {"filters": {"position": "MC"}}),
            call(
                "ask_user",
                {
                    "questions": [
                        {"question": "How likely?", "options": ["25% (recommended)", "50%"]}
                    ]
                },
                index=2,
            ),
            call("propose_objective", {"objective": GOAL}, index=3),
            explained([100, 99]),
        ],
    )
    reply = planner.send("world class midfielders")
    assert reply.kind == "questions" and "a) 25% (recommended)" in reply.text
    reply = planner.send("1b")
    assert reply.kind == "objective" and "Go, or change something?" in reply.text
    assert planner.state == "proposed"
    reply = planner.send("go")
    assert reply.kind == "shortlist"
    assert [row["player_id"] for row in reply.result.shortlist] == [100, 99]
    assert "Profile 100." in reply.text and "Few players qualified." in reply.text
    # the count went back to the LLM as a tool result, the answer "1b" as the user's words
    assert any(m.get("content") == "1b" for m in backend.messages[-2])


def test_an_invalid_objective_goes_back_to_the_planner(store):
    bad = GOAL | {"rank_by": {"target": "shoe_size"}}
    planner, backend = session(
        store,
        [
            call("propose_objective", {"objective": bad}),
            call("propose_objective", {"objective": GOAL}, index=2),
        ],
    )
    reply = planner.send("best midfielders")
    assert reply.kind == "objective"
    tool_results = [m for m in backend.messages[-1] if m.get("role") == "tool"]
    assert "Unknown target" in tool_results[0]["content"]


def test_edits_are_replanned_and_nothing_runs_without_go(store):
    tighter = GOAL | {"count": 1}
    planner, _ = session(
        store,
        [
            call("propose_objective", {"objective": GOAL}),
            call("propose_objective", {"objective": tighter}, index=2),
            explained([100]),
        ],
    )
    planner.send("best midfielders")
    reply = planner.send("just one please")
    assert reply.kind == "objective" and planner.objective.count == 1
    assert [row["player_id"] for row in planner.send("yes").result.shortlist] == [100]


def test_one_shot_mode_never_asks_and_runs_at_once(store):
    planner, backend = session(
        store, [call("propose_objective", {"objective": GOAL}), explained([100, 99])], auto=True
    )
    names = [tool["function"]["name"] for tool in planner.tools]
    assert "ask_user" not in names
    reply = planner.send("world class midfielders")
    assert reply.kind == "shortlist" and reply.text.startswith("Objective")
    assert "never call ask_user" in backend.messages[0][0]["content"]


def test_explanations_must_cover_exactly_the_chosen_players(store):
    planner, _ = session(
        store, [call("propose_objective", {"objective": GOAL}), explained([100, 5])], auto=True
    )
    reply = planner.send("world class midfielders")
    assert "Profile 5." not in reply.text and "Best visible attributes" in reply.text


def test_small_talk_gets_a_plain_answer_and_reset_starts_over(store):
    planner, _ = session(store, [text("Hello! Tell me what kind of player you need.")])
    assert planner.send("hi").kind == "chat"
    assert planner.send("new search").kind == "chat" and len(planner.messages) == 1


def test_go_words():
    assert is_go("Go!") and is_go("yes please") and is_go("OK")
    assert not is_go("go cheaper")
