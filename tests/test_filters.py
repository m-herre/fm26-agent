from __future__ import annotations

import sqlite3

import pytest
from test_agent import FakeBackend, call, final

from fm26_agent.agent import ScoutingAgent
from fm26_agent.tools import ScoutingTools
from fm26_agent.visible_db import club_matches, club_names


@pytest.fixture
def varied(store):
    """Player 1 left-footed, 2 at Real Madrid, 4 at Barcelona, 5 with a contract ending soon."""
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE players SET preferred_foot='left' WHERE player_id=1")
        connection.execute("UPDATE players SET club='Real Madrid C.F.' WHERE player_id=2")
        connection.execute("UPDATE players SET club='FC Barcelona' WHERE player_id=4")
        connection.execute("UPDATE players SET contract_days_remaining=100 WHERE player_id=5")
        connection.execute("UPDATE players SET contract_days_remaining=-3 WHERE player_id=6")
    return store


def ids(result):
    return set(result["player_ids"])


def test_preferred_foot_filter(varied):
    assert ids(varied.search(preferred_foot="left", limit=500)) == {1}
    assert 1 not in ids(varied.search(preferred_foot="right", limit=500))
    assert varied.search(preferred_foot="both")["matching_count"] == 0


def test_several_clubs_match_any_of_them(varied):
    both = varied.search(club=["real madrid", "BARCELONA"], limit=500)
    assert ids(both) == {2, 4}
    assert ids(varied.search(club="barcelona")) == {4}
    assert varied.search(club=[])["matching_count"] == varied.search()["matching_count"]


def test_club_names_are_matched_literally_not_as_patterns(varied):
    assert varied.search(club="F_xture")["matching_count"] == 0  # "_" is not a wildcard
    assert varied.search(club="%")["matching_count"] == 0  # neither is "%"
    assert varied.search(club="Fixture FC")["matching_count"] > 0


def test_contract_expiry_filter_ignores_unknown_and_already_expired(varied):
    assert ids(varied.search(contract_ends_within_days=200)) == {5}  # not the expired -3 days
    assert {1, 5} <= ids(varied.search(contract_ends_within_days=3000, limit=500))
    with sqlite3.connect(varied.path) as connection:
        connection.execute("UPDATE players SET contract_days_remaining=NULL WHERE player_id=5")
    assert varied.search(contract_ends_within_days=200)["matching_count"] == 0


def test_club_helpers():
    assert (
        club_names(None) == []
        and club_names(" A , ") == ["A ,"]
        and club_names(["A", " ", "B"]) == ["A", "B"]
    )
    assert club_matches("Real Madrid C.F.", ["madrid", "porto"]) and not club_matches(
        None, "madrid"
    )
    assert club_matches(None, None)


def test_tools_accept_the_new_filters_and_reject_bad_ones(varied):
    tools = ScoutingTools(varied)
    hit = tools.call("search_players", {"preferred_foot": "left", "club": ["fixture"]})
    assert hit["player_ids"] == [1]
    for bad in (
        {"preferred_foot": "centre"},
        {"club": []},
        {"club": ["a"] * 11},
        {"contract_ends_within_days": -1},
        {"contract_ends_within_days": 4000},
    ):
        with pytest.raises(ValueError):
            tools.call("search_players", bad)


def run_agent(store, predictor, constraints, ids_):
    backend = FakeBackend(
        [
            call("search_players", constraints),
            call("predict_player_potential", {"search_id": "search-1"}),
            final(tuple(ids_), constraints=constraints, requested_count=len(ids_)),
        ]
    )
    return ScoutingAgent(backend, ScoutingTools(store, predictor)).run("query"), backend


def test_agent_and_store_agree_on_every_new_filter(varied, fake_predictor):
    for constraints, expected in (
        ({"preferred_foot": "left"}, [1]),
        ({"club": ["real madrid", "barcelona"]}, [2, 4]),
        ({"contract_ends_within_days": 200}, [5]),
    ):
        result, _ = run_agent(varied, fake_predictor, constraints, expected)
        assert result.error is None, (constraints, result.error)
        assert [row["player_id"] for row in result.recommendations] == expected


def test_a_shortlist_that_ignores_the_foot_filter_is_rejected(varied, fake_predictor):
    backend = FakeBackend(
        [
            call("search_players", {"preferred_foot": "left"}),
            call("predict_player_potential", {"search_id": "search-1"}),
            final((2,), constraints={"preferred_foot": "left"}, requested_count=1),
        ]
    )
    result = ScoutingAgent(backend, ScoutingTools(varied, fake_predictor)).run("left footers")
    assert result.error and "not returned by search" in result.error


def test_the_prompt_tells_the_agent_what_it_can_and_cannot_filter(varied, fake_predictor):
    _, backend = run_agent(varied, fake_predictor, {"preferred_foot": "left"}, [1])
    system = backend.messages[0][0]["content"]
    for phrase in ("preferred_foot", "contract_ends_within_days", "cannot be filtered", "list"):
        assert phrase in system


def test_small_talk_gets_a_plain_reply_without_searching(store, fake_predictor):
    from fm26_agent.agent import render_shortlist

    backend = FakeBackend(
        [final((), constraints={}, requested_count=1, note="I can only help you scout players.")]
    )
    result = ScoutingAgent(backend, ScoutingTools(store, fake_predictor)).run("what's the weather?")
    assert result.error is None and result.chat_only and not result.recommendations
    assert render_shortlist(result) == "I can only help you scout players."


def test_an_empty_shortlist_after_searching_is_still_validated(store, fake_predictor):
    backend = FakeBackend(
        [
            call("search_players", {"age_max": 19}),
            call("predict_player_potential", {"search_id": "search-1"}),
            final((), constraints={"age_max": 19}, requested_count=1),
        ]
    )
    result = ScoutingAgent(backend, ScoutingTools(store, fake_predictor), max_tool_steps=3).run("q")
    assert not result.chat_only and result.error  # players exist, so "nothing" is not accepted
