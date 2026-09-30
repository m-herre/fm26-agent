from __future__ import annotations

import sqlite3
from datetime import date
from types import SimpleNamespace

import pytest
from test_agent import FakeBackend, call, final

from fm26_agent.agent import ScoutingAgent, render_shortlist
from fm26_agent.evaluate import benchmark_pool
from fm26_agent.extract import record_to_player
from fm26_agent.tools import ScoutingTools, tool_schemas
from fm26_agent.visible_db import value_in_range


@pytest.fixture
def mixed_store(store):
    """Player 1 has no stored value; player 2 costs far more than the €8M budget."""
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE players SET value_eur=NULL WHERE player_id=1")
        connection.execute("UPDATE players SET value_eur=50000000 WHERE player_id=2")
    return store


@pytest.mark.parametrize(
    ("value", "lower", "upper", "include_unknown", "expected"),
    [
        (None, None, None, False, True),  # no filter: nobody is dropped
        (None, None, 8.0, True, True),
        (None, None, 8.0, False, False),
        (None, 1.0, None, True, True),
        (5.0, None, 8.0, False, True),
        (9.0, None, 8.0, True, False),
        (8.0, 8.0, 8.0, True, True),  # inclusive bounds
    ],
)
def test_value_in_range(value, lower, upper, include_unknown, expected):
    assert value_in_range(value, lower, upper, include_unknown) is expected


def test_missing_transfer_value_stays_null(records):
    stub = SimpleNamespace(
        uid=9,
        attributes=SimpleNamespace(),
        transfer_value=None,
        contract=None,
        ability=None,
    )
    assert record_to_player(stub, date(2076, 7, 1), 1.0).visible["value_eur"] is None


def test_unknown_value_players_are_listed_and_flagged(mixed_store):
    tools = ScoutingTools(mixed_store, heldout_only=True)
    result = tools.call("search_players", {"value_max_eur": 8_000_000, "position": "MC"})
    ids = set(result["player_ids"])
    assert 1 in ids and 2 not in ids
    assert result["unknown_value_count"] == 1
    assert next(row for row in result["players"] if row["player_id"] == 1)["value_known"] is False
    summary = tools.call("get_database_summary", {})
    assert summary["players_without_stored_value"] == 1
    assert summary["unknown_value_policy"] == "included_and_flagged"
    strict = ScoutingTools(mixed_store, include_unknown_value=False)
    assert 1 not in strict.call("search_players", {"value_max_eur": 8_000_000})["player_ids"]
    assert strict.call("get_database_summary", {})["unknown_value_policy"] == "excluded"


def test_tool_description_states_the_policy():
    def search_description(flag):
        tools = tool_schemas(include_unknown_value=flag)
        return next(t for t in tools if t["function"]["name"] == "search_players")["function"][
            "description"
        ]

    assert "INCLUDED" in search_description(True)
    assert "fail budget filters" in search_description(False)


def test_agent_shortlists_unknown_value_player_and_discloses_it(mixed_store, fake_predictor):
    backend = FakeBackend(
        [
            call("search_players", {"age_max": 19, "value_max_eur": 8_000_000, "position": "MC"}),
            call("predict_wonderkid_probability", {"search_id": "search-1"}),
            final((1, 4, 5, 6, 8)),
        ]
    )
    tools = ScoutingTools(mixed_store, fake_predictor)
    result = ScoutingAgent(backend, tools).run("five central midfielders under 20, max €8M")
    assert result.error is None
    first = next(row for row in result.recommendations if row["player_id"] == 1)
    assert first["value_eur"] is None and first["value_known"] is False
    assert "1 of 5 shortlisted players have no market value" in result.note
    assert "value unknown (not stored in save)" in render_shortlist(result)
    assert "have no market value" in backend.messages[0][0]["content"]


def test_known_values_only_excludes_them_from_the_shortlist(mixed_store, fake_predictor):
    tools = ScoutingTools(mixed_store, fake_predictor, include_unknown_value=False)
    backend = FakeBackend(
        [
            call("search_players", {"age_max": 19, "value_max_eur": 8_000_000, "position": "MC"}),
            final((1, 4, 5, 6, 8)),
        ]
    )
    assert ScoutingAgent(backend, tools).run("strict").error
    tools = ScoutingTools(mixed_store, fake_predictor, include_unknown_value=False)
    backend = FakeBackend(
        [
            call("search_players", {"age_max": 19, "value_max_eur": 8_000_000, "position": "MC"}),
            call("predict_wonderkid_probability", {"search_id": "search-1"}),
            final((4, 5, 6, 8, 9)),
        ]
    )
    result = ScoutingAgent(backend, tools).run("strict")
    assert result.error is None
    assert "no market value" not in result.note


def test_benchmark_pool_matches_what_the_agent_can_see(mixed_store):
    players = mixed_store.get_players(mixed_store.test_ids())
    pool = {row["player_id"] for row in benchmark_pool(players, "MC")}
    assert 1 in pool and 2 not in pool
