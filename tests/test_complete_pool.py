from __future__ import annotations

import json

import pytest
from conftest import ScoreFields
from test_agent import FakeBackend, call, final

from fm26_agent.agent import ScoutingAgent
from fm26_agent.prediction_cache import CachedPredictor
from fm26_agent.tools import ScoutingTools


@pytest.fixture
def large_store(store):
    template = store.get_players([1])[0]
    store.initialize(
        [{**template, "player_id": i, "name": f"Player {i}"} for i in range(1, 1202)],
        {"preparation_id": "large-fixture"},
    )
    return store


class CountingPredictor(ScoreFields):
    def __init__(self):
        self.calls = []

    def predict(self, players):
        self.calls.append([row["player_id"] for row in players])
        return [
            {"player_id": row["player_id"], "predicted_potential": 1 + row["player_id"] / 10}
            for row in players
        ]


def test_whole_pool_one_call_global_leaders_and_small_llm_context(large_store):
    predictor = CountingPredictor()
    backend = FakeBackend(
        [
            call("search_players", {"age_max": 19, "value_max_eur": 8_000_000, "position": "MC"}),
            call("predict_player_potential", {"search_id": "search-1", "top_k": 5}),
            call("get_player_details", {"player_ids": [1201, 1200, 1199, 1198, 1197]}),
            final((1201, 1200, 1199, 1198, 1197)),
        ]
    )
    result = ScoutingAgent(backend, ScoutingTools(large_store, predictor)).run("five midfielders")
    assert result.error is None
    assert [len(ids) for ids in predictor.calls] == [1201]
    assert result.prediction_coverage == {
        "matching_count": 1201,
        "scored_count": 1201,
        "complete": True,
    }
    assert result.recommendations[0]["player_id"] == 1201
    assert len(result.prediction_operations[0]["predictions"]) == 1201
    search_message = json.loads(backend.messages[1][-1]["content"])
    assert search_message["matching_count"] == 1201
    assert "players" not in search_message and "player_ids" not in search_message
    assert "subset" not in result.note


def test_disk_cache_scores_all_uncached_matches_once(large_store, tmp_path):
    predictor = CountingPredictor()
    cached = CachedPredictor(predictor, tmp_path / "cache.sqlite3", "fixed-model")
    cached.predict(large_store.get_players(list(range(1, 501))))
    tools = ScoutingTools(large_store, cached)
    search = tools.call("search_players", {})
    result = tools.call("predict_player_potential", {"search_id": search["search_id"]})
    assert result["scored_count"] == 1201
    assert [len(ids) for ids in predictor.calls] == [500, 701]
    tools.call("predict_player_potential", {"search_id": search["search_id"]})
    assert len(predictor.calls) == 2


def test_stable_pagination_and_offset_validation(large_store):
    pages = [large_store.search(offset=offset, limit=500) for offset in (0, 500, 1000)]
    assert [page["returned_count"] for page in pages] == [500, 500, 201]
    assert [page["next_offset"] for page in pages] == [500, 1000, None]
    assert [i for page in pages for i in page["player_ids"]] == list(range(1, 1202))
    assert large_store.search(offset=1201)["player_ids"] == []
    for offset in (-1, True, 0.5):
        with pytest.raises(ValueError, match="offset"):
            large_store.search(offset=offset)


def test_handles_authorization_empty_pool_and_changed_dataset(store, fake_predictor):
    tools = ScoutingTools(store, fake_predictor)
    with pytest.raises(ValueError, match="Unknown search_id"):
        tools.call("predict_player_potential", {"search_id": "search-1"})
    search = tools.call("search_players", {"club": "nonexistent"})
    result = tools.call("predict_player_potential", {"search_id": search["search_id"]})
    assert result["complete"] and result["scored_count"] == 0
    assert result["ranked_players"] == []
    with pytest.raises(ValueError, match="exactly one"):
        tools.call(
            "predict_player_potential", {"search_id": search["search_id"], "player_ids": [1]}
        )
    store.set_metadata("preparation_id", "replacement")
    with pytest.raises(ValueError, match="Dataset changed"):
        tools.call("predict_player_potential", {"search_id": search["search_id"]})


def test_prediction_failure_never_claims_complete(large_store):
    class FailingPredictor:
        def predict(self, players):
            raise RuntimeError("sensitive provider payload")

    tools = ScoutingTools(large_store, FailingPredictor())
    search = tools.call("search_players", {})
    with pytest.raises(RuntimeError, match="no partial shortlist") as caught:
        tools.call("predict_player_potential", {"search_id": search["search_id"]})
    assert "sensitive" not in str(caught.value)
    assert not tools.queries[search["search_id"]]["complete"]
    assert not tools.prediction_operations


def test_first_page_only_ranking_is_rejected(large_store, fake_predictor):
    backend = FakeBackend(
        [
            call("search_players", {"limit": 1}),
            call("predict_player_potential", {"player_ids": [1]}),
            final((1,), requested_count=1),
        ]
    )
    result = ScoutingAgent(
        backend, ScoutingTools(large_store, fake_predictor), max_tool_steps=2
    ).run("one midfielder")
    assert result.error and "Not all matching players" in result.error
    assert not result.recommendations
