from __future__ import annotations

from test_agent import FakeBackend, call, final

from fm26_agent.agent import ScoutingAgent
from fm26_agent.config import Currency
from fm26_agent.tools import ScoutingTools

RATE = 1.5
BUDGET = 3_000_000  # player 1 stores 2,001,000: within budget at rate 1.0, above it at rate 1.5


class RecordingPredictor:
    def __init__(self):
        self.seen = []

    def predict(self, players):
        self.seen.extend(players)
        return [{"player_id": row["player_id"], "predicted_potential": 150.0} for row in players]


def test_filters_use_euros_and_results_show_euros(store):
    plain = ScoutingTools(store).call("search_players", {"value_max_eur": BUDGET})
    scaled = ScoutingTools(store, currency=Currency(RATE)).call(
        "search_players", {"value_max_eur": BUDGET}
    )
    assert 1 in plain["player_ids"] and 1 not in scaled["player_ids"]
    assert all(row["value_eur"] <= BUDGET for row in scaled["players"] if row["value_known"])
    listed = ScoutingTools(store, currency=Currency(RATE)).call("search_players", {})
    assert (
        next(r for r in listed["players"] if r["player_id"] == 1)["value_eur"] == 2_001_000 * RATE
    )


def test_details_and_summary_report_euros(store):
    tools = ScoutingTools(store, currency=Currency(RATE, calibrated=True))
    detail = tools.call("get_player_details", {"player_ids": [1]})[0]
    assert detail["value_eur"] == 2_001_000 * RATE and detail["wage_eur"] == 2000 * RATE
    summary = tools.call("get_database_summary", {})
    assert summary["eur_per_internal_unit"] == RATE and summary["currency_calibrated"] is True


def test_bounds_are_inclusive_even_with_float_rounding(store):
    scale = 1.1
    euros = store.get_players([1])[0]["value_eur"] * scale
    hit = store.search(value_min_eur=euros, value_max_eur=euros, currency_scale=scale)
    assert 1 in hit["player_ids"]


def test_the_model_always_receives_internal_units(store):
    predictor = RecordingPredictor()
    tools = ScoutingTools(store, predictor, currency=Currency(RATE))
    search = tools.call("search_players", {"age_max": 19, "position": "MC"})
    tools.call("predict_player_potential", {"search_id": search["search_id"]})
    assert predictor.seen
    stored = {
        p["player_id"]: p["value_eur"]
        for p in store.get_players([r["player_id"] for r in predictor.seen])
    }
    assert all(row["value_eur"] == stored[row["player_id"]] for row in predictor.seen)
    assert {row["wage_eur"] for row in predictor.seen} == {2000.0}


def test_agent_validates_and_reports_in_euros(store):
    predictor = RecordingPredictor()
    budget = {"age_max": 19, "value_max_eur": 3_300_000, "position": "MC"}
    tools = ScoutingTools(store, predictor, currency=Currency(RATE))
    eligible = [
        r["player_id"] for r in store.search(**budget, currency_scale=RATE, limit=500)["players"]
    ]
    assert 1 in eligible  # 2,001,000 x 1.5 = 3,001,500 <= 3.3M
    backend = FakeBackend(
        [
            call("search_players", budget),
            call("predict_player_potential", {"search_id": "search-1"}),
            final(tuple(eligible[:5]), constraints=budget),
        ]
    )
    result = ScoutingAgent(backend, tools).run("five midfielders")
    assert result.error is None
    first = next(r for r in result.recommendations if r["player_id"] == 1)
    assert first["value_eur"] == 2_001_000 * RATE
