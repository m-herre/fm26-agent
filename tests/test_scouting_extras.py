from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from test_agent import FakeBackend, call, final
from test_ranges import RangePredictor

from fm26_agent import value_model
from fm26_agent.agent import price, render_shortlist
from fm26_agent.config import DataSettings, LLMSettings, Settings, TrainingSettings
from fm26_agent.features import FeatureSchema
from fm26_agent.finder import AmbiguousPlayerError, describe, find_players, resolve_player
from fm26_agent.runtime import scout
from fm26_agent.tools import ScoutingTools
from fm26_agent.visible_db import profile_attributes


@pytest.fixture
def settings(tmp_path):
    (tmp_path / "config.toml").write_text("")
    return Settings(
        data=DataSettings(
            tmp_path / "visible.sqlite3",
            tmp_path / "private" / "labels.sqlite3",
            tmp_path / "model.json",
            tmp_path / "schema.json",
            tmp_path / "runs",
        ),
        training=TrainingSettings(),
        llm=LLMSettings(),
        eur_per_internal_unit=1.0,
        config_path=tmp_path / "config.toml",
    )


def unvalue(store, ids):
    """Give some fixture players no stored value, like free agents in a real save."""
    import sqlite3

    with sqlite3.connect(store.path) as connection:
        connection.executemany(
            "UPDATE players SET value_eur = NULL WHERE player_id = ?", [(i,) for i in ids]
        )


# ------------------------------------------------------------------ market value estimates


def test_budget_filters_use_the_estimate_when_the_save_has_no_value(store):
    unvalue(store, [1, 2, 3])
    store.set_value_estimates({1: (1e6, 1.5e6, 3e6), 2: (8e6, 9e6, 12e6)})
    page = store.search(value_max_eur=5_000_000, limit=500)
    assert 1 in page["player_ids"]  # estimated 1.5M: within budget
    assert 2 not in page["player_ids"]  # estimated 9M: over budget
    assert 3 in page["player_ids"]  # no value, no estimate: kept and flagged, as before
    assert page["unknown_value_count"] == 1 and page["estimated_value_count"] == 1
    strict = store.search(value_max_eur=5_000_000, include_unknown_value=False, limit=500)
    assert 1 in strict["player_ids"] and 3 not in strict["player_ids"]


def test_estimates_are_display_data_not_model_inputs(store):
    unvalue(store, [1])
    store.set_value_estimates({1: (1e6, 1.5e6, 3e6)})
    plain = store.get_players([1])[0]
    assert "estimated_value" not in plain and plain["value_eur"] is None
    shown = store.get_players([1], with_estimates=True, currency_scale=2.0)[0]
    assert shown["estimated_value"] == 3e6 and shown["estimated_value_high"] == 6e6
    assert "estimated_value" not in FeatureSchema.fit([plain]).columns


def test_price_shows_stored_then_estimated_then_unknown():
    assert price({"value_eur": 4_500_000}) == "€4.5M"
    assert (
        price({"value_eur": None, "estimated_value_low": 2e6, "estimated_value_high": 6.5e6})
        == "est. €2.0M–6.5M"
    )
    assert price({"value_eur": None}) == "value not in save"


class FakeRegressor:
    """Records what the value model sends; predicts log value from the pace column."""

    fitted: dict = {}

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, table, target):
        FakeRegressor.fitted = {
            "columns": list(table.columns),
            "rows": len(table),
            "target": target,
        }

    def predict(self, table, output_type, quantiles):
        FakeRegressor.predicted = len(table)
        middle = np.log(1_000_000.0) + table["pace"].to_numpy() / 100
        return np.array([middle - 0.3, middle, middle + 0.3])


@pytest.mark.value_model
def test_value_model_trains_on_known_values_and_checks_itself(monkeypatch, records):
    monkeypatch.setitem(
        sys.modules, "tabpfn_client", SimpleNamespace(TabPFNRegressor=FakeRegressor)
    )
    monkeypatch.setattr(value_model, "TRAIN_ROWS", 60)
    monkeypatch.setattr(value_model, "CHECK_ROWS", 20)
    players = [row.visible for row in records]
    for player in players[:15]:
        player["value_eur"] = None
    estimates, report = value_model.estimate_values(players, FeatureSchema.fit(players))
    assert "value_eur" not in FakeRegressor.fitted["columns"]  # never its own input
    assert FakeRegressor.fitted["rows"] == 60
    assert np.allclose(np.exp(FakeRegressor.fitted["target"][:1]) > 0, True)
    assert FakeRegressor.predicted == 15 + 20  # missing + check, in ONE request
    assert set(estimates) == {row["player_id"] for row in players[:15]}
    assert all(low < mid < high for low, mid, high in estimates.values())
    check = report["check_on_known_values"]
    assert check["players"] == 20 and 0 <= check["range_coverage_80"] <= 1


# ------------------------------------------------------------------ players like X


@pytest.fixture
def profiles(tmp_path, records):
    """40 players with varied attributes: 2 is a near copy of 1, 3 is a keeper copy of 1."""
    from fm26_agent.schema import VISIBLE_ATTRIBUTES
    from fm26_agent.visible_db import VisibleStore

    rng = np.random.default_rng(0)
    rows = []
    for index, record in enumerate(records[:40], 1):
        row = dict(record.visible, player_id=index, name=f"Profile {index}")
        for name in VISIBLE_ATTRIBUTES:
            row[name] = int(rng.integers(1, 21))
        rows.append(row)
    for name in VISIBLE_ATTRIBUTES:
        rows[1][name] = min(20, rows[0][name] + (1 if name == "pace" else 0))
        rows[2][name] = rows[0][name]
    rows[2]["natural_positions"], rows[2]["accomplished_positions"] = ["GK"], []
    store = VisibleStore(tmp_path / "profiles.sqlite3")
    store.initialize(rows, {"save_date": "2076-07-01", "preparation_id": "p", "model_ready": True})
    return store


def test_similar_to_keeps_the_closest_profiles_at_his_positions(profiles, monkeypatch):
    from fm26_agent import visible_db

    monkeypatch.setattr(visible_db, "PROFILE_POOL", 12)
    page = profiles.search(similar_to=1, limit=500)
    matches = {row["player_id"]: row["profile_match"] for row in page["players"]}
    assert 1 not in matches and page["matching_count"] == 12
    assert max(matches, key=matches.get) == 2 and matches[2] > 0.99  # the near copy wins
    assert 3 not in matches  # same numbers, but a keeper: not his position
    assert all(0 <= match <= 1 for match in matches.values())
    with pytest.raises(ValueError, match="similar_to"):
        profiles.search(similar_to=999_999)


def test_similar_to_combines_with_other_filters(store):
    page = store.search(similar_to=10, age_max=18, limit=500)
    assert page["player_ids"] and all(
        row["age"] <= 18 for row in store.get_players(page["player_ids"])
    )


def test_profiles_compare_keepers_and_outfielders_on_their_own_attributes():
    assert "reflexes" in profile_attributes(True) and "finishing" not in profile_attributes(True)
    assert "finishing" in profile_attributes(False) and "reflexes" not in profile_attributes(False)


def test_agent_finds_a_player_and_ranks_his_lookalikes(settings, store):
    pool = store.search(similar_to=10, limit=500)["player_ids"]
    highs = {i: 160.0 + (i * 3) % 25 for i in pool}
    expected = sorted(pool, key=lambda i: (-150.0, i))[:3]  # all estimates tie at 150
    backend = FakeBackend(
        [
            call("find_player", {"name": "Fixture 10"}),
            call("search_players", {"similar_to": 10}, index=2),
            call("predict_player_potential", {"search_id": "search-1"}, index=3),
            final(expected, constraints={"similar_to": 10}, requested_count=3),
        ]
    )
    result, _ = scout(settings, backend, store, RangePredictor(), "three players like Fixture 10")
    assert result.error is None, result.error
    assert [row["player_id"] for row in result.recommendations] == expected
    assert all(row["profile_match"] is not None for row in result.recommendations)
    assert "profile match" in render_shortlist(result)
    assert highs  # the pool was scored in full, not just the shortlist


def test_a_lookalike_outside_the_pool_is_rejected(settings, profiles, monkeypatch):
    from fm26_agent import visible_db

    monkeypatch.setattr(visible_db, "PROFILE_POOL", 5)
    pool = profiles.search(similar_to=1, limit=500)["player_ids"]
    outsider = next(i for i in range(4, 41) if i not in pool)
    backend = FakeBackend(
        [
            call("search_players", {"similar_to": 1}),
            call("predict_player_potential", {"search_id": "search-1"}, index=2),
            final((outsider,), constraints={"similar_to": 1}, requested_count=1),
        ]
    )
    result = scout(settings, backend, profiles, RangePredictor(), "one player like Profile 1")[0]
    assert result.error is not None and not result.recommendations


# ------------------------------------------------------------------ find without an LLM


def test_find_needs_no_llm_and_passes_the_same_checks(store):
    tools = ScoutingTools(store, RangePredictor())
    result = find_players(store, tools, count=3, rank_by="ceiling", position="MC", age_max=19)
    assert result.error is None and result.ranking == "ceiling"
    highs = [row["potential_high"] for row in result.recommendations]
    assert len(highs) == 3 and highs == sorted(highs, reverse=True)
    assert all("Best visible attributes" in row["explanation"] for row in result.recommendations)


def test_find_like_a_player(store):
    tools = ScoutingTools(store, RangePredictor())
    result = find_players(store, tools, count=2, like="Fixture 10")
    assert len(result.recommendations) == 2 and "Fixture 10" in result.note
    assert all(row["profile_match"] is not None for row in result.recommendations)


def test_resolving_a_name(store):
    assert resolve_player(store, "fixture 7")["player_id"] == 7
    assert resolve_player(store, "42")["player_id"] == 42
    with pytest.raises(AmbiguousPlayerError, match="Several players"):
        resolve_player(store, "ixture 1")  # no exact match; 1, 10, 100, 11 ... all contain it
    with pytest.raises(ValueError, match="No player"):
        resolve_player(store, "Nobody")


def test_describe_uses_the_right_attributes(store):
    player = store.get_players([5])[0]
    text = describe(player)
    assert text.startswith("MC.") and "reflexes" not in text
