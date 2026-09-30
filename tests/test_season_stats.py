from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest
from test_agent import FakeBackend, call, final

from fm26_agent import prepare as prepare_module
from fm26_agent.agent import format_season_stats, render_shortlist
from fm26_agent.config import DataSettings, LLMSettings, Settings, TrainingSettings
from fm26_agent.extract import SEASON_STATS_VERSION, _season_stats
from fm26_agent.features import NUMERIC_FEATURES
from fm26_agent.runtime import scout
from fm26_agent.tools import ScoutingTools
from fm26_agent.value_model import VALUE_MODEL_VERSION

STATS = {
    "appearances": 30,
    "starts": 28,
    "minutes": 2500,
    "goals": 7,
    "assists": 4,
    "average_rating": 7.12,
    "expected_goals": 6.1,
    "expected_assists": 3.0,
    "player_of_the_match": 2,
    "clean_sheets": 0,
}


def row(uid, kind="overall", minutes=900, rating=7.0, rated=10, **extra):
    base = dict(
        player_uid=uid,
        kind=SimpleNamespace(value=kind),
        minutes=minutes,
        starts=10,
        substitute_appearances=2,
        goals=3,
        assists=1,
        player_of_the_match=1,
        clean_sheets=0,
        expected_goals=2.5,
        expected_assists=1.0,
        average_rating=rating,
        rated_appearances=rated,
    )
    return SimpleNamespace(**(base | extra))


def test_season_stats_sum_teams_and_skip_other_rows():
    career = SimpleNamespace(
        player_season_stats=lambda: [
            row(1, minutes=900, rating=7.0, rated=10),
            row(1, minutes=300, rating=8.0, rated=10),
            row(1, kind="league"),  # only the overall total counts
            row(2, minutes=0),  # never played
            row(3, rating=None, rated=0),
        ]
    )
    stats = _season_stats(career)
    assert set(stats) == {1, 3}
    assert stats[1]["minutes"] == 1200 and stats[1]["appearances"] == 24
    assert stats[1]["goals"] == 6 and stats[1]["expected_goals"] == 5.0
    assert stats[1]["average_rating"] == 7.5  # weighted by rated games
    assert stats[3]["average_rating"] is None


def test_stats_are_stored_apart_from_players_and_absent_ones_are_skipped(store):
    store.set_season_stats({5: STATS}, SEASON_STATS_VERSION)
    assert store.season_stats([5, 6]) == {5: STATS}
    assert store.metadata()["season_stats_players"] == 1
    assert "goals" not in store.get_players([5])[0]  # never mixed into the model's rows


def test_database_from_before_stats_existed_still_works(store):
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TABLE season_stats")
    assert store.season_stats([1, 2]) == {}


def test_stats_never_become_model_features():
    assert not {"goals", "assists", "average_rating", "minutes"} & set(NUMERIC_FEATURES)


def test_details_include_stats_or_null(store, fake_predictor):
    store.set_season_stats({1: STATS}, SEASON_STATS_VERSION)
    details = ScoutingTools(store, fake_predictor).call(
        "get_player_details", {"player_ids": [1, 2]}
    )
    assert details[0]["season_stats"] == STATS and details[1]["season_stats"] is None


def test_format_season_stats():
    assert format_season_stats(STATS) == (
        "This season: 30 games · 7 goals · 4 assists · avg rating 7.12"
    )
    keeper = STATS | {"clean_sheets": 11}
    assert "11 clean sheets" in format_season_stats(keeper, goalkeeper=True)
    assert "goals" not in format_season_stats(keeper, goalkeeper=True)
    assert format_season_stats(None) is None
    assert format_season_stats(STATS | {"minutes": 0}) is None


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


def test_shortlist_shows_stats_only_for_players_who_have_them(settings, store, fake_predictor):
    store.set_season_stats({1: STATS}, SEASON_STATS_VERSION)
    backend = FakeBackend(
        [
            call("search_players", {"age_max": 19, "value_max_eur": 8_000_000, "position": "MC"}),
            call("predict_player_potential", {"search_id": "search-1"}),
            final((1, 2, 4, 5, 6)),
        ]
    )
    result, _ = scout(settings, backend, store, fake_predictor, "two midfielders under 20")
    assert result.error is None
    by_id = {r["player_id"]: r for r in result.recommendations}
    assert by_id[1]["season_stats"] == STATS and by_id[2]["season_stats"] is None
    text = render_shortlist(result)
    assert text.count("This season:") == 1 and "7 goals" in text


def test_existing_setup_gets_stats_without_refitting(settings, store, monkeypatch):
    save = settings.project_root / "save.fm"
    save.write_bytes(b"x")
    monkeypatch.setattr(prepare_module, "setup_problem", lambda *args, **kwargs: None)
    monkeypatch.setattr(prepare_module, "read_season_stats", lambda path: {1: STATS})
    assert settings.data.visible_database == store.path
    store.set_metadata("value_model_version", VALUE_MODEL_VERSION)  # only stats are missing
    messages = []
    with sqlite3.connect(settings.data.visible_database) as connection:
        connection.execute("DROP TABLE season_stats")
        connection.execute("DELETE FROM metadata WHERE key = 'season_stats_version'")
    prepare_module.prepare(settings, save, emit=messages.append)
    assert any("stats" in message for message in messages)
    from fm26_agent.visible_db import VisibleStore

    assert VisibleStore(settings.data.visible_database).season_stats([1]) == {1: STATS}
    messages.clear()
    prepare_module.prepare(settings, save, emit=messages.append)
    assert messages == ["This save is already set up."]
