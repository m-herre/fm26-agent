from __future__ import annotations

import json

import pytest
from test_season_stats import STATS

from fm26_agent.config import DataSettings, LLMSettings, Settings, TrainingSettings, demo_settings
from fm26_agent.extract import SEASON_STATS_VERSION
from fm26_agent.sample import export_sample, read_sample


def test_export_then_read_gives_back_the_same_players(tmp_path, store, private):
    store.set_season_stats({3: STATS}, SEASON_STATS_VERSION)
    path = tmp_path / "sample" / "players.csv.gz"
    assert export_sample(store, private, path) == 100
    sample = read_sample(path)
    assert len(sample.players) == 100 and sample.save_date.isoformat() == "2076-07-01"
    original = {row["player_id"]: row for row in store.get_players(store.all_ids())}
    by_name = sorted(sample.players, key=lambda player: player.visible["player_id"])
    first, source = by_name[2].visible, original[3]  # ids are renumbered in order
    for column in ("age", "club", "value_eur", "wage_eur", "preferred_foot", "passing", "pace"):
        assert first[column] == source[column]
    assert first["natural_positions"] == source["natural_positions"]
    assert isinstance(first["on_loan"], bool) and isinstance(first["age"], int)
    labels = {row["player_id"]: row["potential_ability"] for row in private.rows()}
    assert by_name[2].potential_ability == labels[3]
    assert sample.season_stats == {by_name[2].visible["player_id"]: STATS}


def test_names_and_ids_are_replaced_unless_kept(tmp_path, store, private):
    hidden = tmp_path / "hidden.csv.gz"
    export_sample(store, private, hidden)
    assert not any("Fixture" in p.visible["name"] for p in read_sample(hidden).players)
    kept = tmp_path / "kept" / "players.csv.gz"
    export_sample(store, private, kept, keep_names=True)
    assert {p.visible["name"] for p in read_sample(kept).players} >= {"Fixture 1", "Fixture 100"}
    assert json.loads((tmp_path / "sample.json").read_text())["names_replaced"] is True
    assert all(p.visible["player_id"] > 1_000_000 for p in read_sample(hidden).players)


def test_a_sample_without_its_description_is_refused(tmp_path, store, private):
    path = tmp_path / "players.csv.gz"
    export_sample(store, private, path)
    (tmp_path / "sample.json").unlink()
    with pytest.raises(ValueError, match="sample.json"):
        read_sample(path)


def test_demo_keeps_its_data_apart_from_a_real_setup(tmp_path):
    settings = Settings(
        data=DataSettings(
            tmp_path / "data" / "players.sqlite3",
            tmp_path / "data" / "private" / "labels.sqlite3",
            tmp_path / "data" / "model.json",
            tmp_path / "data" / "feature_schema.json",
            tmp_path / "runs",
            tmp_path / "data" / "predictions.sqlite3",
        ),
        training=TrainingSettings(),
        llm=LLMSettings(),
        eur_per_internal_unit=1.0,
        config_path=tmp_path / "config.toml",
    )
    demo = demo_settings(settings)
    for name in ("visible_database", "private_database", "model_reference", "feature_schema"):
        assert getattr(demo.data, name).is_relative_to(tmp_path / "data" / "demo")
        assert getattr(demo.data, name) != getattr(settings.data, name)
    assert demo.data.prediction_cache.is_relative_to(tmp_path / "data" / "demo")
    assert demo.data.runs_directory == settings.data.runs_directory
