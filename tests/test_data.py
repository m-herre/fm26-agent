from __future__ import annotations

import json
import sqlite3
import warnings
from datetime import date
from types import SimpleNamespace

import pytest

from fm26_agent.extract import read_save
from fm26_agent.features import FeatureSchema
from fm26_agent.schema import (
    HIDDEN_ATTRIBUTES,
    VISIBLE_ATTRIBUTES,
    assert_safe_features,
    normalize_position,
)
from fm26_agent.split import stratified_cap
from fm26_agent.tools import ScoutingTools


def test_extraction_currency_and_hidden_fields(records):
    row = records[0]
    assert row.visible["value_eur"] == 2_001_000 * 1.2
    assert row.visible["wage_eur"] == 2400
    assert row.visible["preferred_foot"] == "right"
    assert row.potential_ability == 120
    assert len(VISIBLE_ATTRIBUTES) == 47
    assert not HIDDEN_ATTRIBUTES.intersection(row.visible)
    assert not any(
        key in row.visible
        for key in ("potential_ability", "current_ability", "raw_attributes", "personality")
    )


def test_stratified_cap_keeps_the_class_ratio(records):
    labels = [
        {
            "player_id": row.visible["player_id"],
            "potential_ability": row.potential_ability,
            "wonderkid": int(row.potential_ability >= 160),
        }
        for row in records
    ]
    capped = stratified_cap(labels, 50, 42)
    assert capped == stratified_cap(labels, 50, 42)
    assert len(capped) == 50
    assert sum(row["wonderkid"] for row in capped) == 10
    assert stratified_cap(labels, 500, 42) == labels


def test_feature_schema_fits_on_training_only(records, tmp_path):
    training = [row.visible for row in records[:50]]
    training[0]["traits"] = ["VISIBLE_TRAIT"]
    schema = FeatureSchema.fit(training)
    candidate = dict(
        records[-1].visible, club="New Club", club_uid=999, nation_id=999, traits=["UNSEEN_TRAIT"]
    )
    transformed = schema.transform([candidate])
    assert transformed.loc[0, "club_category"] == "New Club"
    assert transformed.loc[0, "nation_category"] == "nation:999"
    assert transformed.loc[0, "visible_traits_text"] == "UNSEEN_TRAIT"
    assert str(transformed["club_category"].dtype) == "category"
    assert "New Club" not in schema.categories["club_category"]
    assert transformed.shape == (1, 59)
    assert not any(key.startswith("trait_") for key in transformed.columns)
    assert not any(key in transformed.columns for key in ("player_id", "name", "potential_ability"))
    path = tmp_path / "schema.json"
    schema.save(path)
    assert FeatureSchema.load(path).fingerprint == schema.fingerprint


@pytest.mark.parametrize(
    "feature",
    [
        "potential_ability",
        "ability_current",
        "raw_passing",
        "consistency",
        "personality_professionalism",
        "player_id",
        "name",
    ],
)
def test_forbidden_features(feature):
    with pytest.raises(ValueError, match="Forbidden"):
        assert_safe_features(["passing", feature])


def test_visible_database_has_no_hidden_columns(store):
    with sqlite3.connect(store.path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(players)")}
    assert not HIDDEN_ATTRIBUTES.intersection(columns)
    assert "potential_ability" not in columns
    assert "current_ability" not in columns


def test_inclusive_search_and_null_values(store):
    value = store.get_players([1])[0]["value_eur"]
    result = store.search(
        age_min=18,
        age_max=18,
        value_min_eur=value,
        value_max_eur=value,
        position="central midfield",
    )
    assert result["player_ids"] == [1]
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE players SET value_eur=NULL WHERE player_id=1")
    # Players without a stored value are kept by default and flagged; opting out removes them.
    kept = store.search(value_max_eur=value, limit=500)
    assert 1 in kept["player_ids"] and kept["unknown_value_count"] == 1
    assert next(row for row in kept["players"] if row["player_id"] == 1)["value_known"] is False
    assert 1 not in store.search(value_max_eur=value, include_unknown_value=False)["player_ids"]
    assert store.search(value_min_eur=value)["unknown_value_count"] == 1
    result = store.search(limit=2)
    assert result["player_ids"] == [1, 2]
    assert result["truncated"]
    assert 100 not in store.search(limit=500)["player_ids"]
    assert normalize_position("centre-back") == "DC"
    with pytest.raises(ValueError):
        normalize_position("unrecognized")


def test_tool_validation_and_authorization(store, fake_predictor):
    tools = ScoutingTools(store, fake_predictor)
    for args in (
        {"player_ids": [100]},
        {"player_ids": [1] * 26},
        {"player_ids": [True]},
        {"player_ids": [1], "sql": "DROP TABLE players"},
    ):
        with pytest.raises(ValueError):
            tools.call("get_player_details", args)
    with pytest.raises(ValueError):
        tools.call("search_players", {"age_min": 20, "age_max": 19})
    with pytest.raises(ValueError):
        tools.call("search_players", {"limit": 501})
    result = tools.call("predict_wonderkid_probability", {"player_ids": [2, 1]})
    assert [row["player_id"] for row in result] == [1, 2]  # Tied probabilities use ascending IDs.
    assert all(set(row) == {"player_id", "wonderkid_probability"} for row in result)
    assert "potential_ability" not in json.dumps(
        tools.call("get_player_details", {"player_ids": [1]})
    )


def test_reader_warnings_require_explicit_override(monkeypatch, records):
    import fmsave

    class Career:
        info = SimpleNamespace(game_date=date(2076, 7, 1), game="FM26", build="fixture")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def players(self):
            warnings.warn("fixture gate failed", fmsave.ReaderCheckWarning, stacklevel=2)
            return [SimpleNamespace()]

    monkeypatch.setattr(fmsave, "open", lambda *args, **kwargs: Career())
    monkeypatch.setattr("fm26_agent.extract.record_to_player", lambda *args: records[0])
    with pytest.raises(RuntimeError, match="reader checks failed"):
        read_save("fixture.fm", 1.0)
    result = read_save("fixture.fm", 1.0, True)
    assert result.warnings == ["fixture gate failed"]
