from __future__ import annotations

import json

import pytest
from conftest import ScoreFields

from fm26_agent.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TEXT_FEATURES, FeatureSchema
from fm26_agent.prediction_cache import CachedPredictor
from fm26_agent.sampling import representative_sample
from fm26_agent.tools import ScoutingTools


def labelled(records):
    return [
        {
            "player_id": row.visible["player_id"],
            "wonderkid": int(row.potential_ability >= 160)
            if row.potential_ability is not None
            else None,
        }
        for row in records
    ]


def test_reference_is_exact_reproducible_and_preserves_natural_target_ratio(records):
    players = [row.visible for row in records]
    ids, report = representative_sample(players, labelled(records), size=50, seed=42)
    reverse, again = representative_sample(
        list(reversed(players)), list(reversed(labelled(records))), size=50, seed=42
    )
    assert ids == reverse and report == again
    assert len(ids) == len(set(ids)) == 50
    assert report["reference_positive_fraction"] == report["population_positive_fraction"] == 0.2
    assert report["positive_quota"] == 10
    assert report["objective_after"] <= report["objective_before"]
    assert "age" in report["features"] and "ks_distance" in report["features"]["age"]


def test_missing_inputs_and_rare_categories_participate_in_marginal_balance(records):
    for index, row in enumerate(records):
        row.visible["club"] = f"Club {index % 20}"
        row.visible["traits"] = ["VISIBLE_TRAIT"] if index % 4 == 0 else []
        if index % 5 == 0:
            row.visible["value_eur"] = None
    ids, report = representative_sample(
        [row.visible for row in records], labelled(records), size=50, seed=42
    )
    assert len(ids) == 50
    assert report["features"]["club_category"]["category_coverage"] >= 0.75
    assert report["features"]["value_eur"]["missing_population_fraction"] == 0.2
    assert report["features"]["value_eur"]["missing_reference_fraction"] > 0
    assert "trait:VISIBLE_TRAIT" in report["features"]
    assert "trait_text_available" in report["features"]
    assert report["scouting_notes_available"] is False
    matrix = FeatureSchema.fit(
        [row.visible for row in records if row.visible["player_id"] in ids]
    ).transform([row.visible for row in records[:3]])
    assert matrix["club_category"].tolist() == ["Club 0", "Club 1", "Club 2"]
    assert matrix.loc[0, "age"] == records[0].visible["age"]
    assert matrix.loc[0, "visible_traits_text"] == "VISIBLE_TRAIT"
    assert "bin:" not in matrix.to_json()


def test_unknown_targets_excluded_but_missing_predictors_allowed(records):
    records[0].potential_ability = None
    records[1].visible["age"] = None
    ids, report = representative_sample(
        [row.visible for row in records], labelled(records), size=50
    )
    assert 1 not in ids
    assert report["population_rows"] == 99
    assert report["features"]["age"]["missing_population_fraction"] == 1 / 99


def test_reference_fails_if_requested_size_cannot_leave_held_out_population(records):
    with pytest.raises(ValueError, match="fixed reference"):
        representative_sample([row.visible for row in records], labelled(records), size=100)


def test_native_missing_and_unseen_categories_survive_without_encoding(records):
    schema = FeatureSchema.fit([row.visible for row in records[:50]])
    player = dict(
        records[0].visible, club="Unseen FC", nation_id=98765, traits=["REAL_TRAIT"], age=None
    )
    matrix = schema.transform([player])
    assert matrix["age"].isna().all()
    assert matrix.loc[0, "nation_category"] == "nation:98765"
    assert matrix.loc[0, "club_category"] == "Unseen FC"
    assert matrix.loc[0, "visible_traits_text"] == "REAL_TRAIT"
    legacy = FeatureSchema(schema.categories, schema.columns, version=1)
    with pytest.raises(ValueError, match="Outdated feature schema"):
        legacy.transform([player])


def test_compact_schema_keeps_original_information_without_duplicate_flags(records):
    players = [row.visible for row in records]
    players[0]["natural_positions"] = ["MC", "DM"]
    players[0]["accomplished_positions"] = ["AMC", "STC"]
    players[0]["traits"] = ["VISIBLE_TRAIT", "SECOND_TRAIT"]
    players[1]["traits"] = []
    schema = FeatureSchema.fit(players)
    frame = schema.transform(players)
    assert schema.version == 3
    assert schema.columns == list(NUMERIC_FEATURES + CATEGORICAL_FEATURES + TEXT_FEATURES)
    assert frame.shape == (100, 59)
    assert (len(NUMERIC_FEATURES), len(CATEGORICAL_FEATURES), len(TEXT_FEATURES)) == (53, 5, 1)
    assert frame.loc[0, "natural_position_category"] == "DM|MC"
    assert frame.loc[0, "accomplished_position_category"] == "AMC|STC"
    assert frame.loc[0, "visible_traits_text"] == "VISIBLE_TRAIT; SECOND_TRAIT"
    assert frame.loc[1, "visible_traits_text"] is None
    for key in NUMERIC_FEATURES:
        assert frame.loc[0, key] == players[0][key]
    assert not any(column.startswith("trait_") for column in frame.columns)
    assert not any(column in frame.columns for column in ("natural_mc", "accomplished_mc"))
    assert len(schema.transform([]).columns) == 59


@pytest.mark.parametrize("version", [1, 2])
def test_previous_schema_cannot_load_or_reuse_old_fit(records, tmp_path, version):
    schema = FeatureSchema.fit([row.visible for row in records])
    path = tmp_path / "schema.json"
    schema.save(path)
    payload = json.loads(path.read_text())
    payload.update(version=version, traits=["OLD_TRAIT"])
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="59-feature"):
        FeatureSchema.load(path)


def test_every_player_in_the_save_is_searchable_and_scorable(store, fake_predictor):
    tools = ScoutingTools(store, fake_predictor)
    assert 100 in tools.call("search_players", {})["player_ids"]
    assert tools.call("predict_player_potential", {"player_ids": [100]})[0]["player_id"] == 100
    assert tools.call("get_player_details", {"player_ids": [100]})[0]["player_id"] == 100
    with pytest.raises(ValueError, match="must belong to a player in the save"):
        tools.call("get_player_details", {"player_ids": [999]})
    assert "potential_ability" not in json.dumps(tools.call("get_database_summary", {}))


class CountingPredictor(ScoreFields):
    def __init__(self):
        self.batches = []

    def predict(self, players):
        self.batches.append([row["player_id"] for row in players])
        return [{"player_id": row["player_id"], "predicted_potential": 150.0} for row in players]


def test_disk_predictions_reuse_without_fit_and_invalidate_by_dataset_model_version(
    store, tmp_path
):
    path = tmp_path / "cache.sqlite3"
    model = CountingPredictor()
    first = CachedPredictor(model, path, "dataset-1/model-1")
    players = store.get_players([1, 2])
    assert len(first.predict(players)) == 2
    assert first.predict(players) == CachedPredictor(model, path, "dataset-1/model-1").predict(
        players
    )
    assert model.batches == [[1, 2]]
    CachedPredictor(model, path, "dataset-2/model-1").predict(players)
    CachedPredictor(model, path, "dataset-1/model-2").predict(players)
    assert model.batches == [[1, 2], [1, 2], [1, 2]]


def test_cache_namespace_covers_reference_identity_and_schema(tmp_path):
    path = tmp_path / "reference.json"
    path.write_text('{"model_id":"one"}')
    base = CachedPredictor.namespace_for("data-one", path, "schema-one")
    assert base != CachedPredictor.namespace_for("data-two", path, "schema-one")
    assert base != CachedPredictor.namespace_for("data-one", path, "schema-two")
    path.write_text('{"model_id":"two"}')
    assert base != CachedPredictor.namespace_for("data-one", path, "schema-one")


def test_invalid_probabilities_are_never_committed_to_disk_cache(store, tmp_path):
    class InvalidPredictor(ScoreFields):
        def predict(self, players):
            return [
                {"player_id": row["player_id"], "predicted_potential": float("nan")}
                for row in players
            ]

    path = tmp_path / "cache.sqlite3"
    with pytest.raises(ValueError, match="invalid model predictions"):
        CachedPredictor(InvalidPredictor(), path, "version-one").predict(store.get_players([1]))
    valid = CountingPredictor()
    assert (
        CachedPredictor(valid, path, "version-one").predict(store.get_players([1]))[0][
            "predicted_potential"
        ]
        == 150.0
    )
    assert valid.batches == [[1]]
