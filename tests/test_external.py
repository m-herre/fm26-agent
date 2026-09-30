import json
import os
import re
from pathlib import Path

import pytest

from fm26_agent.config import DataSettings, LLMSettings, Settings, TrainingSettings
from fm26_agent.features import FeatureSchema
from fm26_agent.prediction import HostedPredictor
from fm26_agent.prepare import prepare
from fm26_agent.private_db import PrivateStore
from fm26_agent.visible_db import VisibleStore


@pytest.mark.hosted
def test_saved_regression_scores_complete_pool_without_refitting():
    if os.getenv("FM26_RUN_REGRESSION_SMOKE") != "1":
        pytest.skip(
            "Set FM26_RUN_REGRESSION_SMOKE=1 after fitting the regressor to opt into live inference"
        )
    if not os.getenv("TABPFN_TOKEN"):
        pytest.skip("The live regression smoke test needs TABPFN_TOKEN")
    from fm26_agent.config import load_settings
    from fm26_agent.runtime import load_predictor
    from fm26_agent.tools import ScoutingTools

    settings = load_settings(os.getenv("FM26_TEST_CONFIG", "config.toml"))
    store = VisibleStore(settings.data.visible_database)
    predictor = load_predictor(settings, store, True)
    tools = ScoutingTools(store, predictor)
    search = tools.call(
        "search_players", {"age_max": 19, "value_max_eur": 8_000_000, "position": "MC"}
    )
    result = tools.call("predict_player_potential", {"search_id": search["search_id"], "top_k": 5})
    assert result["complete"] and result["scored_count"] == search["matching_count"]
    scores = [r["predicted_potential"] for r in result["ranked_players"]]
    assert scores == sorted(scores, reverse=True)
    assert all(1 <= score <= 200 for score in scores)
    assert not re.search(r"\b(?:CA|PA|potential_ability|actual_pa)\b", json.dumps(result))


@pytest.mark.hosted
def test_saved_model_agent_smoke_without_refitting():
    """Opt-in acceptance query against the existing save/model, not a paid refit."""
    if os.getenv("FM26_RUN_AGENT_SMOKE") != "1":
        pytest.skip("Set FM26_RUN_AGENT_SMOKE=1 to opt into one live scouting query")
    if not os.getenv("DEEPSEEK_API_KEY") or not os.getenv("TABPFN_TOKEN"):
        pytest.skip("The live scouting smoke test needs both API credentials")
    from fm26_agent.agent import ScoutingAgent
    from fm26_agent.config import load_settings
    from fm26_agent.runtime import open_runtime
    from fm26_agent.tools import ScoutingTools

    settings = load_settings(os.getenv("FM26_TEST_CONFIG", "config.toml"))
    backend, store, predictor = open_runtime(settings, True)
    result = ScoutingAgent(
        backend,
        ScoutingTools(store, predictor),
        settings.llm.max_tool_steps,
        final_retries=settings.llm.final_retries,
    ).run("Find me five central midfield wonderkids under 20 for at most €8M.")
    assert result.error is None, result.error
    assert result.constraints == {"age_max": 19, "value_max_eur": 8_000_000, "position": "MC"}
    assert result.requested_count == 5
    assert len(result.recommendations) <= 5
    for row in store.get_players(
        [item["player_id"] for item in result.recommendations], require_test=True
    ):
        assert row["age"] <= 19
        assert row["value_eur"] is None or row["value_eur"] <= 8_000_000  # unknown values are kept
        assert "MC" in row["natural_positions"] + row["accomplished_positions"]
    scores = [row["predicted_potential"] for row in result.recommendations]
    assert scores == sorted(scores, reverse=True)
    assert all(1 <= value <= 200 for value in scores)
    assert not re.search(
        r"\b(?:CA|PA|potential_ability|ability_current)\b", json.dumps(result.to_dict())
    )


@pytest.mark.integration
def test_real_save_extraction(tmp_path):
    source = os.getenv("FM26_TEST_SAVE")
    if not source:
        pytest.skip("Set FM26_TEST_SAVE to run the save integration test")
    settings = Settings(
        data=DataSettings(
            tmp_path / "visible.sqlite3",
            tmp_path / "private.sqlite3",
            tmp_path / "model.json",
            tmp_path / "schema.json",
            tmp_path / "runs",
        ),
        training=TrainingSettings(),
        llm=LLMSettings(),
        eur_per_internal_unit=1.0,
        config_path=tmp_path / "config.toml",
    )
    report = prepare(settings, Path(source), extract_only=True)
    store = VisibleStore(settings.data.visible_database)
    summary = store.summary()
    assert summary["game"] == "FM26"
    assert report["player_count"] > 1000
    assert report["sampling"]["reference_rows"] == 10000
    assert summary["player_counts"]["train"] == 10000
    assert report["sampling"]["objective_after"] <= report["sampling"]["objective_before"]
    assert summary["player_counts"]["test"] > 100
    candidates = store.search(age_max=19, position="MC")
    assert candidates["matching_count"] > 0
    assert all("potential_ability" not in player for player in candidates["players"])
    assert PrivateStore(settings.data.private_database).preparation_id() == report["preparation_id"]


@pytest.mark.hosted
def test_hosted_prediction_roundtrip(records, tmp_path):
    if os.getenv("FM26_RUN_HOSTED_TESTS") != "1" or not os.getenv("TABPFN_TOKEN"):
        pytest.skip("Opt in with FM26_RUN_HOSTED_TESTS=1 and TABPFN_TOKEN")
    train = [row.visible for row in records[:80]]
    test = [row.visible for row in records[80:]]
    schema = FeatureSchema.fit(train)
    schema_path = tmp_path / "schema.json"
    model_path = tmp_path / "model.json"
    schema.save(schema_path)
    predictor = HostedPredictor.fit(
        train, [int(row.potential_ability >= 160) for row in records[:80]], schema
    )
    predictor.save(model_path, "fixture")
    restored = HostedPredictor.load(model_path, schema_path, "fixture")
    predictions = restored.predict(test)
    assert len(predictions) == 20
    assert all(0 <= row["wonderkid_probability"] <= 1 for row in predictions)
