import json
import os
import re
from pathlib import Path

import pytest

from fm26_agent.features import FeatureSchema
from fm26_agent.prediction import HostedPredictor
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
    predictor = load_predictor(settings, store)
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
    backend, store, predictor = open_runtime(settings)
    result = ScoutingAgent(
        backend,
        ScoutingTools(store, predictor),
        settings.llm.max_tool_steps,
        final_retries=settings.llm.final_retries,
    ).run("Find me five central midfield wonderkids under 20 for at most €8M.")
    assert result.error is None, result.error
    constraints = {key: value for key, value in result.constraints.items() if value is not None}
    assert constraints == {"age_max": 19, "value_max_eur": 8_000_000, "position": "MC"}
    assert result.requested_count == 5
    assert len(result.recommendations) <= 5
    for row in store.get_players([item["player_id"] for item in result.recommendations]):
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
def test_real_save_extraction():
    source = os.getenv("FM26_TEST_SAVE")
    if not source:
        pytest.skip("Set FM26_TEST_SAVE to run the save integration test")
    from fm26_agent.extract import inspect_save, read_save
    from fm26_agent.sampling import representative_sample

    assert inspect_save(source).supported
    extracted = read_save(source)
    assert extracted.game == "FM26" and len(extracted.players) > 1000
    assert extracted.pa_below_current_fraction <= 0.01
    labels = [
        {
            "player_id": p.visible["player_id"],
            "potential_ability": p.potential_ability,
            "wonderkid": None if p.potential_ability is None else int(p.potential_ability >= 160),
        }
        for p in extracted.players
    ]
    ids, _ = representative_sample([p.visible for p in extracted.players], labels, 10_000, 42)
    assert len(ids) == len(set(ids)) == 10_000
    assert not any("potential_ability" in p.visible for p in extracted.players)


@pytest.mark.hosted
@pytest.mark.integration
def test_full_first_run_on_a_fresh_folder(tmp_path):
    """What a new user does: empty folder, real save, keys, one question. Fits a real model."""
    source = os.getenv("FM26_TEST_SAVE")
    if os.getenv("FM26_RUN_FULL_SETUP") != "1" or not source:
        pytest.skip("Opt in with FM26_RUN_FULL_SETUP=1 and FM26_TEST_SAVE (fits a real model)")
    if not os.getenv("DEEPSEEK_API_KEY") or not os.getenv("TABPFN_TOKEN"):
        pytest.skip("Needs both API keys")
    from fm26_agent.app import Console, run
    from fm26_agent.config import load_settings

    (tmp_path / "career.fm").hardlink_to(Path(source).resolve())  # no copy of a huge file
    settings = load_settings(tmp_path / "config.toml")
    said: list[str] = []
    console = Console(
        say=said.append, ask=lambda prompt: pytest.fail(f"unexpected prompt {prompt}")
    )
    assert run(settings, console, query="the five best young goalkeeper prospects") == 0
    text = "\n".join(said)
    assert "Teaching the potential model" in text and "All set." in text
    assert "1. " in text and "Potential ≈" in text
    assert settings.data.model_reference.exists() and settings.data.visible_database.exists()
    said.clear()
    assert run(settings, console, query="two strikers under 21") == 0  # second run reuses it all
    assert "already set up" in "\n".join(said) and "Teaching the potential model" not in "\n".join(
        said
    )


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
    predictor = HostedPredictor.fit(train, [row.potential_ability for row in records[:80]], schema)
    predictor.save(model_path, "fixture")
    restored = HostedPredictor.load(model_path, schema_path, "fixture")
    predictions = restored.predict(test)
    assert len(predictions) == 20
    assert all(
        1 <= row["potential_low"] <= row["predicted_potential"] <= row["potential_high"] <= 200
        and 0 <= row["star_chance"] <= 1
        for row in predictions
    )
