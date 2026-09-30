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
    from fm26_agent.cli import _load_predictor
    from fm26_agent.config import load_settings
    from fm26_agent.tools import ScoutingTools

    settings = load_settings(os.getenv("FM26_TEST_CONFIG", "config.toml"))
    store = VisibleStore(settings.data.visible_database)
    predictor = _load_predictor(settings, store, True)
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
    from fm26_agent.cli import _runtime
    from fm26_agent.config import load_settings
    from fm26_agent.tools import ScoutingTools

    settings = load_settings(os.getenv("FM26_TEST_CONFIG", "config.toml"))
    backend, store, predictor = _runtime(settings, True)
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
        assert row["age"] <= 19 and row["value_eur"] <= 8_000_000
        assert "MC" in row["natural_positions"] + row["accomplished_positions"]
    probabilities = [row["wonderkid_probability"] for row in result.recommendations]
    assert probabilities == sorted(probabilities, reverse=True)
    assert all(0 <= value <= 1 for value in probabilities)
    assert not re.search(
        r"\b(?:CA|PA|potential_ability|ability_current)\b", json.dumps(result.to_dict())
    )


@pytest.mark.integration
def test_recorded_real_chat_rejects_natural_only_and_recovers_without_network():
    """Replay actual hosted scores against the visible DB without API calls or hidden labels."""
    source = os.getenv("FM26_TEST_CHAT_LOG")
    if not source:
        pytest.skip("Set FM26_TEST_CHAT_LOG to replay the previous real-save chat")
    from fm26_agent.agent import ScoutingAgent
    from fm26_agent.backend import ChatReply
    from fm26_agent.config import load_settings
    from fm26_agent.tools import ScoutingTools

    log = json.loads(Path(source).read_text())
    settings = load_settings(os.getenv("FM26_TEST_CONFIG", "config.toml"))
    store = VisibleStore(settings.data.visible_database)
    assert store.metadata()["preparation_id"] == log["preparation_id"]
    probabilities = {
        row["player_id"]: row["wonderkid_probability"]
        for trace in log["traces"]
        if trace["tool"] == "predict_wonderkid_probability"
        for row in trace["result"]
    }
    ranked = sorted(probabilities, key=lambda player_id: (-probabilities[player_id], player_id))[:5]

    class RecordedPredictor:
        def predict(self, players):
            return [
                {
                    "player_id": row["player_id"],
                    "wonderkid_probability": probabilities[row["player_id"]],
                }
                for row in players
            ]

    replies = []
    for index, trace in enumerate(log["traces"]):
        replies.append(
            ChatReply(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": f"recorded-{index}",
                            "type": "function",
                            "function": {
                                "name": trace["tool"],
                                "arguments": json.dumps(trace["arguments"]),
                            },
                        }
                    ],
                }
            )
        )

    def shortlist(ids):
        return ChatReply(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "constraints": log["constraints"],
                        "requested_count": 5,
                        "recommendations": [
                            {
                                "player_id": player_id,
                                "explanation": "Ranked using the recorded model probability.",
                            }
                            for player_id in ids
                        ],
                        "note": "Offline regression replay; no new hosted inference.",
                    }
                ),
            }
        )

    replies.append(shortlist([row["player_id"] for row in log["recommendations"]]))
    replies.append(
        ChatReply(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "corrected-details",
                        "type": "function",
                        "function": {
                            "name": "get_player_details",
                            "arguments": json.dumps({"player_ids": ranked}),
                        },
                    }
                ],
            }
        )
    )
    replies.append(shortlist(ranked))

    class RecordedBackend:
        def __init__(self):
            self.replies = iter(replies)

        def complete(self, messages, tools):
            return next(self.replies)

    scope = log.get("candidate_scope", "held_out")
    result = ScoutingAgent(
        RecordedBackend(),
        ScoutingTools(store, RecordedPredictor(), heldout_only=scope == "held_out"),
    ).run(log["query"])
    total = store.search(**log["constraints"], heldout_only=scope == "held_out")["matching_count"]
    if len(probabilities) < total:
        # Historical page-only scores cannot support a complete-pool ranking.
        assert result.error and not result.recommendations
        assert any(
            "Not all matching players" in event["error"] for event in result.validation_events
        )
    else:
        assert result.error is None, result.error
        assert result.validation_events[0]["kind"] == "ranking"
        assert [row["player_id"] for row in result.recommendations] == ranked
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
