from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from fm26_agent.agent import ScoutingAgent, render_shortlist
from fm26_agent.backend import ChatReply, OpenAICompatibleBackend
from fm26_agent.config import LLMSettings
from fm26_agent.features import FeatureSchema
from fm26_agent.prediction import HostedPredictor
from fm26_agent.tools import ScoutingTools, tool_schemas


def normalized(result):
    """Constraints as the agent reported them, without explicit nulls."""
    return {key: value for key, value in result.constraints.items() if value is not None}


def call(name, arguments, index=1):
    return ChatReply(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"call_{index}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        },
        {"total_tokens": 10},
    )


def final(ids=(1,), constraints=None, requested_count=5, note=""):
    return ChatReply(
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "constraints": constraints
                    or {"age_max": 19, "value_max_eur": 8_000_000, "position": "MC"},
                    "requested_count": requested_count,
                    "recommendations": [
                        {"player_id": player_id, "explanation": "Strong observable passing."}
                        for player_id in ids
                    ],
                    "note": note,
                }
            ),
        },
        {"total_tokens": 3},
    )


class FakeBackend:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.messages = []
        self.tool_sets = []

    def complete(self, messages, tools):
        self.messages.append(json.loads(json.dumps(messages)))
        self.tool_sets.append(tools)
        return next(self.replies)


def test_full_agent_tool_loop_has_no_hidden_leakage(store, fake_predictor):
    backend = FakeBackend(
        [
            call(
                "search_players",
                {"age_max": 19, "value_max_eur": 8_000_000, "position": "MC", "limit": 1},
            ),
            call("predict_player_potential", {"search_id": "search-1"}),
            call("get_player_details", {"player_ids": [1, 2, 4, 5, 6]}),
            final((1, 2, 4, 5, 6)),
        ]
    )
    result = ScoutingAgent(backend, ScoutingTools(store, fake_predictor)).run(
        "Find me five central midfield wonderkids under 20 for at most €8M"
    )
    assert result.error is None
    assert result.recommendations[0]["predicted_potential"] == 150.0
    assert result.usage["total_tokens"] == 33
    assert "Potential ≈ 150" in render_shortlist(result)
    assert "estimate" in render_shortlist(result)
    messages = json.dumps(backend.messages)
    for forbidden in ("potential_ability", "ability_current", "consistency", "professionalism"):
        assert forbidden not in messages
    expected = {"age_max": 19, "value_max_eur": 8_000_000, "position": "MC"}
    assert normalized(result) == expected
    assert result.prediction_coverage == {
        "matching_count": 75,
        "scored_count": 75,
        "complete": True,
    }
    assert "subset" not in result.note


@pytest.mark.parametrize("ids", [(100,), (1, 1), (999,)])
def test_rejects_fabricated_or_ineligible_final_ids(store, fake_predictor, ids):
    backend = FakeBackend([call("search_players", {}), final(ids)])
    result = ScoutingAgent(backend, ScoutingTools(store, fake_predictor)).run("five midfielders")
    assert result.error


def test_rejects_unscored_and_constraint_violations(store, fake_predictor):
    backend = FakeBackend([call("search_players", {}), final()])
    result = ScoutingAgent(backend, ScoutingTools(store, fake_predictor)).run("five midfielders")
    assert "unscored" in result.error
    backend = FakeBackend([call("search_players", {}), final((3,))])
    result = ScoutingAgent(backend, ScoutingTools(store)).run("under 20")
    assert "violates" in result.error  # Fixture 3 is age 20.


def test_agent_only_and_parse_errors_are_recorded(store):
    assert "predict_player_potential" not in [
        tool["function"]["name"] for tool in tool_schemas(False)
    ]
    backend = FakeBackend([call("search_players", {}), final()])
    result = ScoutingAgent(backend, ScoutingTools(store)).run("five midfielders")
    assert result.error is None
    assert result.recommendations[0]["predicted_potential"] is None
    assert "Potential ≈" not in render_shortlist(result)


def test_loop_limit_and_malformed_arguments(store):
    backend = FakeBackend([call("search_players", {"limit": 999}), call("search_players", {})])
    result = ScoutingAgent(backend, ScoutingTools(store), max_tool_steps=1).run("query")
    assert "step limit" in result.error
    assert "error" in result.traces[0]["result"]


def test_truncation_disclosed(store):
    backend = FakeBackend([call("search_players", {"limit": 1}), final()])
    result = ScoutingAgent(backend, ScoutingTools(store)).run("query")
    assert "Only some" in result.note


def test_predictor_clips_to_the_game_scale_and_checks_identity(records, tmp_path):
    players = [row.visible for row in records[:3]]
    schema = FeatureSchema.fit(players)
    model = SimpleNamespace(
        predict=lambda matrix, output_type="mean": np.array([150.4, 250.0, -3.0]),
        save_model=lambda: {"fixture_model": True},
    )
    predictor = HostedPredictor(model, schema)
    result = predictor.predict(players)
    assert [row["predicted_potential"] for row in result] == [150.4, 200.0, 1.0]
    path = tmp_path / "model.json"
    predictor.save(path, "fixture")
    assert json.loads(path.read_text())["preparation_id"] == "fixture"
    assert HostedPredictor.check_reference(path, schema, "fixture") is None
    assert "another save" in HostedPredictor.check_reference(path, schema, "other")
    model.predict = lambda matrix, output_type="mean": np.array([np.nan, 1.0, 2.0])
    with pytest.raises(ValueError, match="invalid potential estimates"):
        predictor.predict(players)


def test_backend_preserves_tool_calls_and_provider_reasoning():
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(
            model_dump=lambda **kwargs: {
                "role": "assistant",
                "reasoning_content": "provider field",
                "tool_calls": [],
            }
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 4}),
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    reply = OpenAICompatibleBackend(LLMSettings(), "fixture-secret", client).complete([], [])
    assert captured["model"] == "deepseek-flash"
    assert captured["temperature"] == 0
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["max_tokens"] == 4096
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
    assert reply.message["reasoning_content"] == "provider field"


@pytest.mark.parametrize("content", [None, "", "   ", "Here is my shortlist:", '{"constraints":'])
def test_empty_or_non_json_final_gets_bounded_json_retry(store, content):
    backend = FakeBackend(
        [
            call("search_players", {"age_max": 19, "position": "MC"}),
            ChatReply({"role": "assistant", "content": content}, {"total_tokens": 2}),
            final(),
        ]
    )
    result = ScoutingAgent(backend, ScoutingTools(store)).run("five midfielders under 20")
    assert result.error is None
    assert result.final_retries == 1
    assert result.usage["total_tokens"] == 15
    assert backend.tool_sets[-1] == []
    assert result.validation_events[0]["kind"] == "format"


def test_json_retry_exhaustion_is_actionable_and_bounded(store):
    empty = ChatReply({"role": "assistant", "content": None})
    backend = FakeBackend([call("search_players", {}), empty, empty, empty])
    result = ScoutingAgent(backend, ScoutingTools(store), final_retries=2).run("query")
    assert "bounded retries" in result.error
    assert result.final_retries == 2
    assert len(backend.messages) == 4
    assert backend.tool_sets[-2:] == [[], []]


def test_fenced_json_and_long_notes_do_not_destroy_valid_shortlist(store):
    reply = final(note="A useful caveat. " * 100)
    reply.message["content"] = "```json\n" + reply.message["content"] + "\n```"
    backend = FakeBackend([call("search_players", {}), reply])
    result = ScoutingAgent(backend, ScoutingTools(store)).run("query")
    assert result.error is None
    assert result.final_retries == 0
    assert len(result.note) > 1000


def test_truncated_response_retries_even_if_partial_content_parses(store):
    truncated = final()
    truncated.finish_reason = "length"
    backend = FakeBackend([call("search_players", {}), truncated, final()])
    result = ScoutingAgent(backend, ScoutingTools(store)).run("query")
    assert result.error is None
    assert result.final_retries == 1
    assert result.finish_reasons == ["length"]


def test_format_retry_cannot_change_parsed_constraints(store):
    invalid = final()
    data = json.loads(invalid.message["content"])
    data["recommendations"][0]["explanation"] = "x" * 501
    invalid.message["content"] = json.dumps(data)
    changed = final(constraints={"age_max": 20, "value_max_eur": 8_000_000, "position": "MC"})
    backend = FakeBackend([call("search_players", {}), invalid, changed])
    result = ScoutingAgent(backend, ScoutingTools(store)).run("query")
    assert "changed normalized constraints" in result.error
    assert result.final_retries == 1
    assert "x" * 100 not in json.dumps(result.validation_events)


def test_schema_format_retry_preserves_valid_constraints(store):
    invalid = final()
    data = json.loads(invalid.message["content"])
    del data["note"]
    invalid.message["content"] = json.dumps(data)
    backend = FakeBackend([call("search_players", {}), invalid, final()])
    result = ScoutingAgent(backend, ScoutingTools(store)).run("query")
    assert result.error is None
    assert result.final_retries == 1


def test_invalid_constraint_schema_is_not_repaired(store):
    backend = FakeBackend([call("search_players", {}), final(constraints={"age_max": "19"})])
    result = ScoutingAgent(backend, ScoutingTools(store)).run("under 20")
    assert "semantic constraints cannot be repaired" in result.error
    assert result.final_retries == 0


def test_valid_constraints_are_reported_exactly_as_given(store):
    backend = FakeBackend([call("search_players", {}), final(constraints={"age_max": 20})])
    result = ScoutingAgent(backend, ScoutingTools(store)).run("under 20")
    assert result.error is None
    assert normalized(result) == {"age_max": 20}  # reported as given, never silently repaired
    assert result.final_retries == 0


def two_midfielders(store):
    players = store.get_players([1, 2])
    players[1]["natural_positions"] = ["DM"]
    players[1]["accomplished_positions"] = ["MC"]
    store.initialize(players, store.metadata())
    return players


class RankedPredictor:
    def __init__(self):
        self.batches = []

    def predict(self, players):
        self.batches.append([row["player_id"] for row in players])
        return [
            {
                "player_id": row["player_id"],
                "predicted_potential": {1: 100.0, 2: 160.0}[row["player_id"]],
            }
            for row in players
        ]


def test_natural_mc_narrowing_is_rejected_then_corrected_within_tool_budget(store):
    two_midfielders(store)
    predictor = RankedPredictor()
    backend = FakeBackend(
        [
            call("search_players", {"age_max": 19, "position": "MC"}),
            call("predict_player_potential", {"player_ids": [1, 2]}),
            final((1,), requested_count=1),
            call("get_player_details", {"player_ids": [2]}),
            final((2,), requested_count=1),
        ]
    )
    result = ScoutingAgent(backend, ScoutingTools(store, predictor)).run("the best MC under 20")
    assert result.error is None
    assert [row["player_id"] for row in result.recommendations] == [2]
    assert result.validation_events[0]["kind"] == "ranking"
    assert result.final_retries == 0
    assert predictor.batches == [[1, 2]]


def test_ranking_error_does_not_extend_tool_budget(store):
    two_midfielders(store)
    backend = FakeBackend(
        [
            call("search_players", {}),
            call("predict_player_potential", {"player_ids": [1, 2]}),
            final((1,), requested_count=1),
        ]
    )
    result = ScoutingAgent(backend, ScoutingTools(store, RankedPredictor()), max_tool_steps=2).run(
        "query"
    )
    assert "Ranking validation failed at tool-step limit" in result.error
    assert len(backend.messages) == 3


def test_missing_scores_cannot_hide_higher_scoring_candidates(store):
    two_midfielders(store)
    backend = FakeBackend(
        [
            call("search_players", {}),
            call("predict_player_potential", {"player_ids": [1]}),
            final((1,), requested_count=1),
            call("predict_player_potential", {"player_ids": [1, 2]}),
            final((2,), requested_count=1),
        ]
    )
    predictor = RankedPredictor()
    result = ScoutingAgent(backend, ScoutingTools(store, predictor)).run("query")
    assert result.error is None
    assert predictor.batches == [[1], [2]]


def test_tied_scores_use_stable_id_order_and_return_available_players(store, fake_predictor):
    two_midfielders(store)
    backend = FakeBackend(
        [
            call("search_players", {}),
            call("predict_player_potential", {"player_ids": [2, 1]}),
            final((2, 1)),
        ]
    )
    result = ScoutingAgent(backend, ScoutingTools(store, fake_predictor)).run("five midfielders")
    assert result.error is None
    assert [row["player_id"] for row in result.recommendations] == [1, 2]


def test_prediction_results_are_sorted_and_cached(store):
    predictor = RankedPredictor()
    tools = ScoutingTools(store, predictor)
    first = tools.call("predict_player_potential", {"player_ids": [1, 2]})
    second = tools.call("predict_player_potential", {"player_ids": [2, 1]})
    assert first == second
    assert [row["player_id"] for row in first] == [2, 1]
    assert predictor.batches == [[1, 2]]


def test_every_player_in_the_save_can_be_recommended(store):
    """Players the model was fitted on are ordinary candidates; nothing is labelled or hidden."""
    backend = FakeBackend([call("search_players", {}), final((100,))])
    result = ScoutingAgent(backend, ScoutingTools(store)).run("five midfielders")
    assert result.error is None
    assert [row["player_id"] for row in result.recommendations] == [100]
    shown = render_shortlist(result)
    assert "training" not in shown.lower() and "held-out" not in shown.lower()


def test_backend_other_provider_does_not_receive_deepseek_parameters():
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        model_dump=lambda **kw: {"role": "assistant", "content": "{}"}
                    ),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    backend = OpenAICompatibleBackend(
        LLMSettings(base_url="https://example.test/v1"), "fixture", client
    )
    reply = backend.complete([], tool_schemas(False))
    assert "extra_body" not in captured
    assert "response_format" not in captured
    assert captured["max_tokens"] == 4096
    assert reply.finish_reason == "stop"
