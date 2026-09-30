"""Agent-built prediction tasks: glossary, private targets, fitting, ranking and answer checks."""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest
from test_agent import FakeBackend, call

from fm26_agent import custom_tasks, private_db, tabpfn_backend
from fm26_agent.agent import ScoutingAgent, render_shortlist
from fm26_agent.backend import ChatReply
from fm26_agent.custom_tasks import TaskLab, TaskSpec, quality_report
from fm26_agent.extract import record_to_player
from fm26_agent.features import FeatureSchema
from fm26_agent.finder import find_players
from fm26_agent.private_db import PrivateStore
from fm26_agent.sample import export_sample, read_sample
from fm26_agent.schema import VISIBLE_ATTRIBUTES, assert_safe_features
from fm26_agent.targets import PERSONALITY, TARGETS, get_target
from fm26_agent.tools import BUILD_TASK_TOOL, TASK_PREDICTION_TOOL, ScoutingTools
from fm26_agent.visible_db import VisibleStore

PLAYERS = 120  # 80 reference players, 40 held out


def _player(index: int):
    finishing = 1 + index % 20
    return SimpleNamespace(
        uid=index,
        name=f"Task {index}",
        age=20 + index % 12,
        club_name="Task FC",
        club_uid=7,
        nation_id=3,
        height_cm=180,
        left_foot=5,
        right_foot=20,
        natural_positions=("STC",),
        accomplished_positions=(),
        traits=(),
        on_loan=False,
        transfer_value=1_000_000 + index * 10_000,
        contract=SimpleNamespace(wage=1000, end=date(2078, 6, 30)),
        attributes=SimpleNamespace(
            **{key: 10 for key in VISIBLE_ATTRIBUTES}
            | {"finishing": finishing, "consistency": finishing, "injury_proneness": finishing}
        ),
        # Unique true values (x.137; reports round to 2 decimals) so a leak would be visible.
        ability=SimpleNamespace(current=100 + index + 0.137, potential=190),
        personality=SimpleNamespace(**{name: 1 + index % 20 for name in PERSONALITY}),
    )


class FakeModel:
    """Predicts each player's finishing, with a +-2 spread across the quantiles."""

    def save_model(self):
        return "handle"


@pytest.fixture
def fits(monkeypatch):
    made = []

    def fit(model, backend, table, target):
        made.append(len(table))

    def predict(model, backend, table, levels):
        centre = table["finishing"].to_numpy(dtype=float) * 5
        return np.array([centre + (level - 0.5) * 4 for level in levels])

    monkeypatch.setattr(tabpfn_backend, "new_regressor", lambda backend, seed: FakeModel())
    monkeypatch.setattr(tabpfn_backend, "fit", fit)
    monkeypatch.setattr(tabpfn_backend, "predict_quantiles", predict)
    monkeypatch.setattr(tabpfn_backend, "load_fitted", lambda stored, backend, path: FakeModel())
    monkeypatch.setattr(custom_tasks, "MIN_TRAIN", 50)
    monkeypatch.setattr(private_db, "MIN_TARGET_VALUES", 50)
    return made


@pytest.fixture
def world(tmp_path):
    extracted = [record_to_player(_player(index), date(2076, 7, 1)) for index in range(1, 121)]
    for player in extracted:
        player.visible["split"] = "train" if player.visible["player_id"] <= 80 else "test"
    visible = VisibleStore(tmp_path / "visible.sqlite3")
    visible.initialize(
        [player.visible for player in extracted],
        {"save_date": "2076-07-01", "preparation_id": "tasks", "model_ready": True},
    )
    private = PrivateStore(tmp_path / "private.sqlite3")
    private.initialize(
        [
            {
                "player_id": player.visible["player_id"],
                "potential_ability": player.potential_ability,
                "wonderkid": 1,
                "split": player.visible["split"],
            }
            for player in extracted
        ],
        "tasks",
    )
    private.set_targets({player.visible["player_id"]: player.hidden for player in extracted})
    schema = FeatureSchema.fit(visible.get_players(visible.all_ids()))
    return SimpleNamespace(visible=visible, private=private, schema=schema, extracted=extracted)


def _lab(world, tmp_path):
    return TaskLab(world.visible, world.private, world.schema, "hosted", tmp_path / "tasks")


class FakePotential:
    task, score_field, score_bounds = "pa_regression", "predicted_potential", (1.0, 200.0)

    def predict(self, players):
        return [{"player_id": row["player_id"], "predicted_potential": 150.0} for row in players]


# --- glossary and storage ---------------------------------------------------------------


def test_every_target_is_a_forbidden_feature():
    for name in TARGETS:
        with pytest.raises(ValueError, match="Forbidden"):
            assert_safe_features([name])


def test_hidden_values_are_extracted_privately_and_checked_against_their_scale(world):
    player = world.extracted[0]
    assert player.hidden["current_ability"] == 101.137
    assert player.hidden["consistency"] == 2 and player.hidden["professionalism"] == 2
    assert not set(TARGETS) & set(player.visible)
    odd = _player(5)
    odd.attributes.consistency = 99  # outside 1-20: a misread, never stored
    odd.personality = None
    hidden = record_to_player(odd, date(2076, 7, 1)).hidden
    assert hidden["consistency"] is None and hidden["ambition"] is None


def test_private_store_offers_derived_growth_room(world, fits):
    available = world.private.available_targets()
    assert available[0] == "potential_ability" and "growth_room" in available
    growth = world.private.target_values("growth_room", "train")
    assert len(growth) == 80 and growth[1] == pytest.approx(190 - 101.137)


def test_task_spec_is_validated_against_the_glossary():
    assert TaskSpec("consistency", 15).task_id == "consistency@15"
    assert TaskSpec.from_id("consistency@15") == TaskSpec("consistency", 15.0)
    with pytest.raises(ValueError, match="Unknown target"):
        TaskSpec("shoe_size")
    with pytest.raises(ValueError, match="between 1 and 20"):
        TaskSpec("consistency", 25)
    assert TaskSpec("injury_proneness", 5).describe_goal() == "injury proneness 5 or lower"


def test_threshold_chance_counts_whole_numbers_fairly():
    from fm26_agent.custom_tasks import threshold_chance

    # A player estimated at exactly 15 with bunched percentiles: "15 or better" is likely.
    curve = np.array([8.0] * 5 + [12.0] * 4 + [15.0] * 10)
    assert threshold_chance(curve, 15, "high") > 0.5
    assert threshold_chance(curve, 15, "low") > 0.9
    assert threshold_chance(np.full(19, 5.0), 15, "high") < 0.05


def test_quality_report_compares_with_guessing_the_average():
    actual = np.array([1.0, 5.0, 10.0, 15.0, 20.0])
    exact = np.tile(actual, (19, 1))
    assert quality_report(actual, exact, 10.2, get_target("consistency"))["verdict"] == "useful"
    guess = np.full((19, 5), 10.2)
    report = quality_report(actual, guess, 10.2, get_target("consistency"))
    assert report["verdict"] == "not predictable" and report["better_than_guessing"] == 0


# --- fitting and caching ----------------------------------------------------------------


def test_a_task_is_fitted_once_checked_on_held_out_players_and_reused(world, fits, tmp_path):
    lab = _lab(world, tmp_path)
    report = lab.build(TaskSpec("consistency"))
    assert fits == [80] and report["trained_on"] == 80 and report["checked_on"] == 40
    lab.build(TaskSpec("consistency", 12))  # same target, other threshold: no new fit
    assert lab.fits == 1
    again = _lab(world, tmp_path)  # a later session loads the stored fit
    assert again.build(TaskSpec("consistency")) == report and again.fits == 0
    rows = again.predict(TaskSpec("consistency", 12), world.visible.get_players([2]))
    assert rows[0]["estimate"] == 15 and 0.5 < rows[0]["chance"] <= 1


def test_a_stored_fit_from_another_setup_is_not_reused(world, fits, tmp_path):
    _lab(world, tmp_path).build(TaskSpec("consistency"))
    world.visible.set_metadata("preparation_id", "other")
    other = _lab(world, tmp_path)
    other.build(TaskSpec("consistency"))
    assert other.fits == 1


# --- tools and answer checks ------------------------------------------------------------


def test_low_is_good_targets_rank_the_lowest_estimate_first(world, fits, tmp_path):
    tools = ScoutingTools(world.visible, FakePotential(), lab=_lab(world, tmp_path))
    names = [tool["function"]["name"] for tool in tools.schemas]
    assert BUILD_TASK_TOOL in names and TASK_PREDICTION_TOOL in names
    built = tools.call(BUILD_TASK_TOOL, {"target": "injury_proneness"})
    assert built["better"] == "low" and built["quality"]["checked_on"] == 40
    search = tools.call("search_players", {"position": "STC"})
    ranked = tools.call(
        TASK_PREDICTION_TOOL, {"task_id": built["task_id"], "search_id": search["search_id"]}
    )
    estimates = [row["estimate"] for row in ranked["ranked_players"]]
    assert estimates == sorted(estimates) and estimates[0] == 5
    with pytest.raises(ValueError, match="threshold"):
        tools.call(
            TASK_PREDICTION_TOOL,
            {"task_id": built["task_id"], "search_id": search["search_id"], "rank_by": "chance"},
        )


def test_no_task_tools_without_hidden_targets(store, fake_predictor):
    names = [tool["function"]["name"] for tool in ScoutingTools(store, fake_predictor).schemas]
    assert BUILD_TASK_TOOL not in names


def _answer(ids, task_id, ranking="expected"):
    return ChatReply(
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "constraints": {"position": "STC", "age_min": 24, "age_max": 29},
                    "requested_count": 3,
                    "ranking": ranking,
                    "task_id": task_id,
                    "recommendations": [
                        {"player_id": player_id, "explanation": "Sharp finisher."}
                        for player_id in ids
                    ],
                    "note": "Read 'in his prime' as 24-29.",
                }
            ),
        },
        {"total_tokens": 10},
        "stop",
    )


def test_agent_builds_its_own_task_and_the_answer_is_checked(world, fits, tmp_path):
    tools = ScoutingTools(world.visible, FakePotential(), lab=_lab(world, tmp_path))
    backend = FakeBackend(
        [
            call(BUILD_TASK_TOOL, {"target": "current_ability"}),
            call("search_players", {"position": "STC", "age_min": 24, "age_max": 29}, index=2),
            call(
                TASK_PREDICTION_TOOL,
                {"task_id": "current_ability", "search_id": "search-1"},
                index=3,
            ),
            _answer([4, 5, 6], "current_ability"),  # not the top: rejected
            _answer([18, 19, 79], "current_ability"),
        ]
    )
    result = ScoutingAgent(backend, tools).run("a striker in his prime")
    assert result.error is None, result.error
    assert result.validation_events[0]["kind"] == "ranking"
    # finishing 20 -> 100 for players 19 and 79 (tie: lower id first), then 18 at 95
    assert [row["player_id"] for row in result.recommendations] == [19, 79, 18]
    assert result.task["target"] == "current_ability"
    text = render_shortlist(result)
    assert "Current ability ≈ 100" in text and "players it hadn't seen" in text
    conversation = json.dumps(backend.messages)
    for player in world.extracted:  # true hidden values never reach the LLM
        assert f"{player.hidden['current_ability']}" not in conversation
    assert "BUILDING YOUR OWN PREDICTION" in backend.messages[0][0]["content"]


def test_find_ranks_by_a_task_without_an_llm(world, fits, tmp_path):
    tools = ScoutingTools(world.visible, FakePotential(), lab=_lab(world, tmp_path))
    result = find_players(
        world.visible,
        tools,
        count=2,
        rank_by="chance",
        task=TaskSpec("consistency", 18),
        position="STC",
    )
    assert result.error is None and result.task_id == "consistency@18"
    assert [row["task_chance"] >= 0.5 for row in result.recommendations] == [True, True]
    assert "chance of consistency 18 or higher" in render_shortlist(result)


def test_sample_keeps_the_hidden_targets(world, tmp_path):
    path = tmp_path / "sample" / "players.csv.gz"
    export_sample(world.visible, world.private, path)
    first = sorted(read_sample(path).players, key=lambda p: p.visible["player_id"])[0]
    assert first.hidden["current_ability"] == 101.137 and first.hidden["consistency"] == 2
