"""Objectives: validation, deterministic execution, funnel, suggestions, and fair value."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from fm26_agent import fair_value, tabpfn_backend
from fm26_agent.objective import Objective, execute
from fm26_agent.prediction import QUANTILES
from fm26_agent.present import render_objective, render_result

LEVELS = np.array(QUANTILES)


def curve(centre: float, spread: float = 4.0) -> np.ndarray:
    return centre + (LEVELS - 0.5) * spread


class FakeLab:
    """Deterministic 'predictions' from player ids, so the expected shortlist is known."""

    def __init__(self, targets=None):
        self.targets = targets or ["potential_ability", "current_ability", "injury_proneness"]
        self.values = SimpleNamespace(fair_values=lambda players: {})

    def available(self, include_value=True):
        return self.targets

    def build_quality(self, target):
        return {"target": target, "verdict": "useful", "average_error": 1.0,
                "average_error_if_guessing": 2.0}  # fmt: skip

    def distribution(self, target, players):
        out = {}
        for row in players:
            pid = row["player_id"]
            if target == "potential_ability":
                out[pid] = curve(100 + pid, 20)  # higher id, more potential
            elif target == "current_ability":
                out[pid] = curve(200 - pid)  # higher id, less ability
            elif target == "injury_proneness" and pid % 2 == 0:
                out[pid] = np.clip(curve(pid % 20), 1, 20)
        return out


def objective(**overrides):
    data = {
        "filters": {"position": "MC"},
        "conditions": [{"target": "potential_ability", "at_least": 160, "min_chance": 0.5}],
        "rank_by": {"target": "potential_ability"},
        "count": 3,
    } | overrides
    return Objective.from_dict(data, FakeLab().available())


def test_objectives_are_validated_in_code():
    with pytest.raises(ValueError, match="Unknown target"):
        objective(rank_by={"target": "shoe_size"})
    with pytest.raises(ValueError, match="cannot predict"):
        objective(rank_by={"target": "consistency"})
    with pytest.raises(ValueError, match="between 1 and 200"):
        objective(conditions=[{"target": "potential_ability", "at_least": 250, "min_chance": 0.5}])
    with pytest.raises(ValueError, match="needs at_least"):
        objective(rank_by={"target": "potential_ability", "mode": "chance"})
    with pytest.raises(ValueError, match="age_min cannot exceed"):
        objective(filters={"age_min": 25, "age_max": 20})
    with pytest.raises(ValueError, match="Invalid objective"):
        objective(conditions=[{"target": "potential_ability", "min_chance": 0.5}])
    assert objective(filters={"position": "striker"}).filters == {"position": "STC"}


def test_round_trip_keeps_the_objective(store):
    first = objective(readings=[{"phrase": "world class", "meaning": "potential 160+"}])
    again = Objective.from_dict(json.loads(json.dumps(first.to_dict())), FakeLab().available())
    assert again == first


def test_execution_applies_conditions_in_order_and_ranks(store):
    goal = objective(
        conditions=[
            {"target": "potential_ability", "at_least": 160, "min_chance": 0.5},
            {"target": "current_ability", "at_least": 120, "min_chance": 0.5},
        ],
    )
    result = execute(goal, store, FakeLab())
    # potential 160+ likely: ids >= 60; current ability 120+: ids <= 80
    assert [stage.remaining for stage in result.funnel] == [100, 41, 21]
    assert [row["player_id"] for row in result.shortlist] == [80, 79, 78]
    assert result.shortlist[0]["targets"]["potential_ability"]["estimate"] == 180
    assert again_same(goal, store) == [80, 79, 78]  # deterministic


def again_same(goal, store):
    return [row["player_id"] for row in execute(goal, store, FakeLab()).shortlist]


def test_low_is_good_and_missing_data_are_handled(store):
    goal = objective(
        conditions=[{"target": "injury_proneness", "at_most": 6, "min_chance": 0.5}],
        rank_by={"target": "injury_proneness"},
    )
    result = execute(goal, store, FakeLab())
    assert result.funnel[1].no_data == 50  # odd ids have no value
    estimates = [row["targets"]["injury_proneness"]["estimate"] for row in result.shortlist]
    assert estimates == sorted(estimates)  # lowest first


def test_too_few_players_gives_exact_suggestions(store):
    goal = objective(
        conditions=[{"target": "potential_ability", "at_least": 199, "min_chance": 0.5}],
        count=5,
    )
    result = execute(goal, store, FakeLab())
    assert len(result.shortlist) == 2
    assert any("gives 5 players" in tip for tip in result.suggestions)


def test_card_and_result_show_every_condition(store):
    goal = objective(readings=[{"phrase": "world class", "meaning": "potential 160 or higher"}])
    card = render_objective(goal, {"potential_ability": FakeLab().build_quality("x")}, 100)
    assert "potential 160 or higher, at least 50% likely" in card and "(100 players)" in card
    text = render_result(execute(goal, store, FakeLab()), store)
    assert "100 match the filters → 41 potential 160 or higher" in text
    assert "chance of potential 160 or higher" in text
    assert '"world class" was read as: potential 160 or higher.' in text


# --- fair value ---------------------------------------------------------------------------


def test_fair_value_never_prices_a_player_with_his_own_price(records, monkeypatch):
    players = [row.visible for row in records]
    seen = []
    monkeypatch.setattr(
        fair_value,
        "value_table",
        lambda schema, rows: seen.append([row["player_id"] for row in rows]) or rows,
    )
    monkeypatch.setattr(tabpfn_backend, "new_regressor", lambda backend, seed: object())
    monkeypatch.setattr(tabpfn_backend, "fit", lambda model, backend, table, target: None)
    monkeypatch.setattr(
        tabpfn_backend,
        "predict_quantiles",
        lambda model, backend, table, levels: np.tile(
            np.log([row["value_eur"] for row in table]), (len(levels), 1)
        ),
    )
    curves, report = fair_value.cross_fitted_curves(players, None, "hosted", 42)
    fit_a, score_b, fit_b, score_a = seen
    assert not set(fit_a) & set(score_b) and not set(fit_b) & set(score_a)
    assert set(score_a) | set(score_b) == {row["player_id"] for row in players}
    assert report["cross_fitted"] and report["priced_players"] == 100


def test_ratio_curve_turns_fair_value_into_price_share():
    log_fair = np.log(np.linspace(5e6, 20e6, len(QUANTILES)))
    ratio = fair_value.ratio_curve(10e6, log_fair)
    assert np.all(np.diff(ratio) >= 0)  # still ascending percentiles
    assert ratio[QUANTILES.index(0.5)] == pytest.approx(10e6 / 12.5e6)
