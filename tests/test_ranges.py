from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import ScoreFields
from test_agent import FakeBackend, call, final

from fm26_agent.agent import format_star_chance, render_shortlist
from fm26_agent.config import DataSettings, LLMSettings, Settings, TrainingSettings
from fm26_agent.features import FeatureSchema
from fm26_agent.prediction import QUANTILES, HostedPredictor, chance_of_reaching
from fm26_agent.prediction_cache import CachedPredictor
from fm26_agent.runtime import scout


class RangePredictor(ScoreFields):
    """Same estimate for everyone, but ranges that differ: the order depends on the ranking mode."""

    def predict(self, players):
        return [
            {
                "player_id": row["player_id"],
                "predicted_potential": 150.0,
                "potential_low": 120.0 + row["player_id"] % 7,
                "potential_high": 160.0 + (row["player_id"] * 3) % 25,
                "star_chance": 0.7,
            }
            for row in players
        ]


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


def fitted_predictor(records, curve, calls):
    """A HostedPredictor whose model returns `curve(level)` for every player at each percentile."""
    players = [row.visible for row in records[:4]]

    def predict(matrix, output_type="mean", quantiles=None):
        calls.append(output_type)
        assert output_type == "quantiles" and tuple(quantiles) == QUANTILES
        return np.array([np.full(len(matrix), curve(level)) for level in quantiles])

    return HostedPredictor(SimpleNamespace(predict=predict), FeatureSchema.fit(players)), players


def test_one_call_gives_estimate_range_and_chance(records):
    calls = []
    # Potential spread evenly between 100 and 200: median 150, range 110-190, half reach 150.
    predictor, players = fitted_predictor(records, lambda level: 100.0 + 100.0 * level, calls)
    predictor.star_level = 150
    rows = predictor.predict(players)
    assert calls == ["quantiles"]  # a single hosted request for everything
    for row in rows:
        assert row["predicted_potential"] == 150.0
        assert (row["potential_low"], row["potential_high"]) == (110.0, 190.0)
        assert abs(row["star_chance"] - 0.5) < 1e-9


def test_distribution_is_kept_on_the_game_scale_and_in_order(records):
    # A curve that dips and overshoots the scale is made non-decreasing and clipped to 1-200.
    predictor, players = fitted_predictor(
        records, lambda level: 260.0 if level == 0.3 else 300.0 * level - 20, []
    )
    for row in predictor.predict(players):
        assert (
            1 <= row["potential_low"] <= row["predicted_potential"] <= row["potential_high"] <= 200
        )


def test_invalid_model_output_is_refused(records):
    predictor, players = fitted_predictor(records, lambda level: float("nan"), [])
    with pytest.raises(ValueError, match="invalid potential estimates"):
        predictor.predict(players)


def test_chance_is_capped_where_the_distribution_ends():
    curve = np.array([100.0 + 100.0 * level for level in QUANTILES])
    assert chance_of_reaching(curve, 50) == 0.975
    assert chance_of_reaching(curve, 199.9) == 0.025
    assert chance_of_reaching(curve, 170) == 0.3


def test_format_star_chance():
    assert format_star_chance(0.714, 160) == "71% chance of reaching 160+"
    assert format_star_chance(0.975, 160) == "over 95% chance of reaching 160+"
    assert format_star_chance(0.025, 160) == "under 5% chance of reaching 160+"
    assert format_star_chance(None, 160) is None


def test_cache_keeps_ranges_and_tolerates_models_without_them(tmp_path):
    cached = CachedPredictor(RangePredictor(), tmp_path / "cache.sqlite3", "ns")
    players = [{"player_id": 5}, {"player_id": 9}]
    first = cached.predict(players)
    assert cached.predict(players) == first  # second call is served from disk
    assert first[0]["potential_low"] <= 150.0 <= first[0]["potential_high"]
    assert first[0]["star_chance"] == 0.7

    class NoRange(ScoreFields):
        def predict(self, rows):
            return [{"player_id": row["player_id"], "predicted_potential": 120.0} for row in rows]

    plain = CachedPredictor(NoRange(), tmp_path / "cache.sqlite3", "plain")
    assert plain.predict([{"player_id": 5}]) == [{"player_id": 5, "predicted_potential": 120.0}]


def test_ceiling_ranking_orders_by_the_top_of_the_range(settings, store):
    # Players 1, 2, 4, 5, 6 are the under-20 midfielders in the fixture's first rows; rank the
    # whole matching pool and compare with the brute-force ceiling order.
    from fm26_agent.visible_db import VisibleStore

    pool_ids = VisibleStore(store.path).search(
        age_max=19, value_max_eur=8_000_000, position="MC", limit=500
    )["player_ids"]
    highs = {i: 160.0 + (i * 3) % 25 for i in pool_ids}
    expected = sorted(pool_ids, key=lambda i: (-highs[i], i))[:5]
    backend = FakeBackend(
        [
            call("search_players", {"age_max": 19, "value_max_eur": 8_000_000, "position": "MC"}),
            call("predict_player_potential", {"search_id": "search-1", "rank_by": "ceiling"}),
            final(expected, ranking="ceiling"),
        ]
    )
    result, _ = scout(
        settings, backend, store, RangePredictor(), "five midfielders with the most upside"
    )
    assert result.error is None and result.ranking == "ceiling"
    assert [row["player_id"] for row in result.recommendations] == expected
    assert result.recommendations[0]["potential_high"] == highs[expected[0]]
    assert result.recommendations[0]["star_chance"] == 0.7
    text = render_shortlist(result)
    assert "(likely" in text
    assert "70% chance of reaching 160+" in text
    assert "Ranked by best case" in text


def test_ranking_by_the_wrong_measure_is_rejected(settings, store):
    # The expected order would be by player id (all estimates tie); ceiling differs.
    backend = FakeBackend(
        [
            call("search_players", {"age_max": 19, "value_max_eur": 8_000_000, "position": "MC"}),
            call("predict_player_potential", {"search_id": "search-1"}),
            final((1, 2, 4, 5, 6), ranking="ceiling"),
        ]
    )
    result, _ = scout(settings, backend, store, RangePredictor(), "most upside")
    assert result.error is not None  # ids were ranked by id, not by the ceiling


def test_default_ranking_is_unchanged_and_shows_the_range(settings, store):
    backend = FakeBackend(
        [
            call("search_players", {"age_max": 19, "value_max_eur": 8_000_000, "position": "MC"}),
            call("predict_player_potential", {"search_id": "search-1"}),
            final((1, 2, 4, 5, 6)),
        ]
    )
    result, _ = scout(settings, backend, store, RangePredictor(), "five midfielders")
    assert result.error is None and result.ranking == "expected"
    text = render_shortlist(result)
    assert "Potential ≈ 150 (likely" in text and "Ranked by" not in text
    assert "four times in five" in text


def test_rate_limit_is_waited_out_and_other_errors_are_not(monkeypatch):
    from fm26_agent import prediction

    waits = []
    monkeypatch.setattr(prediction.time, "sleep", waits.append)
    replies = iter(
        [RuntimeError("Fail to call predict: [HTTP 429] Rate limit exceeded. Retry in 4s.."), "ok"]
    )

    def flaky():
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    assert prediction.with_rate_limit_retry(flaky) == "ok" and waits == [5]
    with pytest.raises(RuntimeError, match="HTTP 500"):
        prediction.with_rate_limit_retry(lambda: (_ for _ in ()).throw(RuntimeError("HTTP 500")))
    with pytest.raises(RuntimeError, match="429"):
        prediction.with_rate_limit_retry(
            lambda: (_ for _ in ()).throw(RuntimeError("HTTP 429")), attempts=2
        )
