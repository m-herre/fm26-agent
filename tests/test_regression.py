from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest
from test_agent import FakeBackend, call, final

from fm26_agent.agent import ScoutingAgent, render_shortlist
from fm26_agent.config import DataSettings, LLMSettings, Settings, TrainingSettings
from fm26_agent.extract import ExtractionResult
from fm26_agent.features import FeatureSchema
from fm26_agent.metrics import regression_metrics
from fm26_agent.prediction import HostedPredictor, HostedRegressionPredictor
from fm26_agent.prediction_cache import CachedPredictor
from fm26_agent.prepare import prepare
from fm26_agent.private_db import PrivateStore
from fm26_agent.regression import compare_models, fit_regression, reference_path
from fm26_agent.tools import ScoutingTools, tool_schemas


@pytest.fixture
def prepared_regression(tmp_path, monkeypatch, records):
    import tabpfn_client

    source = tmp_path / "fixture.fm"
    source.write_text("mocked save")
    settings = Settings(
        DataSettings(
            tmp_path / "visible.sqlite3",
            tmp_path / "private.sqlite3",
            tmp_path / "model.json",
            tmp_path / "schema.json",
            tmp_path / "runs",
            tmp_path / "cache.sqlite3",
        ),
        TrainingSettings(max_train_rows=50, max_evaluation_rows=10),
        LLMSettings(),
        1.2,
        source,
    )
    captured = {"classifier_fits": 0, "regression_fits": 0, "regression_predictions": 0}

    class Classifier:
        classes_ = np.array([0, 1])

        def __init__(self, **kwargs):
            pass

        def fit(self, x, y):
            captured["classifier_fits"] += 1

        def predict_proba(self, x):
            return np.tile([0.8, 0.2], (len(x), 1))

        def save_model(self):
            return {"mock": "classifier"}

        @classmethod
        def load_model(cls, payload):
            assert payload == {"mock": "classifier"}
            return cls()

    class Regressor:
        def __init__(self, **kwargs):
            captured["regression_kwargs"] = kwargs

        def fit(self, x, y):
            captured["regression_fits"] += 1
            captured["x"] = x
            captured["y"] = y

        def predict(self, x, output_type="mean"):
            assert output_type == "mean"
            captured["regression_predictions"] += 1
            return np.asarray(x["passing"], dtype=float) + 130

        def save_model(self):
            return {"mock": "regression"}

        @classmethod
        def load_model(cls, payload):
            assert payload == {"mock": "regression"}
            return cls()

    monkeypatch.setenv("TABPFN_TOKEN", "mock-token")
    monkeypatch.setattr(tabpfn_client, "TabPFNClassifier", Classifier)
    monkeypatch.setattr(tabpfn_client, "TabPFNRegressor", Regressor)
    monkeypatch.setattr(
        tabpfn_client, "estimate_cost", lambda *a, **k: SimpleNamespace(estimated_cost=1)
    )
    monkeypatch.setattr(
        "fm26_agent.prepare.read_save",
        lambda *a: ExtractionResult(records, date(2076, 7, 1), "FM26", "fixture", []),
    )
    prepare(settings, source, emit=lambda s: None)
    return settings, captured


def test_regression_fit_uploads_exact_targets_only_and_preserves_classifier(
    prepared_regression, monkeypatch
):
    settings, captured = prepared_regression
    old_model = settings.data.model_reference.read_bytes()
    old_schema = settings.data.feature_schema.read_bytes()
    private = PrivateStore(settings.data.private_database)
    old_membership = private.rows()
    first = fit_regression(settings, emit=lambda s: None)
    assert captured["classifier_fits"] == captured["regression_fits"] == 1
    assert captured["x"].shape == (50, 59)
    assert captured["y"].tolist() == [r["potential_ability"] for r in private.rows("train")]
    assert set(captured["y"]) == {120, 170}
    assert (
        "player_id" not in captured["x"]
        and "potential_ability" not in captured["x"]
        and "name" not in captured["x"]
    )
    assert settings.data.model_reference.read_bytes() == old_model
    assert settings.data.feature_schema.read_bytes() == old_schema
    assert private.rows() == old_membership
    assert captured["regression_kwargs"] == {
        "model_path": "v3.5_default",
        "fit_mode": "fit_with_cache",
        "text_handling": "advanced",
        "random_state": 42,
    }
    assert first["model_metrics"]["eligible_count"] == 10
    assert "mae" in first["model_metrics"]
    monkeypatch.delenv("TABPFN_TOKEN")
    second = fit_regression(settings, emit=lambda s: None)
    assert first["reference_id_hash"] == second["reference_id_hash"]
    assert captured["regression_fits"] == 1
    with pytest.raises(ValueError, match="different prediction task"):
        HostedPredictor.load(reference_path(settings), settings.data.feature_schema)
    with pytest.raises(ValueError, match="regression fit"):
        HostedRegressionPredictor.load(settings.data.model_reference, settings.data.feature_schema)


def test_failed_regression_metrics_keep_fit_and_retry_without_refit(
    prepared_regression, monkeypatch
):
    settings, captured = prepared_regression
    original = HostedRegressionPredictor.predict
    monkeypatch.setattr(
        HostedRegressionPredictor,
        "predict",
        lambda *a: (_ for _ in ()).throw(RuntimeError("private provider payload")),
    )
    first = fit_regression(settings, emit=lambda s: None)
    assert reference_path(settings).exists() and "model_metrics_error" in first
    assert "private provider payload" not in json.dumps(first)
    monkeypatch.setattr(HostedRegressionPredictor, "predict", original)
    second = fit_regression(settings, emit=lambda s: None)
    assert "model_metrics_error" not in second and "model_metrics" in second
    assert captured["regression_fits"] == 1


def test_regression_comparison_uses_identical_pools_and_actual_labels_offline(prepared_regression):
    settings, _ = prepared_regression
    fit_regression(settings, emit=lambda s: None)
    classifier = HostedPredictor.load(settings.data.model_reference, settings.data.feature_schema)
    regressor = HostedRegressionPredictor.load(
        reference_path(settings), settings.data.feature_schema
    )
    report = compare_models(settings, classifier, regressor, emit=lambda s: None)
    assert len(report["cases"]) == 6
    assert report["scope"] == "held_out"
    for case in report["cases"]:
        for mode in ("classifier", "regression"):
            result = case["models"][mode]
            assert "error" not in result
            assert result["scored_count"] == case["eligible_count"]
            assert result["complete"]
            assert all("actual_pa" in row for row in result["shortlist"])


def test_regression_metrics_known_answers_and_empty_pool():
    truth = {
        1: {"potential_ability": 170, "wonderkid": 1},
        2: {"potential_ability": 120, "wonderkid": 0},
    }
    metrics = regression_metrics(
        [
            {"player_id": 1, "predicted_potential": 160},
            {"player_id": 2, "predicted_potential": 130},
        ],
        truth,
    )
    assert metrics["mae"] == metrics["rmse"] == 10
    assert metrics["true_wonderkids_top_5"] == 1
    assert metrics["average_hidden_pa_top_5"] == 145
    assert metrics["r2"] == pytest.approx(0.84)
    assert regression_metrics([], {})["mae"] is None


def test_regression_prediction_shapes_finite_validation_and_scale(records):
    schema = FeatureSchema.fit([r.visible for r in records])
    model = SimpleNamespace(predict=lambda x, output_type: np.asarray([210, -5]))
    predictor = HostedRegressionPredictor(model, schema)
    assert [
        r["predicted_potential"] for r in predictor.predict([r.visible for r in records[:2]])
    ] == [200, 1]
    assert predictor.predict([]) == []
    model.predict = lambda x, output_type: np.asarray([np.nan, 100])
    with pytest.raises(ValueError, match="invalid potential"):
        predictor.predict([r.visible for r in records[:2]])


def test_regression_cache_is_task_isolated_and_keeps_whole_pool(store, tmp_path, fake_predictor):
    class Regressor:
        task, score_field, score_bounds = "pa_regression", "predicted_potential", (1, 200)

        def __init__(self):
            self.calls = 0

        def predict(self, players):
            self.calls += 1
            return [
                {"player_id": r["player_id"], "predicted_potential": 150 + r["player_id"] / 10}
                for r in players
            ]

    path = tmp_path / "cache.sqlite3"
    regression = Regressor()
    binary = CachedPredictor(fake_predictor, path, "same-namespace")
    cached = CachedPredictor(regression, path, "same-namespace")
    players = store.get_players([1, 2])
    binary.predict(players)
    assert cached.predict(players)[0]["predicted_potential"] == 150.1
    cached.predict(players)
    assert regression.calls == 1
    assert binary.predict(players)[0]["wonderkid_probability"] == 0.9


def test_regression_agent_ranks_estimates_not_probabilities_and_never_sees_truth(store):
    class Regressor:
        score_field, score_bounds = "predicted_potential", (1, 200)

        def predict(self, players):
            return [
                {"player_id": r["player_id"], "predicted_potential": 100 + r["player_id"] / 2}
                for r in players
            ]

    constraints = {"age_max": 19, "value_max_eur": 8_000_000, "position": "MC"}
    backend = FakeBackend(
        [
            call("search_players", constraints),
            call("predict_player_potential", {"search_id": "search-1"}),
            final((98, 97, 96, 94, 93)),
        ]
    )
    result = ScoutingAgent(backend, ScoutingTools(store, Regressor())).run("five wonderkids")
    assert result.error is None
    assert result.prediction_task == "pa_regression"
    assert result.recommendations[0]["player_id"] == 98
    assert result.recommendations[0]["predicted_potential"] == 149
    assert "wonderkid_probability" not in json.dumps(result.to_dict())
    assert "regression estimate" in render_shortlist(result)
    messages = json.dumps(backend.messages)
    for forbidden in ("potential_ability", "actual_pa", "actual_wonderkid", "consistency"):
        assert forbidden not in messages
    schemas = tool_schemas(regression=True)
    assert "predict_player_potential" in [t["function"]["name"] for t in schemas]
    assert "predict_wonderkid_probability" not in [t["function"]["name"] for t in schemas]


def test_regression_cannot_silently_change_reference_or_threshold(prepared_regression):
    settings, _ = prepared_regression
    changed = replace(settings, training=replace(settings.training, random_seed=99))
    with pytest.raises(ValueError, match="seed or target threshold"):
        fit_regression(changed, emit=lambda s: None)


def test_changed_evaluation_cap_reuses_fit_but_recalculates_metrics(prepared_regression):
    settings, captured = prepared_regression
    first = fit_regression(settings, emit=lambda s: None)
    larger = replace(settings, training=replace(settings.training, max_evaluation_rows=20))
    second = fit_regression(larger, emit=lambda s: None)
    assert captured["regression_fits"] == 1
    assert first["evaluation_id_hash"] != second["evaluation_id_hash"]
    assert second["model_metrics"]["eligible_count"] == 20
