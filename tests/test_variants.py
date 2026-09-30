from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from test_regression import prepared_regression as prepared_regression

from fm26_agent.variants import compare_variants, variant_parameters


def test_documented_selectors_and_thinking_cache_compatibility():
    plus = variant_parameters("plus", "classifier", 42)
    fast = variant_parameters("fast", "classifier", 42)
    thinking = variant_parameters("thinking", "classifier", 42)
    assert plus["model_path"] == "v3.5_default"
    assert fast["model_path"] == "v3.5-fast_default"
    assert plus["fit_mode"] == fast["fit_mode"] == "fit_with_cache"
    assert thinking["model_path"] == "v3.5_default"
    assert thinking["fit_mode"] == "fit_preprocessors"
    assert thinking["thinking_metric"] == "average_precision"
    assert thinking["thinking_effort"] == "medium"
    assert thinking["thinking_timeout_s"] == 180
    assert all(p["n_estimators"] == 8 and p["random_state"] == 42 for p in (plus, fast, thinking))
    assert variant_parameters("thinking", "regression", 42)["thinking_metric"] == "rmse"
    with pytest.raises(ValueError):
        variant_parameters("thinking", "classifier", 42, timeout=0)


def test_preview_is_local_without_token_and_never_fits(prepared_regression, monkeypatch):
    settings, captured = prepared_regression
    monkeypatch.delenv("TABPFN_TOKEN")
    result = compare_variants(settings, preview=True, emit=lambda s: None)
    assert result["preview"] and result["features"] == 59
    assert set(result["models"]) == {"classifier_plus", "classifier_fast", "classifier_thinking"}
    assert captured["classifier_fits"] == 1
    assert not (settings.data.model_reference.parent / "variants").exists()
    with pytest.raises(ValueError, match="Set TABPFN_TOKEN"):
        compare_variants(settings, emit=lambda s: None)


def test_variants_share_reference_target_and_pools_and_reuse_fits(prepared_regression, monkeypatch):
    import tabpfn_client

    settings, captured = prepared_regression
    baseline = settings.data.model_reference.read_bytes()
    original = tabpfn_client.TabPFNClassifier
    uploads, params = [], []

    class Spy(original):
        def __init__(self, **kwargs):
            params.append(kwargs)

        def fit(self, x, y):
            uploads.append((x.copy(), y.copy()))
            super().fit(x, y)

    monkeypatch.setattr(tabpfn_client, "TabPFNClassifier", Spy)
    first = compare_variants(settings, emit=lambda s: None)
    assert len(uploads) == 3 and captured["classifier_fits"] == 4
    assert settings.data.model_reference.read_bytes() == baseline
    assert first["automatic_model_selection"] is False
    for frame, target in uploads:
        assert frame.shape == (50, 59)
        assert frame.equals(uploads[0][0])
        assert target.tolist() == uploads[0][1].tolist()
        assert set(target) == {0, 1}
        assert not {"player_id", "name", "potential_ability"}.intersection(frame.columns)
    assert [p["model_path"] for p in params[:3]] == [
        "v3.5_default",
        "v3.5-fast_default",
        "v3.5_default",
    ]
    assert len(first["cases"]) == 6
    for case in first["cases"]:
        assert len(case["models"]) == 3
        assert all(
            model["scored_count"] == case["eligible_count"] and model["complete"]
            for model in case["models"].values()
        )
        assert all(model["prediction_wall_seconds"] >= 0 for model in case["models"].values())
    second = compare_variants(settings, emit=lambda s: None)
    assert captured["classifier_fits"] == 4
    assert all(p["fit_reused"] for p in second["variant_plans"].values())
    assert first["reference_id_hash"] == second["reference_id_hash"]
    assert first["model_reference_hashes"] == second["model_reference_hashes"]
    assert len(list(settings.data.runs_directory.glob("variant-comparison-*.json"))) == 2


def test_thinking_configuration_change_does_not_refit_plus_or_fast(prepared_regression):
    settings, captured = prepared_regression
    first = compare_variants(settings, emit=lambda s: None)
    second = compare_variants(settings, thinking_effort="high", emit=lambda s: None)
    assert captured["classifier_fits"] == 5
    for name in ("classifier_plus", "classifier_fast"):
        assert second["variant_plans"][name]["fit_reused"]
        assert (
            first["variant_plans"][name]["reference"] == second["variant_plans"][name]["reference"]
        )
    assert not second["variant_plans"]["classifier_thinking"]["fit_reused"]


def test_failed_variant_is_reported_without_secret_or_fallback(prepared_regression, monkeypatch):
    import tabpfn_client

    settings, _ = prepared_regression
    original = tabpfn_client.TabPFNClassifier

    class FailingFast(original):
        def __init__(self, **kwargs):
            if kwargs.get("model_path") == "v3.5-fast_default":
                raise RuntimeError("SECRET provider payload")

    monkeypatch.setattr(tabpfn_client, "TabPFNClassifier", FailingFast)
    result = compare_variants(settings, emit=lambda s: None)
    assert "SECRET" not in json.dumps(result)
    assert "error" in result["variant_plans"]["classifier_fast"]
    for case in result["cases"]:
        assert "error" in case["models"]["classifier_fast"]
        assert case["models"]["classifier_plus"]["complete"]
        assert case["models"]["classifier_thinking"]["complete"]


def test_regression_variants_are_separate_from_classification(prepared_regression):
    settings, captured = prepared_regression
    result = compare_variants(settings, task="both", emit=lambda s: None)
    assert len(result["variant_plans"]) == 6
    assert captured["classifier_fits"] == 4 and captured["regression_fits"] == 3
    assert set(captured["y"]) == {120, 170}
    assert captured["x"].shape == (50, 59)
    for case in result["cases"]:
        assert len(case["models"]) == 6
        assert all("error" not in model for model in case["models"].values())
        assert "mae" in case["models"]["regression_thinking"]["metrics"]


def test_failed_cost_quote_stops_before_any_fit(prepared_regression, monkeypatch):
    import tabpfn_client

    settings, captured = prepared_regression
    monkeypatch.setattr(
        tabpfn_client,
        "estimate_cost",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("SECRET")),
    )
    with pytest.raises(ValueError, match="No variant fits started") as error:
        compare_variants(settings, emit=lambda s: None)
    assert "SECRET" not in str(error.value)
    assert captured["classifier_fits"] == 1


def test_thinking_fit_quote_has_no_test_matrix(prepared_regression, monkeypatch):
    import tabpfn_client

    settings, captured = prepared_regression
    seen = []

    def estimate(train, test, **kwargs):
        operation = kwargs["operation"]
        if operation == "thinking_fit":
            assert test is None
            assert kwargs["thinking_effort"] == "medium"
        else:
            assert test.shape == (10, 59)
            assert "thinking_effort" not in kwargs
        seen.append(operation)
        return SimpleNamespace(estimated_cost=10000)

    monkeypatch.setattr(tabpfn_client, "estimate_cost", estimate)
    compare_variants(settings, preview=True, emit=lambda s: None)
    assert seen == ["cache_predict", "cache_predict", "thinking_fit", "thinking_predict"]
    assert captured["classifier_fits"] == 1
