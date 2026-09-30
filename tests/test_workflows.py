from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from fm26_agent.backend import ChatReply
from fm26_agent.cli import main
from fm26_agent.config import DataSettings, LLMSettings, Settings, TrainingSettings, load_settings
from fm26_agent.evaluate import BENCHMARKS, evaluate
from fm26_agent.extract import ExtractionResult
from fm26_agent.features import FeatureSchema
from fm26_agent.prediction import HostedPredictor
from fm26_agent.prepare import prepare
from fm26_agent.private_db import PrivateStore
from fm26_agent.schema import assert_safe_features
from fm26_agent.visible_db import VisibleStore


@pytest.fixture
def settings(tmp_path):
    (tmp_path / "config.toml").write_text("fixture save for mocked extraction")
    return Settings(
        data=DataSettings(
            tmp_path / "players.sqlite3",
            tmp_path / "private" / "labels.sqlite3",
            tmp_path / "model.json",
            tmp_path / "schema.json",
            tmp_path / "runs",
        ),
        training=TrainingSettings(max_train_rows=50, max_evaluation_rows=10),
        llm=LLMSettings(),
        eur_per_internal_unit=1.2,
        config_path=tmp_path / "config.toml",
    )


def install_extraction(monkeypatch, records):
    monkeypatch.setattr(
        "fm26_agent.prepare.read_save",
        lambda *args: ExtractionResult(records, date(2076, 7, 1), "FM26", "fixture", []),
    )


def test_prepare_extract_only_and_missing_token(settings, monkeypatch, records):
    install_extraction(monkeypatch, records)
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    with pytest.raises(ValueError, match="TABPFN_TOKEN"):
        prepare(settings, settings.config_path, emit=lambda message: None)
    assert not settings.data.visible_database.exists()
    report = prepare(settings, settings.config_path, extract_only=True, emit=lambda message: None)
    store = VisibleStore(settings.data.visible_database)
    assert report["player_count"] == 100
    assert store.summary()["player_counts"] == {"train": 50, "test": 50}
    assert report["sampling"]["reference_rows"] == 50
    assert not store.metadata()["model_ready"]
    assert (
        PrivateStore(settings.data.private_database).preparation_id()
        == store.metadata()["preparation_id"]
    )


def test_prepare_hosted_upload_excludes_identity_and_raw_labels(settings, monkeypatch, records):
    import tabpfn_client

    install_extraction(monkeypatch, records)
    monkeypatch.setenv("TABPFN_TOKEN", "fixture-token")
    captured = {}

    class Model:
        classes_ = np.array([0, 1])

        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def fit(self, matrix, target):
            captured["matrix"] = matrix
            captured["target"] = target

        def predict_proba(self, matrix):
            return np.tile([0.2, 0.8], (len(matrix), 1))

        def save_model(self):
            return {"fixture": True}

        @classmethod
        def load_model(cls, reference):
            return cls()

    monkeypatch.setattr(tabpfn_client, "TabPFNClassifier", Model)
    monkeypatch.setattr(
        tabpfn_client, "estimate_cost", lambda *args, **kwargs: SimpleNamespace(estimated_cost=12)
    )
    report = prepare(settings, settings.config_path, emit=lambda message: None)
    assert captured["matrix"].shape == (50, 59)
    assert report["model_features"]["groups"] == {"numeric": 53, "categorical": 5, "text": 1}
    assert VisibleStore(settings.data.visible_database).metadata()["feature_schema_version"] == 3
    assert set(captured["target"]) == {0, 1}
    assert_safe_features(list(captured["matrix"].columns))
    assert not any(row.visible["name"] in captured["matrix"].to_json() for row in records)
    assert captured["kwargs"]["random_state"] == 42
    assert captured["kwargs"]["model_path"] == "v3.5_default"
    assert captured["kwargs"]["fit_mode"] == "fit_with_cache"
    assert "categorical_features_indices" not in captured["kwargs"]
    assert str(captured["matrix"]["club_category"].dtype) == "category"
    assert captured["matrix"]["club_category"].iloc[0] == "Fixture FC"
    assert report["evaluation_rows"] == 10
    assert VisibleStore(settings.data.visible_database).metadata()["model_ready"]
    predictor = HostedPredictor.load(
        settings.data.model_reference, settings.data.feature_schema, report["preparation_id"]
    )
    with pytest.raises(ValueError, match="another dataset"):
        HostedPredictor.load(settings.data.model_reference, settings.data.feature_schema, "wrong")
    assert len(predictor.predict([records[0].visible])) == 1
    monkeypatch.delenv("TABPFN_TOKEN")
    monkeypatch.setattr(
        "fm26_agent.prepare.read_save",
        lambda *args: pytest.fail("A reusable preparation must not reread/refit"),
    )
    again = prepare(settings, settings.config_path, emit=lambda message: None)
    assert again["preparation_id"] == report["preparation_id"]


def test_failed_fit_leaves_model_unavailable_and_extraction_report(settings, monkeypatch, records):
    install_extraction(monkeypatch, records)
    monkeypatch.setenv("TABPFN_TOKEN", "fixture-token")

    def fail(*args):
        raise ValueError("Fixture quota failure")

    monkeypatch.setattr(HostedPredictor, "estimate_cost", fail)
    with pytest.raises(ValueError, match="quota"):
        prepare(settings, settings.config_path, emit=lambda message: None)
    assert not VisibleStore(settings.data.visible_database).metadata()["model_ready"]
    assert (
        json.loads((settings.data.runs_directory / "preparation.json").read_text())["player_count"]
        == 100
    )


def test_reference_preview_does_not_replace_databases_or_models(settings, monkeypatch, records):
    install_extraction(monkeypatch, records)
    prepare(settings, settings.config_path, extract_only=True, emit=lambda message: None)
    before = settings.data.visible_database.read_bytes()
    labels_before = settings.data.private_database.read_bytes()
    settings.data.model_reference.write_text("original model sentinel")
    report = prepare(settings, settings.config_path, preview=True, emit=lambda message: None)
    assert report["reference_rows"] == 50
    assert settings.data.visible_database.read_bytes() == before
    assert settings.data.private_database.read_bytes() == labels_before
    assert settings.data.model_reference.read_text() == "original model sentinel"
    assert (settings.data.runs_directory / "reference-preview.json").exists()


def test_metric_failure_retains_fixed_fit_and_retries_without_refitting(
    settings, monkeypatch, records
):
    import tabpfn_client

    install_extraction(monkeypatch, records)
    monkeypatch.setenv("TABPFN_TOKEN", "fixture")
    fit_calls = []
    failing = [True]

    class Model:
        classes_ = np.array([0, 1])

        def predict_proba(self, matrix):
            if failing[0]:
                raise ValueError("provider body with secret-that-must-not-be-logged")
            return np.tile([0.2, 0.8], (len(matrix), 1))

        def save_model(self):
            return {"fixture": True}

        @classmethod
        def load_model(cls, reference):
            return cls()

    def fit(players, targets, schema, seed):
        fit_calls.append(len(players))
        return HostedPredictor(Model(), schema)

    monkeypatch.setattr(HostedPredictor, "fit", fit)
    monkeypatch.setattr(HostedPredictor, "estimate_cost", lambda *args: {"estimated_cost": 1})
    monkeypatch.setattr(tabpfn_client, "TabPFNClassifier", Model)
    first = prepare(settings, settings.config_path, emit=lambda message: None)
    assert "model_metrics_error" in first
    assert "secret-that" not in json.dumps(first)
    assert VisibleStore(settings.data.visible_database).metadata()["model_ready"] is True
    failing[0] = False
    second = prepare(settings, settings.config_path, emit=lambda message: None)
    assert "model_metrics" in second and "model_metrics_error" not in second
    assert first["preparation_id"] == second["preparation_id"]
    assert fit_calls == [50]


class BenchmarkBackend:
    def complete(self, messages, tools):
        query = messages[1]["content"]
        position = next(position for _, label, position in BENCHMARKS if label in query)
        expected = {"age_max": 19, "value_max_eur": 8_000_000, "position": position}
        tool_names = {tool["function"]["name"] for tool in tools}
        if len(messages) == 2:
            name, arguments = "search_players", expected
        else:
            last = json.loads(messages[-1]["content"])
            ids = (
                last.get("player_ids", [])
                if isinstance(last, dict)
                else [row["player_id"] for row in last]
            )
            if (
                isinstance(last, dict)
                and "search_id" in last
                and not last.get("complete")
                and "predict_wonderkid_probability" in tool_names
            ):
                name, arguments = "predict_wonderkid_probability", {"search_id": last["search_id"]}
            else:
                return ChatReply(
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "constraints": expected,
                                "requested_count": 5,
                                "recommendations": [
                                    {"player_id": player_id, "explanation": "Fixture evidence"}
                                    for player_id in ids[:5]
                                ],
                                "note": "",
                            }
                        ),
                    }
                )
        return ChatReply(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "fixture-call",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                ],
            }
        )


def test_fixed_benchmark_records_both_modes_and_private_metrics(
    settings, monkeypatch, records, fake_predictor
):
    install_extraction(monkeypatch, records)
    prepare(settings, settings.config_path, extract_only=True, emit=lambda message: None)
    report = evaluate(
        settings,
        BenchmarkBackend(),
        VisibleStore(settings.data.visible_database),
        fake_predictor,
        emit=lambda message: None,
    )
    assert report["successful_runs"] == {"agent_only": 5, "agent_tabpfn": 5}
    assert len(report["benchmarks"]) == 5
    assert "metrics" in report["benchmarks"][0]["runs"]["agent_only"]
    assert report["benchmarks"][1]["tabpfn_only"]["scored_count"] == 0
    assert len(list(settings.data.runs_directory.glob("benchmark-*.json"))) == 1


@pytest.mark.parametrize("money", ["0", "-1", "inf", "nan"])
def test_invalid_currency_rate_rejected(tmp_path, money):
    path = tmp_path / "config.toml"
    path.write_text(f"[money]\neur_per_internal_unit={money}\n")
    with pytest.raises(ValueError, match="greater than zero"):
        load_settings(path)


def test_cli_missing_configuration_is_actionable(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "missing.toml"), "chat"]) == 1
    assert "Copy config.example.toml" in capsys.readouterr().err


def test_doctor_fails_for_missing_credentials_not_false_success(settings, monkeypatch, capsys):
    from fm26_agent.cli import doctor

    monkeypatch.setattr("fm26_agent.cli.load_settings", lambda path: settings)
    for key in ("DEEPSEEK_API_KEY", "LLM_API_KEY", "TABPFN_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    assert doctor(str(settings.config_path), None) == 1
    assert "LLM API key: missing" in capsys.readouterr().out


def test_doctor_explains_legacy_model_migration(settings, store, monkeypatch, capsys):
    from fm26_agent.cli import doctor

    settings = replace(settings, data=replace(settings.data, visible_database=store.path))
    monkeypatch.setattr("fm26_agent.cli.load_settings", lambda path: settings)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-llm-secret")
    monkeypatch.setenv("TABPFN_TOKEN", "fixture-tabpfn-secret")
    assert doctor(str(settings.config_path), None) == 1
    output = capsys.readouterr().out
    assert "outdated feature preparation" in output
    assert "59-feature model" in output
    assert "fixture-llm-secret" not in output and "fixture-tabpfn-secret" not in output


def test_schema_tampering_is_rejected(records):
    schema = FeatureSchema.fit([row.visible for row in records])
    unsafe = replace(schema, columns=schema.columns + ["ability_potential"])
    with pytest.raises(ValueError, match="allowlist"):
        unsafe.transform([records[0].visible])
