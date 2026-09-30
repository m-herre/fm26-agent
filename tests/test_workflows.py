from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from fm26_agent.config import DataSettings, LLMSettings, Settings, TrainingSettings, load_settings
from fm26_agent.extract import ExtractionResult
from fm26_agent.features import FeatureSchema
from fm26_agent.prediction import HostedPredictor
from fm26_agent.prepare import prepare, setup_problem
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
        training=TrainingSettings(max_train_rows=50),
        llm=LLMSettings(),
        eur_per_internal_unit=1.2,
        config_path=tmp_path / "config.toml",
    )


def install_extraction(monkeypatch, records):
    monkeypatch.setattr(
        "fm26_agent.prepare.read_save",
        lambda *args: ExtractionResult(records, date(2076, 7, 1), "FM26", "fixture", []),
    )


def install_regressor(monkeypatch, captured):
    import tabpfn_client

    class Model:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs
            captured["fits"] = captured.get("fits", 0)

        def fit(self, matrix, target):
            captured["matrix"], captured["target"] = matrix, target
            captured["fits"] += 1

        def predict(self, matrix, output_type="mean"):
            return np.full(len(matrix), 150.0)

        def save_model(self):
            return {"fixture": True}

        @classmethod
        def load_model(cls, reference):
            return cls()

    monkeypatch.setattr(tabpfn_client, "TabPFNRegressor", Model)


def test_prepare_needs_a_key_and_leaves_nothing_behind(settings, monkeypatch, records):
    install_extraction(monkeypatch, records)
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    with pytest.raises(ValueError, match="TabPFN key"):
        prepare(settings, settings.config_path, emit=lambda message: None)
    assert not settings.data.visible_database.exists()


def test_prepare_uploads_features_and_potential_only_then_reuses_the_fit(
    settings, monkeypatch, records
):
    install_extraction(monkeypatch, records)
    monkeypatch.setenv("TABPFN_TOKEN", "fixture-token")
    captured: dict = {}
    install_regressor(monkeypatch, captured)
    messages: list[str] = []
    metadata = prepare(settings, settings.config_path, emit=messages.append)
    assert captured["matrix"].shape == (50, 59)
    assert_safe_features(list(captured["matrix"].columns))
    assert not any(row.visible["name"] in captured["matrix"].to_json() for row in records)
    assert str(captured["matrix"]["club_category"].dtype) == "category"
    assert set(captured["target"]) == {120.0, 170.0}  # exact potential is the target only
    assert captured["kwargs"] == {
        "model_path": "v3.5_default",
        "fit_mode": "fit_with_cache",
        "text_handling": "advanced",
        "random_state": 42,
    }
    store = VisibleStore(settings.data.visible_database)
    assert store.metadata()["model_ready"] and store.metadata()["source"] == "config.toml"
    assert (
        PrivateStore(settings.data.private_database).preparation_id() == metadata["preparation_id"]
    )
    assert any("minute or two" in message for message in messages)
    assert setup_problem(settings, settings.config_path) is None
    HostedPredictor.load(
        settings.data.model_reference, settings.data.feature_schema, metadata["preparation_id"]
    )
    with pytest.raises(ValueError, match="another save"):
        HostedPredictor.load(settings.data.model_reference, settings.data.feature_schema, "wrong")

    # Running again reads nothing and fits nothing, even without a key.
    monkeypatch.delenv("TABPFN_TOKEN")
    monkeypatch.setattr(
        "fm26_agent.prepare.read_save", lambda *args: pytest.fail("must not reread the save")
    )
    again = prepare(settings, settings.config_path, emit=messages.append)
    assert again["preparation_id"] == metadata["preparation_id"]
    assert captured["fits"] == 1 and "This save is already set up." in messages


def test_failed_fit_is_not_mistaken_for_a_finished_setup(settings, monkeypatch, records):
    install_extraction(monkeypatch, records)
    monkeypatch.setenv("TABPFN_TOKEN", "fixture-token")

    def fail(*args, **kwargs):
        raise ValueError("Fixture quota failure")

    monkeypatch.setattr(HostedPredictor, "fit", fail)
    with pytest.raises(ValueError, match="quota"):
        prepare(settings, settings.config_path, emit=lambda message: None)
    assert not VisibleStore(settings.data.visible_database).metadata()["model_ready"]
    assert setup_problem(settings) == "setup did not finish"
    monkeypatch.setattr(HostedPredictor, "fit", classmethod(_working_fit))
    prepare(settings, settings.config_path, emit=lambda message: None)
    assert setup_problem(settings) is None


def _working_fit(cls, players, targets, schema, random_seed=42, backend="hosted"):
    from types import SimpleNamespace

    model = SimpleNamespace(
        predict=lambda matrix, output_type="mean": np.full(len(matrix), 150.0),
        save_model=lambda: {"fixture": True},
    )
    return cls(model, schema)


def test_changed_save_file_needs_a_new_setup(settings, monkeypatch, records):
    install_extraction(monkeypatch, records)
    monkeypatch.setenv("TABPFN_TOKEN", "fixture-token")
    install_regressor(monkeypatch, {})
    prepare(settings, settings.config_path, emit=lambda message: None)
    assert setup_problem(settings, settings.config_path) is None
    settings.config_path.write_text("the player saved again, so the file is different")
    assert setup_problem(settings, settings.config_path) == "the save file has changed since setup"


def test_refit_redoes_the_setup(settings, monkeypatch, records):
    install_extraction(monkeypatch, records)
    monkeypatch.setenv("TABPFN_TOKEN", "fixture-token")
    captured: dict = {}
    install_regressor(monkeypatch, captured)
    first = prepare(settings, settings.config_path, emit=lambda message: None)
    second = prepare(settings, settings.config_path, refit=True, emit=lambda message: None)
    assert captured["fits"] == 2
    assert first["preparation_id"] != second["preparation_id"]


def test_setup_problem_reports_each_missing_piece(settings, monkeypatch, records):
    assert setup_problem(settings) == "no save has been set up yet"
    install_extraction(monkeypatch, records)
    monkeypatch.setenv("TABPFN_TOKEN", "fixture-token")
    install_regressor(monkeypatch, {})
    prepare(settings, settings.config_path, emit=lambda message: None)
    settings.data.model_reference.unlink()
    assert setup_problem(settings) == "no saved model"


@pytest.mark.parametrize("money", ["0", "-1", "inf", "nan"])
def test_invalid_currency_rate_rejected(tmp_path, money):
    path = tmp_path / "config.toml"
    path.write_text(f"[money]\neur_per_internal_unit={money}\n")
    with pytest.raises(ValueError, match="greater than zero"):
        load_settings(path)


def test_no_config_file_is_needed(tmp_path):
    settings = load_settings(tmp_path / "config.toml")
    assert settings.eur_per_internal_unit == 1.0 and not settings.currency_calibrated
    assert settings.data.model_reference == (tmp_path / "data" / "model.json").resolve()


def test_doctor_reports_what_is_missing_without_leaking_keys(settings, monkeypatch, capsys):
    from fm26_agent.cli import doctor

    for key in ("DEEPSEEK_API_KEY", "LLM_API_KEY", "TABPFN_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    doctor(settings)
    output = capsys.readouterr().out
    assert "DeepSeek key: missing" in output and "TabPFN key: missing" in output
    assert "Setup: not ready (no save has been set up yet)" in output
    assert "Only Football Manager 26 saves" in output
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-llm-secret")
    monkeypatch.setenv("TABPFN_TOKEN", "fixture-tabpfn-secret")
    doctor(settings)
    output = capsys.readouterr().out
    assert "DeepSeek key: found" in output
    assert "fixture-llm-secret" not in output and "fixture-tabpfn-secret" not in output


def test_schema_tampering_is_rejected(records):
    schema = FeatureSchema.fit([row.visible for row in records])
    unsafe = replace(schema, columns=schema.columns + ["ability_potential"])
    with pytest.raises(ValueError, match="allowlist"):
        unsafe.transform([records[0].visible])
