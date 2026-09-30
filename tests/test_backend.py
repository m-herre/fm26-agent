from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from fm26_agent import tabpfn_backend
from fm26_agent.features import FeatureSchema
from fm26_agent.prediction import TabPFNPredictor


def test_auto_prefers_local_only_with_a_gpu(monkeypatch):
    tabpfn_backend.resolve.cache_clear()
    monkeypatch.setattr(tabpfn_backend, "local_available", lambda: True)
    monkeypatch.setattr(tabpfn_backend, "device", lambda: "mps")
    assert tabpfn_backend.resolve("auto") == "local"
    tabpfn_backend.resolve.cache_clear()
    monkeypatch.setattr(tabpfn_backend, "device", lambda: "cpu")
    assert tabpfn_backend.resolve("auto") == "hosted"
    tabpfn_backend.resolve.cache_clear()
    monkeypatch.setattr(tabpfn_backend, "local_available", lambda: False)
    assert tabpfn_backend.resolve("auto") == "hosted"
    with pytest.raises(ValueError, match="not installed"):
        tabpfn_backend.resolve("local")
    with pytest.raises(ValueError, match="must be one of"):
        tabpfn_backend.resolve("gpu")
    tabpfn_backend.resolve.cache_clear()


def test_local_needs_no_key(monkeypatch, tmp_path):
    from fm26_agent.config import load_settings

    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    monkeypatch.setattr(tabpfn_backend, "local_available", lambda: True)
    tabpfn_backend.resolve.cache_clear()
    monkeypatch.setenv("FM26_TABPFN_BACKEND", "local")
    settings = load_settings(tmp_path / "config.toml")
    assert settings.tabpfn_backend == "local" and settings.tabpfn_ready
    monkeypatch.setenv("FM26_TABPFN_BACKEND", "hosted")
    assert not settings.tabpfn_ready
    tabpfn_backend.resolve.cache_clear()


def test_config_file_chooses_the_backend(tmp_path, monkeypatch):
    from fm26_agent.config import load_settings

    monkeypatch.delenv("FM26_TABPFN_BACKEND")
    (tmp_path / "config.toml").write_text('[tabpfn]\nbackend = "hosted"\n')
    assert load_settings(tmp_path / "config.toml").tabpfn_backend == "hosted"


def test_weights_stay_in_the_project(tmp_path, monkeypatch):
    monkeypatch.delenv("TABPFN_MODEL_CACHE_DIR", raising=False)
    tabpfn_backend.use_project_weights(tmp_path)
    import os

    assert os.environ["TABPFN_MODEL_CACHE_DIR"] == str(tmp_path / "data" / "tabpfn-weights")


def test_a_local_fit_is_saved_beside_model_json_and_checked(records, tmp_path, monkeypatch):
    players = [row.visible for row in records[:3]]
    schema = FeatureSchema.fit(players)
    saved = {}
    monkeypatch.setattr(
        tabpfn_backend,
        "save_fitted",
        lambda model, backend, path: saved.setdefault("name", path.with_suffix(".tabpfn_fit").name),
    )
    path = tmp_path / "model.json"
    TabPFNPredictor(SimpleNamespace(), schema, backend="local").save(path, "p")
    payload = json.loads(path.read_text())
    assert payload["backend"] == "local" and payload["model"] == "model.tabpfn_fit"
    assert "missing" in TabPFNPredictor.check_reference(path, schema, "p", "local")
    (tmp_path / "model.tabpfn_fit").write_bytes(b"fit")
    assert TabPFNPredictor.check_reference(path, schema, "p", "local") is None
    # switching backend means the saved fit can't be reused
    assert "locally" in TabPFNPredictor.check_reference(path, schema, "p", "hosted")


def test_old_model_files_count_as_hosted(records, tmp_path):
    players = [row.visible for row in records[:3]]
    schema = FeatureSchema.fit(players)
    model = SimpleNamespace(save_model=lambda: "fit-id")
    path = tmp_path / "model.json"
    TabPFNPredictor(model, schema).save(path, "p")
    payload = json.loads(path.read_text())
    payload.pop("backend")
    path.write_text(json.dumps(payload))
    assert TabPFNPredictor.check_reference(path, schema, "p", "hosted") is None
    assert "Prior Labs" in TabPFNPredictor.check_reference(path, schema, "p", "local")
    assert np.isfinite(1.0)
