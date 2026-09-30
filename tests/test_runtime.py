from __future__ import annotations

import json

import pytest
from test_agent import FakeBackend, call, final

from fm26_agent.config import DataSettings, LLMSettings, Settings, TrainingSettings
from fm26_agent.runtime import open_runtime, scout, write_report


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
        eur_per_internal_unit=1.2,
        config_path=tmp_path / "config.toml",
    )


def test_write_report_stays_in_the_runs_directory(settings):
    path = write_report(settings, "chat", {"ok": True})
    assert path.parent == settings.data.runs_directory
    assert path.name.startswith("chat-") and path.suffix == ".json"
    assert json.loads(path.read_text()) == {"ok": True}


def test_scout_runs_one_request_and_saves_its_report(settings, store, fake_predictor):
    backend = FakeBackend(
        [
            call("search_players", {"age_max": 19, "value_max_eur": 8_000_000, "position": "MC"}),
            call("predict_wonderkid_probability", {"search_id": "search-1"}),
            final((1, 2, 4, 5, 6)),
        ]
    )
    progress = []
    result, report_path = scout(
        settings,
        backend,
        store,
        fake_predictor,
        "five central midfielders under 20, max €8M",
        trace=progress.append,
    )
    assert result.error is None and len(result.recommendations) == 5
    assert progress  # front ends get progress through the callback, not print()
    report = json.loads(report_path.read_text())
    assert report["preparation_id"] == "fixture"
    assert report["query"].startswith("five central midfielders")
    assert report_path.parent == settings.data.runs_directory


def test_open_runtime_reports_missing_prerequisites(settings, monkeypatch):
    for key in ("DEEPSEEK_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        open_runtime(settings, False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture")
    with pytest.raises(ValueError, match="run prepare first"):
        open_runtime(settings, False)
