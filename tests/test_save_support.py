from __future__ import annotations

import warnings
from datetime import date
from types import SimpleNamespace

import fmsave
import pytest

from fm26_agent import cli
from fm26_agent.extract import (
    UnsupportedSaveError,
    inspect_save,
    read_save,
    supported_builds,
    supported_saves_message,
)


class Career:
    info = SimpleNamespace(game_date=date(2076, 7, 1), game="FM26", build="26.1.0+1")

    def __init__(self, warning=None):
        self.warning = warning

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def players(self):
        if self.warning:
            warnings.warn("no layout tables for this build", self.warning, stacklevel=2)
        return [SimpleNamespace()]


@pytest.fixture
def one_player(monkeypatch, records):
    monkeypatch.setattr("fm26_agent.extract.record_to_player", lambda *args: records[0])


def test_supported_message_names_the_limit():
    message = supported_saves_message()
    assert "Football Manager 26" in message and "FM25" in message
    assert all(build in message for build in supported_builds())


def test_other_game_versions_get_a_clear_error(monkeypatch):
    def refuse(*args, **kwargs):
        raise fmsave.UnsupportedGameError("career.fm is an FM25 save")

    monkeypatch.setattr(fmsave, "open", refuse)
    with pytest.raises(UnsupportedSaveError, match="FM25 save") as caught:
        read_save("career.fm", 1.0)
    assert "Only Football Manager 26 saves" in str(caught.value)
    with pytest.raises(UnsupportedSaveError, match="Only Football Manager 26 saves"):
        inspect_save("career.fm")


def test_non_save_files_get_a_clear_error(monkeypatch, tmp_path):
    def refuse(*args, **kwargs):
        raise fmsave.NotAFmSaveError("bad header")

    monkeypatch.setattr(fmsave, "open", refuse)
    with pytest.raises(UnsupportedSaveError, match="is not a Football Manager save file"):
        read_save(tmp_path / "notes.txt", 1.0)


def test_unknown_fm26_build_stops_unless_explicitly_allowed(monkeypatch, one_player):
    monkeypatch.setattr(fmsave, "open", lambda *a, **k: Career(fmsave.UnknownBuildWarning))
    with pytest.raises(UnsupportedSaveError, match="--allow-reader-warnings") as caught:
        read_save("old.fm", 1.0)
    assert "no layout tables for this build" in str(caught.value)
    assert "Only Football Manager 26 saves" in str(caught.value)
    assert read_save("old.fm", 1.0, allow_reader_warnings=True).build == "26.1.0+1"


def test_inspect_save_reports_support_without_reading_players(monkeypatch):
    monkeypatch.setattr(fmsave, "open", lambda *a, **k: Career())
    report = inspect_save("ok.fm")
    assert report.supported and report.build == "26.1.0+1" and report.game == "FM26"

    class WarnsOnOpen(Career):
        def __enter__(self):
            warnings.warn("unknown build", fmsave.UnknownBuildWarning, stacklevel=2)
            return self

    monkeypatch.setattr(fmsave, "open", lambda *a, **k: WarnsOnOpen())
    report = inspect_save("old.fm")
    assert not report.supported and report.warnings == ["unknown build"]


def test_chat_defaults_to_regression_and_classifier_is_opt_in(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "load_settings", lambda path: object())
    monkeypatch.setattr(cli, "chat", lambda *args: seen.append(args) or 0)
    assert cli.main(["chat", "--query", "q"]) == 0
    assert cli.main(["chat", "--query", "q", "--classifier"]) == 0
    assert cli.main(["chat", "--query", "q", "--agent-only"]) == 0
    regression = [
        call[3 + 1] for call in seen
    ]  # (settings, query, agent_only, held_out, regression, ...)
    assert regression == [True, False, True]
    assert [call[2] for call in seen] == [False, False, True]


def test_chat_model_flags_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        cli._parser().parse_args(["chat", "--classifier", "--agent-only"])
