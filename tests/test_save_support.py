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
        read_save("career.fm")
    assert "Only Football Manager 26 saves" in str(caught.value)
    with pytest.raises(UnsupportedSaveError, match="Only Football Manager 26 saves"):
        inspect_save("career.fm")


def test_non_save_files_get_a_clear_error(monkeypatch, tmp_path):
    def refuse(*args, **kwargs):
        raise fmsave.NotAFmSaveError("bad header")

    monkeypatch.setattr(fmsave, "open", refuse)
    with pytest.raises(UnsupportedSaveError, match="is not a Football Manager save file"):
        read_save(tmp_path / "notes.txt")


def test_unknown_fm26_build_stops_unless_explicitly_allowed(monkeypatch, one_player):
    monkeypatch.setattr(fmsave, "open", lambda *a, **k: Career(fmsave.UnknownBuildWarning))
    with pytest.raises(UnsupportedSaveError, match="--allow-reader-warnings") as caught:
        read_save("old.fm")
    assert "no layout tables for this build" in str(caught.value)
    assert "Only Football Manager 26 saves" in str(caught.value)
    assert read_save("old.fm", allow_reader_warnings=True).build == "26.1.0+1"


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


def test_running_without_a_command_starts_the_guided_app(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(cli, "load_settings", lambda path: SimpleNamespace(project_root=tmp_path))
    monkeypatch.setattr("fm26_agent.app.run", lambda settings, **kw: seen.append(kw) or 0)
    assert cli.main([]) == 0
    assert cli.main(["--query", "five young midfielders"]) == 0
    assert seen[0] == {
        "query": None,
        "save": None,
        "demo": False,
        "allow_reader_warnings": False,
    }
    assert seen[1]["query"] == "five young midfielders"


def test_research_options_are_gone():
    for old in ("--classifier", "--held-out", "--regression", "--known-values-only"):
        with pytest.raises(SystemExit):
            cli._parser().parse_args([old])
    for command in ("evaluate", "compare-models", "fit-regression", "compare-variants", "chat"):
        with pytest.raises(SystemExit):
            cli._parser().parse_args([command])


def _career_with_ability_flags(flags):
    class Flagged(Career):
        def players(self):
            return [SimpleNamespace() for _ in flags]

    return Flagged()


def _fake_players(monkeypatch, records, flags):
    from fm26_agent.extract import ExtractedPlayer

    iterator = iter(flags)
    monkeypatch.setattr(
        "fm26_agent.extract.record_to_player",
        lambda *args: ExtractedPlayer(records[0].visible, 120, next(iterator)),
    )
    monkeypatch.setattr(fmsave, "open", lambda *a, **k: _career_with_ability_flags(flags))


def test_potential_below_current_is_flagged_in_extraction():
    from fm26_agent.extract import record_to_player

    def player(current, potential):
        return SimpleNamespace(
            uid=1,
            attributes=SimpleNamespace(),
            transfer_value=None,
            contract=None,
            ability=SimpleNamespace(current=current, potential=potential),
        )

    assert record_to_player(player(100, 120), date(2076, 7, 1)).potential_below_current is False
    assert record_to_player(player(100, 100), date(2076, 7, 1)).potential_below_current is False
    assert record_to_player(player(130, 120), date(2076, 7, 1)).potential_below_current is True
    assert record_to_player(player(130, None), date(2076, 7, 1)).potential_below_current is None


def test_misread_potential_stops_extraction_unless_overridden(monkeypatch, records):
    _fake_players(monkeypatch, records, [True] * 3 + [False] * 97)  # 3% impossible
    with pytest.raises(RuntimeError, match="Potential ability looks misread: 3.0%"):
        read_save("bad.fm")
    _fake_players(monkeypatch, records, [True] * 3 + [False] * 97)
    assert read_save("bad.fm", allow_reader_warnings=True).pa_below_current_fraction == 0.03


def test_consistent_potential_passes_and_is_reported(monkeypatch, records):
    _fake_players(monkeypatch, records, [False] * 100)
    assert read_save("ok.fm").pa_below_current_fraction == 0.0
    _fake_players(monkeypatch, records, [None, None])
    assert read_save("unknown.fm").pa_below_current_fraction is None
