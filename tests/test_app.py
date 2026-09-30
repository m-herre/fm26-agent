from __future__ import annotations

import os
import stat
from types import SimpleNamespace

import pytest

from fm26_agent import app
from fm26_agent.agent import AgentResult
from fm26_agent.app import Console, choose_save, ensure_keys, offer_calibration, progress_message
from fm26_agent.config import DataSettings, LLMSettings, Settings, TrainingSettings, load_settings
from fm26_agent.extract import UnsupportedSaveError
from fm26_agent.keys import ENV_FILE, load_env_file, save_key
from fm26_agent.validate import spotcheck_sample


class Script:
    """A scripted user: answers prompts in order and records everything said."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.said: list[str] = []
        self.prompts: list[str] = []

    def _next(self, prompt):
        self.prompts.append(prompt)
        return self.answers.pop(0)

    @property
    def console(self):
        return Console(ask=self._next, secret=self._next, say=self.said.append)

    @property
    def output(self):
        return "\n".join(self.said)


@pytest.fixture
def settings(tmp_path):
    (tmp_path / "config.toml").write_text(
        "[money]\neur_per_internal_unit = 1.0\ncalibrated = false\n"
    )
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


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("DEEPSEEK_API_KEY", "LLM_API_KEY", "TABPFN_TOKEN"):
        monkeypatch.delenv(key, raising=False)


# --- keys ---------------------------------------------------------------------------------


def test_save_key_writes_a_private_env_file_and_replaces_old_values(tmp_path, monkeypatch):
    save_key(tmp_path, "DEEPSEEK_API_KEY", "sk-one")
    save_key(tmp_path, "TABPFN_TOKEN", "tp-two")
    save_key(tmp_path, "DEEPSEEK_API_KEY", "sk-three")
    text = (tmp_path / ENV_FILE).read_text()
    assert text.count("DEEPSEEK_API_KEY=") == 1 and "sk-three" in text and "sk-one" not in text
    assert stat.S_IMODE((tmp_path / ENV_FILE).stat().st_mode) == 0o600
    assert os.environ["DEEPSEEK_API_KEY"] == "sk-three"


@pytest.mark.parametrize("value", ["", "   ", "two words"])
def test_save_key_rejects_unusable_values(tmp_path, value):
    with pytest.raises(ValueError):
        save_key(tmp_path, "TABPFN_TOKEN", value)
    with pytest.raises(ValueError, match="Unknown key"):
        save_key(tmp_path, "OTHER", "x")


def test_env_file_never_overrides_real_environment_variables(tmp_path, monkeypatch):
    (tmp_path / ENV_FILE).write_text(
        "# comment\nTABPFN_TOKEN='from-file'\nDEEPSEEK_API_KEY=file-ds\n\nbroken\n"
    )
    monkeypatch.setenv("TABPFN_TOKEN", "from-shell")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    load_env_file(tmp_path)
    assert (
        os.environ["TABPFN_TOKEN"] == "from-shell" and os.environ["DEEPSEEK_API_KEY"] == "file-ds"
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    load_env_file(tmp_path / "nowhere")  # a missing file is fine


def test_ensure_keys_asks_once_explains_and_remembers(settings):
    script = Script("ds-key", "tabpfn-key")
    ensure_keys(settings, script.console, interactive=True)
    assert settings.deepseek_api_key == "ds-key" and settings.tabpfn_token == "tabpfn-key"
    assert "platform.deepseek.com" in script.output and "platform.priorlabs.ai" in script.output
    assert all("hidden" in prompt for prompt in script.prompts)
    again = Script()
    ensure_keys(settings, again.console, interactive=True)  # nothing left to ask
    assert not again.prompts and not again.said
    assert "ds-key" in (settings.project_root / ENV_FILE).read_text()


def test_ensure_keys_retries_unusable_input_and_refuses_when_not_interactive(settings):
    script = Script("", "has spaces", "ds-key", "tabpfn-key")
    ensure_keys(settings, script.console, interactive=True)
    assert script.output.count("must be one piece of text") + script.output.count("A key") >= 2
    os.environ.pop("DEEPSEEK_API_KEY")
    with pytest.raises(ValueError, match="DeepSeek key is missing"):
        ensure_keys(settings, Script().console, interactive=False)


# --- choosing a save ----------------------------------------------------------------------


def fake_save(folder, name, size=10):
    path = folder / name
    path.write_bytes(b"x" * size)
    return path


def test_an_already_prepared_save_is_used_without_asking(settings, monkeypatch):
    ready = fake_save(settings.project_root, "career.fm")
    fake_save(settings.project_root, "other.fm")
    monkeypatch.setattr(
        app, "setup_problem", lambda s, source: None if source == ready.resolve() else "x"
    )
    script = Script()
    assert choose_save(settings, script.console, None) == ready.resolve()
    assert not script.prompts


def test_a_single_save_in_the_folder_is_picked_automatically(settings, monkeypatch):
    monkeypatch.setattr(app, "setup_problem", lambda s, source: "not set up")
    only = fake_save(settings.project_root, "career.fm")
    script = Script()
    assert choose_save(settings, script.console, None) == only.resolve()
    assert "career.fm" in script.output and not script.prompts


def test_several_saves_show_a_menu_and_survive_bad_input(settings, monkeypatch):
    monkeypatch.setattr(app, "setup_problem", lambda s, source: "not set up")
    old = fake_save(settings.project_root, "old.fm")
    new = fake_save(settings.project_root, "new.fm")
    os.utime(old, (1, 1))
    script = Script("7", "abc", "1")
    assert choose_save(settings, script.console, None) == new.resolve()  # newest is listed first
    assert script.output.count("Please type one of the numbers") == 2
    assert choose_save(settings, Script("").console, None) is None  # Enter cancels


def test_saves_from_the_game_folder_are_copied_in_after_choosing(settings, monkeypatch, tmp_path):
    monkeypatch.setattr(app, "setup_problem", lambda s, source: "not set up")
    game = tmp_path / "game"
    (game / "sub").mkdir(parents=True)
    fake_save(game / "sub", "Career.fm", size=2_000_000)
    monkeypatch.setattr(app, "default_save_folders", lambda: [game])
    project = tmp_path / "project"
    project.mkdir()
    settings = Settings(**{**settings.__dict__, "config_path": project / "config.toml"})
    script = Script("1")
    chosen = choose_save(settings, script.console, None)
    assert chosen == (project / "Career.fm").resolve() and chosen.exists()
    assert "Copying Career.fm (2 MB)" in script.output


def test_no_save_anywhere_asks_for_a_path_and_copies_it_in(settings, monkeypatch, tmp_path):
    monkeypatch.setattr(app, "default_save_folders", lambda: [])
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    dropped = fake_save(outside, "dragged.fm")
    project = tmp_path / "project"
    project.mkdir()
    settings = Settings(**{**settings.__dict__, "config_path": project / "config.toml"})
    script = Script(f"'{dropped}'")
    assert choose_save(settings, script.console, None) == (project / "dragged.fm").resolve()
    assert "Drag your .fm save" in script.output
    empty = tmp_path / "empty"
    empty.mkdir()
    fresh = Settings(**{**settings.__dict__, "config_path": empty / "config.toml"})
    assert choose_save(fresh, Script("").console, None) is None  # Enter cancels


def test_a_requested_save_must_exist_and_outside_files_are_copied(settings, tmp_path):
    with pytest.raises(ValueError, match="can't find the file"):
        choose_save(settings, Script().console, tmp_path / "missing.fm")
    inside = fake_save(settings.project_root, "here.fm")
    assert choose_save(settings, Script().console, inside) == inside


# --- progress wording ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("technical", "friendly"),
    [
        ("search_players: 500", "Searching your save..."),
        ("scoring 3,129 players", "Estimating potential for 3,129 players..."),
        ("get_player_details: 7 IDs", "Taking a closer look at the best candidates..."),
        ("shortlist ranking rejected; checking eligible leaders", "Double-checking the ranking..."),
        ("predict_player_potential: done", None),
        ("final JSON retry 1/2", None),
    ],
)
def test_progress_uses_plain_language(technical, friendly):
    assert progress_message(technical) == friendly


# --- the whole flow -----------------------------------------------------------------------


@pytest.fixture
def flow(settings, monkeypatch):
    """Everything outside the app is faked so the guided flow can be exercised end to end."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds")
    monkeypatch.setenv("TABPFN_TOKEN", "tp")
    save = fake_save(settings.project_root, "career.fm")
    state = SimpleNamespace(
        prepared=[], asked=[], problem="not set up", settings=settings, save=save
    )

    def fake_prepare(settings, source, **kwargs):
        state.prepared.append(source)
        kwargs["emit"]("Reading your save...")
        state.problem = None

    def fake_scout(settings, backend, store, predictor, query, trace=None, **kw):
        state.asked.append(query)
        trace("search_players: 500")
        trace("scoring 10 players")
        trace("scoring 10 players")
        result = AgentResult(query=query)
        result.recommendations = [
            {
                "player_id": 1,
                "name": "Ann Example",
                "age": 17,
                "club": "Fixture FC",
                "value_eur": 4_500_000.0,
                "value_known": True,
                "predicted_potential": 161.4,
                "explanation": "A composed 17-year-old passer.",
            }
        ]
        result.note = "I read central midfield as MC."
        return result, None

    monkeypatch.setattr(app, "setup_problem", lambda s, source=None: state.problem)
    monkeypatch.setattr(app, "prepare", fake_prepare)
    monkeypatch.setattr("fm26_agent.runtime.open_runtime", lambda s: ("backend", "store", "model"))
    monkeypatch.setattr("fm26_agent.runtime.scout", fake_scout)
    monkeypatch.setattr(
        app, "offer_calibration", lambda s, c: state.asked.append("calibration") or s
    )
    return state


def test_one_question_mode_sets_up_answers_and_exits_without_prompting(flow):
    script = Script()
    assert app.run(flow.settings, script.console, query="five young midfielders") == 0
    assert flow.prepared == [flow.save.resolve()] and flow.asked == ["five young midfielders"]
    out = script.output
    assert "1. Ann Example · 17 · Fixture FC · €4.5M" in out and "Potential ≈ 161" in out
    assert "I read central midfield as MC." in out and "off by about 9 points on average" in out
    assert out.count("Searching your save...") == 1
    assert out.count("Estimating potential for 10 players...") == 1  # repeats are not shown twice
    assert "Tip: prices are approximate" in out and not script.prompts
    assert "calibration" not in flow.asked  # never asked in one-question mode


def test_interactive_mode_asks_for_calibration_once_then_chats_until_quit(flow):
    script = Script("best goalkeeper prospects", "", "QUIT")
    assert app.run(flow.settings, script.console) == 0
    assert flow.asked == ["calibration", "best goalkeeper prospects"]
    assert script.prompts.count("\nWhat are you looking for? ") == 3  # an empty line just re-asks


def test_an_existing_setup_is_not_repeated_or_recalibrated(flow):
    flow.problem = None
    script = Script("a good striker", "exit")
    assert app.run(flow.settings, script.console) == 0
    assert flow.asked == ["a good striker"]


def test_missing_keys_are_requested_before_anything_else(flow, monkeypatch):
    monkeypatch.delenv("TABPFN_TOKEN")
    script = Script("new-tabpfn-key", "a good striker", "exit")
    assert app.run(flow.settings, script.console) == 0
    assert script.output.index("TabPFN key") < script.output.index("Reading your save")
    assert os.environ["TABPFN_TOKEN"] == "new-tabpfn-key"


def test_one_question_mode_without_keys_explains_instead_of_prompting(flow, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    script = Script()
    assert app.run(flow.settings, script.console, query="a good striker") == 1
    assert "DeepSeek key is missing" in script.output and not script.prompts


def test_unsupported_saves_get_a_clear_message_and_a_failing_exit(flow, monkeypatch):
    def refuse(*args, **kwargs):
        raise UnsupportedSaveError(
            "career.fm is an FM25 save.\nOnly Football Manager 26 saves work."
        )

    monkeypatch.setattr(app, "prepare", refuse)
    script = Script()
    assert app.run(flow.settings, script.console, query="a good striker") == 1
    assert "This save can't be used." in script.output and "FM25" in script.output


def test_unexpected_failures_are_friendly_and_leak_nothing(flow, monkeypatch):
    def explode(*args, **kwargs):
        raise KeyError("secret-token-value")

    monkeypatch.setattr(app, "prepare", explode)
    script = Script()
    assert app.run(flow.settings, script.console, query="a good striker") == 1
    assert "Something went wrong (KeyError)" in script.output
    assert "secret-token-value" not in script.output


def test_no_save_means_a_polite_exit(flow, monkeypatch):
    monkeypatch.setattr(app, "choose_save", lambda *a, **k: None)
    script = Script()
    assert app.run(flow.settings, script.console, query="a good striker") == 1
    assert "Nothing to do without a save file" in script.output


def test_an_answer_that_failed_shows_a_friendly_message(flow, monkeypatch):
    def failing(settings, backend, store, predictor, query, trace=None, **kw):
        result = AgentResult(query=query)
        result.error = "Final shortlist violates its constraints"
        return result, None

    monkeypatch.setattr("fm26_agent.runtime.scout", failing)
    script = Script()
    assert app.run(flow.settings, script.console, query="a good striker") == 1
    assert "couldn't put together a reliable answer" in script.output


# --- optional calibration -----------------------------------------------------------------


def test_calibration_matches_prices_to_the_game_and_saves_them(settings, store, private):
    sample = spotcheck_sample(store, private, count=3)
    answers = [f"{row['stored_value'] * 1.25 / 1e6:.3f}M" for row in sample]
    script = Script(*answers)
    updated = offer_calibration(settings, script.console)
    assert updated.eur_per_internal_unit == pytest.approx(1.25, rel=1e-3)
    assert updated.currency_calibrated and "Prices now match your game." in script.output
    assert load_settings(settings.config_path).currency_calibrated


def test_calibration_can_be_skipped_or_rejected(settings, store, private):
    script = Script("")
    assert offer_calibration(settings, script.console) is settings
    assert "fm26-agent calibrate" in script.output
    sample = spotcheck_sample(store, private, count=3)
    mismatched = [
        f"{row['stored_value'] * f / 1e6:.3f}M"
        for row, f in zip(sample, (1.0, 2.0, 0.5), strict=True)
    ]
    script = Script(*mismatched)
    assert offer_calibration(settings, script.console) is settings
    assert "don't agree" in script.output
    assert not load_settings(settings.config_path).currency_calibrated
    assert offer_calibration(settings, Script("lots", "", "").console) is settings
