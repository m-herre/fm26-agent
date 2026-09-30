from __future__ import annotations

import pytest

from fm26_agent.cli import calibrate_command, spotcheck
from fm26_agent.config import DataSettings, LLMSettings, Settings, TrainingSettings
from fm26_agent.private_db import PrivateStore
from fm26_agent.validate import apply_to_config, calibrate, parse_amount, spotcheck_sample


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("€4.5M", 4_500_000),
        ("4.5m", 4_500_000),
        ("£850K", 850_000),
        ("1,250,000", 1_250_000),
        ("  12 M", 12_000_000),
        ("2.1B", 2_100_000_000),
        ("750", 750),
    ],
)
def test_parse_amount(text, expected):
    assert parse_amount(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "free", "M4", "€"])
def test_parse_amount_rejects_garbage(text):
    with pytest.raises(ValueError, match="Cannot read an amount"):
        parse_amount(text)


@pytest.fixture
def private(tmp_path, records):
    store = PrivateStore(tmp_path / "private" / "labels.sqlite3")
    store.initialize(
        [
            {
                "player_id": row.visible["player_id"],
                "potential_ability": row.potential_ability,
                "wonderkid": int(row.potential_ability >= 160),
                "split": row.visible["split"],
            }
            for row in records
        ],
        "fixture",
    )
    return store


@pytest.fixture
def settings(tmp_path):
    (tmp_path / "config.toml").write_text(
        "[money]\neur_per_internal_unit = 1.0\ncalibrated = false\n", encoding="utf-8"
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


def test_spotcheck_spreads_players_across_the_pa_range(store, private):
    sample = spotcheck_sample(store, private, count=4)
    pas = [row["extracted_pa"] for row in sample]
    assert pas == sorted(pas) and len(sample) == 4
    assert min(pas) == 120 and max(pas) == 170  # fixture PA is 120 or 170
    assert all(row["stored_value"] >= 2_000_000 for row in sample)


def test_calibrate_recovers_the_multiplier(store):
    stored = {p["name"]: p["value_eur"] for p in store.get_players([1, 2, 4])}
    observed = [(name, value * 1.15) for name, value in stored.items()]
    result = calibrate(store, observed)
    assert result["eur_per_internal_unit"] == pytest.approx(1.15)
    assert result["consistent"] and result["spread"] < 1e-9


def test_calibrate_refuses_disagreeing_or_single_observations(store):
    a, b = (p["value_eur"] for p in store.get_players([1, 2]))
    assert not calibrate(store, [("Fixture 1", a * 1.0), ("Fixture 2", b * 1.5)])["consistent"]
    assert not calibrate(store, [("Fixture 1", a * 1.0)])["consistent"]


def test_calibrate_rejects_unknown_names_and_missing_values(store):
    with pytest.raises(ValueError, match="matches 0 players"):
        calibrate(store, [("Nobody", 1e6)])
    import sqlite3

    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE players SET value_eur=NULL WHERE player_id=1")
    with pytest.raises(ValueError, match="no stored value"):
        calibrate(store, [("Fixture 1", 1e6)])


def test_apply_to_config_edits_only_the_money_lines(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[money]\neur_per_internal_unit = 1.0\ncalibrated = false\n[x]\nk = 1\n")
    apply_to_config(path, 1.15)
    assert (
        path.read_text() == "[money]\neur_per_internal_unit = 1.15\ncalibrated = true\n[x]\nk = 1\n"
    )
    path.write_text("[money]\n")
    with pytest.raises(ValueError, match="needs one"):
        apply_to_config(path, 1.0)


def test_calibrate_command_applies_only_when_asked_and_consistent(store, settings, capsys):
    values = {p["name"]: p["value_eur"] for p in store.get_players([1, 2])}
    args = [f"{name}={value * 1.2:.0f}" for name, value in values.items()]
    assert calibrate_command(settings, args, apply=False) == 0
    assert "calibrated = false" in settings.config_path.read_text()
    assert calibrate_command(settings, args, apply=True) == 0
    text = settings.config_path.read_text()
    assert "eur_per_internal_unit = 1.2" in text and "calibrated = true" in text
    bad = [f"Fixture 1={values['Fixture 1']:.0f}", f"Fixture 2={values['Fixture 2'] * 2:.0f}"]
    assert calibrate_command(settings, bad, apply=True) == 1
    assert "Not applied" in capsys.readouterr().out


def test_spotcheck_command_prints_a_table(store, private, settings, capsys):
    assert spotcheck(settings, 3) == 0
    out = capsys.readouterr().out
    assert "Extracted PA" in out and "fm26-agent calibrate" in out
