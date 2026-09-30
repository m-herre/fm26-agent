from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from fm26_agent.extract import record_to_player
from fm26_agent.schema import VISIBLE_ATTRIBUTES
from fm26_agent.visible_db import VisibleStore


@pytest.fixture
def records():
    players = []
    for index in range(1, 101):
        player = SimpleNamespace(
            uid=index,
            name=f"Fixture {index}",
            age=17 + index % 4,
            club_name="Fixture FC",
            club_uid=123,
            nation_id=45,
            height_cm=175,
            left_foot=8,
            right_foot=20,
            natural_positions=("MC",),
            accomplished_positions=("DM",),
            traits=(),
            on_loan=False,
            transfer_value=2_000_000 + index * 1000,
            contract=SimpleNamespace(wage=2000, end=date(2078, 6, 30)),
            attributes=SimpleNamespace(
                **{key: 8 + index % 10 for key in VISIBLE_ATTRIBUTES}, consistency=20
            ),
            ability=SimpleNamespace(current=199, potential=170 if index % 5 == 0 else 120),
        )
        players.append(record_to_player(player, date(2076, 7, 1), 1.2))
    return players


@pytest.fixture
def store(tmp_path, records):
    for player in records:
        player.visible["split"] = "train" if player.visible["player_id"] == 100 else "test"
    store = VisibleStore(tmp_path / "visible.sqlite3")
    store.initialize(
        [row.visible for row in records],
        {
            "save_date": "2076-07-01",
            "preparation_id": "fixture",
            "eur_per_internal_unit": 1.2,
            "model_ready": True,
        },
    )
    return store


class FakePredictor:
    def predict(self, players):
        return [{"player_id": row["player_id"], "wonderkid_probability": 0.9} for row in players]


@pytest.fixture
def fake_predictor():
    return FakePredictor()
