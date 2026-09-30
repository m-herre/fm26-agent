"""A portable sample of a prepared save, so the tool can run without Football Manager.

`export_sample` writes every player (attributes, positions, traits, value, wage, contract, this
season's stats and the real potential the model learns from) to one compressed CSV.
`read_sample` reads it back in the same shape a save produces, so the normal setup, model and
agent run on it unchanged.
"""

from __future__ import annotations

import json
import random
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .extract import SEASON_STATS_VERSION, ExtractedPlayer, ExtractionResult
from .private_db import PrivateStore
from .schema import VISIBLE_ATTRIBUTES
from .visible_db import SEASON_STAT_COLUMNS, VisibleStore

SAMPLE_FORMAT = 1
JSON_COLUMNS = ("natural_positions", "accomplished_positions", "traits")
PLAIN_COLUMNS = (
    "player_id",
    "name",
    "age",
    "club",
    "club_uid",
    "nation_id",
    "height_cm",
    "value_eur",
    "wage_eur",
    "contract_end",
    "contract_days_remaining",
    "on_loan",
    "preferred_foot",
)
STAT_PREFIX = "stat_"
FIRST_NAMES = (
    "Adrian Alex Andre Bruno Carlos Dani Diego Elias Emil Enzo Felix Gabriel Hugo Ivan Jonas Kai "
    "Leon Luca Marco Mateo Milan Nico Noah Oscar Pablo Rafael Sami Theo Tomas Viktor Yusuf Zane "
    "Amir Bastian Cedric Dario Erik Fabio Gustav Hassan Idris Jakub Karim Lars Mikel Nils Omar "
    "Pedro Rui Stefan Tariq Ulrich Vasco Wilmer Xavi Yannick Zoran"
).split()
LAST_NAMES = (
    "Albrecht Barros Castell Dvorak Eklund Ferreira Gallo Haugen Ibarra Jansen Kovac Lindqvist "
    "Moreau Novak Ortega Petrov Quinn Rossi Silva Torres Ulmer Vidal Weber Yilmaz Zielinski "
    "Abara Berg Costa Duarte Engel Fonseca Grau Hess Ionescu Jovanovic Keller Lopez Marin "
    "Nunez Olsen Pavlovic Reyes Santos Toth Uribe Varga Wolf Xuereb Young Zeman Adeyemi Brandt "
    "Correia Delgado Esposito Fischer Gomez Hartmann Iglesias Jung Krause Lund Mendes Nielsen"
).split()


def _meta_path(path: Path) -> Path:
    return path.with_name("sample.json")


def export_sample(
    visible: VisibleStore,
    private: PrivateStore,
    path: Path,
    *,
    keep_names: bool = False,
    seed: int = 1,
) -> int:
    """Write the prepared players to `path` (a .csv.gz). Names are replaced unless kept."""
    metadata = visible.metadata()
    labels = {row["player_id"]: row["potential_ability"] for row in private.rows()}
    players = visible.get_players(visible.all_ids())
    stats = visible.season_stats([row["player_id"] for row in players])
    rng = random.Random(seed)
    rows = []
    for new_id, player in enumerate(sorted(players, key=lambda row: row["player_id"]), 1_000_001):
        item = {column: player.get(column) for column in PLAIN_COLUMNS}
        item["player_id"] = new_id
        item["name"] = (
            player["name"] if keep_names else f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        )
        item["on_loan"] = None if player["on_loan"] is None else int(player["on_loan"])
        for column in JSON_COLUMNS:
            item[column] = json.dumps(player[column])
        for attribute in VISIBLE_ATTRIBUTES:
            item[attribute] = player.get(attribute)
        item["potential_ability"] = labels.get(player["player_id"])
        for column in SEASON_STAT_COLUMNS:
            item[STAT_PREFIX + column] = stats.get(player["player_id"], {}).get(column)
        rows.append(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
    _meta_path(path).write_text(
        json.dumps(
            {
                "format": SAMPLE_FORMAT,
                "game": metadata.get("game", "FM26"),
                "build": metadata.get("build", "unknown"),
                "save_date": metadata["save_date"],
                "players": len(rows),
                "names_replaced": not keep_names,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return len(rows)


def _clean(value: Any) -> Any:
    return None if pd.isna(value) else value


def read_sample(path: Path) -> ExtractionResult:
    """Read an exported sample as if it came from a save."""
    meta_path = _meta_path(path)
    if not meta_path.exists():
        raise ValueError(f"{path.name} needs its sample.json next to it")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("format") != SAMPLE_FORMAT:
        raise ValueError("This sample file was made by a different version of the tool")
    frame = pd.read_csv(path)
    players: list[ExtractedPlayer] = []
    season_stats: dict[int, dict[str, Any]] = {}
    for record in frame.to_dict("records"):
        record = {key: _clean(value) for key, value in record.items()}
        visible: dict[str, Any] = {"player_id": int(record["player_id"]), "split": "unassigned"}
        for column in PLAIN_COLUMNS[1:]:
            value = record[column]
            if column in ("age", "club_uid", "nation_id", "height_cm", "contract_days_remaining"):
                value = None if value is None else int(value)
            elif column == "on_loan":
                value = None if value is None else bool(value)
            elif column in ("value_eur", "wage_eur"):
                value = None if value is None else float(value)
            visible[column] = value
        for column in JSON_COLUMNS:
            visible[column] = json.loads(record[column])
        for attribute in VISIBLE_ATTRIBUTES:
            visible[attribute] = record[attribute]
        potential = record["potential_ability"]
        players.append(
            ExtractedPlayer(visible, None if potential is None else int(potential), None)
        )
        if record[STAT_PREFIX + "minutes"] is not None:
            season_stats[visible["player_id"]] = {
                column: record[STAT_PREFIX + column] for column in SEASON_STAT_COLUMNS
            }
            for column in ("appearances", "starts", "minutes", "goals", "assists"):
                season_stats[visible["player_id"]][column] = int(
                    season_stats[visible["player_id"]][column] or 0
                )
            for column in ("player_of_the_match", "clean_sheets"):
                season_stats[visible["player_id"]][column] = int(
                    season_stats[visible["player_id"]][column] or 0
                )
    return ExtractionResult(
        players=players,
        save_date=date.fromisoformat(meta["save_date"]),
        game=meta["game"],
        build=meta["build"],
        warnings=["This is sample data, not a live save."],
        season_stats=season_stats,
    )


__all__ = ["SEASON_STATS_VERSION", "export_sample", "read_sample"]
