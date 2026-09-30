from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .schema import VISIBLE_ATTRIBUTES, normalize_position

BASE_COLUMNS = (
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
    "natural_positions",
    "accomplished_positions",
    "preferred_foot",
    "traits",
    "split",
)
ALL_COLUMNS = BASE_COLUMNS + VISIBLE_ATTRIBUTES

VALUE_NOTE = (
    "value_eur is null when the save stores no market value for a player (mostly free agents "
    "and players at clubs the game does not simulate; it saves 0 or a placeholder). "
    "FM calculates those values on the fly, so null means unknown, not worthless."
)


def scale_money(player: dict[str, Any], scale: float) -> dict[str, Any]:
    """Return a copy of the player with value and wage converted from internal units to euros.

    The columns are named *_eur for historical reasons but hold the save's internal units;
    they are the model inputs, so only display and filtering code may scale them.
    """
    if scale == 1.0:
        return player
    item = dict(player)
    for key in ("value_eur", "wage_eur"):
        if item.get(key) is not None:
            item[key] = item[key] * scale
    return item


def value_in_range(
    value: float | None,
    lower: float | None,
    upper: float | None,
    include_unknown: bool = True,
) -> bool:
    """Apply a budget filter; players without a stored value pass only if include_unknown."""
    if value is None:
        return (lower is None and upper is None) or include_unknown
    return (lower is None or value >= lower) and (upper is None or value <= upper)


class VisibleStore:
    """The only database dependency available to runtime scouting tools."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self, players: Sequence[dict[str, Any]], metadata: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        attribute_sql = ",\n".join(f'"{name}" REAL' for name in VISIBLE_ATTRIBUTES)
        with self._connect() as connection:
            connection.executescript(
                f"""
                BEGIN IMMEDIATE;
                DROP TABLE IF EXISTS players;
                DROP TABLE IF EXISTS metadata;
                CREATE TABLE players (
                    player_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    age INTEGER,
                    club TEXT,
                    club_uid INTEGER,
                    nation_id INTEGER,
                    height_cm INTEGER,
                    value_eur REAL,
                    wage_eur REAL,
                    contract_end TEXT,
                    contract_days_remaining INTEGER,
                    on_loan INTEGER,
                    natural_positions TEXT NOT NULL,
                    accomplished_positions TEXT NOT NULL,
                    preferred_foot TEXT NOT NULL,
                    traits TEXT NOT NULL,
                    split TEXT NOT NULL,
                    {attribute_sql}
                );
                CREATE INDEX players_split_age ON players(split, age);
                CREATE INDEX players_split_value ON players(split, value_eur);
                CREATE INDEX players_club ON players(club);
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                """
            )
            placeholders = ",".join("?" for _ in ALL_COLUMNS)
            quoted = ",".join(f'"{column}"' for column in ALL_COLUMNS)
            rows = []
            for player in players:
                item = dict(player)
                item["natural_positions"] = json.dumps(item["natural_positions"])
                item["accomplished_positions"] = json.dumps(item["accomplished_positions"])
                item["traits"] = json.dumps(item["traits"])
                item["on_loan"] = None if item["on_loan"] is None else int(item["on_loan"])
                rows.append(tuple(item.get(column) for column in ALL_COLUMNS))
            connection.executemany(f"INSERT INTO players ({quoted}) VALUES ({placeholders})", rows)
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                [(key, json.dumps(value)) for key, value in metadata.items()],
            )

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        for column in ("natural_positions", "accomplished_positions", "traits"):
            item[column] = json.loads(item[column])
        if item.get("on_loan") is not None:
            item["on_loan"] = bool(item["on_loan"])
        return item

    def metadata(self) -> dict[str, Any]:
        with self._connect() as connection:
            return {
                row["key"]: json.loads(row["value"])
                for row in connection.execute("SELECT key, value FROM metadata")
            }

    def summary(self) -> dict[str, Any]:
        with self._connect() as connection:
            counts = {
                row["split"]: row["count"]
                for row in connection.execute(
                    "SELECT split, COUNT(*) AS count FROM players GROUP BY split"
                )
            }
            missing_values = connection.execute(
                "SELECT COUNT(*) FROM players WHERE value_eur IS NULL"
            ).fetchone()[0]
        metadata = self.metadata()
        return {
            "save_date": metadata.get("save_date"),
            "game": metadata.get("game"),
            "build": metadata.get("build"),
            "player_counts": counts,
            "players_without_stored_value": missing_values,
            "value_note": VALUE_NOTE,
            "model_ready": bool(metadata.get("model_ready", False)),
            "feature_schema_version": metadata.get("feature_schema_version"),
            "model_version": metadata.get("model_version"),
            "reference_rows": metadata.get("reference_rows"),
            "available_positions": metadata.get("available_positions", []),
            "available_filters": ["age", "value_eur", "position", "club"],
        }

    def set_metadata(self, key: str, value: Any) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)),
            )

    def search(
        self,
        *,
        age_min: int | None = None,
        age_max: int | None = None,
        value_min_eur: float | None = None,
        value_max_eur: float | None = None,
        position: str | None = None,
        club: str | None = None,
        include_unknown_value: bool = True,
        currency_scale: float = 1.0,
        limit: int = 200,
        offset: int = 0,
        heldout_only: bool = True,
    ) -> dict[str, Any]:
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        position = normalize_position(position)
        where = ["split = 'test'" if heldout_only else "split IN ('train', 'test', 'unlabeled')"]
        params: list[Any] = []
        for column, operator, value in (
            ("age", ">=", age_min),
            ("age", "<=", age_max),
        ):
            if value is not None:
                where.append(f"{column} {operator} ?")
                params.append(value)
        value_bounds = [
            (operator, value)
            for operator, value in ((">=", value_min_eur), ("<=", value_max_eur))
            if value is not None
        ]
        if value_bounds:
            # Multiply (not divide) so the comparison matches scale_money() bit for bit.
            clause = " AND ".join(f"value_eur * ? {operator} ?" for operator, _ in value_bounds)
            if include_unknown_value:
                clause = f"(value_eur IS NULL OR ({clause}))"
            where.append(clause)
            for _, value in value_bounds:
                params.extend([currency_scale, value])
        if club:
            where.append("LOWER(club) LIKE LOWER(?)")
            params.append(f"%{club.strip()}%")
        if position:
            where.append(
                "(EXISTS (SELECT 1 FROM json_each(natural_positions) WHERE value=?) OR EXISTS (SELECT 1 FROM json_each(accomplished_positions) WHERE value=?))"
            )
            params.extend([position, position])
        predicate = " AND ".join(where)
        with self._connect() as connection:
            total = connection.execute(
                "SELECT COUNT(*) FROM players WHERE " + predicate, params
            ).fetchone()[0]
            unknown_value = connection.execute(
                "SELECT COUNT(*) FROM players WHERE value_eur IS NULL AND " + predicate, params
            ).fetchone()[0]
            selected = [
                self._decode(row)
                for row in connection.execute(
                    "SELECT * FROM players WHERE "
                    + predicate
                    + " ORDER BY player_id LIMIT ? OFFSET ?",
                    [*params, limit, offset],
                )
            ]
        compact = [
            {
                "player_id": row["player_id"],
                "name": row["name"],
                "age": row["age"],
                "club": row["club"],
                "positions": row["natural_positions"] + row["accomplished_positions"],
                "value_eur": scale_money(row, currency_scale)["value_eur"],
                "value_known": row["value_eur"] is not None,
            }
            for row in selected
        ]
        return {
            "matching_count": total,
            "unknown_value_count": unknown_value,
            "returned_count": len(compact),
            "truncated": total > len(compact),
            "offset": offset,
            "has_more": offset + len(compact) < total,
            "next_offset": offset + len(compact) if offset + len(compact) < total else None,
            "player_ids": [row["player_id"] for row in compact],
            "players": compact,
        }

    def get_players(
        self, player_ids: Sequence[int], *, require_test: bool = False, currency_scale: float = 1.0
    ) -> list[dict[str, Any]]:
        if not player_ids:
            return []
        unique_ids = list(dict.fromkeys(int(value) for value in player_ids))
        found: dict[int, dict[str, Any]] = {}
        with self._connect() as connection:
            for start in range(0, len(unique_ids), 900):
                chunk = unique_ids[start : start + 900]
                placeholders = ",".join("?" for _ in chunk)
                split_clause = " AND split='test'" if require_test else ""
                query = f"SELECT * FROM players WHERE player_id IN ({placeholders}){split_clause}"
                for row in connection.execute(query, chunk):
                    decoded = self._decode(row)
                    found[decoded["player_id"]] = decoded
        return [
            scale_money(found[player_id], currency_scale)
            for player_id in unique_ids
            if player_id in found
        ]

    def all_ids(self) -> list[int]:
        with self._connect() as connection:
            return [
                row[0]
                for row in connection.execute("SELECT player_id FROM players ORDER BY player_id")
            ]

    def find_by_name(self, name: str) -> list[dict[str, Any]]:
        """Players whose name equals `name`, ignoring case."""
        with self._connect() as connection:
            return [
                self._decode(row)
                for row in connection.execute(
                    "SELECT * FROM players WHERE LOWER(name) = LOWER(?) ORDER BY player_id",
                    (name.strip(),),
                )
            ]

    def test_ids(self) -> list[int]:
        with self._connect() as connection:
            return [
                row[0]
                for row in connection.execute(
                    "SELECT player_id FROM players WHERE split='test' ORDER BY player_id"
                )
            ]
