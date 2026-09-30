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

SEASON_STAT_COLUMNS = (
    "appearances",
    "starts",
    "minutes",
    "goals",
    "assists",
    "average_rating",
    "expected_goals",
    "expected_assists",
    "player_of_the_match",
    "clean_sheets",
)

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


def club_names(club: str | Sequence[str] | None) -> list[str]:
    """One club name or several, as a clean list; empty when no club filter was given."""
    if club is None:
        return []
    names = [club] if isinstance(club, str) else list(club)
    return [name.strip() for name in names if name and name.strip()]


def club_matches(player_club: str | None, club: str | Sequence[str] | None) -> bool:
    """Whether a player's club contains any of the requested names (case-insensitive)."""
    names = club_names(club)
    return not names or any(name.casefold() in (player_club or "").casefold() for name in names)


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

    def initialize(
        self,
        players: Sequence[dict[str, Any]],
        metadata: dict[str, Any],
        season_stats: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        attribute_sql = ",\n".join(f'"{name}" REAL' for name in VISIBLE_ATTRIBUTES)
        with self._connect() as connection:
            connection.executescript(
                f"""
                BEGIN IMMEDIATE;
                DROP TABLE IF EXISTS players;
                DROP TABLE IF EXISTS metadata;
                DROP TABLE IF EXISTS season_stats;
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
                CREATE INDEX players_age ON players(age);
                CREATE INDEX players_value ON players(value_eur);
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
        self.set_season_stats(season_stats or {}, metadata.get("season_stats_version", 1))

    def set_season_stats(self, stats: dict[int, dict[str, Any]], version: int) -> None:
        """Replace the display-only season stats. They are never read by the model."""
        columns = ",".join(f'"{name}"' for name in SEASON_STAT_COLUMNS)
        with self._connect() as connection:
            connection.executescript(
                "DROP TABLE IF EXISTS season_stats;"
                "CREATE TABLE season_stats (player_id INTEGER PRIMARY KEY, "
                + ",".join(f'"{name}" REAL' for name in SEASON_STAT_COLUMNS)
                + ");"
            )
            connection.executemany(
                f"INSERT INTO season_stats (player_id, {columns}) VALUES (?, "
                + ",".join("?" for _ in SEASON_STAT_COLUMNS)
                + ")",
                [
                    (player_id, *(item.get(name) for name in SEASON_STAT_COLUMNS))
                    for player_id, item in stats.items()
                ],
            )
        self.set_metadata("season_stats_version", version)
        self.set_metadata("season_stats_players", len(stats))

    def season_stats(self, player_ids: Sequence[int]) -> dict[int, dict[str, Any]]:
        """Season stats for the players who have any (others are simply absent)."""
        ids = list(dict.fromkeys(int(value) for value in player_ids))
        found: dict[int, dict[str, Any]] = {}
        with self._connect() as connection:
            for start in range(0, len(ids), 900):
                chunk = ids[start : start + 900]
                try:
                    rows = connection.execute(
                        "SELECT * FROM season_stats WHERE player_id IN ("
                        + ",".join("?" for _ in chunk)
                        + ")",
                        chunk,
                    )
                    for row in rows:
                        item = dict(row)
                        for name in (
                            "appearances",
                            "starts",
                            "minutes",
                            "goals",
                            "assists",
                            "player_of_the_match",
                            "clean_sheets",
                        ):
                            if item[name] is not None:
                                item[name] = int(item[name])
                        found[item.pop("player_id")] = item
                except sqlite3.OperationalError:  # a save set up before stats existed
                    return {}
        return found

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
            total = connection.execute("SELECT COUNT(*) FROM players").fetchone()[0]
            missing_values = connection.execute(
                "SELECT COUNT(*) FROM players WHERE value_eur IS NULL"
            ).fetchone()[0]
        metadata = self.metadata()
        return {
            "save_date": metadata.get("save_date"),
            "game": metadata.get("game"),
            "build": metadata.get("build"),
            "player_count": total,
            "players_without_stored_value": missing_values,
            "value_note": VALUE_NOTE,
            "model_ready": bool(metadata.get("model_ready", False)),
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
        club: str | Sequence[str] | None = None,
        preferred_foot: str | None = None,
        contract_ends_within_days: int | None = None,
        include_unknown_value: bool = True,
        currency_scale: float = 1.0,
        limit: int = 200,
        offset: int = 0,
    ) -> dict[str, Any]:
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        position = normalize_position(position)
        where = ["1 = 1"]
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
        names = club_names(club)
        if names:
            where.append(
                "(" + " OR ".join("LOWER(club) LIKE LOWER(?) ESCAPE '\\'" for _ in names) + ")"
            )
            params.extend(
                "%" + name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
                for name in names
            )
        if preferred_foot:
            where.append("preferred_foot = ?")
            params.append(preferred_foot)
        if contract_ends_within_days is not None:
            where.append("contract_days_remaining BETWEEN 0 AND ?")
            params.append(contract_ends_within_days)
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
        self, player_ids: Sequence[int], *, currency_scale: float = 1.0
    ) -> list[dict[str, Any]]:
        if not player_ids:
            return []
        unique_ids = list(dict.fromkeys(int(value) for value in player_ids))
        found: dict[int, dict[str, Any]] = {}
        with self._connect() as connection:
            for start in range(0, len(unique_ids), 900):
                chunk = unique_ids[start : start + 900]
                placeholders = ",".join("?" for _ in chunk)
                query = f"SELECT * FROM players WHERE player_id IN ({placeholders})"
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
