from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

MIN_TARGET_VALUES = 200  # fewer known values than this: not offered as a prediction target


class PrivateStore:
    """Ground truth used only by preparation and evaluation code."""

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

    def initialize(self, labels: Sequence[dict[str, Any]], preparation_id: str = "fixture") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                BEGIN IMMEDIATE;
                DROP TABLE IF EXISTS labels;
                DROP TABLE IF EXISTS metadata;
                DROP TABLE IF EXISTS targets;
                CREATE TABLE metadata (preparation_id TEXT NOT NULL);
                CREATE TABLE labels (
                    player_id INTEGER PRIMARY KEY,
                    potential_ability INTEGER,
                    wonderkid INTEGER,
                    split TEXT NOT NULL
                );
                CREATE INDEX labels_split ON labels(split);
                """
            )
            connection.execute("INSERT INTO metadata VALUES (?)", (preparation_id,))
            connection.executemany(
                "INSERT INTO labels(player_id, potential_ability, wonderkid, split) VALUES (?, ?, ?, ?)",
                [
                    (row["player_id"], row["potential_ability"], row["wonderkid"], row["split"])
                    for row in labels
                ],
            )

    def set_targets(self, hidden: dict[int, dict[str, Any]]) -> None:
        """Store each player's hidden targets (see targets.py), replacing any stored before."""
        from .targets import STORED_TARGETS

        columns = ", ".join(f"{name} REAL" for name in STORED_TARGETS)
        with self._connect() as connection:
            connection.execute("DROP TABLE IF EXISTS targets")
            connection.execute(f"CREATE TABLE targets (player_id INTEGER PRIMARY KEY, {columns})")
            connection.executemany(
                f"INSERT INTO targets VALUES (?, {', '.join('?' for _ in STORED_TARGETS)})",
                [
                    (player_id, *(values.get(name) for name in STORED_TARGETS))
                    for player_id, values in hidden.items()
                ],
            )

    def has_targets(self) -> bool:
        with self._connect() as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='targets'"
                ).fetchone()
                is not None
            )

    def available_targets(self) -> list[str]:
        """Targets with enough stored values to learn from (derived ones included)."""
        from .targets import STORED_TARGETS, TARGETS

        if not self.has_targets():
            return ["potential_ability"]
        with self._connect() as connection:
            counts = connection.execute(
                "SELECT " + ", ".join(f"COUNT({name})" for name in STORED_TARGETS) + " FROM targets"
            ).fetchone()
        names = {
            name
            for name, count in zip(STORED_TARGETS, counts, strict=True)
            if count >= MIN_TARGET_VALUES
        }
        if "current_ability" in names:
            names.add("growth_room")
        # potential_ability lives in the labels table, which every setup has.
        return ["potential_ability", *[name for name in TARGETS if name in names]]

    def target_values(self, name: str, split: str | None = None) -> dict[int, float]:
        """{player_id: value} for one target, optionally for one split; missing values left out."""
        from .targets import clean_value, get_target

        get_target(name)
        if name == "potential_ability":
            expression, source = "labels.potential_ability", "labels"
        elif name == "growth_room":
            expression = "labels.potential_ability - targets.current_ability"
            source = "labels JOIN targets USING(player_id)"
        else:
            expression, source = f"targets.{name}", "labels JOIN targets USING(player_id)"
        query = f"SELECT labels.player_id, {expression} FROM {source}"
        params: tuple[Any, ...] = ()
        if split:
            query += " WHERE labels.split=?"
            params = (split,)
        query += " ORDER BY labels.player_id"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        result = {}
        for player_id, value in rows:
            cleaned = clean_value(name, value)
            if cleaned is not None:
                result[int(player_id)] = cleaned
        return result

    def preparation_id(self) -> str:
        with self._connect() as connection:
            return connection.execute("SELECT preparation_id FROM metadata").fetchone()[0]

    def rows(self, split: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM labels"
        params: tuple[Any, ...] = ()
        if split:
            query += " WHERE split=?"
            params = (split,)
        query += " ORDER BY player_id"
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query, params)]

    def get(self, player_ids: Sequence[int]) -> dict[int, dict[str, Any]]:
        if not player_ids:
            return {}
        unique_ids = list(dict.fromkeys(int(value) for value in player_ids))
        result: dict[int, dict[str, Any]] = {}
        with self._connect() as connection:
            for start in range(0, len(unique_ids), 900):
                chunk = unique_ids[start : start + 900]
                placeholders = ",".join("?" for _ in chunk)
                for row in connection.execute(
                    f"SELECT * FROM labels WHERE player_id IN ({placeholders})", chunk
                ):
                    result[row["player_id"]] = dict(row)
        return result
