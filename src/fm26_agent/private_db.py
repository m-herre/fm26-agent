from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any


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
