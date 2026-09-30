from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

from .prediction import CHANCE_FIELD, HIGH_FIELD, LOW_FIELD, Predictor

TABLE = "potential_v3"


class CachedPredictor:
    """Prediction-only disk cache, isolated by immutable dataset, task and model versions."""

    def __init__(self, predictor: Predictor, path: Path, namespace: str):
        self.predictor, self.path, self.namespace = predictor, path, namespace
        self.score_field = predictor.score_field
        self.score_bounds = predictor.score_bounds
        self.task = predictor.task
        self.star_level = getattr(predictor, "star_level", None)
        # Keeps scores from different tasks apart and matches caches written by earlier versions.
        self.namespace = self.task + ":" + namespace
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as connection, connection:
            # Tables from earlier versions ("predictions", "estimates") held other estimates; the
            # table name changes whenever what is stored changes, so they are simply ignored.
            connection.execute(
                f"CREATE TABLE IF NOT EXISTS {TABLE} (namespace TEXT NOT NULL, player_id INTEGER NOT NULL, score REAL NOT NULL, low REAL, high REAL, chance REAL, PRIMARY KEY(namespace, player_id))"
            )

    @staticmethod
    def namespace_for(preparation_id: str, reference: Path, schema_fingerprint: str) -> str:
        record = json.loads(reference.read_text())
        return hashlib.sha256(
            json.dumps([preparation_id, schema_fingerprint, record], sort_keys=True).encode()
        ).hexdigest()

    def predict(self, players: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        ids = [row["player_id"] for row in players]
        cached: dict[int, dict[str, Any]] = {}
        with closing(sqlite3.connect(self.path)) as connection, connection:
            for start in range(0, len(ids), 500):
                batch = ids[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                for player_id, *values in connection.execute(
                    f"SELECT player_id, score, low, high, chance FROM {TABLE} WHERE namespace=? AND player_id IN ({placeholders})",
                    [self.namespace, *batch],
                ):
                    cached[player_id] = _row(player_id, self.score_field, *values)
            missing = [row for row in players if row["player_id"] not in cached]
            result = self.predictor.predict(missing) if missing else []
            expected = {row["player_id"] for row in missing}
            if len(result) != len(expected) or {row["player_id"] for row in result} != expected:
                raise ValueError("Cached prediction result IDs differ from requested IDs")
            for row in result:
                if not self._valid(row):
                    raise ValueError("Cannot cache invalid model predictions")
            connection.executemany(
                f"INSERT OR REPLACE INTO {TABLE} VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        self.namespace,
                        row["player_id"],
                        row[self.score_field],
                        row.get(LOW_FIELD),
                        row.get(HIGH_FIELD),
                        row.get(CHANCE_FIELD),
                    )
                    for row in result
                ],
            )
            cached.update(
                {
                    row["player_id"]: _row(
                        row["player_id"],
                        self.score_field,
                        row[self.score_field],
                        row.get(LOW_FIELD),
                        row.get(HIGH_FIELD),
                        row.get(CHANCE_FIELD),
                    )
                    for row in result
                }
            )
        if not all(self._valid(cached[player_id]) for player_id in ids):
            raise ValueError("Prediction cache contains invalid predictions")
        return [cached[player_id] for player_id in ids]

    def _valid(self, row: dict[str, Any]) -> bool:
        low_bound, high_bound = self.score_bounds

        def finite(value: Any, low: float, high: float) -> bool:
            return isinstance(value, (int, float)) and math.isfinite(value) and low <= value <= high

        return (
            finite(row.get(self.score_field), low_bound, high_bound)
            and all(
                row.get(key) is None or finite(row[key], low_bound, high_bound)
                for key in (LOW_FIELD, HIGH_FIELD)
            )
            and (row.get(CHANCE_FIELD) is None or finite(row[CHANCE_FIELD], 0.0, 1.0))
        )


def _row(
    player_id: int,
    score_field: str,
    score: float,
    low: float | None,
    high: float | None,
    chance: float | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"player_id": player_id, score_field: score}
    if low is not None and high is not None:
        row[LOW_FIELD], row[HIGH_FIELD] = low, high
    if chance is not None:
        row[CHANCE_FIELD] = chance
    return row
