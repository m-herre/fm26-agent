from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

from .prediction import Predictor


class CachedPredictor:
    """Prediction-only disk cache, isolated by immutable dataset, task and model versions."""

    def __init__(self, predictor: Predictor, path: Path, namespace: str):
        self.predictor, self.path, self.namespace = predictor, path, namespace
        self.score_field = getattr(predictor, "score_field", "wonderkid_probability")
        self.score_bounds = getattr(predictor, "score_bounds", (0.0, 1.0))
        self.task = getattr(predictor, "task", "binary_classification")
        if self.task != "binary_classification":
            self.namespace = self.task + ":" + namespace
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS predictions (namespace TEXT NOT NULL, player_id INTEGER NOT NULL, probability REAL NOT NULL, PRIMARY KEY(namespace, player_id))"
            )

    @staticmethod
    def namespace_for(preparation_id: str, reference: Path, schema_fingerprint: str) -> str:
        record = json.loads(reference.read_text())
        return hashlib.sha256(
            json.dumps([preparation_id, schema_fingerprint, record], sort_keys=True).encode()
        ).hexdigest()

    def predict(self, players: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        ids = [row["player_id"] for row in players]
        cached: dict[int, float] = {}
        with closing(sqlite3.connect(self.path)) as connection, connection:
            for start in range(0, len(ids), 500):
                batch = ids[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                cached.update(
                    connection.execute(
                        f"SELECT player_id, probability FROM predictions WHERE namespace=? AND player_id IN ({placeholders})",
                        [self.namespace, *batch],
                    )
                )
            missing = [row for row in players if row["player_id"] not in cached]
            result = self.predictor.predict(missing) if missing else []
            expected = {row["player_id"] for row in missing}
            if len(result) != len(expected) or {row["player_id"] for row in result} != expected:
                raise ValueError("Cached prediction result IDs differ from requested IDs")
            if any(
                not isinstance(row[self.score_field], (int, float))
                or not math.isfinite(row[self.score_field])
                or not self.score_bounds[0] <= row[self.score_field] <= self.score_bounds[1]
                for row in result
            ):
                raise ValueError(
                    "Cannot cache invalid model probabilities"
                    if self.score_field == "wonderkid_probability"
                    else "Cannot cache invalid model predictions"
                )
            connection.executemany(
                "INSERT OR REPLACE INTO predictions VALUES (?, ?, ?)",
                [(self.namespace, row["player_id"], row[self.score_field]) for row in result],
            )
            cached.update({row["player_id"]: row[self.score_field] for row in result})
        if any(
            not math.isfinite(cached[player_id])
            or not self.score_bounds[0] <= cached[player_id] <= self.score_bounds[1]
            for player_id in ids
        ):
            raise ValueError("Prediction cache contains invalid predictions")
        return [{"player_id": player_id, self.score_field: cached[player_id]} for player_id in ids]
