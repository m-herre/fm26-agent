from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .features import FeatureSchema


class Predictor(Protocol):
    def predict(self, players: Sequence[dict[str, Any]]) -> list[dict[str, Any]]: ...


class HostedPredictor:
    task = "binary_classification"
    score_field = "wonderkid_probability"
    score_bounds = (0.0, 1.0)

    def __init__(self, model: Any, schema: FeatureSchema):
        self.model = model
        self.schema = schema

    @classmethod
    def fit(
        cls,
        players: Sequence[dict[str, Any]],
        targets: Sequence[int],
        schema: FeatureSchema,
        random_seed: int = 42,
    ) -> HostedPredictor:
        from tabpfn_client import TabPFNClassifier

        matrix = schema.transform(players)
        model = TabPFNClassifier(
            model_path="v3.5_default",
            fit_mode="fit_with_cache",
            text_handling="advanced",
            random_state=random_seed,
        )
        model.fit(matrix, np.asarray(targets, dtype=int))
        return cls(model, schema)

    @staticmethod
    def estimate_cost(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
        from tabpfn_client import estimate_cost

        quote = estimate_cost(train, test, model_version="v3.5", operation="cache_predict")
        if hasattr(quote, "model_dump"):
            return quote.model_dump(mode="json")
        if hasattr(quote, "__dict__"):
            return vars(quote)
        return {"estimate": str(quote)}

    def save(self, path: Path, preparation_id: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        reference = self.model.save_model()
        payload = {
            "version": 2,
            "model_version": "v3.5",
            "fit_mode": "fit_with_cache",
            "preparation_id": preparation_id,
            "feature_fingerprint": self.schema.fingerprint,
            "model": reference,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls, path: Path, schema_path: Path, preparation_id: str | None = None
    ) -> HostedPredictor:
        from tabpfn_client import TabPFNClassifier

        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != 2 or payload.get("model_version") != "v3.5":
            raise ValueError(
                "Legacy model reference; run one new prepare for mixed-type TabPFN 3.5"
            )
        if payload.get("task", "binary_classification") != cls.task:
            raise ValueError("Model reference belongs to a different prediction task")
        schema = FeatureSchema.load(schema_path)
        if payload.get("feature_fingerprint") != schema.fingerprint:
            raise ValueError("Model and feature schema differ; run prepare again")
        if preparation_id is not None and payload.get("preparation_id") != preparation_id:
            raise ValueError("Model belongs to another dataset; run prepare again")
        return cls(TabPFNClassifier.load_model(payload["model"]), schema)

    def predict(self, players: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        if not players:
            return []
        positive_columns = np.flatnonzero(np.asarray(self.model.classes_) == 1)
        if len(positive_columns) != 1:
            raise ValueError("The fitted model must have binary classes including label 1")
        probabilities = np.asarray(self.model.predict_proba(self.schema.transform(players)))[
            :, positive_columns[0]
        ]
        if (
            len(probabilities) != len(players)
            or not np.all(np.isfinite(probabilities))
            or np.any((probabilities < 0) | (probabilities > 1))
        ):
            raise ValueError("TabPFN returned invalid probabilities")
        return [
            {"player_id": row["player_id"], "wonderkid_probability": float(probability)}
            for row, probability in zip(players, probabilities, strict=True)
        ]


class HostedRegressionPredictor(HostedPredictor):
    """Exact potential is an authorized training target, never an input feature."""

    task = "pa_regression"
    score_field = "predicted_potential"
    score_bounds = (1.0, 200.0)

    @classmethod
    def fit(cls, players, targets, schema, random_seed=42):
        from tabpfn_client import TabPFNRegressor

        target = np.asarray(targets, dtype=float)
        if (
            target.shape != (len(players),)
            or not np.all(np.isfinite(target))
            or np.any((target < 1) | (target > 200))
        ):
            raise ValueError("Regression requires one exact 1–200 target per reference player")
        model = TabPFNRegressor(
            model_path="v3.5_default",
            fit_mode="fit_with_cache",
            text_handling="advanced",
            random_state=random_seed,
        )
        model.fit(schema.transform(players), target)
        return cls(model, schema)

    def save(self, path: Path, preparation_id: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "version": 2,
                    "task": self.task,
                    "model_version": "v3.5",
                    "fit_mode": "fit_with_cache",
                    "target_upload": "exact_pa_user_authorized",
                    "prediction_clip": [1, 200],
                    "preparation_id": preparation_id,
                    "feature_fingerprint": self.schema.fingerprint,
                    "model": self.model.save_model(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path, schema_path: Path, preparation_id: str | None = None):
        from tabpfn_client import TabPFNRegressor

        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("version") != 2
            or payload.get("task") != cls.task
            or payload.get("model_version") != "v3.5"
        ):
            raise ValueError("A compatible regression fit is required; run fit-regression")
        schema = FeatureSchema.load(schema_path)
        if payload.get("feature_fingerprint") != schema.fingerprint:
            raise ValueError("Regression feature schema differs; run fit-regression")
        if preparation_id is not None and payload.get("preparation_id") != preparation_id:
            raise ValueError("Regression fit belongs to another dataset; run fit-regression")
        return cls(TabPFNRegressor.load_model(payload["model"]), schema)

    def predict(self, players):
        if not players:
            return []
        estimates = np.asarray(
            self.model.predict(self.schema.transform(players), output_type="mean"), dtype=float
        )
        if estimates.shape != (len(players),) or not np.all(np.isfinite(estimates)):
            raise ValueError("TabPFN returned invalid potential estimates")
        # Bound display/ranking to the game's scale; preserve continuous estimates within it.
        estimates = np.clip(estimates, *self.score_bounds)
        return [
            {"player_id": row["player_id"], self.score_field: float(score)}
            for row, score in zip(players, estimates, strict=True)
        ]
