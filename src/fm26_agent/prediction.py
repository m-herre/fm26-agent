from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .features import FeatureSchema

SCORE_FIELD = "predicted_potential"
SCORE_BOUNDS = (1.0, 200.0)  # the game's potential-ability scale


class Predictor(Protocol):
    def predict(self, players: Sequence[dict[str, Any]]) -> list[dict[str, Any]]: ...


class HostedPredictor:
    """Estimates potential ability with a TabPFN regressor fitted once on the reference players.

    The exact potential of the reference players is the training target and is never an input
    feature. TabPFN takes the raw table as it is: categories, text and missing values need no
    manual preprocessing.
    """

    task = "pa_regression"
    score_field = SCORE_FIELD
    score_bounds = SCORE_BOUNDS

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
        from tabpfn_client import TabPFNRegressor

        target = np.asarray(targets, dtype=float)
        if (
            target.shape != (len(players),)
            or not np.all(np.isfinite(target))
            or np.any((target < SCORE_BOUNDS[0]) | (target > SCORE_BOUNDS[1]))
        ):
            raise ValueError("Fitting needs one exact 1–200 potential per reference player")
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
                    "prediction_clip": list(SCORE_BOUNDS),
                    "preparation_id": preparation_id,
                    "feature_fingerprint": self.schema.fingerprint,
                    "model": self.model.save_model(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def check_reference(path: Path, schema: FeatureSchema, preparation_id: str) -> str | None:
        """Why the saved fit cannot be used for this preparation, or None if it can. Local only."""
        if not path.exists():
            return "no saved model"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("version") != 2
            or payload.get("task") != HostedPredictor.task
            or payload.get("model_version") != "v3.5"
        ):
            return "the saved model is from an incompatible version"
        if payload.get("feature_fingerprint") != schema.fingerprint:
            return "the saved model was fitted on different features"
        if payload.get("preparation_id") != preparation_id:
            return "the saved model belongs to another save"
        return None

    @classmethod
    def load(
        cls, path: Path, schema_path: Path, preparation_id: str | None = None
    ) -> HostedPredictor:
        from tabpfn_client import TabPFNRegressor

        schema = FeatureSchema.load(schema_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("version") != 2
            or payload.get("task") != cls.task
            or payload.get("model_version") != "v3.5"
        ):
            raise ValueError("The saved model is incompatible; run prepare again")
        if payload.get("feature_fingerprint") != schema.fingerprint:
            raise ValueError("The saved model and feature schema differ; run prepare again")
        if preparation_id is not None and payload.get("preparation_id") != preparation_id:
            raise ValueError("The saved model belongs to another save; run prepare again")
        return cls(TabPFNRegressor.load_model(payload["model"]), schema)

    def predict(self, players: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        if not players:
            return []
        estimates = np.asarray(
            self.model.predict(self.schema.transform(players), output_type="mean"), dtype=float
        )
        if estimates.shape != (len(players),) or not np.all(np.isfinite(estimates)):
            raise ValueError("TabPFN returned invalid potential estimates")
        # Bound to the game's scale; estimates stay continuous inside it.
        estimates = np.clip(estimates, *SCORE_BOUNDS)
        return [
            {"player_id": row["player_id"], SCORE_FIELD: float(score)}
            for row, score in zip(players, estimates, strict=True)
        ]
