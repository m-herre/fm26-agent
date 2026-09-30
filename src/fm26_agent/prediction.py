from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from . import tabpfn_backend
from .features import FeatureSchema

SCORE_FIELD = "predicted_potential"
SCORE_BOUNDS = (1.0, 200.0)  # the game's potential-ability scale
LOW_FIELD = "potential_low"
HIGH_FIELD = "potential_high"
CHANCE_FIELD = "star_chance"
STAR_LEVEL = 160  # "chance of reaching 160+": the usual wonderkid line
# One hosted call returns TabPFN's predictive distribution at these percentiles. The estimate is
# the median, the range shown is the 10th to 90th percentile, and the star chance is read off the
# same curve, so every player costs a single request.
QUANTILES = tuple(round(level / 20, 2) for level in range(1, 20))  # 0.05, 0.10, ... 0.95
MEDIAN, LOW_LEVEL, HIGH_LEVEL = QUANTILES.index(0.5), QUANTILES.index(0.1), QUANTILES.index(0.9)


def with_rate_limit_retry(call: Any, attempts: int = 6) -> Any:
    """Run a hosted request, waiting and retrying when the service says to slow down (HTTP 429)."""
    for attempt in range(attempts):
        try:
            return call()
        except RuntimeError as exc:
            message = str(exc)
            if "429" not in message or attempt == attempts - 1:
                raise
            wait = re.search(r"Retry in (\d+)s", message)
            time.sleep(min(60, int(wait.group(1)) + 1 if wait else 5 * (attempt + 1)))


class Predictor(Protocol):
    def predict(self, players: Sequence[dict[str, Any]]) -> list[dict[str, Any]]: ...


class TabPFNPredictor:
    """Estimates potential ability with a TabPFN regressor fitted once on the reference players.

    The regressor runs locally or at Prior Labs (see tabpfn_backend); both are TabPFN-3.5.

    The exact potential of the reference players is the training target and is never an input
    feature. TabPFN takes the raw table as it is: categories, text and missing values need no
    manual preprocessing.
    """

    task = "pa_regression"
    score_field = SCORE_FIELD
    score_bounds = SCORE_BOUNDS

    def __init__(
        self,
        model: Any,
        schema: FeatureSchema,
        star_level: int = STAR_LEVEL,
        backend: str = "hosted",
    ):
        self.model = model
        self.schema = schema
        self.star_level = star_level
        self.backend = backend

    @classmethod
    def fit(
        cls,
        players: Sequence[dict[str, Any]],
        targets: Sequence[int],
        schema: FeatureSchema,
        random_seed: int = 42,
        backend: str = "hosted",
    ) -> TabPFNPredictor:
        target = np.asarray(targets, dtype=float)
        if (
            target.shape != (len(players),)
            or not np.all(np.isfinite(target))
            or np.any((target < SCORE_BOUNDS[0]) | (target > SCORE_BOUNDS[1]))
        ):
            raise ValueError("Fitting needs one exact 1–200 potential per reference player")
        model = tabpfn_backend.new_regressor(backend, random_seed)
        tabpfn_backend.fit(model, backend, schema.transform(players), target)
        return cls(model, schema, backend=backend)

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
                    "backend": self.backend,
                    "model": tabpfn_backend.save_fitted(self.model, self.backend, path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def check_reference(
        path: Path, schema: FeatureSchema, preparation_id: str, backend: str | None = None
    ) -> str | None:
        """Why the saved fit cannot be used for this preparation, or None if it can. Local only."""
        if not path.exists():
            return "no saved model"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("version") != 2
            or payload.get("task") != TabPFNPredictor.task
            or payload.get("model_version") != "v3.5"
        ):
            return "the saved model is from an incompatible version"
        if payload.get("feature_fingerprint") != schema.fingerprint:
            return "the saved model was fitted on different features"
        if payload.get("preparation_id") != preparation_id:
            return "the saved model belongs to another save"
        if backend is not None and payload.get("backend", "hosted") != backend:
            return (
                f"the saved model was fitted {'at Prior Labs' if backend == 'local' else 'locally'}"
            )
        if payload.get("backend") == "local" and not (path.parent / payload["model"]).exists():
            return "the saved local model is missing"
        return None

    @classmethod
    def load(
        cls, path: Path, schema_path: Path, preparation_id: str | None = None
    ) -> TabPFNPredictor:
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
        backend = payload.get("backend", "hosted")
        model = tabpfn_backend.load_fitted(payload["model"], backend, path)
        return cls(model, schema, backend=backend)

    def distribution(self, players: Sequence[dict[str, Any]]) -> np.ndarray:
        """TabPFN's predicted percentiles (one row per QUANTILES level) in a single request."""
        matrix = self.schema.transform(players)
        values = tabpfn_backend.predict_quantiles(self.model, self.backend, matrix, list(QUANTILES))
        if values.shape != (len(QUANTILES), len(players)) or not np.all(np.isfinite(values)):
            raise ValueError("TabPFN returned invalid potential estimates")
        # Percentiles must not decrease; clip to the game's scale.
        return np.clip(np.maximum.accumulate(values, axis=0), *SCORE_BOUNDS)

    def predict(self, players: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Estimate, likely range and star chance for each player, from one hosted call."""
        if not players:
            return []
        curves = self.distribution(players)
        return [
            {
                "player_id": row["player_id"],
                SCORE_FIELD: float(curve[MEDIAN]),
                LOW_FIELD: float(curve[LOW_LEVEL]),
                HIGH_FIELD: float(curve[HIGH_LEVEL]),
                CHANCE_FIELD: chance_of_reaching(curve, self.star_level),
            }
            for row, curve in zip(players, curves.T, strict=True)
        ]


HostedPredictor = TabPFNPredictor  # the name used before local TabPFN existed


def chance_of_reaching(curve: np.ndarray, level: float) -> float:
    """Share of the predicted distribution at or above `level` (0-1), interpolated.

    Beyond the outermost percentiles the answer is capped at 3% and 97%: the curve says no more.
    """
    below = float(
        np.interp(
            level,
            curve,
            QUANTILES,
            left=QUANTILES[0] / 2,
            right=1 - (1 - QUANTILES[-1]) / 2,
        )
    )
    return round(1.0 - below, 3)
