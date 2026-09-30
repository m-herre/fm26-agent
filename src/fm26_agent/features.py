from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import VISIBLE_ATTRIBUTES, assert_safe_features

FEATURE_SCHEMA_VERSION = 3

NUMERIC_FEATURES = (
    "age",
    "height_cm",
    "value_eur",
    "wage_eur",
    "contract_days_remaining",
    "on_loan",
) + VISIBLE_ATTRIBUTES
CATEGORICAL_FEATURES = (
    "club_category",
    "nation_category",
    "preferred_foot",
    "natural_position_category",
    "accomplished_position_category",
)
TEXT_FEATURES = ("visible_traits_text",)


@dataclass(frozen=True)
class FeatureSchema:
    categories: dict[str, list[str]]
    columns: list[str]
    version: int = FEATURE_SCHEMA_VERSION

    @staticmethod
    def categorical_values(player: dict[str, Any]) -> dict[str, str | None]:
        club = player.get("club")
        if not club and player.get("club_uid") is not None:
            club = f"club:{player['club_uid']}"
        return {
            "club_category": club or None,
            "nation_category": f"nation:{player['nation_id']}"
            if player.get("nation_id") is not None
            else None,
            "preferred_foot": player.get("preferred_foot"),
            "natural_position_category": "|".join(sorted(player.get("natural_positions", [])))
            or None,
            "accomplished_position_category": "|".join(
                sorted(player.get("accomplished_positions", []))
            )
            or None,
        }

    @classmethod
    def fit(cls, players: Sequence[dict[str, Any]]) -> FeatureSchema:
        values = [cls.categorical_values(row) for row in players]
        categories = {
            key: sorted({row[key] for row in values if row[key] is not None})
            for key in CATEGORICAL_FEATURES
        }
        columns = list(NUMERIC_FEATURES + CATEGORICAL_FEATURES + TEXT_FEATURES)
        assert_safe_features(columns)
        return cls(categories=categories, columns=columns)

    def transform(self, players: Sequence[dict[str, Any]]) -> pd.DataFrame:
        if self.version != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                "Outdated feature schema; run `fm26-agent prepare --save <your .fm save>` for the 59-feature TabPFN 3.5 model"
            )
        expected = list(NUMERIC_FEATURES + CATEGORICAL_FEATURES + TEXT_FEATURES)
        if self.columns != expected:
            raise ValueError("Feature schema does not match the observable feature allowlist")
        assert_safe_features(self.columns)
        records = []
        for player in players:
            record = {column: player.get(column) for column in NUMERIC_FEATURES}
            record.update(self.categorical_values(player))
            # Genuine extracted trait labels, not generated descriptions or player identity.
            record["visible_traits_text"] = "; ".join(player.get("traits", [])) or None
            records.append(record)
        frame = pd.DataFrame(records, columns=self.columns)
        for key in NUMERIC_FEATURES:
            frame[key] = pd.to_numeric(frame[key], errors="coerce").astype(float)
        for key in CATEGORICAL_FEATURES:
            # Preserve original labels, including unseen categories. Never assign numeric codes.
            frame[key] = pd.Categorical(frame[key])
        for key in TEXT_FEATURES:
            frame[key] = frame[key].astype(object)
        return frame

    @property
    def categorical_indices(self) -> list[int]:
        return [self.columns.index(column) for column in CATEGORICAL_FEATURES]

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> FeatureSchema:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                "Outdated feature schema; run `fm26-agent prepare --save <your .fm save>` for the 59-feature TabPFN 3.5 model"
            )
        result = cls(**payload)
        result.transform([])  # Validate the exact allowlist even before a provider is loaded.
        return result
