"""A second TabPFN model: market value for the players whose save stores none.

About half of a save's players (free agents, clubs the game isn't simulating) have no stored
market value. At setup a TabPFN regressor learns log(value) from players who do have one, using
the same visible features as the potential model minus value itself, and then estimates every
missing value with a 10th-90th percentile range.

The same single prediction request also scores a held-out slice of players whose real value is
known, so every setup measures how good its own estimates are, at no extra cost.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

import numpy as np

from . import tabpfn_backend
from .features import FeatureSchema

VALUE_MODEL_VERSION = 1
LEVELS = (0.1, 0.5, 0.9)
TRAIN_ROWS = 10_000
CHECK_ROWS = 2_000
MIN_KNOWN = 50  # fewer players with a value than this: no estimates


def value_table(schema: FeatureSchema, players: Sequence[dict[str, Any]]):
    """The potential model's inputs without the value column (the thing being estimated)."""
    return schema.transform(players).drop(columns=["value_eur"])


def check_report(actual: np.ndarray, low: np.ndarray, mid: np.ndarray, high: np.ndarray) -> dict:
    ratio = mid / actual
    log_miss = np.log(mid) - np.log(actual)
    total = float(np.sum((np.log(actual) - np.log(actual).mean()) ** 2))
    return {
        "players": int(len(actual)),
        "median_error_percent": round(float(np.median(np.abs(ratio - 1)) * 100), 1),
        "within_25_percent": round(float(np.mean(np.abs(ratio - 1) <= 0.25)), 3),
        "within_50_percent": round(float(np.mean(np.abs(ratio - 1) <= 0.5)), 3),
        "range_coverage_80": round(float(np.mean((actual >= low) & (actual <= high))), 3),
        "median_range_factor": round(float(np.median(high / low)), 2),
        "r2_log_value": round(1 - float(np.sum(log_miss**2)) / total, 3) if total else None,
    }


def estimate_values(
    players: Sequence[dict[str, Any]],
    schema: FeatureSchema,
    seed: int = 42,
    backend: str = "hosted",
) -> tuple[dict[int, tuple[float, float, float]], dict[str, Any]]:
    """Fit on known values and estimate the missing ones: one fit and one prediction request.

    Returns {player_id: (low, mid, high)} for players without a stored value, in the save's
    internal units, and a report on held-out players whose value is known.
    """
    known = sorted(
        (row for row in players if row.get("value_eur") is not None and row["value_eur"] > 0),
        key=lambda row: row["player_id"],
    )
    missing = [row for row in players if row.get("value_eur") is None]
    if len(known) < MIN_KNOWN:
        return {}, {"skipped": "too few players with a stored value"}
    random.Random(seed).shuffle(known)
    train = known[:TRAIN_ROWS]
    check = known[TRAIN_ROWS : TRAIN_ROWS + CHECK_ROWS]
    model = tabpfn_backend.new_regressor(backend, seed)
    target = np.log(np.array([row["value_eur"] for row in train], dtype=float))
    tabpfn_backend.fit(model, backend, value_table(schema, train), target)
    scored = missing + check
    if not scored:
        return {}, {"skipped": "no player needs an estimate"}
    table = value_table(schema, scored)
    curves = tabpfn_backend.predict_quantiles(model, backend, table, list(LEVELS))
    if curves.shape != (len(LEVELS), len(scored)) or not np.all(np.isfinite(curves)):
        raise ValueError("TabPFN returned invalid value estimates")
    low, mid, high = np.exp(np.maximum.accumulate(curves, axis=0))
    estimates = {
        row["player_id"]: (float(low[i]), float(mid[i]), float(high[i]))
        for i, row in enumerate(missing)
        if all(math.isfinite(v) and v > 0 for v in (low[i], mid[i], high[i]))
    }
    report: dict[str, Any] = {
        "version": VALUE_MODEL_VERSION,
        "backend": backend,
        "trained_on": len(train),
        "estimated": len(estimates),
    }
    if check:
        offset = len(missing)
        actual = np.array([row["value_eur"] for row in check], dtype=float)
        report["check_on_known_values"] = check_report(
            actual, low[offset:], mid[offset:], high[offset:]
        )
    return estimates, report
