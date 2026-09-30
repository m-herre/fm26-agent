"""Two readings of "undervalued", both built on TabPFN.

Fair value: what a player's visible profile is worth. Players with a stored price are split into
two halves; each half is priced by a TabPFN log-value regressor fitted on the other half, so no
player is ever priced by a model that saw his own price (cross-fitting). The result is a full
distribution per player, so "priced at most 80% of his fair value" comes with a real chance.

Peers: his price against the median price of players at his position whose estimated current
ability (the agent-built current-ability task, never the hidden true value) is within 5 points.
A comparison with no distribution of its own.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from . import tabpfn_backend
from .features import FeatureSchema
from .prediction import HIGH_LEVEL, LOW_LEVEL, MEDIAN, QUANTILES
from .targets import get_target
from .value_model import check_report, value_table
from .visible_db import VisibleStore

FAIR_VALUE_VERSION = 1
FIT_ROWS = 10_000
MIN_PRICED = 200
PEER_WINDOW = 5.0  # current-ability points either side
MIN_PEERS = 10

Curves = dict[int, np.ndarray]


def cross_fitted_curves(
    priced: Sequence[dict[str, Any]], schema: FeatureSchema, backend: str, seed: int
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """log(fair value) percentiles for every priced player, each from the half he wasn't in."""
    players = sorted(priced, key=lambda row: row["player_id"])
    random.Random(seed).shuffle(players)
    halves = (players[0::2], players[1::2])
    curves: dict[int, np.ndarray] = {}
    for fit_half, score_half in ((halves[0], halves[1]), (halves[1], halves[0])):
        train = fit_half[:FIT_ROWS]
        model = tabpfn_backend.new_regressor(backend, seed)
        target = np.log(np.array([row["value_eur"] for row in train], dtype=float))
        tabpfn_backend.fit(model, backend, value_table(schema, train), target)
        values = tabpfn_backend.predict_quantiles(
            model, backend, value_table(schema, score_half), list(QUANTILES)
        )
        if values.shape != (len(QUANTILES), len(score_half)) or not np.all(np.isfinite(values)):
            raise ValueError("TabPFN returned invalid fair values")
        values = np.maximum.accumulate(values, axis=0)
        for row, curve in zip(score_half, values.T, strict=True):
            curves[row["player_id"]] = curve
        del model
        tabpfn_backend.free_memory()
    ids = sorted(curves)
    actual = np.array([row["value_eur"] for row in sorted(players, key=lambda r: r["player_id"])])
    matrix = np.exp(np.array([curves[i] for i in ids]).T)
    report = {
        "version": FAIR_VALUE_VERSION,
        "cross_fitted": True,
        "fitted_on_per_half": min(FIT_ROWS, len(halves[0])),
        "priced_players": len(ids),
        "check": check_report(actual, matrix[LOW_LEVEL], matrix[MEDIAN], matrix[HIGH_LEVEL]),
    }
    return curves, report


def ratio_curve(price: float, log_fair: np.ndarray) -> np.ndarray:
    """Percentiles of price / fair value. The ratio falls as fair value rises, so the p-th
    percentile of the ratio is price divided by the (1-p)-th percentile of fair value."""
    low, high = get_target("price_vs_fair_value").scale
    return np.clip(price / np.exp(log_fair[::-1]), low, high)


class FairValues:
    """Computes (once per save, lazily) and serves the two value targets."""

    def __init__(
        self,
        store: VisibleStore,
        schema: FeatureSchema,
        backend: str,
        seed: int,
        current_ability: Callable[[Sequence[dict[str, Any]]], Curves],
        progress: Callable[[str], None] | None = None,
    ):
        self.store, self.schema, self.backend, self.seed = store, schema, backend, seed
        self.current_ability = current_ability
        self.progress = progress
        self._priced: list[dict[str, Any]] | None = None

    def priced(self) -> list[dict[str, Any]]:
        if self._priced is None:
            self._priced = self.store.priced_players()
        return self._priced

    def available(self) -> bool:
        return len(self.priced()) >= MIN_PRICED

    def report(self) -> dict[str, Any]:
        """Build the fair values if this save has none yet; return their quality report."""
        metadata = self.store.metadata()
        if metadata.get("fair_value_version") != FAIR_VALUE_VERSION:
            if not self.available():
                raise ValueError("This save has too few players with a price to learn values")
            if self.progress:
                self.progress("building task price_vs_fair_value")
            ids = [row["player_id"] for row in self.priced()]
            players = self.store.get_players(ids)
            curves, report = cross_fitted_curves(players, self.schema, self.backend, self.seed)
            self.store.set_fair_values(curves)
            self.store.set_metadata("fair_value_report", report)
            self.store.set_metadata("fair_value_version", FAIR_VALUE_VERSION)
            metadata = self.store.metadata()
        return metadata["fair_value_report"]

    def fair_value_curves(self, players: Sequence[dict[str, Any]]) -> Curves:
        """Percentiles of price / fair value for players with a stored price."""
        self.report()
        stored = self.store.fair_values([row["player_id"] for row in players])
        return {
            row["player_id"]: ratio_curve(row["value_eur"], np.asarray(stored[row["player_id"]]))
            for row in players
            if row.get("value_eur") and row["player_id"] in stored
        }

    def fair_values(
        self, players: Sequence[dict[str, Any]]
    ) -> dict[int, tuple[float, float, float]]:
        """(low, mid, high) fair value in internal units, for display."""
        stored = self.store.fair_values([row["player_id"] for row in players])
        result = {}
        for player_id, curve in stored.items():
            values = np.exp(np.asarray(curve))
            result[player_id] = (values[LOW_LEVEL], values[MEDIAN], values[HIGH_LEVEL])
        return result

    def peer_curves(self, players: Sequence[dict[str, Any]]) -> Curves:
        """price / median price of priced players sharing a natural position whose estimated
        current ability is within PEER_WINDOW. A constant curve: it's a comparison."""
        candidates = [row for row in players if row.get("value_eur")]
        positions = {pos for row in candidates for pos in row["natural_positions"]}
        peers = [row for row in self.priced() if positions.intersection(row["natural_positions"])]
        if not candidates or not peers:
            return {}
        everyone = {row["player_id"]: row for row in [*peers, *candidates]}
        ability = self.current_ability(self.store.get_players(sorted(everyone)))
        median_ca = {pid: float(curve[MEDIAN]) for pid, curve in ability.items()}
        by_position: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for position in positions:
            group = [
                row
                for row in peers
                if position in row["natural_positions"] and row["player_id"] in median_ca
            ]
            group.sort(key=lambda row: median_ca[row["player_id"]])
            by_position[position] = (
                np.array([median_ca[row["player_id"]] for row in group]),
                np.array([row["value_eur"] for row in group], dtype=float),
                np.array([row["player_id"] for row in group]),
            )
        low, high = get_target("price_vs_peers").scale
        result: Curves = {}
        for row in candidates:
            if row["player_id"] not in median_ca:
                continue
            own = median_ca[row["player_id"]]
            prices: dict[int, float] = {}
            for position in row["natural_positions"]:
                abilities, values, ids = by_position[position]
                start = np.searchsorted(abilities, own - PEER_WINDOW, "left")
                end = np.searchsorted(abilities, own + PEER_WINDOW, "right")
                for pid, value in zip(ids[start:end], values[start:end], strict=True):
                    if pid != row["player_id"]:
                        prices[int(pid)] = float(value)
            if len(prices) >= MIN_PEERS:
                ratio = row["value_eur"] / float(np.median(list(prices.values())))
                result[row["player_id"]] = np.full(len(QUANTILES), np.clip(ratio, low, high))
        return result
