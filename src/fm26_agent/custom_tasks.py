"""Prediction tasks the agent defines itself, fitted with TabPFN on the spot.

The agent picks a target from the glossary (targets.py), e.g. current ability for "a striker in
his prime", and optionally a threshold ("consistency 15 or better"). Code does the rest:

1. Fit a TabPFN-3.5 regressor on the 10,000 reference players' visible features against that
   target. No training loop or tuning, which is why a model can be made mid-conversation.
2. Check it on 2,000 held-out players and compare it with guessing the average, so every task
   reports how far it can be trusted before anyone relies on it.
3. Predict the candidates: estimate, 80% range and, with a threshold, the chance of meeting it.

The agent only ever sees the glossary and the quality report, never training rows or the spread
of a target. Features are the fixed visible schema; targets are forbidden features.
"""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import tabpfn_backend
from .features import FeatureSchema
from .prediction import HIGH_LEVEL, LOW_LEVEL, MEDIAN, QUANTILES, chance_of_reaching
from .private_db import PrivateStore
from .targets import VALUE_TARGETS, Target, get_target
from .visible_db import VisibleStore

TASK_VERSION = 1
CHECK_ROWS = 2_000
MIN_TRAIN = 200
PEER_WINDOW = 5.0  # mirrors fair_value.PEER_WINDOW, for the report text
KEPT_IN_MEMORY = 1  # fitted models held at once; others reload from disk (GPU memory is the limit)
# How much better than guessing the average a task must be, measured on held-out players.
USEFUL_GAIN = 0.15
WEAK_GAIN = 0.05

ESTIMATE, LOW, HIGH, CHANCE = "estimate", "low", "high", "chance"


@dataclass(frozen=True)
class TaskSpec:
    """What to predict: a glossary target and, optionally, the level that counts as good enough."""

    target: str
    threshold: float | None = None

    def __post_init__(self) -> None:
        target = get_target(self.target)
        if self.threshold is not None:
            low, high = target.scale
            if not low <= self.threshold <= high:
                raise ValueError(
                    f"threshold for {self.target} must be between {low:g} and {high:g}"
                )

    @property
    def info(self) -> Target:
        return get_target(self.target)

    @property
    def task_id(self) -> str:
        return self.target if self.threshold is None else f"{self.target}@{self.threshold:g}"

    @classmethod
    def from_id(cls, task_id: str) -> TaskSpec:
        target, _, threshold = task_id.partition("@")
        try:
            return cls(target, float(threshold) if threshold else None)
        except ValueError as exc:
            raise ValueError(f"Unknown task_id {task_id!r}; build it first") from exc

    def describe_goal(self) -> str:
        """e.g. 'consistency 15 or better' (low-is-good targets count downwards)."""
        target = self.info
        if self.threshold is None:
            return f"{'lowest' if target.better == 'low' else 'highest'} {target.label}"
        side = "or lower" if target.better == "low" else "or higher"
        return f"{target.label} {self.threshold:g} {side}"


def quality_report(
    actual: np.ndarray, curves: np.ndarray, baseline: float, target: Target
) -> dict[str, Any]:
    """How good a task's model is on held-out players, in terms a player can follow."""
    mid, low, high = curves[MEDIAN], curves[LOW_LEVEL], curves[HIGH_LEVEL]
    error = float(np.mean(np.abs(mid - actual)))
    guess = float(np.mean(np.abs(baseline - actual)))
    gain = 1 - error / guess if guess else 0.0
    order = np.argsort(np.argsort(mid))
    truth = np.argsort(np.argsort(actual))
    spearman = float(np.corrcoef(order, truth)[0, 1]) if len(actual) > 2 else 0.0
    verdict = (
        "useful" if gain >= USEFUL_GAIN else "weak" if gain >= WEAK_GAIN else "not predictable"
    )
    return {
        "checked_on": int(len(actual)),
        "average_error": round(error, 2),
        "average_error_if_guessing": round(guess, 2),
        "better_than_guessing": round(gain, 3),
        "rank_correlation": round(spearman, 3),
        "range_coverage_80": round(float(np.mean((actual >= low) & (actual <= high))), 3),
        "scale": list(target.scale),
        "verdict": verdict,
    }


def chance_of(curve: np.ndarray, level: float, side: str, whole: bool = True) -> float:
    """Chance (0-1) that the value is at least / at most `level`, read off the percentiles.

    On a whole-number scale "15 or better" counts from 14.5 (or up to 15.5). Whole numbers make
    percentiles repeat, so ties are broken by a hair to keep the interpolation well defined.
    """
    strict = curve + np.arange(len(curve)) * 1e-6
    margin = 0.5 if whole else 0.0
    if side == "at_least":
        return chance_of_reaching(strict, level - margin)
    return round(1 - chance_of_reaching(strict, level + margin), 3)


def threshold_chance(curve: np.ndarray, threshold: float, better: str) -> float:
    """Chance of meeting a threshold in the target's good direction (see chance_of)."""
    return chance_of(curve, threshold, "at_least" if better == "high" else "at_most")


def fair_value_quality(report: dict[str, Any]) -> dict[str, Any]:
    """The fair-value model's self-check in the shape of a task report."""
    check = report.get("check", {})
    r2 = check.get("r2_log_value") or 0.0
    return {
        "target": "price_vs_fair_value",
        "trained_on": report.get("fitted_on_per_half"),
        "checked_on": report.get("priced_players"),
        "cross_fitted": True,
        "typical_error_percent": check.get("median_error_percent"),
        "range_coverage_80": check.get("range_coverage_80"),
        "r2_log_value": r2,
        "verdict": "useful" if r2 >= 0.5 else "weak" if r2 >= 0.2 else "not predictable",
    }


def check_sample(held_out: dict[int, float], seed: int) -> list[int]:
    """The held-out players a task is checked on (the same ones for any comparison)."""
    ids = sorted(held_out)
    random.Random(seed).shuffle(ids)
    return sorted(ids[:CHECK_ROWS])


class TaskLab:
    """Builds, checks, caches and runs agent-defined TabPFN tasks for one prepared save."""

    def __init__(
        self,
        visible: VisibleStore,
        private: PrivateStore,
        schema: FeatureSchema,
        backend: str,
        folder: Path,
        *,
        seed: int = 42,
        progress: Callable[[str], None] | None = None,
    ):
        self.visible, self.private, self.schema = visible, private, schema
        self.backend, self.folder, self.seed = backend, folder, seed
        self.progress = progress
        self.preparation_id = visible.metadata().get("preparation_id", "")
        self._models: OrderedDict[str, Any] = OrderedDict()
        self._reports: dict[str, dict[str, Any]] = {}
        self._available: list[str] | None = None
        self._cache: dict[str, dict[int, np.ndarray]] = {}  # predicted percentiles per player
        self.fits = 0  # TabPFN fits made by this lab (cached tasks cost none)
        from .fair_value import FairValues

        self.values = FairValues(
            visible,
            schema,
            backend,
            seed,
            current_ability=lambda players: self.distribution("current_ability", players),
            progress=self._say,
        )

    def available(self, include_value: bool = True) -> list[str]:
        """Targets this save can predict. Value targets need enough priced players (and, for
        peers, current ability)."""
        if self._available is None:
            base = self.private.available_targets()
            value = []
            if self.values.available():
                value.append("price_vs_fair_value")
                if "current_ability" in base:
                    value.append("price_vs_peers")
            self._available = base + value
        if include_value:
            return self._available
        return [name for name in self._available if name not in VALUE_TARGETS]

    def build(self, spec: TaskSpec) -> dict[str, Any]:
        """Fit (or reuse) the model for spec.target and return its quality report."""
        if spec.target not in self.available():
            raise ValueError(f"This save has no {spec.info.label} values to learn from")
        if spec.target in self._reports:
            return self._reports[spec.target]
        if spec.target == "price_vs_fair_value":
            self._reports[spec.target] = fair_value_quality(self.values.report())
        elif spec.target == "price_vs_peers":
            ability = self.build(TaskSpec("current_ability"))
            self._reports[spec.target] = {
                "target": spec.target,
                "verdict": "comparison",
                "peers": f"same position, estimated current ability within "
                f"{PEER_WINDOW:g} points (itself off by {ability.get('average_error', '?')} "
                "on average)",
            }
        elif not self._load(spec.target):
            self._fit(spec.target)
        return self._reports[spec.target]

    def build_quality(self, target: str) -> dict[str, Any]:
        """The quality report for a target (builds it if needed)."""
        return self.build(TaskSpec(target))

    def distribution(self, target: str, players: Sequence[dict[str, Any]]) -> dict[int, np.ndarray]:
        """Predicted percentiles (QUANTILES) per player. Players a value target can't judge (no
        stored price, too few peers) are left out. Cached for the session."""
        self.build(TaskSpec(target))
        cache = self._cache.setdefault(target, {})
        missing = [row for row in players if row["player_id"] not in cache]
        if missing:
            if target == "price_vs_fair_value":
                found = self.values.fair_value_curves(missing)
            elif target == "price_vs_peers":
                found = self.values.peer_curves(missing)
            else:
                curves = self._curves(self._model(target), missing, get_target(target))
                found = {
                    row["player_id"]: curve for row, curve in zip(missing, curves.T, strict=True)
                }
            for row in missing:  # remember "no data" too, so it isn't asked again
                cache[row["player_id"]] = found.get(row["player_id"])
        return {
            row["player_id"]: cache[row["player_id"]]
            for row in players
            if cache.get(row["player_id"]) is not None
        }

    def predict(self, spec: TaskSpec, players: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Estimate, 80% range and (with a threshold) chance of meeting it, per player."""
        self.build(spec)
        if not players:
            return []
        target = spec.info
        found = self.distribution(spec.target, players)
        rows = []
        for row, curve in (
            (row, found[row["player_id"]]) for row in players if row["player_id"] in found
        ):
            item = {
                "player_id": row["player_id"],
                ESTIMATE: float(curve[MEDIAN]),
                LOW: float(curve[LOW_LEVEL]),
                HIGH: float(curve[HIGH_LEVEL]),
            }
            if spec.threshold is not None:
                side = "at_least" if target.better == "high" else "at_most"
                item[CHANCE] = chance_of(curve, spec.threshold, side, target.whole)
            rows.append(item)
        return rows

    # -- fitting and storage -------------------------------------------------------------

    def _say(self, message: str) -> None:
        if self.progress:
            self.progress(message)

    def _curves(self, model: Any, players: Sequence[dict[str, Any]], target: Target) -> np.ndarray:
        values = tabpfn_backend.predict_quantiles(
            model, self.backend, self.schema.transform(players), list(QUANTILES)
        )
        if values.shape != (len(QUANTILES), len(players)) or not np.all(np.isfinite(values)):
            raise ValueError("TabPFN returned invalid estimates")
        return np.clip(np.maximum.accumulate(values, axis=0), *target.scale)

    def _fit(self, name: str) -> None:
        target = get_target(name)
        train = self.private.target_values(name, "train")
        if len(train) < MIN_TRAIN:
            raise ValueError(f"Too few players with a known {target.label} to learn from")
        self._say(f"building task {name}")
        ids = sorted(train)
        players = self.visible.get_players(ids)
        values = np.array([train[player_id] for player_id in ids], dtype=float)
        model = tabpfn_backend.new_regressor(self.backend, self.seed)
        tabpfn_backend.fit(model, self.backend, self.schema.transform(players), values)
        self.fits += 1
        held_out = self.private.target_values(name, "test")
        check_ids = check_sample(held_out, self.seed)
        if check_ids:
            curves = self._curves(model, self.visible.get_players(check_ids), target)
            actual = np.array([held_out[player_id] for player_id in check_ids], dtype=float)
            report = quality_report(actual, curves, float(np.mean(values)), target)
        else:
            report = {"checked_on": 0, "verdict": "unchecked", "scale": list(target.scale)}
        report = {"target": name, "trained_on": len(ids), **report}
        self._remember(name, model, report)
        self._save(name, model, report)

    def _remember(self, name: str, model: Any, report: dict[str, Any]) -> None:
        self._reports[name] = report
        self._models[name] = model
        self._models.move_to_end(name)
        while len(self._models) > KEPT_IN_MEMORY:
            self._models.popitem(last=False)
            tabpfn_backend.free_memory()

    def _model(self, name: str) -> Any:
        if name not in self._models and not self._load(name):
            self._fit(name)
        self._models.move_to_end(name)
        return self._models[name]

    def _record(self, name: str) -> Path:
        return self.folder / f"{name}.json"

    def _identity(self, name: str) -> dict[str, Any]:
        return {
            "version": TASK_VERSION,
            "target": name,
            "model_version": "v3.5",
            "preparation_id": self.preparation_id,
            "feature_fingerprint": self.schema.fingerprint,
            "backend": self.backend,
            "seed": self.seed,
        }

    def _save(self, name: str, model: Any, report: dict[str, Any]) -> None:
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            path = self._record(name)
            stored = tabpfn_backend.save_fitted(model, self.backend, path)
            path.write_text(
                json.dumps(self._identity(name) | {"model": stored, "report": report}, indent=2),
                encoding="utf-8",
            )
        except Exception:  # a task that can't be stored is simply fitted again next time
            pass

    def _load(self, name: str) -> bool:
        path = self._record(name)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if {key: payload.get(key) for key in self._identity(name)} != self._identity(name):
                return False
            model = tabpfn_backend.load_fitted(payload["model"], self.backend, path)
        except Exception:
            return False
        self._remember(name, model, payload["report"])
        return True
